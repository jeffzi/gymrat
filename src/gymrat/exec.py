"""Run shell commands as asyncio subprocesses with bounded, decoded capture.

``exec`` spawns a command through the shell, captures stdout and stderr
separately as decoded text plus raw byte counts, and settles when the child's
stdio pipes close — not merely when the shell exits, so output flushed by a
background descendant is still captured. A run is bounded three ways:

- a per-stream 64 MiB text cap (byte counts keep counting past it),
- an optional timeout that resolves an :class:`ExecTimeoutError` value, and
- an optional abort :class:`asyncio.Event` that resolves a failed
  :class:`ExecResult`.

Timeout and abort snapshot whatever has been captured so far and kill the whole
process group, so a grandchild the shell left running is reaped too. Every
returned value is a frozen dataclass, so a caller cannot mutate one run's result
into a landmine for the next.
"""

import asyncio
import codecs
import contextlib
from collections.abc import Iterable
from dataclasses import dataclass

from gymrat.process_group import current_platform, kill_process_group
from gymrat.signals import deferring_termination_signals

FAILURE_EXIT_CODE = 1
"""Exit code reported when a run fails without a positive child exit code."""

OUTPUT_CAP = 64 * 1024 * 1024
"""Per-stream cap, in bytes, on retained text. Byte counts keep counting past it."""

_READ_CHUNK = 65536
"""Bytes requested per pipe read."""

_live_process_groups: set[int] = set()
"""Process-group leader PIDs of bench children currently alive under :func:`exec`.

A PID is present only for the child's lifetime: it is added once the spawn
succeeds and removed on every settle path (normal, abort, timeout, reader
error). :func:`kill_live_process_groups` reads it to tear down surviving groups
on the signal path before a worktree sweep runs.
"""


def kill_live_process_groups() -> None:
    """Kill every process group with a live bench child, never raising.

    Iterates a snapshot so a run settling on another task can deregister its PID
    mid-sweep without disturbing the loop. :func:`kill_process_group` already
    tolerates an already-dead group and warns on other failures; the extra guard
    keeps an unexpected ``OSError`` from escaping into the signal-path cleanup
    that calls this.
    """
    for pid in list(_live_process_groups):
        with contextlib.suppress(OSError):
            kill_process_group(pid)


def reset_live_process_groups(groups: Iterable[int] = ()) -> set[int]:
    """Replace the live-group registry, returning what it held before.

    Test-only escape hatch for isolating and seeding :data:`_live_process_groups`
    across test runs without reaching into module-private state directly.
    """
    previous = set(_live_process_groups)
    _live_process_groups.clear()
    _live_process_groups.update(groups)
    return previous


@dataclass(frozen=True, slots=True)
class ExecOptions:
    """Inputs for a single :func:`exec` run.

    Args:
        cwd: Working directory the command runs in.
        timeout_ms: Wall-clock budget in milliseconds; ``None`` waits forever.
        abort: Event that, once set, kills the run and resolves a failed result.
        stdin: Text delivered to the command's standard input, then closed;
            ``None`` gives an immediately closed (EOF) input.
    """

    cwd: str
    timeout_ms: int | None = None
    abort: asyncio.Event | None = None
    stdin: str | None = None


@dataclass(frozen=True, slots=True)
class ExecResult:
    """A completed run's captured output and exit code."""

    stdout: str
    stderr: str
    exit_code: int
    stdout_bytes: int
    stderr_bytes: int


@dataclass(frozen=True, slots=True)
class ExecTimeoutError:
    """A run that exceeded its timeout, with whatever was captured beforehand.

    This is a returned value, not a raised exception: callers distinguish it from
    :class:`ExecResult` by type (``isinstance``).
    """

    stdout: str
    stderr: str
    timeout_ms: int
    stdout_bytes: int
    stderr_bytes: int


@dataclass(slots=True)
class OutputBuffer:
    """Accumulates decoded text up to :data:`OUTPUT_CAP` while counting all bytes.

    A chunk is appended only while the bytes received *before* it are still under
    the cap; the chunk that crosses the cap is kept whole, and every later chunk
    is dropped from the text. ``byte_count`` always reflects every byte received,
    so a caller can tell that output was truncated.
    """

    text: str = ""
    byte_count: int = 0

    def append(self, chunk: str, chunk_bytes: int) -> None:
        """Add ``chunk`` (worth ``chunk_bytes`` raw bytes) subject to the cap."""
        self.byte_count += chunk_bytes
        if self.byte_count - chunk_bytes >= OUTPUT_CAP:
            return
        self.text += chunk

    def append_failure(self, message: str) -> None:
        """Append a failure ``message`` and newline unconditionally, past the cap.

        An error explaining why a run failed must always reach the caller, so it
        bypasses the truncation cap that governs ordinary output.
        """
        line = f"{message}\n"
        self.text += line
        self.byte_count += len(line.encode())


def _exit_code(returncode: int | None) -> int:
    """Map a child's raw return code to a reported exit code.

    A signal kill surfaces as a negative return code, and a child whose status
    has not been collected yet as ``None``; both collapse to
    :data:`FAILURE_EXIT_CODE` rather than leaking a negative or missing value.
    """
    if returncode is None or returncode < 0:
        return FAILURE_EXIT_CODE
    return returncode


def _terminate(proc: asyncio.subprocess.Process) -> None:
    """Tear down a run that will not settle on its own.

    Killing the process group stops the child and any descendant it left
    running; dropping the stdio pipes then releases the reader tasks still
    waiting on EOF, so the run can be snapshotted without awaiting a natural
    end of stream.
    """
    kill_process_group(proc.pid)
    _close_pipes(proc)


def _close_pipes(proc: asyncio.subprocess.Process) -> None:
    """Close the child's stdio pipes so a surviving descendant cannot grow buffers.

    Closing the transport drops the read ends, and feeding EOF to each reader
    settles them deterministically for a snapshot taken without waiting for the
    natural end of stream.
    """
    # asyncio.subprocess.Process exposes no public accessor for its pipe
    # transports; reaching the undocumented transport via getattr is the only way
    # to drop the read ends before the child is reaped.
    transport = getattr(proc, "_transport", None)
    if transport is not None:
        transport.close()
    for reader in (proc.stdout, proc.stderr):
        if reader is not None and not reader.at_eof():
            reader.feed_eof()


async def _read_stream(reader: asyncio.StreamReader, buffer: OutputBuffer) -> None:
    """Read ``reader`` to EOF, decoding UTF-8 incrementally into ``buffer``.

    One incremental decoder per stream reassembles a multi-byte character split
    across pipe reads; the raw byte length of each read drives the byte counts.
    """
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    while True:
        chunk = await reader.read(_READ_CHUNK)
        if not chunk:
            tail = decoder.decode(b"", final=True)
            if tail:
                buffer.append(tail, 0)
            return
        buffer.append(decoder.decode(chunk), len(chunk))


async def _feed_stdin(proc: asyncio.subprocess.Process, data: str | None) -> None:
    """Write ``data`` to the child's stdin then close it, swallowing broken pipes.

    A child that exits without reading a large payload breaks the write; that is
    an expected end-of-run condition, not an error to surface.
    """
    stdin = proc.stdin
    if stdin is None:
        return
    try:
        if data:
            stdin.write(data.encode())
            await stdin.drain()
    except (BrokenPipeError, ConnectionResetError):
        pass
    finally:
        with contextlib.suppress(BrokenPipeError, ConnectionResetError, OSError):
            stdin.close()


async def _await_normal(
    proc: asyncio.subprocess.Process,
    stdout_task: asyncio.Task[None],
    stderr_task: asyncio.Task[None],
) -> None:
    """Wait for both readers to reach EOF and the child to be reaped.

    Settling on stdio close rather than process exit is what captures output a
    background descendant flushes after the shell itself has returned.
    """
    await asyncio.gather(stdout_task, stderr_task)
    await proc.wait()


async def _cancel_all(tasks: list[asyncio.Task[object]]) -> None:
    """Cancel every task and await their settling, discarding their outcomes."""
    for task in tasks:
        if not task.done():
            task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)


def _build_result(stdout_buf: OutputBuffer, stderr_buf: OutputBuffer, exit_code: int) -> ExecResult:
    return ExecResult(
        stdout_buf.text,
        stderr_buf.text,
        exit_code,
        stdout_buf.byte_count,
        stderr_buf.byte_count,
    )


async def exec(command: str, options: ExecOptions) -> ExecResult | ExecTimeoutError:  # noqa: A001 -- names the subprocess executor `exec`
    """Run ``command`` through the shell and capture its output.

    The run never raises for a spawn or child failure: a shell that cannot be
    spawned, a non-zero exit, a signal kill, an abort, and a stream error all
    resolve as an :class:`ExecResult`. Only exceeding ``options.timeout_ms``
    resolves the distinct :class:`ExecTimeoutError` value.

    Args:
        command: The shell command line to run.
        options: Working directory, timeout, abort event, and stdin.

    Returns:
        An :class:`ExecResult` for any completed or cancelled run, or an
        :class:`ExecTimeoutError` when the timeout is exceeded.
    """
    if options.abort is not None and options.abort.is_set():
        return ExecResult("", "", FAILURE_EXIT_CODE, 0, 0)

    # Mask termination signals across the spawn + registration pair so a
    # signal delivered between the two still finds the child in the live
    # registry when the deferred handler fires kill_live_process_groups.
    try:
        with deferring_termination_signals():
            proc = await asyncio.create_subprocess_shell(
                command,
                cwd=options.cwd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=current_platform() != "win32",
            )
            # Register the group leader before lifting the mask; the enclosing
            # finally guarantees deregistration even if a later step raises.
            _live_process_groups.add(proc.pid)
    except OSError as error:
        stderr = f"{error}\n"
        return ExecResult("", stderr, FAILURE_EXIT_CODE, 0, len(stderr.encode()))
    try:
        stdout_buf = OutputBuffer()
        stderr_buf = OutputBuffer()
        assert proc.stdout is not None  # noqa: S101 -- guaranteed by stdout=PIPE
        assert proc.stderr is not None  # noqa: S101 -- guaranteed by stderr=PIPE
        stdout_task = asyncio.create_task(_read_stream(proc.stdout, stdout_buf))
        stderr_task = asyncio.create_task(_read_stream(proc.stderr, stderr_buf))
        stdin_task = asyncio.create_task(_feed_stdin(proc, options.stdin))
        normal_task = asyncio.create_task(_await_normal(proc, stdout_task, stderr_task))
        abort_task = (
            asyncio.create_task(options.abort.wait()) if options.abort is not None else None
        )

        waiters: list[asyncio.Task[object]] = [normal_task]
        if abort_task is not None:
            waiters.append(abort_task)
        timeout = options.timeout_ms / 1000 if options.timeout_ms is not None else None

        try:
            done, _ = await asyncio.wait(
                waiters,
                timeout=timeout,
                return_when=asyncio.FIRST_COMPLETED,
            )

            if normal_task in done:
                reader_error = normal_task.exception()
                if reader_error is not None:
                    _terminate(proc)
                    stderr_buf.append_failure(str(reader_error))
                    return _build_result(stdout_buf, stderr_buf, FAILURE_EXIT_CODE)
                return _build_result(stdout_buf, stderr_buf, _exit_code(proc.returncode))

            _terminate(proc)
            if abort_task is not None and abort_task in done:
                return _build_result(stdout_buf, stderr_buf, FAILURE_EXIT_CODE)
            if options.timeout_ms is not None:
                return ExecTimeoutError(
                    stdout_buf.text,
                    stderr_buf.text,
                    options.timeout_ms,
                    stdout_buf.byte_count,
                    stderr_buf.byte_count,
                )
            # asyncio.wait only returns empty-handed when a timeout elapsed, so with
            # no timeout set this branch cannot be reached.
            msg = "exec settled without a normal, abort, or timeout outcome"
            raise RuntimeError(msg)
        finally:
            pending: list[asyncio.Task[object]] = [
                stdout_task,
                stderr_task,
                stdin_task,
                normal_task,
            ]
            if abort_task is not None:
                pending.append(abort_task)
            await _cancel_all(pending)
    finally:
        _live_process_groups.discard(proc.pid)
