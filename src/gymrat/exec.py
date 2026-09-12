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
process group, so a grandchild the shell left running is killed too. Every
returned value is a frozen dataclass, so a caller cannot mutate one run's result
into a landmine for the next.
"""

import asyncio
import codecs
import contextlib
import signal as _signal_module
import sys
from dataclasses import dataclass, field

from gymrat.process_group import kill_process_group
from gymrat.signals import TERMINATION_SIGNALS, deferring_termination_signals, pthread_sigmask

FAILURE_EXIT_CODE = 1
"""Exit code reported when a run fails without a positive child exit code."""

OUTPUT_CAP = 64 * 1024 * 1024
"""Per-stream cap, in bytes, on retained text. Byte counts keep counting past it."""

READ_CHUNK = 65536
"""Bytes requested per pipe read."""

_CANCEL_REAP_TIMEOUT_S = 2.0
"""Seconds an interrupted run waits for its killed child to be reaped before giving up."""

_live_process_groups: set[int] = set()
"""Process-group leader PIDs of bench children currently alive under :func:`exec`.

A PID is present only for the child's lifetime: it is added once the spawn
succeeds and removed on every settle path (normal, abort, timeout, reader
error, cancellation). :func:`kill_live_process_groups` reads it to tear down
surviving groups on the signal path before a worktree sweep runs.
"""


def _child_reset_signal_mask() -> None:
    """Unblock termination signals in the child, reversing the parent's mask."""
    if pthread_sigmask is not None:
        pthread_sigmask(_signal_module.SIG_UNBLOCK, TERMINATION_SIGNALS)


def kill_live_process_groups() -> None:
    """Kill every process group with a live bench child, never raising.

    Iterates a snapshot so a run settling on another task can deregister its PID
    mid-sweep without disturbing the loop. :func:`kill_process_group` already
    tolerates an already-dead group and warns on other failures; the extra guard
    keeps any other unexpected exception (for example a warning escalated to an
    error by a warnings filter) from escaping into the signal-path cleanup that
    calls this.
    """
    for pid in list(_live_process_groups):
        with contextlib.suppress(Exception):
            kill_process_group(pid)


@dataclass(frozen=True, slots=True)
class ExecOptions:
    """Inputs for a single :func:`exec` run.

    Attributes:
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

    Internally the buffer accumulates into a list and joins on access, avoiding
    O(n^2) re-copy from repeated string concatenation toward the 64 MiB cap.
    """

    _chunks: list[str] = field(default_factory=list)
    byte_count: int = 0

    def append(self, chunk: str, chunk_bytes: int) -> None:
        """Add ``chunk`` (worth ``chunk_bytes`` raw bytes) subject to the cap.

        Args:
            chunk: The decoded text to append.
            chunk_bytes: The raw byte count the chunk is worth, counted toward
                :data:`OUTPUT_CAP` even when the chunk itself is dropped past it.
        """
        self.byte_count += chunk_bytes
        if self.byte_count - chunk_bytes >= OUTPUT_CAP:
            return
        self._chunks.append(chunk)

    def append_failure(self, message: str) -> None:
        """Append a failure ``message`` and newline unconditionally, past the cap.

        An error explaining why a run failed must always reach the caller, so it
        bypasses the truncation cap that governs ordinary output. The diagnostic
        is not counted as command output — ``byte_count`` is left unchanged.

        Args:
            message: The failure text to append.
        """
        self._chunks.append(f"{message}\n")

    @property
    def text(self) -> str:
        """The accumulated text, joined from internal chunks."""
        return "".join(self._chunks)


def _exit_code(returncode: int | None) -> int:
    """Map a child's raw return code to a reported exit code.

    A signal kill surfaces as a negative return code, and a child whose status
    has not been collected yet as ``None``; both collapse to
    :data:`FAILURE_EXIT_CODE` rather than leaking a negative or missing value.

    Returns:
        The non-negative exit code, or :data:`FAILURE_EXIT_CODE` for abnormal
        terminations.
    """
    if returncode is None or returncode < 0:
        return FAILURE_EXIT_CODE
    return returncode


def _terminate(proc: asyncio.subprocess.Process) -> bool:
    """Tear down a run that will not settle on its own.

    Killing the process group stops the child and any descendant it left
    running; dropping the stdio pipes then releases the reader tasks still
    waiting on EOF, so the run can be snapshotted without awaiting a natural
    end of stream.

    Args:
        proc: The child whose process group to kill.

    Returns:
        Whether the group kill was refused and deferred, as
        :func:`kill_process_group` reports it.
    """
    refused = kill_process_group(proc.pid, defer_refusal=True)
    _close_pipes(proc)
    return refused


async def _terminate_and_reap(
    proc: asyncio.subprocess.Process,
    *,
    reap_timeout: float | None = None,
) -> None:
    """Tear down a run that will not settle on its own, then reap the child.

    A group that refused the kill because its members were all still exiting is
    signaled once more after the reap, which stays silent when the group is gone
    and warns only when the refusal is genuine. When the reap does not land
    within ``reap_timeout``, the group is signaled again anyway, so a refusal
    that outlasts the wait still warns.

    Args:
        proc: The child whose process group to kill and reap.
        reap_timeout: Seconds to wait for the reap, or ``None`` to wait until it
            lands.
    """
    refused = _terminate(proc)
    if reap_timeout is None:
        await proc.wait()
    else:
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(proc.wait(), reap_timeout)
    if refused:
        kill_process_group(proc.pid)


async def _terminate_reap_and_drain(
    proc: asyncio.subprocess.Process,
    stdout_task: asyncio.Task[None],
    stderr_task: asyncio.Task[None],
) -> None:
    """Terminate and reap the child, then await the readers it just fed EOF.

    Both settle paths that abandon the normal wait -- a reader error and a
    timeout/abort -- need the same sequence: kill and reap first, then let the
    readers, unblocked by the pipe close, finish before the outcome is built.
    """
    await _terminate_and_reap(proc)
    await asyncio.gather(stdout_task, stderr_task, return_exceptions=True)


def _close_pipes(proc: asyncio.subprocess.Process) -> None:
    """Close the child's stdio pipes so a surviving descendant cannot grow buffers.

    Closing the pipe transports drops the read ends, and feeding EOF to each
    reader settles them deterministically for a snapshot taken without waiting
    for the natural end of stream.

    Only the pipe transports are closed, never the subprocess transport itself:
    closing that before the exit is recorded polls the child, reaping it behind
    asyncio's child watcher, which then logs an unknown child and reports
    returncode 255. asyncio closes the subprocess transport once the exit lands.

    Args:
        proc: The child whose stdio pipes to close.
    """
    # asyncio.subprocess.Process exposes no public accessor for its pipe
    # transports; reaching the undocumented transport via getattr is the only way
    # to drop the read ends before the child is reaped.
    transport = getattr(proc, "_transport", None)
    if transport is not None:
        for fd in (0, 1, 2):
            pipe = transport.get_pipe_transport(fd)
            if pipe is not None:
                pipe.close()
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
        chunk = await reader.read(READ_CHUNK)
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


async def _spawn(command: str, options: ExecOptions) -> asyncio.subprocess.Process | ExecResult:
    """Spawn the command, registering its process group. Returns ExecResult on failure."""
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
                start_new_session=sys.platform != "win32",
                preexec_fn=_child_reset_signal_mask if sys.platform != "win32" else None,
            )
            _live_process_groups.add(proc.pid)
    except OSError as error:
        stderr = f"{error}\n"
        return ExecResult("", stderr, FAILURE_EXIT_CODE, 0, len(stderr.encode()))
    return proc


async def _settle(
    proc: asyncio.subprocess.Process,
    options: ExecOptions,
    stdout_buf: OutputBuffer,
    stderr_buf: OutputBuffer,
) -> ExecResult | ExecTimeoutError:
    """Wait for the run to complete, abort, or time out, and return the outcome."""
    assert proc.stdout is not None  # noqa: S101 -- guaranteed by stdout=PIPE
    assert proc.stderr is not None  # noqa: S101 -- guaranteed by stderr=PIPE
    stdout_task = asyncio.create_task(_read_stream(proc.stdout, stdout_buf))
    stderr_task = asyncio.create_task(_read_stream(proc.stderr, stderr_buf))
    stdin_task = asyncio.create_task(_feed_stdin(proc, options.stdin))
    normal_task = asyncio.create_task(_await_normal(proc, stdout_task, stderr_task))
    abort_task = asyncio.create_task(options.abort.wait()) if options.abort is not None else None

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
                await _terminate_reap_and_drain(proc, stdout_task, stderr_task)
                stderr_buf.append_failure(str(reader_error))
                return _build_result(stdout_buf, stderr_buf, FAILURE_EXIT_CODE)
            return _build_result(stdout_buf, stderr_buf, _exit_code(proc.returncode))

        await _terminate_reap_and_drain(proc, stdout_task, stderr_task)
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
        msg = "exec settled without a normal, abort, or timeout outcome"
        raise RuntimeError(msg)
    finally:
        pending: list[asyncio.Task[object]] = [stdout_task, stderr_task, stdin_task, normal_task]
        if abort_task is not None:
            pending.append(abort_task)
        await _cancel_all(pending)


async def exec(command: str, options: ExecOptions) -> ExecResult | ExecTimeoutError:  # noqa: A001 -- names the subprocess executor `exec`
    """Run ``command`` through the shell and capture its output.

    The run never raises for a spawn or child failure: a shell that cannot be
    spawned, a non-zero exit, a signal kill, an abort, and a stream error all
    resolve as an :class:`ExecResult`. Only exceeding ``options.timeout_ms``
    resolves the distinct :class:`ExecTimeoutError` value.

    Args:
        command: The shell command line to run.
        options: Spawn, timeout, and abort settings for the run.

    Returns:
        An :class:`ExecResult` on completion (including failures), or an
        :class:`ExecTimeoutError` when the timeout is exceeded.
    """
    if options.abort is not None and options.abort.is_set():
        return ExecResult("", "", FAILURE_EXIT_CODE, 0, 0)

    spawn_result = await _spawn(command, options)
    if isinstance(spawn_result, ExecResult):
        return spawn_result
    proc = spawn_result

    try:
        return await _settle(proc, options, OutputBuffer(), OutputBuffer())
    finally:
        try:
            if proc.returncode is None:
                await _terminate_and_reap(proc, reap_timeout=_CANCEL_REAP_TIMEOUT_S)
        finally:
            _live_process_groups.discard(proc.pid)
