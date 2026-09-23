"""Run commands as asyncio subprocesses with bounded, decoded capture.

``exec`` spawns a command through the shell; ``exec_argv`` runs an executable
directly with an argument vector and no shell interpretation.  Both capture
stdout and stderr separately as decoded text plus raw byte counts, and settle
when the child's stdio pipes close — not merely when the child exits, so output
flushed by a background descendant is still captured. A run is bounded three
ways:

- a per-stream 64 MiB text cap (byte counts keep counting past it),
- an optional timeout that resolves an :class:`ExecTimeoutError` value, and
- an optional abort :class:`asyncio.Event` that resolves a failed
  :class:`ExecResult`.

Timeout and abort snapshot whatever has been captured so far and stop the whole
process tree — asked first, killed if it does not go — so a grandchild the child
left running, or a bench the child started in a session of its own, dies with
the run instead of outliving it. Every returned value is a frozen dataclass, so
a caller cannot mutate one run's result into a landmine for the next.
"""

import asyncio
import codecs
import contextlib
import signal as _signal_module
import sys
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from gymrat.process_group import (
    TERMINATE_GRACE_S,
    attach_process_group,
    kill_process_group,
    release_process_group,
    resume_process_group,
    terminate_process_group,
    wait_for_process_group_exit,
)
from gymrat.signals import TERMINATION_SIGNALS, deferring_termination_signals, pthread_sigmask

FAILURE_EXIT_CODE = 1
"""Exit code reported when a run fails without a positive child exit code."""

OUTPUT_CAP = 64 * 1024 * 1024
"""Per-stream cap, in bytes, on retained text. Byte counts keep counting past it."""

READ_CHUNK = 65536
"""Bytes requested per pipe read."""

_CANCEL_REAP_TIMEOUT_S = 2.0
"""Seconds an interrupted run waits for its killed child to be reaped before giving up."""

_FINAL_OUTPUT_GRACE_S = 0.1
"""Seconds a child that stopped on request gets to have its last output read to EOF."""

_CREATE_SUSPENDED: int = 0x4 if sys.platform == "win32" else 0
"""Win32 ``CREATE_SUSPENDED``: the child exists but runs nothing until resumed.

Zero on POSIX, where ``creationflags`` is rejected outright. The platform is
read once, when this module is imported, because the flags a spawn accepts are
fixed by the interpreter's own host and cannot change under a running process.
"""

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
    """Stop every process tree with a live bench child, never raising.

    Every tree is asked to stop first, then the sweep waits out a single shared
    grace, then the stragglers are killed: a child that is itself a gymrat run
    needs that window to tear down the bench it started, and batching the
    request keeps the wait one grace long however many children are live. The
    caller is a signal handler with no event loop left to await on, so the wait
    blocks.

    Iterates a snapshot so a run settling on another task can deregister its PID
    mid-sweep without disturbing the loop. The process-group calls already
    tolerate an already-dead tree and warn on other failures; the extra guard
    keeps any other unexpected exception (for example a warning escalated to an
    error by a warnings filter) from escaping into the signal-path cleanup that
    calls this.
    """
    leaders = list(_live_process_groups)
    for pid in leaders:
        with contextlib.suppress(Exception):
            terminate_process_group(pid)
    with contextlib.suppress(Exception):
        wait_for_process_group_exit(leaders, TERMINATE_GRACE_S)
    for pid in leaders:
        with contextlib.suppress(Exception):
            kill_process_group(pid)


@dataclass(frozen=True, slots=True)
class ExecOptions:
    """Inputs for a single :func:`exec` or :func:`exec_argv` run.

    Attributes:
        cwd: Working directory the command runs in.
        timeout_ms: Wall-clock budget in milliseconds; ``None`` waits forever.
        abort: Event that, once set, kills the run and resolves a failed result.
        stdin: Text delivered to the command's standard input, then closed;
            ``None`` gives an immediately closed (EOF) input.
        env: Environment variables for the child process. When set, the child
            sees exactly this mapping; when ``None``, it inherits the parent's
            environment.
    """

    cwd: str
    timeout_ms: int | None = None
    abort: asyncio.Event | None = None
    stdin: str | None = None
    env: Mapping[str, str] | None = None


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


async def _wait_for_exit(proc: asyncio.subprocess.Process, grace_s: float) -> bool:
    """Wait up to ``grace_s`` seconds for the child to exit, reporting whether it did."""
    with contextlib.suppress(TimeoutError):
        await asyncio.wait_for(proc.wait(), grace_s)
    return proc.returncode is not None


async def _terminate_and_reap(
    proc: asyncio.subprocess.Process,
    *,
    readers: tuple[asyncio.Task[None], asyncio.Task[None]] | None = None,
    reap_timeout: float | None = None,
) -> None:
    """Tear down a run that will not settle on its own, then reap the child.

    The tree is first asked to stop and given :data:`TERMINATE_GRACE_S` to act
    on the request, so a child that runs cleanup of its own — killing benches it
    started in sessions this process cannot reach, writing a last diagnostic —
    gets to finish. Whatever is still standing afterwards is killed, which also
    covers a descendant that ignored the request. Dropping the stdio pipes then
    releases the reader tasks still waiting on EOF, so the run can be
    snapshotted without awaiting a natural end of stream.

    A group that refused a signal because its members were all still exiting is
    signaled once more after the reap, which stays silent when the group is gone
    and warns only when the refusal is genuine. When the reap does not land
    within ``reap_timeout``, the group is signaled again anyway, so a refusal
    that outlasts the wait still warns.

    Args:
        proc: The child whose process tree to stop and reap.
        readers: The stdout and stderr reader tasks, when the caller still owns
            them. A child that stopped on request has closed its write ends, so
            the readers reach a real EOF on their own; closing the pipes before
            they do would drop the output the child flushed on its way out.
        reap_timeout: Seconds to wait for the reap, or ``None`` to wait until it
            lands.
    """
    refused = terminate_process_group(proc.pid, defer_refusal=True)
    exited = await _wait_for_exit(proc, TERMINATE_GRACE_S)
    if exited and readers is not None:
        await asyncio.wait(readers, timeout=_FINAL_OUTPUT_GRACE_S)
    refused = kill_process_group(proc.pid, defer_refusal=True) or refused
    _close_pipes(proc)
    if not exited:
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
    timeout/abort -- need the same sequence: stop and reap first, then let the
    readers, unblocked by the pipe close, finish before the outcome is built.
    """
    await _terminate_and_reap(proc, readers=(stdout_task, stderr_task))
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


def _subprocess_kwargs(options: ExecOptions) -> dict[str, Any]:
    """Common keyword arguments for both subprocess creation functions.

    The win32 child is created suspended so :func:`_spawn` can contain it before
    it runs; any further creation flag has to be combined with that one rather
    than replace it.

    Args:
        options: Spawn settings for the run.

    Returns:
        The keyword arguments to hand the subprocess creation function.
    """
    posix = sys.platform != "win32"
    kwargs: dict[str, Any] = {
        "cwd": options.cwd,
        "stdin": asyncio.subprocess.PIPE,
        "stdout": asyncio.subprocess.PIPE,
        "stderr": asyncio.subprocess.PIPE,
        "start_new_session": posix,
        "preexec_fn": _child_reset_signal_mask if posix else None,
    }
    if _CREATE_SUSPENDED:
        kwargs["creationflags"] = _CREATE_SUSPENDED
    if options.env is not None:
        kwargs["env"] = dict(options.env)
    return kwargs


def _spawn_failure(message: str) -> ExecResult:
    """Build the failed :class:`ExecResult` a spawn error resolves to."""
    stderr = f"{message}\n"
    return ExecResult("", stderr, FAILURE_EXIT_CODE, 0, len(stderr.encode()))


async def _discard_suspended_child(proc: asyncio.subprocess.Process) -> ExecResult:
    """Kill a child the host refused to resume and resolve the run as a spawn failure.

    On win32 the child is created suspended, so one that never starts would hold
    the run open with a process that can neither run nor exit on its own.

    Args:
        proc: The suspended child to kill and reap.

    Returns:
        The failed :class:`ExecResult` the run resolves to.
    """
    try:
        await _terminate_and_reap(proc, reap_timeout=_CANCEL_REAP_TIMEOUT_S)
    finally:
        _live_process_groups.discard(proc.pid)
        release_process_group(proc.pid)
    return _spawn_failure(f"child process {proc.pid} could not be resumed")


async def _spawn(
    create_child: Callable[[], Awaitable[asyncio.subprocess.Process]],
) -> asyncio.subprocess.Process | ExecResult:
    """Spawn a child via ``create_child``, registering and containing its process group.

    Masks termination signals across the spawn-and-register pair so a signal
    delivered between them still finds the child in the live registry when the
    deferred handler fires :func:`kill_live_process_groups`. Containment
    (:func:`attach_process_group`) and the resume that lets a win32 child start
    running happen in the same masked block, in that order: the child reaches
    its first instruction already inside the container that tears it down.

    Args:
        create_child: No-arg callable returning the subprocess creation
            awaitable (``create_subprocess_shell`` or
            ``create_subprocess_exec``).

    Returns:
        The spawned process, or an :class:`ExecResult` with exit code 1 when
        the spawn itself fails or the child cannot be resumed.
    """
    try:
        with deferring_termination_signals():
            proc = await create_child()
            _live_process_groups.add(proc.pid)
            attach_process_group(proc.pid)
            resumed = resume_process_group(proc.pid)
    except (OSError, ValueError) as error:
        # ValueError covers what CPython rejects while marshalling the spawn
        # arguments, before any fork: a NUL byte in an argument, in cwd, or in
        # an env value, and an env name containing "=".
        return _spawn_failure(str(error))
    if not resumed:
        return await _discard_suspended_child(proc)
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


async def _run(
    create_child: Callable[[], Awaitable[asyncio.subprocess.Process]],
    options: ExecOptions,
) -> ExecResult | ExecTimeoutError:
    """Shared entry point: spawn via ``create_child``, then settle.

    Both :func:`exec` (shell) and :func:`exec_argv` (direct) delegate here
    after building their creation callable, so the abort short-circuit, settle
    loop, and teardown are written once.

    Args:
        create_child: No-arg callable returning the subprocess creation
            awaitable.
        options: Spawn, timeout, and abort settings for the run.

    Returns:
        An :class:`ExecResult` on completion (including failures), or an
        :class:`ExecTimeoutError` when the timeout is exceeded.
    """
    if options.abort is not None and options.abort.is_set():
        return ExecResult("", "", FAILURE_EXIT_CODE, 0, 0)

    spawn_result = await _spawn(create_child)
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
            release_process_group(proc.pid)


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
    kwargs = _subprocess_kwargs(options)
    return await _run(
        lambda: asyncio.create_subprocess_shell(command, **kwargs),
        options,
    )


async def exec_argv(argv: Sequence[str], options: ExecOptions) -> ExecResult | ExecTimeoutError:
    """Run ``argv[0]`` with the remaining items as arguments, no shell.

    Identical to :func:`exec` in every respect except the child is spawned
    directly: arguments containing spaces, ``$HOME``, pipes, or quotes reach
    the child as exactly one ``sys.argv`` entry with those characters intact.

    Args:
        argv: The program and its arguments. ``argv[0]`` is the executable.
        options: Spawn, timeout, and abort settings for the run.

    Returns:
        An :class:`ExecResult` on completion (including failures), or an
        :class:`ExecTimeoutError` when the timeout is exceeded.
    """
    if not argv:
        return _spawn_failure("argv is empty: no program to run")
    kwargs = _subprocess_kwargs(options)
    return await _run(
        lambda: asyncio.create_subprocess_exec(*argv, **kwargs),
        options,
    )
