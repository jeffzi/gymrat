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
import os
import signal as _signal_module
import sys
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

from gymrat.process_group import (
    TERMINATE_GRACE_S,
    attach_process_group,
    kill_process_group,
    release_process_group,
    resume_process_group,
    terminate_process_group,
    wait_for_process_group_exit,
    wait_for_process_group_exit_async,
)
from gymrat.signals import (
    TERMINATION_SIGNALS,
    deferring_termination_signals,
    install_termination_escalation,
    pthread_sigmask,
)
from gymrat.utils import MS_PER_SECOND

FAILURE_EXIT_CODE = 1
"""Exit code reported when a run fails without a positive child exit code."""

OUTPUT_CAP = 64 * 1024 * 1024
"""Per-stream cap, in bytes, on retained text. Byte counts keep counting past it."""

_READ_CHUNK = 65536
"""Bytes requested per pipe read."""

_CANCEL_REAP_TIMEOUT_S = 2.0
"""Seconds an interrupted run waits for its killed child to be reaped before giving up."""

_FINAL_OUTPUT_GRACE_S = 0.1
"""Seconds a child that stopped on request gets to have its last output read to EOF."""

_ESCALATION_GRACE_S = 0.2
"""Seconds the outermost gymrat run's double-signal sweep waits for a re-terminated child.

Kept well under :data:`TERMINATE_GRACE_S`: the user is already waiting on a second Ctrl-C.
A nested run waits less, halved per nesting level (see :func:`_kill_live_process_groups_now`).
"""

_NESTING_DEPTH_ENV = "GYMRAT_NESTING_DEPTH"
"""Environment variable telling a gymrat run how many gymrat runs sit above it.

:func:`spawn_contained` sets it on every child, so a nested run can wait a shorter
escalation grace than the run that spawned it.
"""


def _read_nesting_depth() -> int:
    """This process's nesting depth: 0 when the variable is missing, invalid, or negative."""
    try:
        return max(int(os.environ.get(_NESTING_DEPTH_ENV, "0")), 0)
    except ValueError:
        return 0


_NESTING_DEPTH = _read_nesting_depth()
"""How many gymrat runs sit above this process.

Read once because the environment a process starts with is fixed.
"""


def _terminate_grace_s() -> float:
    # A run that is itself a gymrat run answers a stop request by sweeping the
    # benches it started in sessions of their own, and a bench that ignores the
    # request costs that sweep its whole grace before the kill. The run above
    # must still be waiting when that kill lands: its own kill reaches the
    # nested run's group but not those sessions, so landing first orphans the
    # benches. Halving per nesting level keeps every level strictly longer than
    # the one below it at any depth, with no maximum depth to know, and leaves
    # the outermost run on the full TERMINATE_GRACE_S.
    return TERMINATE_GRACE_S * 0.5**_NESTING_DEPTH


_live_process_groups: set[int] = set()
"""Process-group leader PIDs of the children :func:`spawn_contained` started and still holds.

A PID is present only for the child's lifetime: it is added once the spawn
succeeds and removed by :func:`release_contained` on every settle path of its
owner (normal, abort, timeout, reader error, cancellation).
:func:`kill_live_process_groups` reads it to tear down surviving groups on the
signal path before a worktree sweep runs.
"""


def reset() -> None:
    """Forget every registered process group, so a later kill targets none of them.

    Test-only seam: production code never calls this, since a group leaves the
    registry when its owner settles. Tests use it to isolate the registry
    between cases instead of reaching into the private set directly.
    """
    _live_process_groups.clear()


class SpawnError(Exception):
    """A child that could not be started, or was killed because it could not run.

    The message is the reason, ready to show a user: the interpreter's or the
    host's rejection of the spawn, the resume failure of a win32 child, or the
    error that interrupted registering, containing, or resuming a spawned child.
    """


def _child_reset_signal_mask() -> None:
    """Unblock termination signals in the child, reversing the parent's mask."""
    if pthread_sigmask is not None:
        pthread_sigmask(_signal_module.SIG_UNBLOCK, TERMINATION_SIGNALS)


def kill_live_process_groups() -> None:
    """Stop every process tree :func:`spawn_contained` started and still holds, never raising.

    Every tree is asked to stop first, then the sweep waits out a single shared
    grace, then the stragglers are killed: a child that is itself a gymrat run
    needs that window to tear down the bench it started, and batching the
    request keeps the wait one grace long however many children are live. The
    grace is :data:`TERMINATE_GRACE_S` for the outermost run and half as long
    per nesting level below it, so a nested run's own sweep has killed its
    benches before the run above kills the nested run. The caller is a signal
    handler with no event loop left to await on, so the wait blocks.

    Groups are signaled without deferring a refusal: the signal path has no
    event loop to reap a leader on, so there is no later signal to defer to.
    """
    _sweep_live_process_groups(_terminate_grace_s())


def _kill_live_process_groups_now() -> None:
    # The escalation for a second termination signal: the sweep with a much
    # shorter grace, so a tree ignoring the polite request cannot outlive a double
    # Ctrl-C. Terminating again first hands a child that is itself a gymrat run
    # its own second signal, so it kills the benches it started in sessions of
    # its own instead of being killed mid-grace and orphaning them.
    #
    # A parent must outwait its child's whole escalation, or its SIGKILL lands
    # first and the child dies before killing the benches it started in sessions
    # the parent's group kill cannot reach. Halving per nesting level keeps every
    # level strictly longer than the one below it at any depth, with no maximum
    # depth to know. The margin at depth d is _ESCALATION_GRACE_S / 2**(d + 1), so
    # past depth 3 it drops under the 10 ms liveness poll and the ordering is no
    # longer guaranteed in practice.
    _sweep_live_process_groups(_ESCALATION_GRACE_S * 0.5**_NESTING_DEPTH)


def _sweep_live_process_groups(grace_s: float) -> None:
    # Iterate a snapshot: a run settling on another task may deregister its PID
    # mid-sweep. The process-group calls already tolerate an already-dead tree and
    # warn on other failures; each suppress(Exception) guard keeps any other
    # unexpected exception (for example a warning escalated to an error by a
    # warnings filter) from escaping into the signal-path cleanup that calls this.
    leaders = list(_live_process_groups)
    for pid in leaders:
        with contextlib.suppress(Exception):
            terminate_process_group(pid)
    with contextlib.suppress(Exception):
        wait_for_process_group_exit(leaders, grace_s)
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
        env: Environment variables for the child process; ``None`` inherits the
            parent's environment. Either way the child also sees
            ``GYMRAT_NESTING_DEPTH``.
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


async def _wait_for_exit(proc: asyncio.subprocess.Process, grace_s: float) -> bool:
    """Wait up to ``grace_s`` seconds for the child's whole group to exit.

    The child is awaited first so its exit status is collected; the rest of the
    grace then goes to any member it leaves behind, such as a nested run still
    cleaning up after the child itself died on the request.

    Args:
        proc: The child leading the group.
        grace_s: Seconds the child and its group get, shared between them.

    Returns:
        Whether the child's exit status was collected within the grace.
    """
    loop = asyncio.get_running_loop()
    deadline = loop.time() + grace_s
    with contextlib.suppress(TimeoutError):
        await asyncio.wait_for(proc.wait(), grace_s)
    await wait_for_process_group_exit_async(proc.pid, deadline - loop.time())
    return proc.returncode is not None


async def _terminate_and_reap(
    proc: asyncio.subprocess.Process,
    *,
    readers: tuple[asyncio.Task[None], asyncio.Task[None]] | None = None,
    reap_timeout: float | None = None,
) -> None:
    """Tear down a run that will not settle on its own, then reap the child.

    The tree is first asked to stop and given a grace to act on the request,
    so a child that runs cleanup of its own — killing benches it started in
    sessions this process cannot reach, writing a last diagnostic — gets to
    finish. The grace is :data:`TERMINATE_GRACE_S` for the outermost run and
    half as long per nesting level below it, so a nested run's cleanup ends
    before the run above it stops waiting. Whatever is still standing afterwards
    is killed, which also covers a descendant that ignored the request. Dropping
    the stdio pipes then releases the reader tasks still waiting on EOF, so the
    run can be snapshotted without awaiting a natural end of stream.

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
    exited = await _wait_for_exit(proc, _terminate_grace_s())
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

    Args:
        reader: The child pipe to read.
        buffer: The buffer that receives the decoded text and byte counts.
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

    Args:
        proc: The child whose stdin receives ``data``.
        data: The text to write; ``None`` or empty closes stdin without writing.
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
        with contextlib.suppress(OSError):
            stdin.close()


async def _await_normal(
    proc: asyncio.subprocess.Process,
    stdout_task: asyncio.Task[None],
    stderr_task: asyncio.Task[None],
) -> None:
    """Wait for both readers to reach EOF and the child to be reaped.

    Settling on stdio close rather than process exit is what captures output a
    background descendant flushes after the shell itself has returned.

    Args:
        proc: The child to reap.
        stdout_task: The reader draining the child's stdout.
        stderr_task: The reader draining the child's stderr.
    """
    await asyncio.gather(stdout_task, stderr_task)
    await proc.wait()


async def _cancel_all(tasks: list[asyncio.Task[object]]) -> None:
    """Cancel every task and await their settling, discarding their outcomes."""
    for task in tasks:
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


def _containment_kwargs() -> dict[str, Any]:
    """The creation arguments that put a child in a tree this process can tear down.

    A POSIX child leads its own session and starts with the termination signals
    the parent deferred around the spawn unblocked again. A win32 child is
    created suspended so it runs nothing before it has joined its job.

    Returns:
        The keyword arguments :func:`spawn_contained` adds to every spawn.
    """
    if sys.platform == "win32":
        # CREATE_SUSPENDED: the child exists but runs nothing until resumed.
        # POSIX rejects ``creationflags`` outright, so only win32 passes it.
        create_suspended = 0x4
        return {"creationflags": create_suspended}
    return {"start_new_session": True, "preexec_fn": _child_reset_signal_mask}


def release_contained(pid: int) -> None:
    """Forget a child :func:`spawn_contained` started, and drop its container.

    Called once the child's owner has stopped and reaped it. On win32 dropping
    the job is the last sweep of the tree, so a descendant that outlived the
    child does not outlive its owner either.

    Args:
        pid: The process ID :func:`spawn_contained` returned the child with.
    """
    _live_process_groups.discard(pid)
    release_process_group(pid)


async def _discard_failed_spawn(proc: asyncio.subprocess.Process) -> None:
    """Tear down a child that was spawned but never made it into its container, then release it.

    The spawn cannot hand such a child to a caller, so nothing else would ever
    stop it: a POSIX child would run on unsupervised, and a win32 child, created
    suspended, would hold its owner open with a process that can neither run nor
    exit on its own. So when stopping the tree fails part-way, the child itself
    is still killed and reaped, and it is released whatever happens.

    Args:
        proc: The child to stop, reap, and release.

    Raises:
        Exception: Whatever interrupted signaling the tree or releasing the
            container, such as a warning a warnings filter escalated to an
            error. It is raised only once the child is down and released.
    """
    try:
        await _terminate_and_reap(proc, reap_timeout=_CANCEL_REAP_TIMEOUT_S)
    except Exception:
        # Signaling the whole group failed, so kill and reap the child alone.
        with contextlib.suppress(ProcessLookupError):
            proc.kill()
        _close_pipes(proc)
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(proc.wait(), _CANCEL_REAP_TIMEOUT_S)
        raise
    finally:
        release_contained(proc.pid)


async def spawn_contained(
    create_child: Callable[..., Awaitable[asyncio.subprocess.Process]],
    *args: object,
    **kwargs: object,
) -> asyncio.subprocess.Process:
    """Start a child that cannot outlive this process's teardown of it.

    The child is created through ``create_child(*args, **kwargs)`` plus the
    containment arguments: its own POSIX session, or suspended on win32. It is
    then registered for :func:`kill_live_process_groups`, joined to its
    kill-on-close job, and resumed, all while termination signals are deferred,
    so a signal landing mid-spawn still finds the child registered and the child
    runs its first instruction already inside its container. The caller owns the
    child from then on and must hand it to :func:`release_contained` once it has
    stopped and reaped it.

    Args:
        create_child: The asyncio creation function,
            ``asyncio.create_subprocess_exec`` or
            ``asyncio.create_subprocess_shell``.
        *args: Positional arguments for ``create_child``: the program and its
            arguments, or the shell command line.
        **kwargs: Keyword arguments for ``create_child`` — pipes, ``cwd``,
            ``env``. The containment arguments are added
            here and must not be passed. The child's environment, ``env`` or
            this process's when absent, also gains :data:`_NESTING_DEPTH_ENV`.

    Returns:
        The running, registered, contained child.

    Raises:
        SpawnError: The interpreter or the host rejected the spawn before any
            process existed; or registering, containing, or resuming the child
            failed — the host would not resume it, or any exception was raised
            on the way, such as a containment warning a warnings filter
            escalated to an error. A child that exists is killed, reaped, and
            released before this raises. When that teardown fails too, the
            message reports both failures and the containment error stays the
            cause: the teardown's exception is never raised in its place.
    """
    with deferring_termination_signals():
        try:
            # The child starts one nesting level deeper, in the environment the
            # caller asked for; only an absent one inherits this process's.
            requested_env = cast("Mapping[str, str] | None", kwargs.get("env"))
            base_env = os.environ if requested_env is None else requested_env
            child_env = {**base_env, _NESTING_DEPTH_ENV: str(_NESTING_DEPTH + 1)}
            child_kwargs = {**kwargs, "env": child_env}
            # Every asyncio creation function forwards unknown keywords to Popen,
            # which accepts the containment arguments.
            proc = await create_child(*args, **child_kwargs, **_containment_kwargs())
        except (OSError, ValueError) as error:
            # ValueError covers what CPython rejects while marshalling the spawn
            # arguments, before any fork: a NUL byte in an argument, in cwd, or
            # in an env value, and an env name containing "=".
            raise SpawnError(str(error)) from error
        # Once the child exists, any failure must end in its teardown: a child
        # the caller never receives is one nothing else would ever stop.
        containment_error: Exception | None = None
        try:
            _live_process_groups.add(proc.pid)
            # Registered with every live group rather than once at import, so
            # the escalation is in place whenever there is a group to kill.
            install_termination_escalation(_kill_live_process_groups_now)
            attach_process_group(proc.pid)
            resumed = resume_process_group(proc.pid)
        except Exception as error:  # noqa: BLE001 -- re-raised as SpawnError once the child is torn down
            containment_error = error
            resumed = False
    if not resumed:
        teardown_error: Exception | None = None
        try:
            await _discard_failed_spawn(proc)
        except Exception as error:  # noqa: BLE001 -- reported in the SpawnError below, which callers expect instead
            teardown_error = error
        reason = (
            f"child process {proc.pid} could not be resumed"
            if containment_error is None
            else str(containment_error)
        )
        if teardown_error is not None:
            reason = f"{reason} (tearing the child down also failed: {teardown_error})"
        raise SpawnError(reason) from containment_error or teardown_error
    return proc


def _spawn_failure(message: str) -> ExecResult:
    """Build the failed :class:`ExecResult` a spawn error resolves to."""
    stderr = f"{message}\n"
    return ExecResult("", stderr, FAILURE_EXIT_CODE, 0, len(stderr.encode()))


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
    timeout = options.timeout_ms / MS_PER_SECOND if options.timeout_ms is not None else None

    try:
        done, _ = await asyncio.wait(
            waiters,
            timeout=timeout,
            return_when=asyncio.FIRST_COMPLETED,
        )

        # Read off ``done`` as the wait returned it, never ``normal_task.done()``
        # later: the teardown below closes the pipes, so the normal wait can
        # finish during it, and an abort or timeout would then be misreported as
        # a normal completion.
        reader_error = normal_task.exception() if normal_task in done else None
        if normal_task in done and reader_error is None:
            # A signal kill surfaces as a negative return code, and a child whose
            # status has not been collected yet as None; both collapse to the
            # failure code rather than leaking a negative or missing value.
            returncode = proc.returncode
            if returncode is None or returncode < 0:
                returncode = FAILURE_EXIT_CODE
            return _build_result(stdout_buf, stderr_buf, returncode)

        # Every other path abandons the normal wait: stop and reap first, then
        # let the readers, unblocked by the pipe close, finish before the outcome
        # is built.
        await _terminate_and_reap(proc, readers=(stdout_task, stderr_task))
        await asyncio.gather(stdout_task, stderr_task, return_exceptions=True)
        if reader_error is not None:
            stderr_buf.append_failure(str(reader_error))
            return _build_result(stdout_buf, stderr_buf, FAILURE_EXIT_CODE)
        if abort_task in done:
            return _build_result(stdout_buf, stderr_buf, FAILURE_EXIT_CODE)
        assert options.timeout_ms is not None  # noqa: S101 -- only a timeout ends the wait with nothing done
        return ExecTimeoutError(
            stdout_buf.text,
            stderr_buf.text,
            options.timeout_ms,
            stdout_buf.byte_count,
            stderr_buf.byte_count,
        )
    finally:
        await _cancel_all([stdout_task, stderr_task, stdin_task, *waiters])


async def _run(
    create_child: Callable[..., Awaitable[asyncio.subprocess.Process]],
    args: Sequence[str],
    options: ExecOptions,
) -> ExecResult | ExecTimeoutError:
    """Shared entry point: spawn ``args`` through ``create_child``, then settle.

    Both :func:`exec` (shell) and :func:`exec_argv` (direct) delegate here, so
    the abort short-circuit, the contained spawn, the settle loop, and the
    teardown are written once.

    Args:
        create_child: The asyncio creation function the child is spawned
            through; called only when the run is not already aborted.
        args: Its positional arguments: the shell command line, or the program
            and its arguments.
        options: Spawn, timeout, and abort settings for the run.

    Returns:
        An :class:`ExecResult` on completion (including failures), or an
        :class:`ExecTimeoutError` when the timeout is exceeded.
    """
    if options.abort is not None and options.abort.is_set():
        return ExecResult("", "", FAILURE_EXIT_CODE, 0, 0)

    try:
        proc = await spawn_contained(
            create_child,
            *args,
            cwd=options.cwd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=options.env,
        )
    except SpawnError as error:
        return _spawn_failure(str(error))

    try:
        return await _settle(proc, options, OutputBuffer(), OutputBuffer())
    finally:
        try:
            if proc.returncode is None:
                await _terminate_and_reap(proc, reap_timeout=_CANCEL_REAP_TIMEOUT_S)
        finally:
            release_contained(proc.pid)


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
    return await _run(asyncio.create_subprocess_shell, (command,), options)


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
    return await _run(asyncio.create_subprocess_exec, argv, options)
