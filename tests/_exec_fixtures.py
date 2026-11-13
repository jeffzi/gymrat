"""Helpers for tests of and around ``exec``: results, spawn recording, ways to end a run, both entry points, exec stand-ins."""

import asyncio
import contextlib
import dataclasses
import os
import shlex
import signal
from collections.abc import AsyncGenerator, Callable, Coroutine, Generator
from pathlib import Path
from typing import Any
from unittest.mock import create_autospec

import pytest

from gymrat import exec as exec_mod
from gymrat.exec import ExecOptions, ExecResult, ExecTimeoutError, exec_argv
from gymrat.exec import exec as run_exec
from tests._process_helpers import capture_spawns, poll_until

type ExecTask = asyncio.Task[ExecResult | ExecTimeoutError]

#: How long :func:`wait_for_spawned` waits for the first child.
_SPAWN_WAIT_S = 3.0

#: How long the in-loop reap waits on each surviving child.
_REAP_WAIT_S = 5


def expected_result(stdout: str = "", stderr: str = "", exit_code: int = 0) -> ExecResult:
    """Build an ``ExecResult`` with byte counts derived from the strings."""
    return ExecResult(
        stdout=stdout,
        stderr=stderr,
        exit_code=exit_code,
        stdout_bytes=len(stdout.encode()),
        stderr_bytes=len(stderr.encode()),
    )


#: The bytes each channel of :func:`capped_result` wrote before exec's cap cut it.
CAPPED_STDOUT_BYTES = 200_000
CAPPED_STDERR_BYTES = 150_000


def capped_result(*, exit_code: int = 0) -> ExecResult:
    """Build an ``ExecResult`` whose output overran exec's cap, keeping the pre-cap byte counts.

    Args:
        exit_code: The exit code the capped run reports.

    Returns:
        A result whose short captured text sits beside ``CAPPED_STDOUT_BYTES`` and
        ``CAPPED_STDERR_BYTES``.
    """
    return ExecResult(
        stdout="capped stdout",
        stderr="capped stderr",
        exit_code=exit_code,
        stdout_bytes=CAPPED_STDOUT_BYTES,
        stderr_bytes=CAPPED_STDERR_BYTES,
        stdout_truncated=True,
        stderr_truncated=True,
    )


#: The ``exec`` the sampling module calls, which tests of sampling and its callers replace.
SAMPLING_EXEC = "gymrat.sampling.exec"


#: An output cap small enough that a few bytes overrun it.
SMALL_CAP = 2

#: The three invalid bytes :func:`overrun_with_invalid_kept` writes before it waits.
OVERRUN_KEPT_BYTES = b"\xff\xff\xff"


def wait_for_gate(gate: Path) -> str:
    """A shell snippet that blocks until ``gate`` exists.

    Args:
        gate: The file the test creates to let the command go on.

    Returns:
        The shell snippet.
    """
    return f"until [ -e {shlex.quote(str(gate))} ]; do sleep 0.01; done"


def open_gate_after_reading(monkeypatch: pytest.MonkeyPatch, gate: Path, marker: bytes) -> None:
    """Create ``gate`` once a stream read hands the reader a chunk ending in ``marker``.

    A command that writes ``marker`` and then waits on the gate puts its next
    bytes in a later read, so the chunk boundary comes from the reader, not
    from a sleep in the child.

    Args:
        monkeypatch: The fixture that wraps ``asyncio.StreamReader.read``.
        gate: The file to create.
        marker: The bytes that end the chunk the command waits to be read.
    """
    real_read = asyncio.StreamReader.read

    async def read(self: asyncio.StreamReader, n: int = -1) -> bytes:
        chunk = await real_read(self, n)
        if chunk.endswith(marker):
            await asyncio.to_thread(gate.touch)
        return chunk

    monkeypatch.setattr(asyncio.StreamReader, "read", read)


def overrun_with_invalid_kept(gate: Path) -> str:
    """Write three invalid bytes, wait on ``gate``, then write ``ab``.

    The three bytes, read before the cap is reached and so kept whole, decode to
    three U+FFFD that re-encode to nine bytes; the later ``ab`` is dropped. The
    kept text re-encodes longer than the five bytes written, yet the stream
    overran.

    Args:
        gate: The file :func:`open_gate_after_reading` creates once the three
            bytes are read.

    Returns:
        The shell command.
    """
    return f"printf '\\377\\377\\377'; {wait_for_gate(gate)}; printf ab"


def stdout_overruns(gate: Path) -> str:
    """Write one short invalid byte to stderr, then overrun the cap on stdout.

    Args:
        gate: The gate :func:`overrun_with_invalid_kept` waits on.

    Returns:
        The shell command.
    """
    return f"{SHORT_WITH_INVALID} >&2; {overrun_with_invalid_kept(gate)}"


def stderr_overruns(gate: Path) -> str:
    """Write one short invalid byte to stdout, then overrun the cap on stderr.

    Args:
        gate: The gate :func:`overrun_with_invalid_kept` waits on.

    Returns:
        The shell command.
    """
    return f"{SHORT_WITH_INVALID}; {{ {overrun_with_invalid_kept(gate)}; }} >&2"


#: One invalid byte, short of the cap, decodes to a U+FFFD that re-encodes to
#: three bytes: longer than the one byte written, yet nothing was dropped.
SHORT_WITH_INVALID = "printf '\\377'"


async def wait_for_spawned(
    processes: list[asyncio.subprocess.Process], *, count: int = 1, spawner: str = "exec"
) -> asyncio.subprocess.Process:
    """Return the most recent child spawned, once ``count`` spawns have happened.

    Args:
        processes: The list a spawn recorder appends each child to.
        count: How many children must have been spawned.
        spawner: The function expected to spawn, named in the timeout message.

    Returns:
        The last process in ``processes``.

    Raises:
        TimeoutError: Fewer than ``count`` children were spawned within three
            seconds.
    """
    await poll_until(
        lambda: len(processes) >= count,
        _SPAWN_WAIT_S,
        lambda: TimeoutError(f"{spawner}() has spawned {len(processes)} of {count} children"),
    )
    return processes[-1]


@contextlib.contextmanager
def fresh_live_process_groups() -> Generator[None]:
    """Swap the live process-group registry for an empty one, putting the original back after.

    A private patcher rather than the ``monkeypatch`` fixture, so an autouse
    fixture built on it does not pull every test's ``monkeypatch`` setup ahead of
    the environment baseline.

    Yields:
        Nothing: the empty registry is in place while the block runs.
    """
    with pytest.MonkeyPatch.context() as patcher:
        patcher.setattr(exec_mod, "_live_process_groups", set())
        yield


@contextlib.asynccontextmanager
async def recorded_spawns(
    monkeypatch: pytest.MonkeyPatch,
) -> AsyncGenerator[list[asyncio.subprocess.Process]]:
    """Record every child ``exec`` or ``exec_argv`` spawns, reaping survivors in-loop on exit.

    Reaping with ``proc.wait()`` while the loop still runs lets asyncio finalize
    each child's transport in-loop, so no orphaned transport lingers for a later
    test's forced garbage collection to finalize against a closed loop (which
    would surface as a warning about an exception that cannot propagate).

    Args:
        monkeypatch: Patches both asyncio spawners for the duration.

    Yields:
        The spawned processes, in spawn order.
    """
    processes: list[asyncio.subprocess.Process] = []
    for spawner in ("create_subprocess_shell", "create_subprocess_exec"):
        capture_spawns(monkeypatch, spawner, processes)
    try:
        yield processes
    finally:
        for proc in processes:
            if proc.returncode is None and proc.pid:
                with contextlib.suppress(OSError):
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            with contextlib.suppress(TimeoutError, ProcessLookupError):
                await asyncio.wait_for(proc.wait(), _REAP_WAIT_S)


# ---------------------------------------------------------------------------
# Ending a running exec
# ---------------------------------------------------------------------------


def leave_to_timeout(
    task: ExecTask, proc: asyncio.subprocess.Process | None, abort: asyncio.Event
) -> None:
    """Leave the run alone: its own timeout tears it down."""


def set_abort(
    task: ExecTask, proc: asyncio.subprocess.Process | None, abort: asyncio.Event
) -> None:
    """Tear the run down through its abort event."""
    abort.set()


def cancel_task(
    task: ExecTask, proc: asyncio.subprocess.Process | None, abort: asyncio.Event
) -> None:
    """Tear the run down by cancelling the task awaiting it."""
    task.cancel()


async def settle(task: "asyncio.Task[object]", timeout_s: float = 5) -> None:
    """Wait for ``task`` to finish unwinding, treating its cancellation as done."""
    with contextlib.suppress(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout_s)


async def cancel_and_settle(task: "asyncio.Task[object]", timeout_s: float = 5) -> None:
    """Cancel ``task`` and wait for it to finish unwinding."""
    task.cancel()
    await settle(task, timeout_s)


@dataclasses.dataclass(frozen=True, slots=True)
class Teardown:
    """How a test ends a run: an optional timeout plus an action on the running exec."""

    timeout_ms: int | None
    trigger: Callable[[ExecTask, asyncio.subprocess.Process | None, asyncio.Event], None]


# ---------------------------------------------------------------------------
# The two entry points, for the behaviors they share
# ---------------------------------------------------------------------------

type ExecRun = Callable[[Any, ExecOptions], Coroutine[Any, Any, ExecResult | ExecTimeoutError]]


async def run_shell(args: Any, options: ExecOptions) -> ExecResult | ExecTimeoutError:
    """Run ``args``, a shell command line, through ``exec``."""
    return await run_exec(args, options)


async def run_argv(args: Any, options: ExecOptions) -> ExecResult | ExecTimeoutError:
    """Run ``args``, a program and its arguments, through ``exec_argv``."""
    return await exec_argv(args, options)


def shell_leaving_a_descendant(pid_file: Path, *, detached: bool = False) -> str:
    """Build a shell command that backgrounds a ``sleep``, writes its pid to ``pid_file``, and exits.

    Args:
        pid_file: Where the ``sleep`` writes its pid.
        detached: Send all three of the ``sleep``'s streams to ``/dev/null``;
            otherwise it holds the shell's stderr open.

    Returns:
        The shell command line.
    """
    redirects = "</dev/null >/dev/null 2>&1" if detached else ">/dev/null"
    return f"sleep 30 {redirects} & echo $! > {shlex.quote(str(pid_file))}; exit 0"


# ---------------------------------------------------------------------------
# A stand-in for ``exec``
# ---------------------------------------------------------------------------


type ExecAnswer = ExecResult | ExecTimeoutError | Callable[[int], ExecResult | ExecTimeoutError]
"""A fixed result for every call, or a builder answering the n-th call (1-based)."""


@dataclasses.dataclass(frozen=True, slots=True)
class ExecRecorder:
    """The calls a stand-in for ``exec`` received and how it answers them.

    Attributes:
        answer: The result every call returns, or a builder handed each
            call's 1-based number that returns that call's result.
        calls: Each call's command and options, in call order. Tests read it to
            assert a command ran (or never did) and to see the working directory
            and timeout it was handed.
    """

    answer: ExecAnswer
    calls: list[tuple[str, ExecOptions]] = dataclasses.field(default_factory=list)

    async def record(self, command: str, options: ExecOptions) -> ExecResult | ExecTimeoutError:
        """Record one call and answer it.

        Args:
            command: The shell command line ``exec`` was asked to run.
            options: The options it was handed.

        Returns:
            The fixed result, or what the builder returns for this call.
        """
        self.calls.append((command, options))
        if callable(self.answer):
            return self.answer(len(self.calls))
        return self.answer


def install_exec(monkeypatch: pytest.MonkeyPatch, target: str, answer: ExecAnswer) -> ExecRecorder:
    """Replace the ``exec`` a module calls with a stand-in answering with ``answer``.

    The stand-in carries the real ``exec`` signature, so a call the real one
    would reject fails the test.

    Args:
        monkeypatch: Patches ``target`` for the duration of the test.
        target: The dotted path of the ``exec`` name to replace, such as
            ``"gymrat.loop.keep.exec"``.
        answer: The result every call returns, or a builder handed each
            call's 1-based number that returns that call's result.

    Returns:
        The recorder holding the calls the stand-in received.
    """
    recorder = ExecRecorder(answer)
    monkeypatch.setattr(target, create_autospec(run_exec, side_effect=recorder.record))
    return recorder
