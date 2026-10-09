"""Helpers for tests of and around ``exec``: results, spawn recording, ways to end a run, both entry points, exec stand-ins."""

import asyncio
import contextlib
import dataclasses
import os
import signal
import sys
from collections.abc import AsyncGenerator, Callable, Coroutine
from pathlib import Path
from typing import Any
from unittest.mock import create_autospec

import pytest

from gymrat.exec import ExecOptions, ExecResult, ExecTimeoutError, exec_argv
from gymrat.exec import exec as run_exec
from tests._process_helpers import SLEEPER_ARGV, capture_spawns, poll_until

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
    )


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


def fail_stderr_read(
    task: ExecTask, proc: asyncio.subprocess.Process | None, abort: asyncio.Event
) -> None:
    """Tear the run down by failing the pending stderr read."""
    assert proc is not None
    assert proc.stderr is not None
    proc.stderr.set_exception(RuntimeError("stream exploded"))


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


def shell_grandchild(pid_file: Path) -> str:
    """A shell command that backgrounds a ``sleep``, writes its pid to ``pid_file``, and waits."""
    return f"sleep 30 & echo $! > '{pid_file}'; wait"


def _argv_grandchild(pid_file: Path) -> tuple[str, ...]:
    script = (
        "import subprocess, time\n"
        f"p = subprocess.Popen({list(SLEEPER_ARGV)!r})\n"
        f"open({str(pid_file)!r}, 'w').write(str(p.pid) + '\\n')\n"
        "time.sleep(30)\n"
    )
    return (sys.executable, "-c", script)


@dataclasses.dataclass(frozen=True, slots=True)
class Runner:
    """One entry point and the arguments a shared test hands it.

    Attributes:
        run: The entry point.
        sleeper: Arguments for a child that sleeps for 30 s without writing.
        partial_output: Arguments for a child that writes ``line 1``, then
            sleeps for 10 s.
        grandchild: Builds the arguments for a child that starts a sleeping
            grandchild, writes the grandchild's pid to the given file, then
            sleeps for 30 s itself.
    """

    run: ExecRun
    sleeper: str | tuple[str, ...]
    partial_output: str | tuple[str, ...]
    grandchild: Callable[[Path], str | tuple[str, ...]]


RUNNERS = [
    pytest.param(
        Runner(run_shell, "sleep 30", "echo 'line 1'; sleep 10", shell_grandchild),
        id="exec",
    ),
    pytest.param(
        Runner(
            run_argv,
            SLEEPER_ARGV,
            (sys.executable, "-c", "import time; print('line 1', flush=True); time.sleep(10)"),
            _argv_grandchild,
        ),
        id="exec_argv",
    ),
]


# ---------------------------------------------------------------------------
# A stand-in for ``exec``
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class ExecRecorder:
    """The calls a stand-in for ``exec`` received and the fixed result it answers with.

    Attributes:
        result: What every call returns.
        calls: Each call's command and options, in call order. Tests read it to
            assert a command ran (or never did) and to see the working directory
            and timeout it was handed.
    """

    result: ExecResult | ExecTimeoutError
    calls: list[tuple[str, ExecOptions]] = dataclasses.field(default_factory=list)

    async def record(self, command: str, options: ExecOptions) -> ExecResult | ExecTimeoutError:
        """Record one call and answer it.

        Args:
            command: The shell command line ``exec`` was asked to run.
            options: The options it was handed.

        Returns:
            The recorder's fixed result.
        """
        self.calls.append((command, options))
        return self.result


def install_exec(
    monkeypatch: pytest.MonkeyPatch, target: str, result: ExecResult | ExecTimeoutError
) -> ExecRecorder:
    """Replace the ``exec`` a module calls with a stand-in answering ``result``.

    The stand-in carries the real ``exec`` signature, so a call the real one
    would reject fails the test.

    Args:
        monkeypatch: Patches ``target`` for the duration of the test.
        target: The dotted path of the ``exec`` name to replace, such as
            ``"gymrat.loop.keep.exec"``.
        result: What every call returns.

    Returns:
        The recorder holding the calls the stand-in received.
    """
    recorder = ExecRecorder(result)
    monkeypatch.setattr(target, create_autospec(run_exec, side_effect=recorder.record))
    return recorder
