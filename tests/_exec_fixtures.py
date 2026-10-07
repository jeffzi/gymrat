"""Helpers for tests of and around ``exec``: results, spawn recording, ways to end a run, exec stand-ins."""

import asyncio
import contextlib
import dataclasses
import os
import signal
from collections.abc import AsyncGenerator, Callable
from pathlib import Path

import pytest

from gymrat.exec import ExecOptions, ExecResult, ExecTimeoutError
from tests._process_helpers import capture_spawns

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


def physical_path(path: Path) -> str:
    """Resolve symlinks so a directory compares equal to ``pwd -P`` output.

    Wrapped in a sync helper so the resolution stays out of the async test body,
    where a blocking filesystem call would trip the async-blocking-call lint.

    Args:
        path: The directory to resolve.

    Returns:
        The symlink-free absolute path, as a string.
    """
    return str(path.resolve())


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
    loop = asyncio.get_running_loop()
    deadline = loop.time() + _SPAWN_WAIT_S
    while len(processes) < count:
        if loop.time() > deadline:
            msg = f"{spawner}() has spawned {len(processes)} of {count} children"
            raise TimeoutError(msg)
        await asyncio.sleep(0.01)
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
# A stand-in for ``exec``
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class ExecRecorder:
    """A stand-in for ``exec`` that records its calls and answers with a fixed result.

    Tests reach into ``calls`` to assert a command ran (or never did) and to read
    the working directory and timeout it was handed.
    """

    result: ExecResult | ExecTimeoutError
    calls: list[tuple[str, ExecOptions]] = dataclasses.field(default_factory=list)

    async def __call__(self, command: str, options: ExecOptions) -> ExecResult | ExecTimeoutError:
        self.calls.append((command, options))
        return self.result


def install_exec(
    monkeypatch: pytest.MonkeyPatch, target: str, result: ExecResult | ExecTimeoutError
) -> ExecRecorder:
    """Replace the ``exec`` a module calls with a recorder answering ``result``.

    Args:
        monkeypatch: Patches ``target`` for the duration of the test.
        target: The dotted path of the ``exec`` name to replace, such as
            ``"gymrat.loop.keep.exec"``.
        result: What every call returns.

    Returns:
        The recorder now standing in for ``exec``.
    """
    recorder = ExecRecorder(result)
    monkeypatch.setattr(target, recorder)
    return recorder
