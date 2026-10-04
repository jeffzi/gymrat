"""Helpers for tests of and around ``exec``: live-group isolation, results, spawn waits."""

import asyncio
from pathlib import Path

import pytest

from gymrat import exec as exec_mod
from gymrat.exec import ExecResult


@pytest.fixture(autouse=True)
def isolate_live_groups(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the module-level live-group registry from bleeding across tests."""
    monkeypatch.setattr(exec_mod, "_live_process_groups", set())


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
    """
    return str(path.resolve())


async def wait_for_spawned(
    processes: list[asyncio.subprocess.Process],
    timeout_s: float = 3.0,
    *,
    spawner: str = "exec",
) -> asyncio.subprocess.Process:
    """Return the most recent child spawned, once the spawn has happened.

    Args:
        processes: The list a spawn recorder appends each child to.
        timeout_s: How long to wait for the first child.
        spawner: The function expected to spawn, named in the timeout message.

    Returns:
        The last process in ``processes``.

    Raises:
        TimeoutError: No child was spawned within ``timeout_s``.
    """
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_s
    while not processes:
        if loop.time() > deadline:
            msg = f"{spawner}() has not spawned a child yet"
            raise TimeoutError(msg)
        await asyncio.sleep(0.01)
    return processes[-1]
