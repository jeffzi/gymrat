"""Shared process helpers for subprocess-driven tests."""

import asyncio
import os
import subprocess
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pytest


def dead_pid() -> int:
    """Return a pid that is certainly gone: the child ran and was reaped."""
    proc = subprocess.Popen([sys.executable, "-c", ""])
    proc.wait()
    return proc.pid


def is_alive(pid: int) -> bool:
    """True while a process with ``pid`` exists."""
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


async def wait_until_dead(pid: int, timeout_s: float = 5.0) -> None:
    """Poll until the process with ``pid`` no longer exists."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_s
    while is_alive(pid):
        if loop.time() > deadline:
            message = f"process {pid} was still alive after {timeout_s}s"
            raise AssertionError(message)
        await asyncio.sleep(0.025)


def capture_spawns(
    monkeypatch: "pytest.MonkeyPatch",
    attr: str,
) -> list[asyncio.subprocess.Process]:
    """Wrap ``asyncio.<attr>`` to record every spawned ``Process``.

    The wrapper leaves the spawn itself real, so a test can reach into the
    captured child's stdio pipes or reap survivors on teardown.
    """
    processes: list[asyncio.subprocess.Process] = []
    real = getattr(asyncio, attr)

    async def wrapper(*args: object, **kwargs: object) -> asyncio.subprocess.Process:
        proc = await real(*args, **kwargs)
        processes.append(proc)
        return proc

    monkeypatch.setattr(asyncio, attr, wrapper)
    return processes
