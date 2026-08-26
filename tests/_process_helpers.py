"""Shared process-liveness polling helpers for subprocess-driven tests."""

import asyncio
import os


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
