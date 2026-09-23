"""Shared process helpers for subprocess-driven tests."""

import asyncio
import os
import pathlib
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


def _has_exited(pid: int) -> bool:
    """Whether ``pid`` has exited: a zombie waiting to be reaped, or already reaped.

    ``os.kill(pid, 0)`` still succeeds for a zombie, so a grandchild killed with
    its process group looks alive until whoever inherited it calls ``wait``.
    That reap is scheduled by the kernel, not by the test, so treating a zombie
    as alive makes every kill assertion race against an unrelated reaper. The
    reap can also land between that probe and this check, so a process whose
    entry has already vanished counts as exited too.
    """
    if sys.platform == "win32":
        # Windows has no zombie state: a handle outlives the process, the pid does not.
        return False
    if sys.platform == "linux":
        try:
            stat = pathlib.Path(f"/proc/{pid}/stat").read_text()
        except OSError:
            return True
        # The state letter is the first field after the comm field, which is
        # parenthesized and may itself contain spaces and parentheses.
        _, _, after_comm = stat.rpartition(")")
        return after_comm.split()[:1] == ["Z"]
    state = subprocess.run(  # noqa: S603 -- argv is a fixed list, not shell-injected
        ["/bin/ps", "-o", "state=", "-p", str(pid)],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    return not state or state.startswith("Z")


def is_alive(pid: int) -> bool:
    """True while a process with ``pid`` exists and has not yet exited.

    A zombie counts as dead: it has run its last instruction and only its exit
    status survives.
    """
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return not _has_exited(pid)


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
