"""Shared process helpers for subprocess-driven tests."""

import asyncio
import contextlib
import os
import pathlib
import signal
import subprocess
import sys
import time
from collections.abc import Iterable
from typing import TYPE_CHECKING

from gymrat import exec as gymrat_exec

if TYPE_CHECKING:
    import pytest

# Interval a polling loop sleeps between checks.
_POLL_INTERVAL_S = 0.025

# Default upper bound a polling loop waits before giving up.
_DEFAULT_WAIT_S = 5.0

# A child that writes nothing, sleeps, and never exits on its own — a stand-in
# for any long-running leader a test needs to kill or wait out.
SLEEPER_ARGV: tuple[str, ...] = (sys.executable, "-c", "import time; time.sleep(30)")

KILLPG_FAILED = "killpg failed"

# A script whose process group ends up holding only a zombie once its own
# process is killed and reaped. It forks a holder; the holder forks a member
# that exits at once, waits for that exit without reaping it, moves itself into
# a group of its own, and writes its pid to ``argv[1]``. The member stays a
# zombie in the script's group for as long as the holder lives, the state an
# orphaned grandchild is in while it waits for init to reap it. The holder
# detaches from the inherited stdio so it never keeps a caller's pipes open.
ZOMBIE_ONLY_GROUP_SCRIPT = """
import os, sys, time
pid_file = sys.argv[1]
if os.fork() == 0:
    member = os.fork()
    if member == 0:
        os._exit(0)
    if hasattr(os, 'waitid'):
        os.waitid(os.P_PID, member, os.WEXITED | os.WNOWAIT)
    else:
        # macOS before Python 3.13: getpgid stops finding a process once it
        # is a zombie, which is how its exit is awaited without reaping it.
        while True:
            try:
                os.getpgid(member)
            except ProcessLookupError:
                break
            time.sleep(0.005)
    os.setpgid(0, 0)
    null = os.open(os.devnull, os.O_RDWR)
    for fd in (0, 1, 2):
        os.dup2(null, fd)
    with open(pid_file, 'w') as out:
        out.write(str(os.getpid()) + '\\n')
    time.sleep(30)
    os._exit(0)
time.sleep(30)
"""


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


async def wait_until_dead(pid: int, timeout_s: float = _DEFAULT_WAIT_S) -> None:
    """Poll until the process with ``pid`` no longer exists.

    Args:
        pid: The process ID to wait on.
        timeout_s: Seconds to poll before giving up.

    Raises:
        AssertionError: The process is still alive after ``timeout_s``.
    """
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_s
    while is_alive(pid):
        if loop.time() > deadline:
            message = f"process {pid} was still alive after {timeout_s}s"
            raise AssertionError(message)
        await asyncio.sleep(_POLL_INTERVAL_S)


def wait_until_dead_blocking(pid: int, timeout_s: float = _DEFAULT_WAIT_S) -> None:
    """Block until the process with ``pid`` no longer exists.

    The synchronous twin of ``wait_until_dead``, for tests that drive a
    subprocess without an event loop.

    Args:
        pid: The process ID to wait on.
        timeout_s: Seconds to poll before giving up.

    Raises:
        AssertionError: The process is still alive after ``timeout_s``.
    """
    deadline = time.monotonic() + timeout_s
    while is_alive(pid):
        if time.monotonic() > deadline:
            message = f"process {pid} was still alive after {timeout_s}s"
            raise AssertionError(message)
        time.sleep(_POLL_INTERVAL_S)


def read_pid_file(pid_path: pathlib.Path) -> int | None:
    """Read the pid a process wrote to ``pid_path``, as ``echo $$ >`` writes it.

    The trailing newline marks the write as complete, so a partial file reads as
    no pid rather than a truncated one. A pid of 0 or below also reads as no
    pid: passed to ``os.kill`` or ``os.killpg`` it would signal the test's own
    process group instead of the child.

    Args:
        pid_path: The pid file to read.

    Returns:
        The process ID, or ``None`` while the file is absent, still being
        written, or holds no positive pid.
    """
    try:
        raw = pid_path.read_text()
    except (FileNotFoundError, PermissionError):
        # Windows refuses the read while the writer still holds the file open.
        return None
    if not raw.endswith("\n"):
        return None
    pid = int(raw)
    return pid if pid > 0 else None


async def wait_for_pid_file(pid_path: pathlib.Path, timeout_s: float = _DEFAULT_WAIT_S) -> int:
    """Poll until ``pid_path`` holds a pid ``read_pid_file`` accepts, then return it.

    Args:
        pid_path: The pid file to poll.
        timeout_s: Seconds to poll before giving up.

    Returns:
        The process ID read from the file.

    Raises:
        TimeoutError: No complete pid appears within ``timeout_s``.
    """
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_s
    while (pid := read_pid_file(pid_path)) is None:
        if loop.time() > deadline:
            message = f"pid never appeared at {pid_path}"
            raise TimeoutError(message)
        await asyncio.sleep(_POLL_INTERVAL_S)
    return pid


def wait_for_pid_file_blocking(pid_path: pathlib.Path, timeout_s: float = _DEFAULT_WAIT_S) -> int:
    """Block until ``pid_path`` holds a pid ``read_pid_file`` accepts, then return it.

    The synchronous twin of ``wait_for_pid_file``, for tests that drive a
    subprocess without an event loop.

    Args:
        pid_path: The pid file to poll.
        timeout_s: Seconds to poll before giving up.

    Returns:
        The process ID read from the file.

    Raises:
        TimeoutError: No complete pid appears within ``timeout_s``.
    """
    deadline = time.monotonic() + timeout_s
    while (pid := read_pid_file(pid_path)) is None:
        if time.monotonic() > deadline:
            message = f"pid never appeared at {pid_path}"
            raise TimeoutError(message)
        time.sleep(_POLL_INTERVAL_S)
    return pid


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


def kill_surviving_groups(processes: Iterable[asyncio.subprocess.Process]) -> None:
    """SIGKILL the process group of every process in ``processes`` that is still alive."""
    for proc in processes:
        if proc.returncode is None and proc.pid:
            with contextlib.suppress(OSError):
                os.killpg(proc.pid, signal.SIGKILL)


def killpg_warnings(recorded: "pytest.WarningsRecorder") -> list[str]:
    """Return the messages of recorded warnings about a failed group kill."""
    return [str(w.message) for w in recorded if KILLPG_FAILED in str(w.message)]


def refuse_resume(_pid: int) -> bool:
    """Stand in for a host that cannot resume the suspended child."""
    return False


def record_registry_sweep(monkeypatch: "pytest.MonkeyPatch") -> list[int]:
    """Stand in for the group signals ``kill_live_process_groups`` sends, recording each pid."""
    attempted: list[int] = []

    def record(pid: int, *_args: object, **_kwargs: object) -> bool:
        attempted.append(pid)
        return True

    monkeypatch.setattr(gymrat_exec, "terminate_process_group", record)
    monkeypatch.setattr(gymrat_exec, "kill_process_group", record)
    return attempted
