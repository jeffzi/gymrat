"""Shared process helpers for subprocess-driven tests."""

import asyncio
import contextlib
import os
import pathlib
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Iterable
from io import StringIO
from typing import TYPE_CHECKING, Any, Literal, overload, override

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
    """Report whether a process is still running.

    A zombie counts as dead: it has run its last instruction and only its exit
    status survives.

    Args:
        pid: The process ID to probe.

    Returns:
        True while the process exists and has not yet exited.
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


@overload
def run_with_closed_reader(
    argv: list[str],
    *,
    stream: Literal["stdout", "stderr"],
    text: Literal[True],
    **run_kwargs: Any,
) -> subprocess.CompletedProcess[str]: ...


@overload
def run_with_closed_reader(
    argv: list[str],
    *,
    stream: Literal["stdout", "stderr"],
    text: Literal[False] | None = None,
    **run_kwargs: Any,
) -> subprocess.CompletedProcess[bytes]: ...


def run_with_closed_reader(
    argv: list[str],
    *,
    stream: Literal["stdout", "stderr"],
    **run_kwargs: Any,
) -> subprocess.CompletedProcess[Any]:
    """Run ``argv`` with ``stream`` writing into a pipe whose reader has already gone.

    The other output stream is captured through a pipe. The pipe is closed on
    return whether or not the child failed.

    Args:
        argv: The command to run.
        stream: The output stream wired to the closed pipe.
        **run_kwargs: Extra ``subprocess.run`` arguments such as ``cwd``, ``env``,
            ``input``, ``text`` and ``timeout``.

    Returns:
        The finished child, run with ``check=False``.
    """
    read_end, write_end = os.pipe()
    os.close(read_end)
    stdout = write_end if stream == "stdout" else subprocess.PIPE
    stderr = write_end if stream == "stderr" else subprocess.PIPE

    try:
        return subprocess.run(  # noqa: S603 -- caller passes a fixed argv
            argv,
            stdout=stdout,
            stderr=stderr,
            check=False,
            **run_kwargs,
        )
    finally:
        os.close(write_end)


def capture_spawns(
    monkeypatch: "pytest.MonkeyPatch",
    attr: str,
    into: list[asyncio.subprocess.Process] | None = None,
) -> list[asyncio.subprocess.Process]:
    """Wrap ``asyncio.<attr>`` to record every spawned ``Process``.

    The wrapper leaves the spawn itself real, so a test can reach into the
    captured child's stdio pipes or reap survivors on teardown.

    Args:
        monkeypatch: Patches ``asyncio.<attr>`` for the duration of the test.
        attr: The asyncio spawner to wrap, such as ``create_subprocess_exec``.
        into: A list to record into, so several spawners can share one;
            ``None`` records into a fresh list.

    Returns:
        The list each spawned process is appended to.
    """
    processes: list[asyncio.subprocess.Process] = [] if into is None else into
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


def fake_install(
    registered: list[Callable[[], None]],
) -> Callable[[Callable[[], None]], Callable[[], None]]:
    """A fake ``install_termination_cleanup`` that records each cleanup it is handed.

    Args:
        registered: The list each installed cleanup is appended to, so a test
            can call one directly to drive the termination path without a
            real signal.

    Returns:
        An installer whose uninstall callable does nothing.
    """

    def install(cleanup: Callable[[], None]) -> Callable[[], None]:
        registered.append(cleanup)
        return lambda: None

    return install


class CleanupRegistry:
    """The termination cleanups installed through a patched seam and not yet uninstalled.

    ``install_termination_cleanup`` hands back an uninstall callable, so recording
    both halves gives the live set at any moment: read it from inside a seam to see
    what is armed, and read it afterwards to see what was left behind.
    """

    def __init__(self) -> None:
        self._live: list[Callable[[], None]] = []

    def install(self, cleanup: Callable[[], None]) -> Callable[[], None]:
        """Record *cleanup* as armed.

        Args:
            cleanup: The termination cleanup being installed.

        Returns:
            A callable that removes *cleanup* from the armed set.
        """
        self._live.append(cleanup)

        def uninstall() -> None:
            self._live = [live for live in self._live if live is not cleanup]

        return uninstall

    def live(self) -> list[Callable[[], None]]:
        """Return the cleanups installed and not yet uninstalled, in install order."""
        return list(self._live)


def track_mounted_cleanups(monkeypatch: "pytest.MonkeyPatch") -> CleanupRegistry:
    """Swap the cleanup installer ``mount_live`` uses for a registry a test can read.

    Args:
        monkeypatch: The fixture that installs the swap.

    Returns:
        The registry recording every erase cleanup a display mounts.
    """
    registry = CleanupRegistry()
    monkeypatch.setattr("gymrat.cli.live_display.install_termination_cleanup", registry.install)
    return registry


class ProcessExit(BaseException):
    """Stands in for the ``os._exit`` that ends the process once a signal is handled."""


class InterruptedTerminal(StringIO):
    """A terminal file that runs a signal handler at one chosen write, then exits.

    ``interrupt_write`` arms it for the next write whose text contains *marker*.
    The handler runs in place of that write, or right after the text lands when
    *lands* is true. ``at_exit`` then keeps what reached the terminal, and the
    write raises ``ProcessExit`` where the real process would end, so nothing
    written while the exception unwinds counts.
    """

    def __init__(self) -> None:
        super().__init__()
        self.at_exit = ""
        self._handler: Callable[[], object] | None = None
        self._marker = ""
        self._lands = False

    def interrupt_write(
        self, handler: Callable[[], object], *, marker: str = "", lands: bool = False
    ) -> None:
        """Arm the terminal to run *handler* at the next matching write.

        Args:
            handler: Runs in place of the matching write, or right after its text
                lands when *lands* is true.
            marker: Text the write must contain to trigger the handler; the empty
                string matches every write.
            lands: Whether the write's text reaches the terminal before the
                handler runs.
        """
        self._handler, self._marker, self._lands = handler, marker, lands

    @override
    def write(self, text: str) -> int:
        handler = self._handler
        if handler is None or self._marker not in text:
            return super().write(text)
        self._handler = None
        if self._lands:
            super().write(text)
        handler()
        self.at_exit = self.getvalue()
        raise ProcessExit


def record_registry_sweep(monkeypatch: "pytest.MonkeyPatch") -> list[int]:
    """Stand in for the group signals ``kill_live_process_groups`` sends, recording each pid."""
    attempted: list[int] = []

    def record(pid: int, *_args: object, **_kwargs: object) -> bool:
        attempted.append(pid)
        return True

    monkeypatch.setattr(gymrat_exec, "terminate_process_group", record)
    monkeypatch.setattr(gymrat_exec, "kill_process_group", record)
    return attempted
