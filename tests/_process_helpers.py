"""Shared process helpers for subprocess-driven tests."""

import asyncio
import contextlib
import ctypes
import dataclasses
import errno
import importlib.util
import os
import pathlib
import subprocess
import sys
import time
import types
from collections.abc import Callable, Generator
from io import StringIO
from typing import TYPE_CHECKING, Any, Literal, overload, override

from gymrat import process_group

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

# Stand-in win32 handles for the faked Job Object layer, which never touches a
# real process: distinct values so a process-handle close can be told apart
# from a job close.
JOB_HANDLE = 777
PROCESS_HANDLE = 4242

# Hard cap on how often a faked job may be polled for its live process count.
# The wait sleeps between polls, so the millisecond graces the job tests give
# admit a handful of polls, and fewer still on a slow machine. Only a wait that
# stopped honouring its deadline reaches this, and it turns that runaway loop
# into a failure instead of a hung suite.
_FAKE_JOB_QUERY_CAP = 200

# ``ctypes.WinError`` is bound only on Windows, but the production refusal paths
# format it into their warning, so the faked platform hands them a stand-in.
_FAKE_WIN_ERROR = "the host refused the call"

# NTSTATUS codes the faked ``NtResumeProcess`` answers with: ``STATUS_SUCCESS``
# and the ``STATUS_ACCESS_DENIED`` a locked-down host replies with.
_NT_STATUS_SUCCESS = 0
_NT_STATUS_ACCESS_DENIED = 0xC0000022

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


# The repository root, so a child script can import the ``tests`` package
# alongside the installed ``gymrat``.
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


def spawn_child_script(
    directory: pathlib.Path, name: str, source: str, *args: str
) -> subprocess.Popen[str]:
    """Write ``source`` to ``<directory>/<name>.py`` and start it as a child.

    The child runs from the repository root with the root on ``PYTHONPATH``, so
    it can import the ``tests`` package as well as ``gymrat``. Its stderr is
    piped in text mode.

    Args:
        directory: Where the script is written.
        name: The script's file stem.
        source: The script's Python source.
        *args: The script's command-line arguments.

    Returns:
        The started child.
    """
    script = directory / f"{name}.py"
    script.write_text(source, encoding="utf-8")
    env = dict(os.environ)
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = f"{_REPO_ROOT}{os.pathsep}{existing}" if existing else str(_REPO_ROOT)
    return subprocess.Popen(  # noqa: S603 -- argv is a fixed list, not shell-injected
        [sys.executable, str(script), *args],
        cwd=str(_REPO_ROOT),
        env=env,
        stderr=subprocess.PIPE,
        text=True,
    )


def dead_pid() -> int:
    """Return a pid that is certainly gone: the child ran and was reaped."""
    proc = subprocess.Popen([sys.executable, "-c", ""])
    proc.wait()
    return proc.pid


def _has_exited(pid: int) -> bool:
    """Probe whether ``pid`` has exited, counting a zombie as exited.

    ``os.kill(pid, 0)`` still succeeds for a zombie, so a grandchild killed with
    its process group looks alive until whoever inherited it calls ``wait``.
    That reap is scheduled by the kernel, not by the test, so treating a zombie
    as alive makes every kill assertion race against an unrelated reaper. The
    reap can also land between that probe and this check, so a process whose
    entry has already vanished counts as exited too.

    Args:
        pid: The process to probe.

    Returns:
        True when ``pid`` is a zombie waiting to be reaped or is no longer
        listed; always False on Windows, which has no zombie state.
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


async def poll_until(
    ready: Callable[[], bool],
    timeout_s: float,
    on_timeout: Callable[[], Exception],
) -> None:
    """Poll ``ready`` on the running loop until it holds.

    Args:
        ready: The condition to wait for; it must not block.
        timeout_s: Seconds to poll before giving up.
        on_timeout: Builds the error to raise, called only once the wait has
            expired so its message can describe the state at that point.

    Raises:
        Exception: Whatever ``on_timeout`` builds, once ``ready`` has not held
            within ``timeout_s``.
    """
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_s
    while not ready():
        if loop.time() > deadline:
            raise on_timeout()
        await asyncio.sleep(_POLL_INTERVAL_S)


async def wait_until_dead(pid: int, timeout_s: float = _DEFAULT_WAIT_S) -> None:
    """Poll until the process with ``pid`` no longer exists.

    Args:
        pid: The process ID to wait on.
        timeout_s: Seconds to poll before giving up.

    Raises:
        AssertionError: The process is still alive after ``timeout_s``.
    """
    await poll_until(
        lambda: not is_alive(pid),
        timeout_s,
        lambda: AssertionError(f"process {pid} was still alive after {timeout_s}s"),
    )


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


async def wait_for_file(path: pathlib.Path, timeout_s: float = _DEFAULT_WAIT_S) -> None:
    """Poll until ``path`` exists.

    Args:
        path: The file to wait for.
        timeout_s: Seconds to poll before giving up.

    Raises:
        TimeoutError: ``path`` has not appeared within ``timeout_s``.
    """
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_s
    while not await asyncio.to_thread(path.exists):
        if loop.time() > deadline:
            message = f"file never appeared at {path}"
            raise TimeoutError(message)
        await asyncio.sleep(_POLL_INTERVAL_S)


def wait_for_file_blocking(path: pathlib.Path, timeout_s: float = _DEFAULT_WAIT_S) -> None:
    """Block until ``path`` exists.

    The synchronous twin of ``wait_for_file``, for tests that drive a
    subprocess without an event loop.

    Args:
        path: The file to wait for.
        timeout_s: Seconds to poll before giving up.

    Raises:
        TimeoutError: ``path`` has not appeared within ``timeout_s``.
    """
    deadline = time.monotonic() + timeout_s
    while not path.exists():
        if time.monotonic() > deadline:
            message = f"file never appeared at {path}"
            raise TimeoutError(message)
        time.sleep(_POLL_INTERVAL_S)


@contextlib.contextmanager
def reaped[P: subprocess.Popen[Any]](proc: P) -> Generator[P]:
    """Hand ``proc`` back, then kill and reap it on the way out if it is still running.

    Whatever path the block takes, no child outlives it and none of its pipes
    stays open: a child still running is killed and drained through
    ``communicate``, and the pipes of one that already exited are closed.

    Args:
        proc: The child to guard.

    Yields:
        ``proc`` itself.
    """
    try:
        yield proc
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.communicate()
        for stream in (proc.stdin, proc.stdout, proc.stderr):
            if stream is not None:
                stream.close()


def record_subprocess_runs(
    monkeypatch: "pytest.MonkeyPatch",
    run: Callable[..., subprocess.CompletedProcess[Any]] | None = None,
) -> list[list[str]]:
    """Stub ``subprocess.run`` to record every call's argv before handing the call to ``run``.

    Args:
        monkeypatch: Patches ``subprocess.run`` for the duration of the test.
        run: What each recorded call returns or raises; ``None`` reports a
            clean exit without running anything.

    Returns:
        The list each call's argv is appended to, in call order.
    """
    argv_calls: list[list[str]] = []

    def record_run(
        args: list[str], *rest: object, **kwargs: object
    ) -> subprocess.CompletedProcess[Any]:
        argv_calls.append(args)
        if run is None:
            return subprocess.CompletedProcess(args, 0)
        return run(args, *rest, **kwargs)

    monkeypatch.setattr(subprocess, "run", record_run)
    return argv_calls


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


def track_cleanups(monkeypatch: "pytest.MonkeyPatch", module: str) -> CleanupRegistry:
    """Swap the cleanup installer ``module`` imported for a registry a test can read.

    Args:
        monkeypatch: The fixture that installs the swap.
        module: The dotted name of the module whose ``install_termination_cleanup``
            binding is swapped, e.g. ``"gymrat.sampling"``.

    Returns:
        The registry recording every cleanup ``module`` installs.
    """
    registry = CleanupRegistry()
    monkeypatch.setattr(f"{module}.install_termination_cleanup", registry.install)
    return registry


# The real ``os.killpg``, captured at import so a stand-in can still reach it.
REAL_KILLPG = os.killpg


def _refuse_every_signal(_signal_number: int) -> bool:
    return True


@dataclasses.dataclass
class KillpgRefusal:
    """Stand-in ``os.killpg`` that refuses with ``EPERM`` every signal ``refuses`` accepts.

    Each signal it is handed lands in ``signals``. A signal it does not refuse,
    or any signal once ``refusing`` is cleared, goes to the real ``killpg``, so a
    test can still tear its run down after the refusal it asserted on.
    """

    refuses: Callable[[int], bool] = _refuse_every_signal
    """Whether to refuse a given signal number."""
    refusing: bool = True
    """Whether refusals are switched on at all."""
    signals: list[int] = dataclasses.field(default_factory=list)
    """Every signal number handed to the stand-in, in call order."""

    def __call__(self, group_pid: int, signal_number: int) -> None:
        """Refuse or forward one ``killpg`` call.

        Args:
            group_pid: The process group to signal.
            signal_number: The signal to send.

        Raises:
            PermissionError: When refusals are on and ``refuses`` accepts the signal.
        """
        self.signals.append(signal_number)
        if self.refusing and self.refuses(signal_number):
            raise PermissionError(errno.EPERM, os.strerror(errno.EPERM))
        REAL_KILLPG(group_pid, signal_number)


def refuse_killpg(
    monkeypatch: "pytest.MonkeyPatch",
    refuses: Callable[[int], bool] = _refuse_every_signal,
    *,
    refusing: bool = True,
) -> KillpgRefusal:
    """Install a :class:`KillpgRefusal` as ``os.killpg``.

    Args:
        monkeypatch: The fixture that installs the swap.
        refuses: Whether to refuse a given signal number; refuses every one by default.
        refusing: Whether refusals start switched on.

    Returns:
        The installed stand-in, for reading ``signals`` or switching ``refusing``.
    """
    refusal = KillpgRefusal(refuses=refuses, refusing=refusing)
    monkeypatch.setattr(os, "killpg", refusal)
    return refusal


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


# ---------------------------------------------------------------------------
# win32 Job Objects, faked so a POSIX run reaches them
# ---------------------------------------------------------------------------


@dataclasses.dataclass(slots=True)
class FakeJobs:
    """Answers the Job Object calls of one child and records what was asked of it.

    ``active_counts`` is handed out one per accounting query, so a test can walk
    a job down to empty; once it runs out, every further query reports
    ``final_active``.
    """

    active_counts: list[int] = dataclasses.field(default_factory=list)
    final_active: int = 0
    creation_granted: bool = True
    """Whether ``CreateJobObjectW`` hands out a job, as a locked-down host would not."""

    assignment_granted: bool = True
    """Whether ``AssignProcessToJobObject`` accepts the child, as a locked-down host would not."""

    resume_granted: bool = True
    """Whether ``NtResumeProcess`` accepts the handle, as a locked-down host would not."""

    queried: list[int] = dataclasses.field(default_factory=list)
    closed: list[int] = dataclasses.field(default_factory=list)
    terminated: list[int] = dataclasses.field(default_factory=list)
    limited: list[tuple[int, int]] = dataclasses.field(default_factory=list)
    """Info class and ``BasicLimitInformation.LimitFlags`` of each limits call."""

    opened: dict[int, int] = dataclasses.field(default_factory=dict)
    """Pid behind each process handle handed out by ``OpenProcess``."""

    assigned: list[tuple[int, int]] = dataclasses.field(default_factory=list)
    """Job handle and pid of each successful assignment."""

    resumed: list[int] = dataclasses.field(default_factory=list)
    """Pid behind the handle of each successful resume."""


def fake_kernel32(jobs: FakeJobs) -> types.SimpleNamespace:
    """A ``kernel32`` stub that grants every job call and answers queries from ``jobs``."""

    def query_information_job_object(
        job: object,
        info_class: object,
        info: Any,
        *_args: object,
    ) -> int:
        active = jobs.active_counts.pop(0) if jobs.active_counts else jobs.final_active
        jobs.queried.append(active)
        assert len(jobs.queried) <= _FAKE_JOB_QUERY_CAP, (
            "the wait polled the job past every plausible grace instead of giving up"
        )
        # The production call passes ``ctypes.byref(struct)``; the struct it
        # reads the count back out of is what ``_obj`` reaches.
        info._obj.ActiveProcesses = active
        return 1

    def close_handle(handle: int) -> int:
        jobs.closed.append(handle)
        return 1

    def terminate_job_object(job: int, *_args: object) -> int:
        jobs.terminated.append(job)
        return 1

    def create_job_object(*_args: object) -> int:
        # A NULL handle is how ``CreateJobObjectW`` reports a refusal.
        return JOB_HANDLE if jobs.creation_granted else 0

    def open_process(_access: int, _inherit: bool, pid: int) -> int:
        jobs.opened[PROCESS_HANDLE] = pid
        return PROCESS_HANDLE

    def set_information_job_object(
        _job: int,
        info_class: int,
        info: Any,
        *_args: object,
    ) -> int:
        # ``ctypes.byref(struct)`` again: ``_obj`` is the struct the production
        # code filled in before handing it over.
        jobs.limited.append((info_class, info._obj.BasicLimitInformation.LimitFlags))
        return 1

    def assign_process_to_job_object(job: int, process: int) -> int:
        if not jobs.assignment_granted:
            return 0
        jobs.assigned.append((job, jobs.opened[process]))
        return 1

    return types.SimpleNamespace(
        CreateJobObjectW=create_job_object,
        SetInformationJobObject=set_information_job_object,
        OpenProcess=open_process,
        AssignProcessToJobObject=assign_process_to_job_object,
        CloseHandle=close_handle,
        TerminateJobObject=terminate_job_object,
        QueryInformationJobObject=query_information_job_object,
    )


def fake_ntdll(jobs: FakeJobs) -> types.SimpleNamespace:
    """An ``ntdll`` stub that resumes the process behind a handle ``jobs`` handed out."""

    def resume_process(process: int) -> int:
        if not jobs.resume_granted:
            return _NT_STATUS_ACCESS_DENIED
        jobs.resumed.append(jobs.opened[process])
        return _NT_STATUS_SUCCESS

    return types.SimpleNamespace(NtResumeProcess=resume_process)


def win32_process_group(monkeypatch: "pytest.MonkeyPatch", jobs: FakeJobs) -> types.ModuleType:
    """Load a private copy of ``gymrat.process_group`` with its win32 job path bound.

    The job functions exist only under ``sys.platform == "win32"``, so reaching
    them from a POSIX run means executing the module source again with the
    platform faked and ``kernel32`` stubbed. The copy is the test's own; the
    imported module keeps its POSIX bindings. The platform stays faked for the
    rest of the test.

    Args:
        monkeypatch: Fakes the platform and the ``ctypes`` win32 bindings.
        jobs: Answers the copy's Job Object calls and records them.

    Returns:
        The private ``gymrat.process_group`` copy.
    """
    windll = types.SimpleNamespace(kernel32=fake_kernel32(jobs), ntdll=fake_ntdll(jobs))
    monkeypatch.setattr(ctypes, "windll", windll, raising=False)
    monkeypatch.setattr(ctypes, "WinError", lambda: OSError(_FAKE_WIN_ERROR), raising=False)
    monkeypatch.setattr(sys, "platform", "win32")
    spec = importlib.util.spec_from_file_location("process_group_win32", process_group.__file__)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
