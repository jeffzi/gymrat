"""Stop a spawned child's whole process tree, cross-platform.

A child spawned into its own session or job can leave grandchildren running
when it is torn down, and a child that is itself a gymrat run holds benches
that only it can reach. Teardown is therefore two steps:
:func:`terminate_process_group` asks the tree to stop, the caller gives it a
grace to act on that (:data:`TERMINATE_GRACE_S`, less in a nested run), and
:func:`kill_process_group` takes down whatever is still standing. Neither
raises into the caller: a tree that is already gone is silent, and any other
failure surfaces as a
:class:`RuntimeWarning` — unless the caller opts into ``defer_refusal``, in
which case a POSIX ``EPERM`` refusal is returned silently instead, and the
caller must signal again after reaping the leader.

POSIX children are spawned into their own session, so signaling the group
reaches every descendant. Windows has neither sessions nor a graceful group
signal: :func:`attach_process_group` puts each child in its own Job Object with
kill-on-close, so terminating the job takes the whole tree — including a
descendant whose own parent has already exited — and losing this process
outright does the same through the closing handle. A child assigned while it is
already running can spawn a grandchild that never reaches the job, so the win32
spawn creates the child suspended and :func:`resume_process_group` starts it
once it is assigned: nothing the child does happens before containment.
``subprocess`` closes the child's thread handle, which rules out
``ResumeThread``; the resume therefore goes through ``NtResumeProcess``, which
takes the process handle. A child that cannot be assigned falls back to
``taskkill /T /F``, which walks the parent-child tree instead.
"""

import asyncio
import ctypes
import errno
import functools
import os
import signal
import struct
import subprocess
import sys
import time
import warnings
from collections.abc import Awaitable, Callable, Iterable
from pathlib import Path

# ---------------------------------------------------------------------------
# Tuning constants and clock hooks
# ---------------------------------------------------------------------------

TERMINATE_GRACE_S = 1.0
"""Seconds a tree gets to act on a stop request before the outermost gymrat run kills it.

A nested run waits less, so its own teardown ends inside the wait of the run above it.
"""

monotonic: Callable[[], float] = time.monotonic
"""The clock every group wait measures its grace against.

The waits read it, :data:`sleep` and :data:`async_sleep` from this module at
call time, so replacing all three with a fake clock whose sleeps advance it runs
a wait in no real time.
"""

sleep: Callable[[float], None] = time.sleep
"""The blocking pause between polls of the group waits and the POSIX kill's settle loop.

The group waits are the signal path's group wait and the win32 job wait.
"""

async_sleep: Callable[[float], Awaitable[None]] = asyncio.sleep
"""The pause between polls of :func:`wait_for_process_group_exit_async`."""

_TASKKILL_GONE = 128
"""``taskkill`` exit status meaning the process was already gone."""

EXIT_POLL_S = 0.01
"""Seconds between liveness polls while a grace is waited out or a kill settles."""

KILL_SETTLE_S = 0.1
"""Seconds a POSIX group kill keeps re-killing members that survived its first ``SIGKILL``.

macOS can leave a child that a member forked at the instant of the kill alive
in the group, and every caller drops the group once the kill returns.
"""

_DARWIN_PROC_PGRP_MIB = (1, 14, 2)
"""``CTL_KERN``, ``KERN_PROC``, ``KERN_PROC_PGRP``: the sysctl listing one group's processes.

The group id completes the name. The listing covers zombies too, which is what
lets a refusal from a group holding nothing but zombies be told apart.
"""

_DARWIN_KINFO_PROC_SIZE = 648
"""``sizeof(struct kinfo_proc)``, identical on arm64 and x86_64 macOS."""

_DARWIN_FLAG_AND_STATE = struct.Struct("=iB")
"""``kp_proc.p_flag`` then ``kp_proc.p_stat``, read from their offset in a ``kinfo_proc``."""

_DARWIN_FLAG_AND_STATE_OFFSET = 32
"""Offset of ``kp_proc.p_flag`` in a ``kinfo_proc``; ``p_stat`` follows it directly."""

_DARWIN_P_WEXIT = 0x2000
"""``P_WEXIT``: the process is working on its exit."""

_DARWIN_SZOMB = 5
"""``SZOMB``: the process has exited and waits for its parent to reap it."""

_DARWIN_LISTING_HEADROOM = 4
"""Extra ``kinfo_proc`` slots for members that join between the size query and the read."""

PROC_ROOT = Path("/proc")
"""Where Linux lists every process, as a ``<pid>/stat`` file per process.

Read at call time, so a test can point it at a directory of fake ``stat`` files.
"""

_LINUX_GONE_STATES = frozenset({"Z", "X", "x"})
"""``/proc/<pid>/stat`` states of a process that has exited: zombie or dead."""

_KILL_SIGNAL = signal.SIGTERM if sys.platform == "win32" else signal.SIGKILL
"""Signal that kills a POSIX process group outright.

Windows defines no ``SIGKILL``, and its teardown goes through the child's job
rather than a signal, so the value bound there is never delivered — reading the
attribute at all is what has to be avoided.
"""

# ---------------------------------------------------------------------------
# Windows job objects
# ---------------------------------------------------------------------------

if sys.platform == "win32":
    from ctypes import wintypes

    _JOB_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
    _JOB_EXTENDED_LIMIT_INFORMATION = 9
    _JOB_BASIC_ACCOUNTING_INFORMATION = 1
    _PROCESS_SET_QUOTA = 0x0100
    _PROCESS_TERMINATE = 0x0001
    _PROCESS_SUSPEND_RESUME = 0x0800
    _JOB_KILL_EXIT_CODE = 1
    _NT_STATUS_SUCCESS = 0

    class _BasicLimitInformation(ctypes.Structure):
        """Win32 job-object basic limit information."""

        _fields_ = (
            ("PerProcessUserTimeLimit", wintypes.LARGE_INTEGER),
            ("PerJobUserTimeLimit", wintypes.LARGE_INTEGER),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        )

    class _IoCounters(ctypes.Structure):
        """``IO_COUNTERS``."""

        _fields_ = (
            ("ReadOperationCount", ctypes.c_uint64),
            ("WriteOperationCount", ctypes.c_uint64),
            ("OtherOperationCount", ctypes.c_uint64),
            ("ReadTransferCount", ctypes.c_uint64),
            ("WriteTransferCount", ctypes.c_uint64),
            ("OtherTransferCount", ctypes.c_uint64),
        )

    class _ExtendedLimitInformation(ctypes.Structure):
        """Win32 job-object extended limit information: basic limits plus accounting."""

        _fields_ = (
            ("BasicLimitInformation", _BasicLimitInformation),
            ("IoInfo", _IoCounters),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        )

    class _BasicAccountingInformation(ctypes.Structure):
        """Win32 job-object basic accounting information, carrying the live process count."""

        _fields_ = (
            ("TotalUserTime", wintypes.LARGE_INTEGER),
            ("TotalKernelTime", wintypes.LARGE_INTEGER),
            ("ThisPeriodTotalUserTime", wintypes.LARGE_INTEGER),
            ("ThisPeriodTotalKernelTime", wintypes.LARGE_INTEGER),
            ("TotalPageFaultCount", wintypes.DWORD),
            ("TotalProcesses", wintypes.DWORD),
            ("ActiveProcesses", wintypes.DWORD),
            ("TotalTerminatedProcesses", wintypes.DWORD),
        )

    def _create_kill_on_close_job() -> int | None:
        """A fresh Job Object that kills its members when the last handle closes."""
        kernel32 = ctypes.windll.kernel32
        job = kernel32.CreateJobObjectW(None, None)
        if not job:
            return None
        limits = _ExtendedLimitInformation()
        limits.BasicLimitInformation.LimitFlags = _JOB_LIMIT_KILL_ON_JOB_CLOSE
        limited = kernel32.SetInformationJobObject(
            job,
            _JOB_EXTENDED_LIMIT_INFORMATION,
            ctypes.byref(limits),
            ctypes.sizeof(limits),
        )
        if not limited:
            kernel32.CloseHandle(job)
            return None
        return job

    def _assign_to_job(job: int, pid: int) -> bool:
        """Place ``pid`` in ``job``.

        Nested jobs are supported on every Windows release gymrat runs on, so a
        supervisor that is itself inside a job still gets its child assigned.

        Args:
            job: Handle of the Job Object to place the process in.
            pid: The process ID to assign.

        Returns:
            Whether the assignment succeeded.
        """
        kernel32 = ctypes.windll.kernel32
        inherit_handle = False
        process = kernel32.OpenProcess(_PROCESS_SET_QUOTA | _PROCESS_TERMINATE, inherit_handle, pid)
        if not process:
            return False
        try:
            return bool(kernel32.AssignProcessToJobObject(job, process))
        finally:
            kernel32.CloseHandle(process)

    def _attach_job(pid: int) -> None:
        """Put ``pid`` in a fresh kill-on-close job, warning once when that is refused."""
        job = _create_kill_on_close_job()
        if job is None:
            _warn(f"job creation refused for pid {pid}: {ctypes.WinError()}")
            return
        if not _assign_to_job(job, pid):
            ctypes.windll.kernel32.CloseHandle(job)
            _warn(f"job assignment refused for pid {pid}: {ctypes.WinError()}")
            return
        _job_handles[pid] = job

    def _resume_child(pid: int) -> bool:
        """Start a child that was created suspended, warning once when that is refused.

        Args:
            pid: The suspended child's process ID.

        Returns:
            Whether the child is now running.
        """
        kernel32 = ctypes.windll.kernel32
        inherit_handle = False
        process = kernel32.OpenProcess(_PROCESS_SUSPEND_RESUME, inherit_handle, pid)
        if not process:
            _warn(f"could not resume pid {pid}: opening it was refused: {ctypes.WinError()}")
            return False
        try:
            status = ctypes.windll.ntdll.NtResumeProcess(process)
        finally:
            kernel32.CloseHandle(process)
        if status != _NT_STATUS_SUCCESS:
            _warn(f"could not resume pid {pid}: NTSTATUS 0x{status & 0xFFFFFFFF:08X}")
            return False
        return True

    def _release_job(pid: int) -> None:
        """Tear down the job holding ``pid``, returning once nothing is left in it.

        Closing the last handle kills the members asynchronously, so the kill is
        driven explicitly and waited on instead: a caller that returns from here
        must be able to treat the whole tree as gone.

        Args:
            pid: The process whose job to tear down.
        """
        job = _job_handles.pop(pid, None)
        if job is not None:
            _terminate_job(job, pid)
            ctypes.windll.kernel32.CloseHandle(job)

    def _wait_for_empty_job(job: int) -> None:
        """Block until the job reports no active process, or the grace elapses."""
        kernel32 = ctypes.windll.kernel32
        info = _BasicAccountingInformation()
        deadline = monotonic() + TERMINATE_GRACE_S
        while True:
            queried = kernel32.QueryInformationJobObject(
                job,
                _JOB_BASIC_ACCOUNTING_INFORMATION,
                ctypes.byref(info),
                ctypes.sizeof(info),
                None,
            )
            if not queried or info.ActiveProcesses == 0 or monotonic() >= deadline:
                return
            sleep(EXIT_POLL_S)

    def _terminate_job(job: int, pid: int) -> None:
        """Kill every process in ``job``, returning once the job reports itself empty."""
        if not ctypes.windll.kernel32.TerminateJobObject(job, _JOB_KILL_EXIT_CODE):
            _warn(f"job terminate failed for pid {pid}: {ctypes.WinError()}")
            _taskkill(pid)
            return
        _wait_for_empty_job(job)

    # Bound only on Windows, so every platform branch below stays reachable
    # under a faked ``sys.platform`` — which is how the taskkill fallback is
    # exercised from a POSIX test run.
    _attach_job_impl: Callable[[int], None] | None = _attach_job
    _release_job_impl: Callable[[int], None] | None = _release_job
    _terminate_job_impl: Callable[[int, int], None] | None = _terminate_job
    _resume_child_impl: Callable[[int], bool] | None = _resume_child
else:
    _attach_job_impl: Callable[[int], None] | None = None
    _release_job_impl: Callable[[int], None] | None = None
    _terminate_job_impl: Callable[[int, int], None] | None = None
    _resume_child_impl: Callable[[int], bool] | None = None


_job_handles: dict[int, int] = {}
"""Job Object handle per child PID, for the children this process placed in a job.

Only ever populated on Windows. A PID is present from the moment
:func:`attach_process_group` assigns the child until
:func:`release_process_group` drops the handle, which is also what kills any
descendant the child left behind.
"""


# ---------------------------------------------------------------------------
# Public process-group lifecycle
# ---------------------------------------------------------------------------


def _warn(message: str) -> None:
    """Emit a :class:`RuntimeWarning` attributed to the caller of the helper that warns."""
    warnings.warn(message, RuntimeWarning, stacklevel=3)


def attach_process_group(pid: int) -> None:
    """Put ``pid`` and everything it spawns under this process's control.

    POSIX children already come with their own session, so this is a no-op
    there. On Windows the child joins a fresh kill-on-close Job Object; a
    refusal warns once and leaves the run on the ``taskkill`` fallback.

    Args:
        pid: The freshly spawned child's process ID.
    """
    if _attach_job_impl is not None:
        _attach_job_impl(pid)


def resume_process_group(pid: int) -> bool:
    """Start the child ``pid``, which the win32 spawn created suspended.

    POSIX children run from the moment they are spawned, so this reports success
    there without touching the child. On Windows the child runs no instruction
    until this lands, which is what keeps anything it spawns inside the job it
    was just assigned to; a refusal warns once and leaves the child suspended for
    the caller to kill.

    Args:
        pid: The freshly spawned child's process ID.

    Returns:
        Whether the child is running.
    """
    if _resume_child_impl is None:
        return True
    return _resume_child_impl(pid)


def release_process_group(pid: int) -> None:
    """Drop the container holding ``pid``, ending any descendant it left behind.

    A no-op on POSIX. On Windows the job's kill-on-close limit turns this into
    the last sweep of the tree, so a descendant that outlived the child does not
    outlive the run.

    Args:
        pid: The process ID the run was spawned with.
    """
    if _release_job_impl is not None:
        _release_job_impl(pid)


def _stop_group(pid: int, signal_number: int, *, defer_refusal: bool, settle_s: float) -> bool:
    """Dispatch to the Windows job teardown, or signal the POSIX group."""
    if sys.platform == "win32":
        _stop_win32_tree(pid)
        return False
    return _signal_group(pid, signal_number, defer_refusal=defer_refusal, settle_s=settle_s)


def terminate_process_group(pid: int, *, defer_refusal: bool = False) -> bool:
    """Ask the whole tree led by ``pid`` to stop, never raising into the caller.

    On POSIX this signals the group with ``SIGTERM``, so a child that installed
    its own cleanup gets to run it — that is what reaches a bench the child
    started in a session of its own. Windows has no graceful equivalent: the
    child's job is terminated outright, which already takes the whole tree.

    Args:
        pid: The process ID leading the tree to stop.
        defer_refusal: Return instead of warning when the POSIX group refuses
            the signal with ``EPERM``. The caller must then reap the leader and
            signal again without this flag so a genuine failure still warns.

    Returns:
        ``True`` when ``defer_refusal`` held back an ``EPERM`` refusal,
        ``False`` otherwise.
    """
    return _stop_group(pid, signal.SIGTERM, defer_refusal=defer_refusal, settle_s=0.0)


def kill_process_group(pid: int, *, defer_refusal: bool = False) -> bool:
    """Kill the whole tree led by ``pid``, never raising into the caller.

    On POSIX this signals the group with ``SIGKILL``, staying silent when the
    group is already gone and warning on any other failure. On Windows it
    terminates the child's job, or delegates to ``taskkill /T /F`` when the
    child never made it into one.

    A POSIX kill then settles the group: while a member still runs it polls
    every :data:`EXIT_POLL_S` and kills the group again, for at most
    :data:`KILL_SETTLE_S`. A child a member forked at the instant of the first
    kill can survive it on macOS, and nothing else would stop it once this
    returns. The settle ends as soon as no member runs, at the bound, or at a
    refusal, which is handled like a refusal of the first kill.

    macOS refuses the signal with ``EPERM`` while every member of the group is
    still exiting or is a zombie not yet reaped. That refusal is silent: the
    group's members are listed, and a group holding nothing but zombies and
    exiting processes has nothing left to stop, whether the zombie is the
    leader or an orphaned descendant waiting for init to reap it. Only a
    refusal while some member is still running warns.

    Args:
        pid: The process ID leading the tree to kill.
        defer_refusal: Return instead of warning when the POSIX group refuses
            the signal with ``EPERM``. The caller must then reap the leader and
            call again without this flag so a genuine failure still warns.

    Returns:
        ``True`` when ``defer_refusal`` held back an ``EPERM`` refusal,
        ``False`` otherwise.
    """
    return _stop_group(pid, _KILL_SIGNAL, defer_refusal=defer_refusal, settle_s=KILL_SETTLE_S)


# ---------------------------------------------------------------------------
# Group exit waits
# ---------------------------------------------------------------------------


def wait_for_process_group_exit(leaders: Iterable[int], timeout_s: float) -> None:
    """Block until no member of any group led by ``leaders`` is running, or ``timeout_s`` elapses.

    This is the signal path's grace, where there is no event loop to await on: a
    child asked to stop gets this long to tear down benches of its own before it
    is killed. The whole group is waited for, not just its leader: a leader
    that dies on the request can leave a nested run behind in its group, still
    cleaning up. A member that exited counts as gone even though nothing has
    reaped it yet, so the wait ends as soon as nothing in the groups still runs.
    Where a group's members cannot be listed, the wait falls back to its leader
    alone. Windows offers no pid probe that is not itself destructive, and its
    terminate already waits for the job to empty, so there the wait is a no-op.

    Args:
        leaders: Process IDs of the group leaders to wait for.
        timeout_s: Seconds to wait before giving up on the stragglers.
    """
    if sys.platform == "win32":
        return
    waited = list(leaders)
    deadline = monotonic() + timeout_s
    while any(_group_running(pid) for pid in waited):
        if monotonic() >= deadline:
            return
        sleep(EXIT_POLL_S)


async def wait_for_process_group_exit_async(leader: int, timeout_s: float) -> None:
    """Wait until no member of the group led by ``leader`` is running, or ``timeout_s`` elapses.

    The event-loop twin of :func:`wait_for_process_group_exit`, for one group:
    it polls without blocking the loop, and never reaps the leader, so the loop
    that owns it still collects its exit status.

    Args:
        leader: Process ID of the group leader to wait for.
        timeout_s: Seconds to wait before giving up on the stragglers.
    """
    if sys.platform == "win32":
        return
    deadline = monotonic() + timeout_s
    while _group_running(leader):
        if monotonic() >= deadline:
            return
        await async_sleep(EXIT_POLL_S)


# ---------------------------------------------------------------------------
# Liveness probing
# ---------------------------------------------------------------------------


def _group_running(group_id: int) -> bool:
    """Whether the leader ``group_id``, or any other member of its group, is still running.

    Zombies and exiting members count as gone. Only macOS and Linux list a
    group's members; elsewhere, and whenever the listing fails, the leader alone
    decides.

    Args:
        group_id: The process group, whose id is also its leader's pid.

    Returns:
        ``True`` while the leader or a listed member still runs.
    """
    # A zombie leader awaiting its reap still answers ``kill`` but runs nothing.
    if _pid_exists(group_id) and not _has_exited(group_id):
        return True
    try:
        if sys.platform == "darwin":
            return not all(
                _darwin_member_gone(flag, state) for flag, state in _darwin_group_members(group_id)
            )
        if sys.platform == "linux":
            return any(state not in _LINUX_GONE_STATES for state in _linux_group_states(group_id))
    # A listing that fails or cannot be parsed leaves the leader alone to decide.
    except (OSError, ValueError):
        return False
    return False


def _darwin_member_gone(flag: int, state: int) -> bool:
    # A zombie or a member already exiting no longer runs any code of its own.
    return state == _DARWIN_SZOMB or bool(flag & _DARWIN_P_WEXIT)


def _linux_group_states(group_id: int) -> list[str]:
    """List the ``/proc/<pid>/stat`` state of every member of group ``group_id``, zombies included.

    Linux has no call listing one group, so every process is scanned. A process
    that exits between the scan and the read of its ``stat`` file is skipped.

    Args:
        group_id: The process group to list.

    Returns:
        One state letter per member; empty when the group no longer exists.

    Raises:
        OSError: ``/proc`` could not be listed or a ``stat`` file could not be read.
        ValueError: A ``stat`` file does not hold the fields Linux writes there.
    """
    states: list[str] = []
    for entry in PROC_ROOT.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat = (entry / "stat").read_bytes()
        except (FileNotFoundError, ProcessLookupError):
            continue
        # The command name before the state is parenthesized and may itself
        # hold spaces and parentheses, so the fields start after the last ")".
        # The name is never decoded: the kernel cuts it at 15 bytes, which can
        # split a multi-byte character, so only the fields after it are text.
        fields = stat.rpartition(b")")[2].decode("ascii")
        state, _parent, member_group = fields.split(maxsplit=3)[:3]
        if int(member_group) == group_id:
            states.append(state)
    return states


def _pid_exists(pid: int) -> bool:
    """Whether ``pid`` names a process, running or zombie, that ``kill`` can still reach."""
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _has_exited(pid: int) -> bool:
    """Whether ``pid`` has exited, its status collected or not, without collecting it."""
    if sys.platform != "darwin" or sys.version_info >= (3, 13):
        try:
            state = os.waitid(os.P_PID, pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
        except ChildProcessError:
            # Either not a child of this process, or a child another thread (the
            # event loop's child watcher) reaped since the caller probed it. Only
            # the first is still running, and only the first is still there.
            return not _pid_exists(pid)
        return state is not None
    # macOS gains waitid only in Python 3.13. Until then a zombie is the process
    # kill(pid, 0) still reaches but getpgid no longer finds: the kernel looks
    # getpgid up among running processes only.
    try:
        os.getpgid(pid)
    except ProcessLookupError:
        return True
    return False


# ---------------------------------------------------------------------------
# POSIX signaling and refusal handling
# ---------------------------------------------------------------------------


def _signal_group(pid: int, signal_number: int, *, defer_refusal: bool, settle_s: float) -> bool:
    """Signal the POSIX process group led by ``pid``, warning only when a live member refuses.

    macOS skips zombies and exiting processes when it signals a group, and
    answers ``EPERM`` when that leaves nothing to signal. Such a group has no
    member left to stop, so its refusal is silent whoever the zombie is — the
    leader not yet reaped, or an orphaned descendant waiting for init to reap it.

    Args:
        pid: The process ID leading the group.
        signal_number: The signal to send.
        defer_refusal: Return instead of warning on an ``EPERM`` refusal.
        settle_s: Seconds to keep signalling again, every :data:`EXIT_POLL_S`,
            while a member still runs; ``0`` signals once.

    Returns:
        ``True`` when ``defer_refusal`` held back an ``EPERM`` refusal,
        ``False`` otherwise.
    """
    deadline = monotonic() + settle_s
    while True:
        try:
            os.killpg(pid, signal_number)
        except OSError as error:
            return _handle_refusal(pid, error, defer_refusal=defer_refusal)
        if monotonic() >= deadline or not _group_running(pid):
            return False
        sleep(EXIT_POLL_S)


def _handle_refusal(pid: int, error: OSError, *, defer_refusal: bool) -> bool:
    """Stay silent on a gone or settled group, defer or warn on any other ``killpg`` failure.

    Args:
        pid: The process ID leading the group that refused.
        error: The failure ``killpg`` raised.
        defer_refusal: Return instead of warning on an ``EPERM`` refusal.

    Returns:
        ``True`` when ``defer_refusal`` held back an ``EPERM`` refusal,
        ``False`` otherwise.
    """
    if isinstance(error, ProcessLookupError):
        return False
    if error.errno == errno.EPERM:
        if defer_refusal:
            return True
        if _group_settled(pid):
            return False
    _warn(f"killpg failed for pid {pid}: {error}")
    return False


def _group_settled(group_id: int) -> bool:
    """Whether the group ``group_id`` has nothing left to stop after an ``EPERM`` refusal.

    Only macOS lists a group's members with their state, and only macOS refuses
    a group holding nothing but zombies. Linux delivers a group signal to
    zombies, so a refusal there always comes from a live member.

    Args:
        group_id: The process group that refused the signal.

    Returns:
        ``True`` when every member is a zombie or exiting, or the group is
        confirmed gone; ``False`` whenever a live member cannot be ruled out,
        including when the listing fails.
    """
    return sys.platform == "darwin" and _darwin_group_settled(group_id)


def _darwin_group_settled(group_id: int) -> bool:
    """Whether macOS lists the group ``group_id`` as holding only zombies, or as gone.

    An empty listing means every member vanished since the refusal — a zombie
    reaped in the meantime — or that the refusal did not come from the kernel
    at all. A probe tells them apart: the kernel answers ``ESRCH`` for a group
    that is really gone.

    Kept apart from ``_group_settled`` so type checkers running for another
    platform still analyze this body instead of treating it as unreachable.

    Args:
        group_id: The process group that refused the signal.

    Returns:
        ``True`` when every member is a zombie or exiting, or the group is
        confirmed gone; ``False`` whenever a live member cannot be ruled out,
        including when the listing fails.
    """
    try:
        members = _darwin_group_members(group_id)
    except OSError:
        return False
    if members:
        return all(_darwin_member_gone(flag, state) for flag, state in members)
    try:
        os.killpg(group_id, 0)
    except ProcessLookupError:
        return True
    except OSError:
        return False
    return False


# ---------------------------------------------------------------------------
# macOS sysctl listing
# ---------------------------------------------------------------------------


@functools.cache
def c_library() -> ctypes.CDLL:
    """Return this process's C library, with ``sysctl`` typed, for the macOS member listing.

    The listing looks this function up at call time, so a test can replace it
    with a stub whose ``sysctl`` lists fake members or fails.

    Returns:
        The C library, loaded once and cached.
    """
    library = ctypes.CDLL(None, use_errno=True)
    library.sysctl.argtypes = (
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_uint,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_void_p,
        ctypes.c_size_t,
    )
    library.sysctl.restype = ctypes.c_int
    return library


def _sysctl(
    mib: ctypes.Array[ctypes.c_int], buffer: ctypes.Array[ctypes.c_char] | None, size: int
) -> int:
    """Read the sysctl ``mib`` into ``buffer``, or only measure it when ``buffer`` is ``None``.

    Args:
        mib: The sysctl name.
        buffer: Where the value is copied, or ``None`` to query its size.
        size: The capacity of ``buffer``.

    Returns:
        The number of bytes the value holds, or was copied.

    Raises:
        OSError: ``sysctl`` failed, including a value outgrowing ``buffer``.
    """
    length = ctypes.c_size_t(size)
    if c_library().sysctl(mib, len(mib), buffer, ctypes.byref(length), None, 0) != 0:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code))
    return length.value


def _darwin_group_members(group_id: int) -> list[tuple[int, int]]:
    """List the ``(p_flag, p_stat)`` pair of every member of group ``group_id``, zombies included.

    Args:
        group_id: The process group to list.

    Returns:
        One pair per member; empty when the group no longer exists.

    Raises:
        OSError: ``sysctl`` failed, including a group that outgrew its headroom
            between the size query and the read.
    """
    mib = (ctypes.c_int * (len(_DARWIN_PROC_PGRP_MIB) + 1))(*_DARWIN_PROC_PGRP_MIB, group_id)
    capacity = _sysctl(mib, None, 0) + _DARWIN_LISTING_HEADROOM * _DARWIN_KINFO_PROC_SIZE
    buffer = ctypes.create_string_buffer(capacity)
    length = _sysctl(mib, buffer, capacity)
    # A trailing partial record holds bytes the kernel never wrote: the zeroed
    # buffer would decode as a running member.
    whole_records_length = length - length % _DARWIN_KINFO_PROC_SIZE
    return [
        _DARWIN_FLAG_AND_STATE.unpack_from(buffer, offset + _DARWIN_FLAG_AND_STATE_OFFSET)
        for offset in range(0, whole_records_length, _DARWIN_KINFO_PROC_SIZE)
    ]


# ---------------------------------------------------------------------------
# Windows fallback
# ---------------------------------------------------------------------------


def _taskkill(pid: int) -> None:
    """Walk ``pid``'s parent-child tree with ``taskkill /T /F``, warning on real failures."""
    argv = ["taskkill", "/F", "/T", "/PID", str(pid)]
    try:
        subprocess.run(argv, capture_output=True, check=True)  # noqa: S603 -- argv is a fixed list, not shell-injected
    except subprocess.CalledProcessError as error:
        if error.returncode != _TASKKILL_GONE:
            _warn(f"taskkill failed for pid {pid}: {error}")
    except OSError as os_error:
        _warn(f"taskkill unavailable while killing pid {pid}: {os_error}")


def _stop_win32_tree(pid: int) -> None:
    """Terminate the job holding ``pid``, or fall back to ``taskkill`` when it has none.

    The fallback is the whole win32 path for a child that was never assigned —
    an assignment the host refused, or a platform faked after a POSIX spawn.

    Args:
        pid: The root process of the tree to stop.
    """
    job = _job_handles.get(pid)
    if job is None or _terminate_job_impl is None:
        _taskkill(pid)
        return
    _terminate_job_impl(job, pid)
