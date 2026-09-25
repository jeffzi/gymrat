"""Stop a spawned child's whole process tree, cross-platform.

A child spawned into its own session or job can leave grandchildren running
when it is torn down, and a child that is itself a gymrat run holds benches
that only it can reach. Teardown is therefore two steps:
:func:`terminate_process_group` asks the tree to stop and gives it
:data:`TERMINATE_GRACE_S` to act on that, and :func:`kill_process_group` takes
down whatever is still standing. Neither raises into the caller: a tree that is
already gone is silent, and any other failure surfaces as a
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
from collections.abc import Callable, Iterable

TERMINATE_GRACE_S = 1.0
"""Seconds a tree gets to act on a stop request before it is killed."""

_TASKKILL_GONE = 128
"""``taskkill`` exit status meaning the process was already gone."""

_EXIT_POLL_S = 0.01
"""Seconds between liveness polls while a grace is waited out."""

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

_KILL_SIGNAL = signal.SIGTERM if sys.platform == "win32" else signal.SIGKILL
"""Signal that kills a POSIX process group outright.

Windows defines no ``SIGKILL``, and its teardown goes through the child's job
rather than a signal, so the value bound there is never delivered — reading the
attribute at all is what has to be avoided.
"""

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
        deadline = time.monotonic() + TERMINATE_GRACE_S
        while True:
            queried = kernel32.QueryInformationJobObject(
                job,
                _JOB_BASIC_ACCOUNTING_INFORMATION,
                ctypes.byref(info),
                ctypes.sizeof(info),
                None,
            )
            if not queried or info.ActiveProcesses == 0 or time.monotonic() >= deadline:
                return
            time.sleep(_EXIT_POLL_S)

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


def _warn(message: str) -> None:
    """Raise a :class:`RuntimeWarning` attributed to this module's caller."""
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


def _stop_group(pid: int, signal_number: int, *, defer_refusal: bool) -> bool:
    """Dispatch to the Windows job teardown, or signal the POSIX group."""
    if sys.platform == "win32":
        _stop_win32_tree(pid)
        return False
    return _signal_group(pid, signal_number, defer_refusal=defer_refusal)


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
    return _stop_group(pid, signal.SIGTERM, defer_refusal=defer_refusal)


def kill_process_group(pid: int, *, defer_refusal: bool = False) -> bool:
    """Kill the whole tree led by ``pid``, never raising into the caller.

    On POSIX this signals the group with ``SIGKILL``, staying silent when the
    group is already gone and warning on any other failure. On Windows it
    terminates the child's job, or delegates to ``taskkill /T /F`` when the
    child never made it into one.

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
    return _stop_group(pid, _KILL_SIGNAL, defer_refusal=defer_refusal)


def wait_for_process_group_exit(leaders: Iterable[int], timeout_s: float) -> None:
    """Block until every leader in ``leaders`` has exited, or ``timeout_s`` elapses.

    This is the signal path's grace, where there is no event loop to await on: a
    child asked to stop gets this long to tear down benches of its own before it
    is killed. A leader that exited counts as gone even though nothing has
    reaped it yet, so the wait ends as soon as every leader has stopped. Windows
    offers no pid probe that is not itself destructive, and its terminate
    already waits for the job to empty, so there the wait is a no-op.

    Args:
        leaders: Process IDs of the group leaders to wait for.
        timeout_s: Seconds to wait before giving up on the stragglers.
    """
    if sys.platform == "win32":
        return
    waited = list(leaders)
    deadline = time.monotonic() + timeout_s
    while any(_leader_alive(pid) for pid in waited):
        if time.monotonic() >= deadline:
            return
        time.sleep(_EXIT_POLL_S)


def _leader_alive(pid: int) -> bool:
    """Whether a process with ``pid`` is still running; a zombie awaiting its reap is not."""
    return _pid_exists(pid) and not _has_exited(pid)


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


def _signal_group(pid: int, signal_number: int, *, defer_refusal: bool) -> bool:
    """Signal the POSIX process group led by ``pid``, warning only when a live member refuses.

    macOS skips zombies and exiting processes when it signals a group, and
    answers ``EPERM`` when that leaves nothing to signal. Such a group has no
    member left to stop, so its refusal is silent whoever the zombie is — the
    leader not yet reaped, or an orphaned descendant waiting for init to reap it.

    Args:
        pid: The process ID leading the group.
        signal_number: The signal to send.
        defer_refusal: Return instead of warning on an ``EPERM`` refusal.

    Returns:
        ``True`` when ``defer_refusal`` held back an ``EPERM`` refusal,
        ``False`` otherwise.
    """
    try:
        os.killpg(pid, signal_number)
    except ProcessLookupError:
        pass
    except OSError as error:
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
        return all(state == _DARWIN_SZOMB or flag & _DARWIN_P_WEXIT for flag, state in members)
    try:
        os.killpg(group_id, 0)
    except ProcessLookupError:
        return True
    except OSError:
        return False
    return False


@functools.cache
def _c_library() -> ctypes.CDLL:
    """This process's C library, with ``sysctl`` typed."""
    c_library = ctypes.CDLL(None, use_errno=True)
    c_library.sysctl.argtypes = (
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_uint,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_void_p,
        ctypes.c_size_t,
    )
    c_library.sysctl.restype = ctypes.c_int
    return c_library


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
    if _c_library().sysctl(mib, len(mib), buffer, ctypes.byref(length), None, 0) != 0:
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
    return [
        _DARWIN_FLAG_AND_STATE.unpack_from(buffer, offset + _DARWIN_FLAG_AND_STATE_OFFSET)
        for offset in range(0, length, _DARWIN_KINFO_PROC_SIZE)
    ]


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
