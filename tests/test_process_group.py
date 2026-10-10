"""Behavioral tests for probing and signalling POSIX process groups."""

import asyncio
import ctypes
import dataclasses
import errno
import os
import signal
import struct
import subprocess
import sys
import types
import warnings
from collections.abc import Awaitable, Callable, Iterator
from pathlib import Path

import pytest

from gymrat import process_group
from gymrat.process_group import (
    kill_process_group,
    wait_for_process_group_exit,
    wait_for_process_group_exit_async,
)
from tests._clock import WaitClock, install_wait_clock, within_hang_guard
from tests._process_helpers import (
    JOB_HANDLE,
    PROCESS_HANDLE,
    SLEEPER_ARGV,
    ZOMBIE_ONLY_GROUP_SCRIPT,
    FakeJobs,
    dead_pid,
    is_alive,
    killpg_warnings,
    record_subprocess_runs,
    refuse_killpg,
    wait_for_pid_file,
    wait_for_pid_file_blocking,
    wait_until_dead,
    wait_until_dead_blocking,
    win32_process_group,
)
from tests._process_helpers import (
    KILLPG_FAILED as _KILLPG_FAILED,
)

# The win32 Job Object tests at the end fake the platform and run everywhere;
# every test before them needs real POSIX process groups.
_POSIX_ONLY = pytest.mark.skipif(
    sys.platform == "win32", reason="POSIX-only zombie and reap semantics"
)

# The C signature of ``sysctl``, so a stand-in receives real pointers to write through.
_SysctlFunction = ctypes.CFUNCTYPE(
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_int),
    ctypes.c_uint,
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_size_t),
    ctypes.c_void_p,
    ctypes.c_size_t,
)

# Exit status of the short-lived child; a reap by anyone else would make
# ``Popen.wait`` report 0 instead.
_CHILD_STATUS = 7

# Grace a wait on real time is given, far longer than any probe takes.
_LONG_WAIT_S = 10.0

# Grace a wait on the fake clock is given. It costs no real time, so it is sized
# to sit far above the real time a fake-clock wait takes.
_WAIT_S = 10.0

# How far past its grace a wait may run on the fake clock: one poll of the
# group, which the wait sleeps between checks.
_ONE_POLL_S = 0.01

# Real seconds a fake-clock wait may take: far below ``_WAIT_S``, so a wait
# that slept on real time instead fails rather than passing slowly.
_HANG_GUARD_S = _WAIT_S / 2

# What a wait shows on the fake clock: it returned without sleeping, or it slept
# until the clock reached its grace, give or take one poll.
_NO_SLEEP = 0.0
_TIMED_OUT = pytest.approx(_WAIT_S, abs=_ONE_POLL_S)

# The macOS ``kinfo_proc`` layout, spelled out from <sys/sysctl.h> and
# <sys/proc.h> rather than read from the module under test, so a wrong value
# there cannot also shape the records these tests feed it.
_DARWIN_KINFO_PROC_SIZE = 648
_DARWIN_FLAG_AND_STATE = struct.Struct("=iB")  # p_flag, then p_stat right after it
_DARWIN_FLAG_AND_STATE_OFFSET = 32
_DARWIN_P_WEXIT = 0x2000
_DARWIN_SZOMB = 5
_DARWIN_RUNNING = 2

# How long a group member outlives its leader: long enough to still be running
# when the wait starts, short enough to keep the test cheap.
_MEMBER_LIFETIME_S = 1.0

# Group id and member pid listed in a fake ``/proc``. Every probe of them goes
# through a stand-in ``os``, so they never reach a real process.
_FAKE_GROUP = 424242
_FAKE_MEMBER = 424243

# The fields a Linux ``/proc/<pid>/stat`` line carries after the process group
# id: session, tty, terminal foreground group, flags, then fault and time counters.
_STAT_TAIL = "0 -1 4194560 120 0 0 0 3 1 0 0 20 0 1 0 5000"

# A process name the kernel cut at its 15-byte limit in the middle of a
# multi-byte character: the first two of the three UTF-8 bytes of "€".
_CUT_NAME = b"bench-costs-5" + "€".encode()[:2]


@dataclasses.dataclass(frozen=True, slots=True)
class FakeProcess:
    """A process listed in a fake ``/proc``, in the group ``group_id`` (``_FAKE_GROUP`` by default)."""

    pid: int
    state: str
    comm: bytes = b"bench"
    group_id: int = _FAKE_GROUP

    def write(self, proc_root: Path) -> None:
        """Write this process's ``stat`` file under ``proc_root``, laid out as Linux lays it out."""
        entry = proc_root / str(self.pid)
        entry.mkdir(exist_ok=True)
        head = f"{self.pid} (".encode()
        tail = f") {self.state} 1 {self.group_id} {_STAT_TAIL}\n".encode()
        (entry / "stat").write_bytes(head + self.comm + tail)


# The exited, not yet reaped, leader every fake group below starts from.
_ZOMBIE_LEADER = FakeProcess(_FAKE_GROUP, state="Z")


@dataclasses.dataclass(frozen=True, slots=True)
class ProbedLeader:
    """Stand-in ``os`` whose probes find every process present, and exited when ``exited`` is set.

    ``kill`` and ``killpg`` with signal 0 succeed, as they do on Linux for a
    zombie as much as for a running process; ``waitid`` reports an exit, and
    ``getpgid`` fails the lookup, only when ``exited`` is set. Every other
    attribute forwards to the real ``os`` module.
    """

    exited: bool

    P_PID: int = getattr(os, "P_PID", 0)
    WEXITED: int = getattr(os, "WEXITED", 0)
    WNOHANG: int = getattr(os, "WNOHANG", 0)
    WNOWAIT: int = getattr(os, "WNOWAIT", 0)

    def __getattr__(self, name: str) -> object:
        return getattr(os, name)

    def kill(self, pid: int, signal_number: int, /) -> None:
        pass

    def killpg(self, group_id: int, signal_number: int, /) -> None:
        pass

    def getpgid(self, pid: int, /) -> int:
        # macOS before Python 3.13 has no waitid, so an exit shows as a lookup failure.
        if self.exited:
            raise ProcessLookupError(pid)
        return pid

    def waitid(self, id_type: int, id_val: int, options: int, /) -> object:
        if not self.exited:
            return None
        return types.SimpleNamespace(si_pid=id_val, si_status=0, si_code=1)


@dataclasses.dataclass(frozen=True, slots=True)
class LingeringGroup:
    """A process group whose leader has exited while another member still runs."""

    leader_pid: int
    member_pid: int


@dataclasses.dataclass
class ReapedMidProbe:
    """Stand-in ``os`` whose child is reaped by another thread during the zombie probe.

    ``kill(pid, 0)`` finds the process until ``waitid`` runs; ``waitid`` then
    fails with ``ChildProcessError`` because the reap won the race, and from
    that point on the pid no longer exists. Every other attribute forwards to
    the real ``os`` module.
    """

    reaped: bool = False

    P_PID: int = getattr(os, "P_PID", 0)
    WEXITED: int = getattr(os, "WEXITED", 0)
    WNOHANG: int = getattr(os, "WNOHANG", 0)
    WNOWAIT: int = getattr(os, "WNOWAIT", 0)

    def __getattr__(self, name: str) -> object:
        return getattr(os, name)

    def kill(self, pid: int, signal_number: int, /) -> None:
        if self.reaped:
            raise ProcessLookupError(pid)

    def waitid(self, id_type: int, id_val: int, options: int, /) -> object:
        self.reaped = True
        raise ChildProcessError(id_val)


def _refuses_all_but_probe(signal_number: int) -> bool:
    return signal_number != 0


@dataclasses.dataclass
class StubCLibrary:
    """Stand-in C library whose ``sysctl`` fails with ``failure_errno``, or lists ``listing`` when it is ``None``.

    A query without a buffer is told the byte length of ``listing``; a query
    with one gets as much of ``listing`` as fits, and is told its full length.
    """

    failure_errno: int | None
    listing: bytes = b""
    sysctl: object = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        self.sysctl = _SysctlFunction(self._sysctl)

    def _sysctl(
        self,
        _name: object,
        _name_length: int,
        old: ctypes.Array[ctypes.c_char] | None,
        old_length: "ctypes._Pointer[ctypes.c_size_t]",
        _new: object,
        _new_length: int,
    ) -> int:
        if self.failure_errno is not None:
            ctypes.set_errno(self.failure_errno)
            return -1
        if old is not None:
            ctypes.memmove(old, self.listing, min(old_length[0], len(self.listing)))
        old_length[0] = len(self.listing)
        return 0


def darwin_record(flag: int, state: int) -> bytes:
    """One macOS ``kinfo_proc`` record carrying ``flag`` as ``p_flag`` and ``state`` as ``p_stat``."""
    record = bytearray(_DARWIN_KINFO_PROC_SIZE)
    _DARWIN_FLAG_AND_STATE.pack_into(record, _DARWIN_FLAG_AND_STATE_OFFSET, flag, state)
    return bytes(record)


@dataclasses.dataclass
class SurvivingMember:
    """Stand-in ``os`` and macOS C library for a group whose one member survives early SIGKILLs.

    The member runs until the group has been sent more than ``kills_survived``
    SIGKILLs, then lists as exiting. Every signal delivered to the group, probes
    with signal 0 aside, lands in ``signals``. The leader has exited, as
    :class:`ProbedLeader` shows it; every other attribute forwards there.
    """

    kills_survived: int
    signals: list[int] = dataclasses.field(default_factory=list)
    library: StubCLibrary = dataclasses.field(
        default_factory=lambda: StubCLibrary(failure_errno=None)
    )
    _leader: ProbedLeader = dataclasses.field(default_factory=lambda: ProbedLeader(exited=True))

    def __post_init__(self) -> None:
        self._list_member()

    def __getattr__(self, name: str) -> object:
        return getattr(self._leader, name)

    def killpg(self, _group_id: int, signal_number: int, /) -> None:
        if signal_number == 0:
            return
        self.signals.append(signal_number)
        self._list_member()

    def _list_member(self) -> None:
        running = self.signals.count(signal.SIGKILL) <= self.kills_survived
        flag = 0 if running else _DARWIN_P_WEXIT
        self.library.listing = darwin_record(flag, _DARWIN_RUNNING)


@pytest.fixture
def sleeping_group_leader() -> Iterator[subprocess.Popen[bytes]]:
    """A running child leading a process group of its own, killed and reaped on teardown."""
    proc = subprocess.Popen(SLEEPER_ARGV, start_new_session=True)  # noqa: S603 -- argv is a fixed list, not shell-injected
    yield proc
    proc.kill()
    proc.wait()


@pytest.fixture
async def orphaned_zombie_group(tmp_path: Path, stray_process_ids: list[int]) -> int:
    """The id of a group whose leader was killed and reaped, leaving only a zombie member."""
    holder_pid_file = tmp_path / "holder.pid"
    leader = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        ZOMBIE_ONLY_GROUP_SCRIPT,
        str(holder_pid_file),
        start_new_session=True,
    )
    stray_process_ids.append(await wait_for_pid_file(holder_pid_file))
    leader.kill()
    await leader.wait()
    return leader.pid


@pytest.fixture
def group_with_lingering_member(
    tmp_path: Path, stray_process_ids: list[int]
) -> Iterator[LingeringGroup]:
    """A group whose leader has exited, not yet reaped, while a member it started keeps running."""
    member_pid_file = tmp_path / "member.pid"
    leader = subprocess.Popen(  # noqa: S603 -- fixed argv
        [
            "/bin/sh",
            "-c",
            f'sleep {_MEMBER_LIFETIME_S} & echo $! > "$1"',
            "sh",
            str(member_pid_file),
        ],
        start_new_session=True,
    )
    member_pid = wait_for_pid_file_blocking(member_pid_file)
    stray_process_ids.append(member_pid)
    wait_until_dead_blocking(leader.pid)
    yield LingeringGroup(leader_pid=leader.pid, member_pid=member_pid)
    leader.wait()


@pytest.fixture
def linux_proc_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """An empty fake ``/proc`` the module reads as the host's, under a faked Linux platform."""
    proc_root = tmp_path / "proc"
    proc_root.mkdir()
    (proc_root / "meminfo").write_text("MemTotal: 1024 kB\n")
    (proc_root / "sys").mkdir()
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(process_group, "PROC_ROOT", proc_root)
    return proc_root


@pytest.fixture
def wait_clock(monkeypatch: pytest.MonkeyPatch) -> WaitClock:
    """Run the process-group waits on a fake clock that advances only when they sleep."""
    return install_wait_clock(monkeypatch, process_group)


@pytest.fixture
def exiting_child() -> Iterator[subprocess.Popen[bytes]]:
    """A child that exits with ``_CHILD_STATUS`` at once, reaped on teardown if the test did not."""
    proc = subprocess.Popen([sys.executable, "-c", f"raise SystemExit({_CHILD_STATUS})"])  # noqa: S603 -- fixed argv
    yield proc
    proc.wait()


@_POSIX_ONLY
def test_wait_for_process_group_exit_when_leader_reaped_during_probe_does_return_without_sleeping(
    monkeypatch: pytest.MonkeyPatch,
    wait_clock: WaitClock,
) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(process_group, "os", ReapedMidProbe())

    with within_hang_guard(_HANG_GUARD_S):
        wait_for_process_group_exit([_FAKE_GROUP], _WAIT_S)

    assert wait_clock.sleeps == []


def _child_leader(leader: subprocess.Popen[bytes]) -> int:
    return leader.pid


def _parent_process(_leader: subprocess.Popen[bytes]) -> int:
    return os.getppid()


@_POSIX_ONLY
@pytest.mark.parametrize(
    "leader_of",
    [
        pytest.param(_child_leader, id="child-leader"),
        pytest.param(_parent_process, id="non-child-leader"),
    ],
)
def test_wait_for_process_group_exit_when_leader_running_does_sleep_until_the_timeout(
    sleeping_group_leader: subprocess.Popen[bytes],
    wait_clock: WaitClock,
    leader_of: Callable[[subprocess.Popen[bytes]], int],
) -> None:
    with within_hang_guard(_HANG_GUARD_S):
        wait_for_process_group_exit([leader_of(sleeping_group_leader)], _WAIT_S)

    assert wait_clock.now == _TIMED_OUT


@_POSIX_ONLY
async def test_wait_for_process_group_exit_when_leader_is_zombie_does_return_without_sleeping_or_reaping(
    exiting_child: subprocess.Popen[bytes],
    wait_clock: WaitClock,
) -> None:
    await wait_until_dead(exiting_child.pid)

    with within_hang_guard(_HANG_GUARD_S):
        wait_for_process_group_exit([exiting_child.pid], _WAIT_S)

    assert wait_clock.sleeps == []
    assert exiting_child.wait() == _CHILD_STATUS


@_POSIX_ONLY
def test_wait_for_process_group_exit_when_leader_exited_but_member_running_does_wait_for_member(
    group_with_lingering_member: LingeringGroup,
) -> None:
    wait_for_process_group_exit([group_with_lingering_member.leader_pid], _LONG_WAIT_S)

    assert not is_alive(group_with_lingering_member.member_pid)


@_POSIX_ONLY
@pytest.mark.parametrize(
    ("listed", "slept"),
    [
        pytest.param(
            [_ZOMBIE_LEADER, FakeProcess(_FAKE_MEMBER, state="S")], _TIMED_OUT, id="sleeping-member"
        ),
        pytest.param(
            [_ZOMBIE_LEADER, FakeProcess(_FAKE_MEMBER, state="R")], _TIMED_OUT, id="running-member"
        ),
        pytest.param(
            [
                _ZOMBIE_LEADER,
                FakeProcess(_FAKE_MEMBER, state="S", comm=f"a) Z 1 {_FAKE_GROUP}".encode()),
            ],
            _TIMED_OUT,
            id="member-name-mimics-zombie-fields",
        ),
        pytest.param(
            [
                _ZOMBIE_LEADER,
                FakeProcess(_FAKE_MEMBER, state="S"),
                FakeProcess(_FAKE_MEMBER + 1, state="S", comm=_CUT_NAME, group_id=_FAKE_GROUP + 10),
            ],
            _TIMED_OUT,
            id="unrelated-name-cut-mid-character",
        ),
        pytest.param(
            [_ZOMBIE_LEADER, FakeProcess(_FAKE_MEMBER, state="S", comm=b"a) (Z " + _CUT_NAME)],
            _TIMED_OUT,
            id="member-name-cut-mid-character-with-parentheses",
        ),
        pytest.param(
            [_ZOMBIE_LEADER, FakeProcess(_FAKE_MEMBER, state="Z")],
            _NO_SLEEP,
            id="every-member-a-zombie",
        ),
        pytest.param(
            [_ZOMBIE_LEADER, FakeProcess(_FAKE_MEMBER, state="X")], _NO_SLEEP, id="member-exiting"
        ),
        pytest.param(
            [_ZOMBIE_LEADER, FakeProcess(_FAKE_MEMBER, state="x")],
            _NO_SLEEP,
            id="member-exiting-old-kernel-letter",
        ),
        pytest.param(
            [_ZOMBIE_LEADER, FakeProcess(_FAKE_MEMBER, state="S", group_id=_FAKE_GROUP + 10)],
            _NO_SLEEP,
            id="only-another-group-running",
        ),
    ],
)
def test_wait_for_process_group_exit_when_linux_zombie_leader_does_wait_only_for_live_members(
    linux_proc_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    wait_clock: WaitClock,
    listed: list[FakeProcess],
    slept: float,
) -> None:
    for process in listed:
        process.write(linux_proc_root)
    monkeypatch.setattr(process_group, "os", ProbedLeader(exited=True))

    with within_hang_guard(_HANG_GUARD_S):
        wait_for_process_group_exit([_FAKE_GROUP], _WAIT_S)

    assert wait_clock.now == slept


async def _wait_blocking(group_id: int) -> None:
    wait_for_process_group_exit([group_id], _WAIT_S)


async def _wait_async(group_id: int) -> None:
    await wait_for_process_group_exit_async(group_id, _WAIT_S)


# The blocking and the async group wait, each given the fake-clock grace.
_WAITS = [
    pytest.param(_wait_blocking, id="blocking"),
    pytest.param(_wait_async, id="async"),
]

# How many polls the wait sleeps through before the last live member exits.
_SLEEPS_BEFORE_EXIT = 3


@_POSIX_ONLY
@pytest.mark.parametrize("wait", _WAITS)
async def test_wait_for_process_group_exit_blocking_or_async_when_last_member_exits_mid_wait_does_return_on_the_next_poll(
    linux_proc_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    wait_clock: WaitClock,
    wait: Callable[[int], Awaitable[None]],
) -> None:
    _ZOMBIE_LEADER.write(linux_proc_root)
    FakeProcess(_FAKE_MEMBER, state="S").write(linux_proc_root)
    exited_member = FakeProcess(_FAKE_MEMBER, state="Z")

    def exit_member_during_last_sleep() -> None:
        if len(wait_clock.sleeps) == _SLEEPS_BEFORE_EXIT:
            exited_member.write(linux_proc_root)

    wait_clock.on_sleep = exit_member_during_last_sleep
    monkeypatch.setattr(process_group, "os", ProbedLeader(exited=True))

    with within_hang_guard(_HANG_GUARD_S):
        await wait(_FAKE_GROUP)

    assert len(wait_clock.sleeps) == _SLEEPS_BEFORE_EXIT


def _without_proc(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(process_group, "PROC_ROOT", tmp_path / "missing")


def _malformed_proc(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(process_group, "PROC_ROOT", tmp_path)
    _ZOMBIE_LEADER.write(tmp_path)
    # A well-formed running member beside the malformed one: skipping the
    # malformed line would find it and wait, so only a fallback to the leader
    # alone returns at once when the leader has exited.
    FakeProcess(_FAKE_MEMBER, state="S").write(tmp_path)
    malformed = tmp_path / str(_FAKE_MEMBER + 1)
    malformed.mkdir()
    (malformed / "stat").write_text(f"{_FAKE_MEMBER + 1} (bench) S\n")


def _platform_without_listing(_tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "freebsd14")


def _failing_sysctl(_tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(
        process_group, "c_library", lambda: StubCLibrary(failure_errno=errno.EINVAL)
    )


def _empty_sysctl(_tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(process_group, "c_library", lambda: StubCLibrary(failure_errno=None))


_GROUPS_WITHOUT_LISTING = [
    pytest.param(_without_proc, id="proc-unreadable"),
    pytest.param(_malformed_proc, id="stat-line-malformed"),
    pytest.param(_failing_sysctl, id="sysctl-failing"),
    pytest.param(_platform_without_listing, id="platform-without-listing"),
]


# Whether the leader has exited, and what the wait shows on the fake clock.
_LEADER_OUTCOMES = [
    pytest.param(False, _TIMED_OUT, id="leader-running-sleeps-until-the-timeout"),
    pytest.param(True, _NO_SLEEP, id="leader-exited-returns-without-sleeping"),
]


@_POSIX_ONLY
@pytest.mark.parametrize("wait", _WAITS)
@pytest.mark.parametrize("hide_members", _GROUPS_WITHOUT_LISTING)
@pytest.mark.parametrize(("exited", "slept"), _LEADER_OUTCOMES)
async def test_wait_for_process_group_exit_blocking_or_async_when_listing_unavailable_does_wait_on_the_leader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    wait_clock: WaitClock,
    wait: Callable[[int], Awaitable[None]],
    hide_members: Callable[[Path, pytest.MonkeyPatch], None],
    *,
    exited: bool,
    slept: float,
) -> None:
    hide_members(tmp_path, monkeypatch)
    monkeypatch.setattr(process_group, "os", ProbedLeader(exited=exited))

    with within_hang_guard(_HANG_GUARD_S):
        await wait(_FAKE_GROUP)

    assert wait_clock.now == slept


# A few bytes past the last whole record, as a listing cut mid-record reports them.
_PARTIAL_RECORD = bytes(16)


@_POSIX_ONLY
@pytest.mark.parametrize(
    ("listing", "slept"),
    [
        pytest.param(_PARTIAL_RECORD, _NO_SLEEP, id="shorter-than-a-record"),
        pytest.param(
            darwin_record(0, _DARWIN_SZOMB) + _PARTIAL_RECORD,
            _NO_SLEEP,
            id="zombie-record-then-partial",
        ),
        pytest.param(
            darwin_record(0, _DARWIN_RUNNING) + _PARTIAL_RECORD,
            _TIMED_OUT,
            id="running-record-then-partial",
        ),
        pytest.param(darwin_record(0, _DARWIN_SZOMB), _NO_SLEEP, id="whole-zombie-record"),
        pytest.param(darwin_record(0, _DARWIN_RUNNING), _TIMED_OUT, id="whole-running-record"),
        pytest.param(
            darwin_record(_DARWIN_P_WEXIT, _DARWIN_RUNNING), _NO_SLEEP, id="whole-exiting-record"
        ),
    ],
)
def test_wait_for_process_group_exit_when_darwin_leader_exited_does_wait_only_for_whole_running_records(
    monkeypatch: pytest.MonkeyPatch,
    wait_clock: WaitClock,
    listing: bytes,
    slept: float,
) -> None:
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(
        process_group, "c_library", lambda: StubCLibrary(failure_errno=None, listing=listing)
    )
    monkeypatch.setattr(process_group, "os", ProbedLeader(exited=True))

    with within_hang_guard(_HANG_GUARD_S):
        wait_for_process_group_exit([_FAKE_GROUP], _WAIT_S)

    assert wait_clock.now == slept


@pytest.mark.skipif(sys.platform != "darwin", reason="only macOS's C library has sysctl")
def test_c_library_when_called_again_does_return_the_library_it_loaded_first() -> None:
    first = process_group.c_library()

    again = process_group.c_library()

    assert again is first, "every member listing reloaded the C library"


# ---------------------------------------------------------------------------
# kill_process_group
# ---------------------------------------------------------------------------


@_POSIX_ONLY
def test_kill_process_group_when_live_member_refuses_does_warn(
    sleeping_group_leader: subprocess.Popen[bytes],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    refuse_killpg(monkeypatch)

    with pytest.warns(RuntimeWarning, match=_KILLPG_FAILED):
        kill_process_group(sleeping_group_leader.pid)


@_POSIX_ONLY
async def test_kill_process_group_when_only_a_zombie_member_is_left_does_not_warn(
    orphaned_zombie_group: int,
    recwarn: pytest.WarningsRecorder,
) -> None:
    kill_process_group(orphaned_zombie_group)

    assert killpg_warnings(recwarn) == []


@_POSIX_ONLY
@pytest.mark.parametrize(
    "hide_members",
    [
        pytest.param(_failing_sysctl, id="listing-fails"),
        pytest.param(_empty_sysctl, id="listing-empty-but-group-exists"),
    ],
)
def test_kill_process_group_when_refusing_group_cannot_be_shown_settled_does_warn(
    sleeping_group_leader: subprocess.Popen[bytes],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    hide_members: Callable[[Path, pytest.MonkeyPatch], None],
) -> None:
    refuse_killpg(monkeypatch, _refuses_all_but_probe)
    hide_members(tmp_path, monkeypatch)

    with pytest.warns(RuntimeWarning, match=_KILLPG_FAILED):
        kill_process_group(sleeping_group_leader.pid)


@_POSIX_ONLY
@pytest.mark.skipif(sys.platform != "darwin", reason="only macOS lists a refusing group's members")
def test_kill_process_group_when_refusing_group_is_gone_by_the_listing_does_not_warn(
    monkeypatch: pytest.MonkeyPatch,
    recwarn: pytest.WarningsRecorder,
) -> None:
    # The refusal stands in for a group whose last zombie is reaped between the
    # signal and the listing: by then the kernel no longer knows the group.
    gone = dead_pid()
    refuse_killpg(monkeypatch, _refuses_all_but_probe)

    kill_process_group(gone)

    assert killpg_warnings(recwarn) == []


@pytest.fixture
def surviving_member(monkeypatch: pytest.MonkeyPatch) -> Callable[[int], SurvivingMember]:
    """Install, under a faked macOS, a group whose member survives a given number of SIGKILLs."""

    def install(kills_survived: int) -> SurvivingMember:
        group = SurvivingMember(kills_survived)
        monkeypatch.setattr(sys, "platform", "darwin")
        monkeypatch.setattr(process_group, "os", group)
        monkeypatch.setattr(process_group, "c_library", lambda: group.library)
        return group

    return install


# A member that no SIGKILL ends within the settle bound.
_NEVER_GOES = sys.maxsize


@_POSIX_ONLY
@pytest.mark.parametrize(
    ("kills_survived", "kills", "sleeps"),
    [
        pytest.param(0, 1, [], id="group-empty-after-the-kill"),
        pytest.param(1, 2, [process_group.EXIT_POLL_S], id="member-survives-the-first-kill"),
    ],
)
def test_kill_process_group_when_member_survives_the_kill_does_kill_again_until_none_runs(
    surviving_member: Callable[[int], SurvivingMember],
    wait_clock: WaitClock,
    kills_survived: int,
    kills: int,
    sleeps: list[float],
) -> None:
    group = surviving_member(kills_survived)

    with within_hang_guard(_HANG_GUARD_S):
        kill_process_group(_FAKE_GROUP)

    assert group.signals == [signal.SIGKILL] * kills
    assert wait_clock.sleeps == sleeps


@_POSIX_ONLY
def test_kill_process_group_when_member_never_goes_does_stop_at_the_settle_bound(
    surviving_member: Callable[[int], SurvivingMember],
    wait_clock: WaitClock,
) -> None:
    surviving_member(_NEVER_GOES)

    with within_hang_guard(_HANG_GUARD_S):
        kill_process_group(_FAKE_GROUP)

    assert wait_clock.now == pytest.approx(process_group.KILL_SETTLE_S, abs=_ONE_POLL_S)


# ---------------------------------------------------------------------------
# kill_process_group — the win32 taskkill fallback
# ---------------------------------------------------------------------------

#: A pid that no job handle is registered for, so the win32 path falls back to taskkill.
_PLAIN_PID = 424242


def _taskkill_exits(returncode: int) -> Callable[..., subprocess.CompletedProcess[bytes]]:
    def fake_run(
        args: list[str], *_args: object, **_kwargs: object
    ) -> subprocess.CompletedProcess[bytes]:
        raise subprocess.CalledProcessError(returncode=returncode, cmd=args)

    return fake_run


def _taskkill_cannot_start(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[bytes]:
    raise PermissionError(13, "Permission denied")


@_POSIX_ONLY
@pytest.mark.parametrize(
    ("run", "warnings_raised"),
    [
        pytest.param(_taskkill_exits(128), [], id="tree-already-gone-stays-silent"),
        pytest.param(
            _taskkill_exits(5),
            [
                (
                    f"taskkill failed for pid {_PLAIN_PID}: Command '['taskkill', '/F', '/T', "
                    f"'/PID', '{_PLAIN_PID}']' returned non-zero exit status 5."
                )
            ],
            id="other-failure-warns",
        ),
        pytest.param(
            _taskkill_cannot_start,
            [
                (
                    f"taskkill unavailable while killing pid {_PLAIN_PID}: "
                    "[Errno 13] Permission denied"
                )
            ],
            id="launch-failure-warns",
        ),
    ],
)
def test_kill_process_group_when_win32_tree_has_no_job_does_fall_back_to_taskkill(
    monkeypatch: pytest.MonkeyPatch,
    run: Callable[..., subprocess.CompletedProcess[bytes]],
    warnings_raised: list[str],
) -> None:
    calls = record_subprocess_runs(monkeypatch, run)
    monkeypatch.setattr(sys, "platform", "win32")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        kill_process_group(_PLAIN_PID)

    assert calls == [["taskkill", "/F", "/T", "/PID", str(_PLAIN_PID)]]
    assert [str(w.message) for w in caught if issubclass(w.category, RuntimeWarning)] == (
        warnings_raised
    )


# ---------------------------------------------------------------------------
# win32 Job Objects, reached from any host through a faked platform
# ---------------------------------------------------------------------------

# A stand-in pid for the job tests, which never touch a real process.
_CHILD_PID = 4321

# Win32 ABI values, spelled out here rather than read back from the module under
# test: a build that set the wrong ones has to fail this, not agree with itself.
# ``JobObjectExtendedLimitInformation`` and ``JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE``.
_WIN32_EXTENDED_LIMIT_INFORMATION = 9
_WIN32_LIMIT_KILL_ON_JOB_CLOSE = 0x2000

# Grace given to a faked job, apart from the real one so a wait that reaches it
# shows the grace is read at call time, and small so the fake clock reaches it
# in a few polls.
_FAKE_JOB_GRACE_S = 0.05


def test_attach_process_group_when_child_gets_a_job_does_limit_it_to_kill_on_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    jobs = FakeJobs()
    module = win32_process_group(monkeypatch, jobs)

    module.attach_process_group(_CHILD_PID)

    assert len(jobs.limited) == 1, "the child's job was created without a limits call"
    info_class, limit_flags = jobs.limited[0]
    assert info_class == _WIN32_EXTENDED_LIMIT_INFORMATION
    assert limit_flags & _WIN32_LIMIT_KILL_ON_JOB_CLOSE, (
        "the job does not kill its members when its last handle closes"
    )


@pytest.mark.parametrize("entry", ["terminate_process_group", "kill_process_group"])
@pytest.mark.parametrize(
    ("refused_step", "closed"),
    [
        pytest.param("assignment", [PROCESS_HANDLE, JOB_HANDLE], id="assignment-refused"),
        # No job was ever created, so there is no handle to close.
        pytest.param("creation", [], id="creation-refused"),
    ],
)
def test_terminate_or_kill_process_group_when_job_refused_does_fall_back_to_taskkill_with_one_warning(
    monkeypatch: pytest.MonkeyPatch,
    recwarn: pytest.WarningsRecorder,
    entry: str,
    refused_step: str,
    closed: list[int],
) -> None:
    jobs = FakeJobs(
        creation_granted=refused_step != "creation",
        assignment_granted=refused_step != "assignment",
    )
    module = win32_process_group(monkeypatch, jobs)
    argv_calls = record_subprocess_runs(monkeypatch)
    module.attach_process_group(_CHILD_PID)

    getattr(module, entry)(_CHILD_PID)

    runtime_warnings = [str(w.message) for w in recwarn if w.category is RuntimeWarning]
    assert len(runtime_warnings) == 1, "a refused job has to warn exactly once"
    assert f"job {refused_step} refused" in runtime_warnings[0], (
        "the one warning is not the job refusal"
    )
    assert argv_calls == [["taskkill", "/F", "/T", "/PID", str(_CHILD_PID)]], (
        "a child that never reached a job was not torn down through taskkill"
    )
    assert sorted(jobs.closed) == sorted(closed), "a refused job left a handle open"


def test_kill_process_group_when_host_has_no_sigkill_does_terminate_the_job(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    jobs = FakeJobs()
    monkeypatch.delattr(signal, "SIGKILL", raising=False)
    module = win32_process_group(monkeypatch, jobs)
    module.attach_process_group(_CHILD_PID)

    module.kill_process_group(_CHILD_PID)

    assert jobs.terminated == [JOB_HANDLE], (
        "the kill never reached the job: a host without SIGKILL cannot be asked for one"
    )


@pytest.mark.parametrize(
    ("entry", "closed"),
    [
        # A terminated job stays on record, so a later kill can still reach it.
        pytest.param("terminate_process_group", [PROCESS_HANDLE], id="terminate-keeps-the-job"),
        pytest.param(
            "release_process_group", [PROCESS_HANDLE, JOB_HANDLE], id="release-closes-the-job"
        ),
    ],
)
def test_terminate_or_release_process_group_when_job_still_emptying_does_wait_for_it_then_keep_or_close_the_handle(
    monkeypatch: pytest.MonkeyPatch,
    entry: str,
    closed: list[int],
) -> None:
    jobs = FakeJobs(active_counts=[2, 1, 0])
    module = win32_process_group(monkeypatch, jobs)
    wait_clock = install_wait_clock(monkeypatch, module)
    module.attach_process_group(_CHILD_PID)

    with within_hang_guard(_HANG_GUARD_S):
        getattr(module, entry)(_CHILD_PID)

    assert jobs.terminated == [JOB_HANDLE]
    assert len(wait_clock.sleeps) == 2, (
        "the wait did not return on the first poll after the job emptied"
    )
    assert sorted(jobs.closed) == sorted(closed), "the settle path left the job handle open"


@pytest.mark.parametrize(
    ("final_active", "slept"),
    [
        pytest.param(0, _NO_SLEEP, id="job-already-empty-returns-without-sleeping"),
        pytest.param(
            1,
            pytest.approx(_FAKE_JOB_GRACE_S, abs=_ONE_POLL_S),
            id="job-never-empties-sleeps-until-the-grace",
        ),
    ],
)
def test_terminate_process_group_when_job_settles_or_not_does_wait_only_while_it_has_processes(
    monkeypatch: pytest.MonkeyPatch,
    final_active: int,
    slept: float,
) -> None:
    jobs = FakeJobs(final_active=final_active)
    module = win32_process_group(monkeypatch, jobs)
    monkeypatch.setattr(module, "TERMINATE_GRACE_S", _FAKE_JOB_GRACE_S)
    wait_clock = install_wait_clock(monkeypatch, module)
    module.attach_process_group(_CHILD_PID)

    with within_hang_guard(_HANG_GUARD_S):
        module.terminate_process_group(_CHILD_PID)

    assert wait_clock.now == slept


def test_kill_process_group_when_job_already_released_does_fall_back_to_taskkill(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    jobs = FakeJobs()
    module = win32_process_group(monkeypatch, jobs)
    argv_calls = record_subprocess_runs(monkeypatch)
    module.attach_process_group(_CHILD_PID)
    module.release_process_group(_CHILD_PID)

    module.kill_process_group(_CHILD_PID)

    assert argv_calls == [["taskkill", "/F", "/T", "/PID", str(_CHILD_PID)]], (
        "the released job was still on record, so the kill reached a closed handle"
    )
    assert jobs.terminated == [JOB_HANDLE], "the kill reached the job after its release"


@pytest.mark.parametrize(
    ("resume_granted", "resumed", "warnings_raised"),
    [
        pytest.param(True, [_CHILD_PID], [], id="suspended-child-is-resumed"),
        pytest.param(
            False,
            [],
            [f"could not resume pid {_CHILD_PID}: NTSTATUS 0xC0000022"],
            id="refused-resume-warns",
        ),
    ],
)
def test_resume_process_group_when_child_suspended_does_report_the_outcome_without_leaking_its_handle(
    monkeypatch: pytest.MonkeyPatch,
    *,
    resume_granted: bool,
    resumed: list[int],
    warnings_raised: list[str],
) -> None:
    jobs = FakeJobs(resume_granted=resume_granted)
    module = win32_process_group(monkeypatch, jobs)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = module.resume_process_group(_CHILD_PID)

    assert result is resume_granted
    assert jobs.resumed == resumed
    assert [str(w.message) for w in caught if issubclass(w.category, RuntimeWarning)] == (
        warnings_raised
    )
    assert jobs.closed == [PROCESS_HANDLE], "the child's handle was leaked"
