"""Behavioral tests for probing and signalling POSIX process groups."""

import asyncio
import ctypes
import dataclasses
import errno
import os
import struct
import subprocess
import sys
import time
import types
import warnings
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from gymrat import process_group
from gymrat.process_group import (
    kill_process_group,
    wait_for_process_group_exit,
    wait_for_process_group_exit_async,
)
from tests._process_helpers import (
    KILLPG_FAILED as _KILLPG_FAILED,
)
from tests._process_helpers import (
    ZOMBIE_ONLY_GROUP_SCRIPT,
    dead_pid,
    is_alive,
    killpg_warnings,
    wait_for_pid_file,
    wait_for_pid_file_blocking,
    wait_until_dead,
    wait_until_dead_blocking,
)

if sys.platform == "win32":
    pytest.skip("POSIX-only zombie and reap semantics", allow_module_level=True)

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

# Grace a wait expected to end early is given, far longer than any probe takes.
_LONG_WAIT_S = 10.0

# The macOS ``kinfo_proc`` layout, spelled out from <sys/sysctl.h> and
# <sys/proc.h> rather than read from the module under test, so a wrong value
# there cannot also shape the records these tests feed it.
_DARWIN_KINFO_PROC_SIZE = 648
_DARWIN_FLAG_AND_STATE = struct.Struct("=iB")  # p_flag, then p_stat right after it
_DARWIN_FLAG_AND_STATE_OFFSET = 32
_DARWIN_P_WEXIT = 0x2000
_DARWIN_SZOMB = 5
_DARWIN_RUNNING = 2

# Grace a wait expected to run its full course is given, kept short so it stays cheap.
_SHORT_WAIT_S = 0.2

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


@dataclasses.dataclass(frozen=True)
class FakeProcess:
    """A process listed in a fake ``/proc``, in the group ``group_id`` (``_FAKE_GROUP`` by default)."""

    pid: int
    state: str
    comm: bytes = b"bench"
    group_id: int = _FAKE_GROUP

    def write(self, proc_root: Path) -> None:
        """Write this process's ``stat`` file under ``proc_root``, laid out as Linux lays it out."""
        entry = proc_root / str(self.pid)
        entry.mkdir()
        head = f"{self.pid} (".encode()
        tail = f") {self.state} 1 {self.group_id} {_STAT_TAIL}\n".encode()
        (entry / "stat").write_bytes(head + self.comm + tail)


# The exited, not yet reaped, leader every fake group below starts from.
_ZOMBIE_LEADER = FakeProcess(_FAKE_GROUP, state="Z")


@dataclasses.dataclass
class ProbedLeader:
    """Stand-in ``os`` whose probes find every process present, and exited when ``exited`` is set.

    ``kill`` and ``killpg`` with signal 0 succeed, as they do on Linux for a
    zombie as much as for a running process; ``waitid`` reports an exit, and
    ``getpgid`` fails the lookup, only when ``exited`` is set. Every other attribute forwards to the real ``os``
    module.
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


@dataclasses.dataclass(frozen=True)
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


_real_killpg = os.killpg


def refuse_all_but_probe(group_pid: int, signal_number: int) -> None:
    """Stand in for ``os.killpg``: refuse every signal with ``EPERM``, but let the ``0`` probe through."""
    if signal_number != 0:
        raise PermissionError(errno.EPERM, os.strerror(errno.EPERM))
    _real_killpg(group_pid, signal_number)


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


@pytest.fixture
def sleeping_group_leader() -> Iterator[subprocess.Popen[bytes]]:
    """A running child leading a process group of its own, killed and reaped on teardown."""
    proc = subprocess.Popen(["sleep", "30"], start_new_session=True)  # noqa: S607 -- fixed argv, sleep on PATH
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
        [  # noqa: S607 -- sh on PATH
            "sh",
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
    monkeypatch.setattr(process_group, "_PROC_ROOT", proc_root)
    return proc_root


@pytest.fixture
def exiting_child() -> Iterator[subprocess.Popen[bytes]]:
    """A child that exits with ``_CHILD_STATUS`` at once, reaped on teardown if the test did not."""
    proc = subprocess.Popen(["sh", "-c", f"exit {_CHILD_STATUS}"])  # noqa: S603, S607 -- fixed argv
    yield proc
    proc.wait()


def test_wait_for_process_group_exit_when_leader_reaped_during_probe_does_return_before_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(process_group, "os", ReapedMidProbe())
    started = time.monotonic()

    wait_for_process_group_exit([424242], _LONG_WAIT_S)

    assert time.monotonic() - started < _LONG_WAIT_S / 2


def _child_leader(leader: subprocess.Popen[bytes]) -> int:
    return leader.pid


def _parent_process(_leader: subprocess.Popen[bytes]) -> int:
    return os.getppid()


@pytest.mark.parametrize(
    "leader_of",
    [
        pytest.param(_child_leader, id="child-leader"),
        pytest.param(_parent_process, id="non-child-leader"),
    ],
)
def test_wait_for_process_group_exit_when_leader_running_does_wait_out_timeout(
    sleeping_group_leader: subprocess.Popen[bytes],
    leader_of: Callable[[subprocess.Popen[bytes]], int],
) -> None:
    started = time.monotonic()

    wait_for_process_group_exit([leader_of(sleeping_group_leader)], _SHORT_WAIT_S)

    assert time.monotonic() - started >= _SHORT_WAIT_S


async def test_wait_for_process_group_exit_when_leader_is_zombie_does_return_without_reaping(
    exiting_child: subprocess.Popen[bytes],
) -> None:
    await wait_until_dead(exiting_child.pid)
    started = time.monotonic()

    wait_for_process_group_exit([exiting_child.pid], _LONG_WAIT_S)

    assert time.monotonic() - started < _LONG_WAIT_S / 2
    assert exiting_child.wait() == _CHILD_STATUS


def test_wait_for_process_group_exit_when_leader_exited_but_member_running_does_wait_for_member(
    group_with_lingering_member: LingeringGroup,
) -> None:
    wait_for_process_group_exit([group_with_lingering_member.leader_pid], _LONG_WAIT_S)

    assert not is_alive(group_with_lingering_member.member_pid)


@pytest.mark.parametrize(
    ("listed", "timeout_s", "waits"),
    [
        pytest.param(
            [_ZOMBIE_LEADER, FakeProcess(_FAKE_MEMBER, state="S")],
            _SHORT_WAIT_S,
            True,
            id="sleeping-member",
        ),
        pytest.param(
            [_ZOMBIE_LEADER, FakeProcess(_FAKE_MEMBER, state="R")],
            _SHORT_WAIT_S,
            True,
            id="running-member",
        ),
        pytest.param(
            [
                _ZOMBIE_LEADER,
                FakeProcess(_FAKE_MEMBER, state="S", comm=f"a) Z 1 {_FAKE_GROUP}".encode()),
            ],
            _SHORT_WAIT_S,
            True,
            id="member-name-mimics-zombie-fields",
        ),
        pytest.param(
            [
                _ZOMBIE_LEADER,
                FakeProcess(_FAKE_MEMBER, state="S"),
                FakeProcess(_FAKE_MEMBER + 1, state="S", comm=_CUT_NAME, group_id=_FAKE_GROUP + 10),
            ],
            _SHORT_WAIT_S,
            True,
            id="unrelated-name-cut-mid-character",
        ),
        pytest.param(
            [_ZOMBIE_LEADER, FakeProcess(_FAKE_MEMBER, state="S", comm=b"a) (Z " + _CUT_NAME)],
            _SHORT_WAIT_S,
            True,
            id="member-name-cut-mid-character-with-parentheses",
        ),
        pytest.param(
            [_ZOMBIE_LEADER, FakeProcess(_FAKE_MEMBER, state="Z")],
            _LONG_WAIT_S,
            False,
            id="every-member-a-zombie",
        ),
        pytest.param(
            [_ZOMBIE_LEADER, FakeProcess(_FAKE_MEMBER, state="X")],
            _LONG_WAIT_S,
            False,
            id="member-exiting",
        ),
        pytest.param(
            [_ZOMBIE_LEADER, FakeProcess(_FAKE_MEMBER, state="x")],
            _LONG_WAIT_S,
            False,
            id="member-exiting-old-kernel-letter",
        ),
        pytest.param(
            [_ZOMBIE_LEADER, FakeProcess(_FAKE_MEMBER, state="S", group_id=_FAKE_GROUP + 10)],
            _LONG_WAIT_S,
            False,
            id="only-another-group-running",
        ),
    ],
)
def test_wait_for_process_group_exit_when_linux_zombie_leader_does_wait_only_for_live_members(
    linux_proc_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    listed: list[FakeProcess],
    *,
    timeout_s: float,
    waits: bool,
) -> None:
    for process in listed:
        process.write(linux_proc_root)
    monkeypatch.setattr(process_group, "os", ProbedLeader(exited=True))
    started = time.monotonic()

    wait_for_process_group_exit([_FAKE_GROUP], timeout_s)

    assert (time.monotonic() - started >= _SHORT_WAIT_S) is waits


def _without_proc(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(process_group, "_PROC_ROOT", tmp_path / "missing")


def _malformed_proc(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(process_group, "_PROC_ROOT", tmp_path)
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
        process_group, "_c_library", lambda: StubCLibrary(failure_errno=errno.EINVAL)
    )


def _empty_sysctl(_tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(process_group, "_c_library", lambda: StubCLibrary(failure_errno=None))


def _undecodable_sysctl(_tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def listed(_group_id: int) -> list[tuple[int, int]]:
        raise struct.error

    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(process_group, "_darwin_group_members", listed)


_GROUPS_WITHOUT_LISTING = [
    pytest.param(_without_proc, id="proc-unreadable"),
    pytest.param(_malformed_proc, id="stat-line-malformed"),
    pytest.param(_failing_sysctl, id="sysctl-failing"),
    pytest.param(_undecodable_sysctl, id="sysctl-records-undecodable"),
    pytest.param(_platform_without_listing, id="platform-without-listing"),
]


# Whether the leader has exited, the grace the wait is given, and whether it runs that grace out.
_LEADER_OUTCOMES = [
    pytest.param(False, _SHORT_WAIT_S, True, id="leader-running-waits-out-the-timeout"),
    pytest.param(True, _LONG_WAIT_S, False, id="leader-exited-returns-at-once"),
]


@pytest.mark.parametrize("hide_members", _GROUPS_WITHOUT_LISTING)
@pytest.mark.parametrize(("exited", "timeout_s", "waits"), _LEADER_OUTCOMES)
def test_wait_for_process_group_exit_when_listing_unavailable_does_wait_on_the_leader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    hide_members: Callable[[Path, pytest.MonkeyPatch], None],
    *,
    exited: bool,
    timeout_s: float,
    waits: bool,
) -> None:
    hide_members(tmp_path, monkeypatch)
    monkeypatch.setattr(process_group, "os", ProbedLeader(exited=exited))
    started = time.monotonic()

    wait_for_process_group_exit([_FAKE_GROUP], timeout_s)

    assert (time.monotonic() - started >= _SHORT_WAIT_S) is waits


@pytest.mark.parametrize(("exited", "timeout_s", "waits"), _LEADER_OUTCOMES)
async def test_wait_for_process_group_exit_async_when_listing_unavailable_does_wait_on_the_leader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    exited: bool,
    timeout_s: float,
    waits: bool,
) -> None:
    _platform_without_listing(tmp_path, monkeypatch)
    monkeypatch.setattr(process_group, "os", ProbedLeader(exited=exited))
    started = time.monotonic()

    await wait_for_process_group_exit_async(_FAKE_GROUP, timeout_s)

    assert (time.monotonic() - started >= _SHORT_WAIT_S) is waits


# A few bytes past the last whole record, as a listing cut mid-record reports them.
_PARTIAL_RECORD = bytes(16)


@pytest.mark.parametrize(
    ("listing", "still_running"),
    [
        pytest.param(_PARTIAL_RECORD, False, id="shorter-than-a-record"),
        pytest.param(
            darwin_record(0, _DARWIN_SZOMB) + _PARTIAL_RECORD,
            False,
            id="zombie-record-then-partial",
        ),
        pytest.param(
            darwin_record(0, _DARWIN_RUNNING) + _PARTIAL_RECORD,
            True,
            id="running-record-then-partial",
        ),
        pytest.param(darwin_record(0, _DARWIN_SZOMB), False, id="whole-zombie-record"),
        pytest.param(darwin_record(0, _DARWIN_RUNNING), True, id="whole-running-record"),
        pytest.param(
            darwin_record(_DARWIN_P_WEXIT, _DARWIN_RUNNING), False, id="whole-exiting-record"
        ),
    ],
)
def test_wait_for_process_group_exit_when_darwin_leader_exited_does_wait_only_for_whole_running_records(
    monkeypatch: pytest.MonkeyPatch,
    listing: bytes,
    still_running: bool,
) -> None:
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(
        process_group, "_c_library", lambda: StubCLibrary(failure_errno=None, listing=listing)
    )
    monkeypatch.setattr(process_group, "os", ProbedLeader(exited=True))
    started = time.monotonic()

    wait_for_process_group_exit([_FAKE_GROUP], _SHORT_WAIT_S)

    assert (time.monotonic() - started >= _SHORT_WAIT_S) == still_running


# ---------------------------------------------------------------------------
# kill_process_group
# ---------------------------------------------------------------------------


def test_kill_process_group_when_live_member_refuses_does_warn(
    sleeping_group_leader: subprocess.Popen[bytes],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def raise_eperm(group_pid: int, sig: int) -> None:
        raise PermissionError(errno.EPERM, os.strerror(errno.EPERM))

    monkeypatch.setattr(os, "killpg", raise_eperm)

    with pytest.warns(RuntimeWarning, match=_KILLPG_FAILED):
        kill_process_group(sleeping_group_leader.pid)


async def test_kill_process_group_when_only_a_zombie_member_is_left_does_not_warn(
    orphaned_zombie_group: int,
    recwarn: pytest.WarningsRecorder,
) -> None:
    kill_process_group(orphaned_zombie_group)

    assert killpg_warnings(recwarn) == []


@pytest.mark.parametrize(
    "hide_members",
    [
        pytest.param(_failing_sysctl, id="listing-fails"),
        pytest.param(_empty_sysctl, id="listing-empty-but-group-exists"),
        pytest.param(_undecodable_sysctl, id="listing-records-undecodable"),
    ],
)
def test_kill_process_group_when_refusing_group_cannot_be_shown_settled_does_warn(
    sleeping_group_leader: subprocess.Popen[bytes],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    hide_members: Callable[[Path, pytest.MonkeyPatch], None],
) -> None:
    monkeypatch.setattr(os, "killpg", refuse_all_but_probe)
    hide_members(tmp_path, monkeypatch)

    with pytest.warns(RuntimeWarning, match=_KILLPG_FAILED):
        kill_process_group(sleeping_group_leader.pid)


@pytest.mark.skipif(sys.platform != "darwin", reason="only macOS lists a refusing group's members")
def test_kill_process_group_when_refusing_group_is_gone_by_the_listing_does_not_warn(
    monkeypatch: pytest.MonkeyPatch,
    recwarn: pytest.WarningsRecorder,
) -> None:
    # The refusal stands in for a group whose last zombie is reaped between the
    # signal and the listing: by then the kernel no longer knows the group.
    gone = dead_pid()
    monkeypatch.setattr(os, "killpg", refuse_all_but_probe)

    kill_process_group(gone)

    assert killpg_warnings(recwarn) == []


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
    calls: list[list[str]] = []

    def recording_run(
        args: list[str], *rest: object, **kwargs: object
    ) -> subprocess.CompletedProcess[bytes]:
        calls.append(args)
        return run(args, *rest, **kwargs)

    monkeypatch.setattr(subprocess, "run", recording_run)
    monkeypatch.setattr(sys, "platform", "win32")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        kill_process_group(_PLAIN_PID)

    assert calls == [["taskkill", "/F", "/T", "/PID", str(_PLAIN_PID)]]
    assert [str(w.message) for w in caught if issubclass(w.category, RuntimeWarning)] == (
        warnings_raised
    )
