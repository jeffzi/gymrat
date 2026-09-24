"""Behavioral tests for probing and signalling POSIX process groups."""

import asyncio
import ctypes
import dataclasses
import errno
import os
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

from gymrat import process_group
from gymrat.process_group import kill_process_group, wait_for_process_group_exit
from tests._process_helpers import (
    KILLPG_FAILED as _KILLPG_FAILED,
)
from tests._process_helpers import (
    ZOMBIE_ONLY_GROUP_SCRIPT,
    dead_pid,
    killpg_warnings,
    wait_for_pid_file,
    wait_until_dead,
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

# Grace a wait expected to run its full course is given, kept short so it stays cheap.
_SHORT_WAIT_S = 0.2


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
    """Stand-in C library whose ``sysctl`` fails with ``failure_errno``, or lists nothing when it is ``None``."""

    failure_errno: int | None
    sysctl: object = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        self.sysctl = _SysctlFunction(self._sysctl)

    def _sysctl(
        self,
        _name: object,
        _name_length: int,
        _old: object,
        old_length: "ctypes._Pointer[ctypes.c_size_t]",
        _new: object,
        _new_length: int,
    ) -> int:
        if self.failure_errno is not None:
            ctypes.set_errno(self.failure_errno)
            return -1
        old_length[0] = 0
        return 0


@pytest.fixture
def sleeping_child() -> Iterator[subprocess.Popen[bytes]]:
    """A running child process, killed and reaped on teardown."""
    proc = subprocess.Popen(["sleep", "30"])  # noqa: S607 -- fixed argv, sleep on PATH
    yield proc
    proc.kill()
    proc.wait()


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


def test_wait_for_process_group_exit_when_child_leader_running_does_wait_out_timeout(
    sleeping_child: subprocess.Popen[bytes],
) -> None:
    started = time.monotonic()

    wait_for_process_group_exit([sleeping_child.pid], _SHORT_WAIT_S)

    assert time.monotonic() - started >= _SHORT_WAIT_S


def test_wait_for_process_group_exit_when_non_child_leader_running_does_wait_out_timeout() -> None:
    started = time.monotonic()

    wait_for_process_group_exit([os.getppid()], _SHORT_WAIT_S)

    assert time.monotonic() - started >= _SHORT_WAIT_S


async def test_wait_for_process_group_exit_when_leader_is_zombie_does_return_without_reaping(
    exiting_child: subprocess.Popen[bytes],
) -> None:
    await wait_until_dead(exiting_child.pid)
    started = time.monotonic()

    wait_for_process_group_exit([exiting_child.pid], _LONG_WAIT_S)

    assert time.monotonic() - started < _LONG_WAIT_S / 2
    assert exiting_child.wait() == _CHILD_STATUS


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
    "c_library",
    [
        pytest.param(StubCLibrary(failure_errno=errno.EINVAL), id="listing-fails"),
        pytest.param(StubCLibrary(failure_errno=None), id="listing-empty-but-group-exists"),
    ],
)
@pytest.mark.skipif(sys.platform != "darwin", reason="only macOS lists a refusing group's members")
def test_kill_process_group_when_refusing_group_cannot_be_shown_settled_does_warn(
    sleeping_group_leader: subprocess.Popen[bytes],
    monkeypatch: pytest.MonkeyPatch,
    c_library: StubCLibrary,
) -> None:
    monkeypatch.setattr(os, "killpg", refuse_all_but_probe)
    monkeypatch.setattr(process_group, "_c_library", lambda: c_library)

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
