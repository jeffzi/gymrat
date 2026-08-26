"""Behavioral tests for the single-flight repository lock.

Every scenario drives the public :func:`acquire_lock` and its release handle,
exercising the hard-link takeover protocol through real filesystem operations
and patched system calls at the exact seams a concurrent run would race on.
"""

import errno
import json
import os
import re
import subprocess
import sys
import tempfile
from collections.abc import Buffer
from pathlib import Path

import pytest

from gymrat_py.errors import GymratError
from gymrat_py.session import lock as lock_module
from gymrat_py.session.lock import acquire_lock

# ---------------------------------------------------------------------------
# Constants and helpers
# ---------------------------------------------------------------------------

# The ISO-8601 shape a freshly published holder record stamps into ``at``.
AT_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$")

# The ``at`` value ``write_lockfile`` stamps every fixture lockfile with.
WRITTEN_LOCK_AT = "2026-01-01T00:00:00.000Z"

# The remedy for a lock whose holder answered a liveness probe: the holder is
# provably alive, so deleting the lockfile is never offered.
LIVE_HOLDER_HINT = "Another gymrat run is active in this repo. Wait for it to finish."


def fresh_lock_path(*segments: str) -> str:
    """Return a lock path inside its own temp dir, so tests never share a file."""
    directory = tempfile.mkdtemp(prefix="lock-test-")
    return str(Path(directory, *segments, "gymrat.lock.json"))


def dead_pid() -> int:
    """Return a pid that is certainly gone: the child ran and was reaped."""
    proc = subprocess.Popen([sys.executable, "-c", ""])
    proc.wait()
    return proc.pid


def write_lockfile(lock_path: str, pid: object, command: str) -> None:
    """Write a holder record for ``pid``/``command`` stamped with the fixed time."""
    path = Path(lock_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"pid": pid, "command": command, "at": WRITTEN_LOCK_AT}),
        encoding="utf-8",
    )


def write_raw_lockfile(lock_path: str, contents: str) -> None:
    """Write ``contents`` verbatim, standing in for a run that died mid-write."""
    path = Path(lock_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents, encoding="utf-8")


def read_lockfile(lock_path: str) -> object:
    """Parse the JSON record currently at ``lock_path``."""
    return json.loads(Path(lock_path).read_text(encoding="utf-8"))


def stale_lock_path(pid: int | None = None) -> tuple[str, int]:
    """Return a fresh lockfile whose holder process has already exited."""
    holder_pid = dead_pid() if pid is None else pid
    lock_path = fresh_lock_path()
    write_lockfile(lock_path, holder_pid, "measure")
    return lock_path, holder_pid


def claim_path_for(lock_path: str) -> str:
    """Return the claim-link path derived from the current lockfile identity."""
    info = Path(lock_path).stat()
    return f"{lock_path}.{info.st_dev}-{info.st_ino}.claim"


def wedge_takeover(lock_path: str) -> str:
    """Leave behind the claim link a run that died mid-takeover would have left.

    The link is the lockfile's own inode under the claim name derived from that
    inode, so every steal attempt is blocked by an occupied claim path while the
    identity behind the lockfile stays constant. Returns the claim path.
    """
    claim_path = claim_path_for(lock_path)
    os.link(lock_path, claim_path)
    return claim_path


def replace_lockfile(lock_path: str, pid: int, command: str) -> None:
    """Put a different run's lockfile where ``lock_path`` is, as a takeover would.

    The replacement is built beside the lock and renamed over it, so the
    filesystem cannot hand it the inode of the file it displaces.
    """
    scratch_path = f"{lock_path}.replacement"
    write_lockfile(scratch_path, pid, command)
    Path(scratch_path).replace(lock_path)


def refuse_open(monkeypatch: pytest.MonkeyPatch, lock_path: str, failure: OSError) -> None:
    """Make every ``os.open`` of ``lock_path`` fail, leaving other paths alone."""
    real_open = os.open

    def spy_open(path: str, *args: int) -> int:
        if path == lock_path:
            raise failure
        return real_open(path, *args)

    monkeypatch.setattr(os, "open", spy_open)


def assert_holder_record(
    record: object, *, pid: int | None = None, command: str = "compare"
) -> None:
    """Assert ``record`` is exactly a live holder record for this process."""
    expected_pid = os.getpid() if pid is None else pid
    assert isinstance(record, dict)
    assert record.keys() == {"pid", "command", "at"}
    assert record["pid"] == expected_pid
    assert record["command"] == command
    assert AT_PATTERN.match(record["at"])


# ---------------------------------------------------------------------------
# acquire_lock — no lockfile exists
# ---------------------------------------------------------------------------


def test_acquire_lock_when_no_lockfile_does_record_holder_command_and_start_time():
    lock_path = fresh_lock_path()

    acquire_lock(lock_path, "compare")

    assert_holder_record(read_lockfile(lock_path))


def test_acquire_lock_when_no_lockfile_does_create_leading_directories():
    lock_path = fresh_lock_path("nested", "deeper")

    acquire_lock(lock_path, "compare")

    assert Path(lock_path).exists()


def test_acquire_lock_when_publishing_does_expose_only_whole_record_and_leave_no_scratch(
    monkeypatch: pytest.MonkeyPatch,
):
    # A rival reader peeks at the lock path each time the record is written out,
    # so a half-written lockfile would be caught in the act.
    lock_path = fresh_lock_path()
    real_write = os.write
    peeks: list[str | None] = []

    def spy_write(fd: int, data: Buffer) -> int:
        result = real_write(fd, data)
        payload = data if isinstance(data, bytes) else bytes(data)
        if b'"pid"' in payload:
            peeks.append(
                Path(lock_path).read_text(encoding="utf-8") if Path(lock_path).exists() else None
            )
        return result

    monkeypatch.setattr(os, "write", spy_write)

    acquire_lock(lock_path, "compare")

    assert peeks == [None]
    assert [entry.name for entry in Path(lock_path).parent.iterdir()] == [Path(lock_path).name]
    assert_holder_record(read_lockfile(lock_path))


# ---------------------------------------------------------------------------
# acquire_lock — a rival publishes first
# ---------------------------------------------------------------------------


def test_acquire_lock_when_rival_publishes_first_does_leave_rival_in_place_and_report_held(
    monkeypatch: pytest.MonkeyPatch,
):
    # The rival's lockfile lands after our record is written but before it is
    # published into place.
    lock_path = fresh_lock_path()
    rival_pid = os.getpid()
    real_write = os.write
    published = False

    def spy_write(fd: int, data: Buffer) -> int:
        nonlocal published
        result = real_write(fd, data)
        payload = data if isinstance(data, bytes) else bytes(data)
        if not published and b'"pid"' in payload:
            published = True
            write_lockfile(lock_path, rival_pid, "rival")
        return result

    monkeypatch.setattr(os, "write", spy_write)

    with pytest.raises(GymratError) as caught:
        acquire_lock(lock_path, "compare")

    assert f"PID {rival_pid}" in str(caught.value)
    assert "rival" in str(caught.value)
    assert read_lockfile(lock_path) == {"pid": rival_pid, "command": "rival", "at": WRITTEN_LOCK_AT}


# ---------------------------------------------------------------------------
# acquire_lock — held by a live process
# ---------------------------------------------------------------------------


def test_acquire_lock_when_held_by_live_process_does_name_holder_and_tell_caller_to_wait():
    lock_path = fresh_lock_path()
    write_lockfile(lock_path, os.getpid(), "measure")

    with pytest.raises(GymratError) as caught:
        acquire_lock(lock_path, "compare")

    assert f"PID {os.getpid()}" in str(caught.value)
    assert "measure" in str(caught.value)
    assert caught.value.hint == LIVE_HOLDER_HINT


def test_acquire_lock_when_holder_has_unknown_extra_keys_does_treat_as_live_holder():
    # Forward compatibility: a lockfile carrying a field this version does not
    # know is still a live holder, not debris to steal.
    lock_path = fresh_lock_path()
    record = json.dumps(
        {
            "pid": os.getpid(),
            "command": "measure",
            "at": WRITTEN_LOCK_AT,
            "future_field": "from a newer gymrat",
        }
    )
    write_raw_lockfile(lock_path, record)

    with pytest.raises(GymratError) as caught:
        acquire_lock(lock_path, "compare")

    assert f"PID {os.getpid()}" in str(caught.value)
    assert caught.value.hint == LIVE_HOLDER_HINT


def test_acquire_lock_when_liveness_probe_unreadable_does_leave_holder_record_in_place(
    monkeypatch: pytest.MonkeyPatch,
):
    lock_path, holder_pid = stale_lock_path()

    monkeypatch.setattr(lock_module, "is_alive", lambda _pid: True)  # pyrefly: ignore

    with pytest.raises(GymratError) as caught:
        acquire_lock(lock_path, "compare")

    assert "measure" in str(caught.value)
    assert read_lockfile(lock_path) == {
        "pid": holder_pid,
        "command": "measure",
        "at": WRITTEN_LOCK_AT,
    }


# ---------------------------------------------------------------------------
# acquire_lock — held by an exited process
# ---------------------------------------------------------------------------


def test_acquire_lock_when_held_by_exited_process_does_steal_and_overwrite_stale_record():
    lock_path, _ = stale_lock_path()

    acquire_lock(lock_path, "compare")

    assert_holder_record(read_lockfile(lock_path))


# ---------------------------------------------------------------------------
# acquire_lock — lockfile cannot be read
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("contents", ["", '{"pid":4242,"comm'])
def test_acquire_lock_when_record_incomplete_does_reclaim_lockfile(contents: str):
    # A whole record is what a reader sees, so an incomplete one is the leftover
    # of a run that died mid-write.
    lock_path = fresh_lock_path()
    write_raw_lockfile(lock_path, contents)

    acquire_lock(lock_path, "compare")

    assert_holder_record(read_lockfile(lock_path))


# ---------------------------------------------------------------------------
# acquire_lock — lockfile names an impossible process
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pid", [0, -1, 3.5, 4_294_967_296])
def test_acquire_lock_when_holder_pid_impossible_does_reclaim_lockfile(pid: float):
    # No run can be identified by this pid, so the record is damaged exactly as a
    # truncated one is — and probing it would answer for something else.
    lock_path = fresh_lock_path()
    write_lockfile(lock_path, pid, "measure")

    acquire_lock(lock_path, "compare")

    assert_holder_record(read_lockfile(lock_path))


# ---------------------------------------------------------------------------
# acquire_lock — lockfile cannot be opened
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("code", [errno.EACCES, errno.EPERM])
def test_acquire_lock_when_open_forbidden_does_name_lockfile_and_manual_remedy(
    monkeypatch: pytest.MonkeyPatch, code: int
):
    # The lockfile is another user's, readable by them alone, so gymrat cannot
    # even learn whose run holds it.
    lock_path, _ = stale_lock_path()
    refuse_open(monkeypatch, lock_path, OSError(code, "permission denied"))

    with pytest.raises(GymratError) as caught:
        acquire_lock(lock_path, "compare")

    assert lock_path in str(caught.value)
    assert re.search("remove", caught.value.hint or "", re.IGNORECASE)
    assert lock_path in (caught.value.hint or "")


def test_acquire_lock_when_open_fails_unexpectedly_does_propagate_error_unwrapped(
    monkeypatch: pytest.MonkeyPatch,
):
    lock_path, _ = stale_lock_path()
    failure = OSError(errno.EIO, "input/output error")
    refuse_open(monkeypatch, lock_path, failure)

    with pytest.raises(OSError) as caught:  # noqa: PT011
        acquire_lock(lock_path, "compare")

    assert caught.value is failure


# ---------------------------------------------------------------------------
# acquire_lock — stale lockfile belongs to another user
# ---------------------------------------------------------------------------


def test_acquire_lock_when_stale_lock_belongs_to_other_user_does_name_lockfile_and_manual_remedy(
    monkeypatch: pytest.MonkeyPatch,
):
    # A sticky /tmp — the stale file is another user's, so gymrat cannot claim it
    # out of the way.
    lock_path, _ = stale_lock_path()

    def refuse_rename(src: str, dst: str) -> None:
        raise OSError(errno.EPERM, "operation not permitted")

    monkeypatch.setattr(os, "rename", refuse_rename)

    with pytest.raises(GymratError) as caught:
        acquire_lock(lock_path, "compare")

    assert lock_path in str(caught.value)
    assert re.search("remove", caught.value.hint or "", re.IGNORECASE)
    assert lock_path in (caught.value.hint or "")


# ---------------------------------------------------------------------------
# acquire_lock — steal race is lost
# ---------------------------------------------------------------------------


def test_acquire_lock_when_steal_race_lost_does_leave_winner_in_place_and_report_held(
    monkeypatch: pytest.MonkeyPatch,
):
    # A rival takes the lock between our claim of the stale file and our
    # exclusive re-creation of it.
    lock_path, _ = stale_lock_path()
    rival_pid = os.getpid()
    real_rename = os.rename
    displaced = False

    def spy_rename(src: str, dst: str) -> None:
        nonlocal displaced
        real_rename(src, dst)
        if not displaced:
            displaced = True
            write_lockfile(lock_path, rival_pid, "rival")

    monkeypatch.setattr(os, "rename", spy_rename)

    with pytest.raises(GymratError) as caught:
        acquire_lock(lock_path, "compare")

    assert f"PID {rival_pid}" in str(caught.value)
    assert "rival" in str(caught.value)
    assert caught.value.hint == LIVE_HOLDER_HINT
    assert read_lockfile(lock_path) == {"pid": rival_pid, "command": "rival", "at": WRITTEN_LOCK_AT}


def test_acquire_lock_when_winner_releases_does_acquire_the_lock(monkeypatch: pytest.MonkeyPatch):
    # A rival claims the stale file first — our claim finds it gone — and has
    # released the lock by the time we retry.
    lock_path, _ = stale_lock_path()
    real_rename = os.rename
    real_unlink = os.unlink
    stolen = False

    def spy_rename(src: str, dst: str) -> None:
        nonlocal stolen
        if not stolen:
            stolen = True
            real_unlink(src)
            raise OSError(errno.ENOENT, "no such file or directory")
        real_rename(src, dst)

    monkeypatch.setattr(os, "rename", spy_rename)

    acquire_lock(lock_path, "compare")

    assert_holder_record(read_lockfile(lock_path))


# ---------------------------------------------------------------------------
# acquire_lock — rival reaches the same staleness verdict first
# ---------------------------------------------------------------------------


def test_acquire_lock_when_rival_reaches_same_verdict_does_leave_rival_in_place_and_report_held(
    monkeypatch: pytest.MonkeyPatch,
):
    # A rival run takes the very same stale lockfile over, and holds it, in the
    # window between our read of that lockfile and our own steal.
    lock_path, _ = stale_lock_path()
    real_is_alive = lock_module.is_alive
    rival_has_run = False

    def spy_is_alive(pid: int) -> bool:
        nonlocal rival_has_run
        if not rival_has_run:
            rival_has_run = True
            acquire_lock(lock_path, "rival")
        return real_is_alive(pid)

    monkeypatch.setattr(lock_module, "is_alive", spy_is_alive)

    with pytest.raises(GymratError) as caught:
        acquire_lock(lock_path, "compare")

    assert f"PID {os.getpid()}" in str(caught.value)
    assert "rival" in str(caught.value)
    assert_holder_record(read_lockfile(lock_path), command="rival")


# ---------------------------------------------------------------------------
# acquire_lock — a takeover died and left its claim behind
# ---------------------------------------------------------------------------


def test_acquire_lock_when_takeover_wedged_does_name_dead_holder_and_both_paths():
    lock_path, holder_pid = stale_lock_path()
    claim_path = wedge_takeover(lock_path)

    with pytest.raises(GymratError) as caught:
        acquire_lock(lock_path, "compare")

    assert str(caught.value) == (
        f"Lock at {lock_path} was left behind by a run that died while taking it over."
    )
    assert caught.value.hint == (
        f"No gymrat process holds this lock (PID {holder_pid} is dead). "
        f"To unblock, delete {lock_path} and {claim_path}, then rerun."
    )


def test_acquire_lock_when_takeover_wedged_and_record_unreadable_does_leave_holder_unnamed():
    # A truncated record names no process, so the remedy cannot either.
    lock_path = fresh_lock_path()
    write_raw_lockfile(lock_path, '{"pid":4242,"comm')
    claim_path = wedge_takeover(lock_path)

    with pytest.raises(GymratError) as caught:
        acquire_lock(lock_path, "compare")

    assert str(caught.value) == (
        f"Lock at {lock_path} was left behind by a run that died while taking it over."
    )
    assert caught.value.hint == (
        f"No gymrat process holds this lock. "
        f"To unblock, delete {lock_path} and {claim_path}, then rerun."
    )


def test_acquire_lock_when_attempt_fails_for_another_reason_does_report_contention(
    monkeypatch: pytest.MonkeyPatch,
):
    # The first steal fails because the lockfile itself went missing — not
    # because a leftover claim blocked it — so the run is contended, not wedged.
    lock_path, _ = stale_lock_path()
    claim_path = wedge_takeover(lock_path)
    real_link = os.link
    claim_refused = False

    def spy_link(existing: str, target: str) -> None:
        nonlocal claim_refused
        if target == claim_path and not claim_refused:
            claim_refused = True
            raise OSError(errno.ENOENT, "no such file or directory")
        real_link(existing, target)

    monkeypatch.setattr(os, "link", spy_link)

    with pytest.raises(GymratError) as caught:
        acquire_lock(lock_path, "compare")

    assert str(caught.value) == (
        f"Lock at {lock_path} was claimed by another process on every attempt."
    )
    assert caught.value.hint == (
        f"{LIVE_HOLDER_HINT} If no gymrat process is running, delete {lock_path}."
    )


def test_acquire_lock_when_wedged_file_no_longer_on_disk_does_report_contention(
    monkeypatch: pytest.MonkeyPatch,
):
    # The leftover claim wedges every steal, and a rival publishes its own
    # lockfile as each claim is refused — so the file at the lock path is not the
    # one the attempts wedged on, and no remedy may name it for deletion. The
    # wedged lockfile, still reachable through its claim link, is put back
    # whenever a further attempt begins.
    lock_path, holder_pid = stale_lock_path()
    claim_path = wedge_takeover(lock_path)
    real_link = os.link

    def restore_wedged_lockfile() -> None:
        scratch_path = f"{lock_path}.restored"
        real_link(claim_path, scratch_path)
        Path(scratch_path).replace(lock_path)

    def spy_link(existing: str, target: str) -> None:
        if target == lock_path:
            restore_wedged_lockfile()
        try:
            real_link(existing, target)
        finally:
            if target == claim_path:
                replace_lockfile(lock_path, holder_pid, "rival")

    monkeypatch.setattr(os, "link", spy_link)

    with pytest.raises(GymratError) as caught:
        acquire_lock(lock_path, "compare")

    assert str(caught.value) == (
        f"Lock at {lock_path} was claimed by another process on every attempt."
    )
    assert caught.value.hint == (
        f"{LIVE_HOLDER_HINT} If no gymrat process is running, delete {lock_path}."
    )


# ---------------------------------------------------------------------------
# the release handle
# ---------------------------------------------------------------------------


def test_release_when_called_does_remove_the_lockfile():
    lock_path = fresh_lock_path()
    release = acquire_lock(lock_path, "compare")

    release()

    assert not Path(lock_path).exists()


def test_release_when_unlink_fails_does_warn_instead_of_throwing(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    lock_path = fresh_lock_path()
    release = acquire_lock(lock_path, "compare")

    def refuse_unlink(path: str) -> None:
        raise OSError(errno.EIO, "disk error")

    monkeypatch.setattr(os, "unlink", refuse_unlink)

    release()

    assert "disk error" in capsys.readouterr().err


def test_release_when_lockfile_already_gone_does_stay_silent():
    lock_path = fresh_lock_path()
    release = acquire_lock(lock_path, "compare")
    release()

    release()

    assert not Path(lock_path).exists()


def test_release_when_lock_replaced_does_leave_replacement_in_place_however_often_called():
    # Our lock was taken over, so the file at the lock path is now the new
    # holder's and deleting it would unlock a live run.
    lock_path = fresh_lock_path()
    release = acquire_lock(lock_path, "compare")
    replace_lockfile(lock_path, os.getpid(), "rival")

    release()
    release()

    assert read_lockfile(lock_path) == {
        "pid": os.getpid(),
        "command": "rival",
        "at": WRITTEN_LOCK_AT,
    }


def test_release_when_stolen_lock_replaced_does_leave_replacement_in_place():
    # The same takeover, but our own lock came from stealing a stale lockfile
    # rather than from publishing a fresh one.
    lock_path, _ = stale_lock_path()
    release = acquire_lock(lock_path, "compare")
    replace_lockfile(lock_path, os.getpid(), "rival")

    release()

    assert read_lockfile(lock_path) == {
        "pid": os.getpid(),
        "command": "rival",
        "at": WRITTEN_LOCK_AT,
    }
