"""Behavioral tests for the single-flight repository lock.

Every scenario drives the public :func:`acquire_lock` and its release handle
through real :class:`filelock.FileLock` operations and patched system calls at
the exact seams a permission error or release failure would hit.
"""

import errno
import json
import os
import re
import sys
import threading
import time
from collections.abc import Callable, Generator, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest
from filelock import FileLock
from filelock import Timeout as FileLockTimeout

from gymrat.errors import GymratError
from gymrat.session.lock import (
    LockContentionError,
    LockHolder,
    acquire_lock,
    is_held,
    read_holder,
)
from tests._lock import (
    FIXED_HOLDER_AT,
    HOLDER_AT_PATTERN,
    hold_lock,
    os_lock_file,
    publish_lock_file,
)

# ---------------------------------------------------------------------------
# Constants and helpers
# ---------------------------------------------------------------------------

LIVE_HOLDER_HINT = "Another gymrat run is active in this repo. Wait for it to finish."

# The transient-contention test widens the production acquisition budget to
# ``TRANSIENT_WAIT_BUDGET_SECONDS`` and polls every
# ``TRANSIENT_POLL_INTERVAL_SECONDS``, so the acquirer outlasts the
# ``TRANSIENT_HOLD_SECONDS`` rival hold by orders of magnitude and scheduling
# stalls under parallel test runs cannot tip the result.
TRANSIENT_HOLD_SECONDS = 0.05
TRANSIENT_WAIT_BUDGET_SECONDS = 5.0
TRANSIENT_POLL_INTERVAL_SECONDS = 0.01

# How long ``held_briefly`` waits for its holder thread to take the lock.
HOLDER_READY_TIMEOUT_SECONDS = 5


@pytest.fixture
def lock_path(tmp_path: Path) -> str:
    """A lock path inside the test's own temp dir, so tests never share a file."""
    return str(tmp_path / "gymrat.lock.json")


def read_holder_json(lock_path: str) -> dict[str, object]:
    """Parse the JSON holder record stamped into the lock file."""
    return json.loads(Path(lock_path).read_text(encoding="utf-8"))


def assert_holder_record(
    record: object, *, pid: int | None = None, command: str = "compare"
) -> None:
    """Assert ``record`` is a valid holder record for this process."""
    expected_pid = os.getpid() if pid is None else pid
    assert isinstance(record, dict)
    assert record.keys() == {"pid", "command", "at"}
    assert record["pid"] == expected_pid
    assert record["command"] == command
    assert HOLDER_AT_PATTERN.match(record["at"])


def refuse_open(monkeypatch: pytest.MonkeyPatch, target_path: str) -> None:
    """Make every ``os.open`` of ``target_path`` raise PermissionError.

    Used for both the OS lock file (``lock_path + ".lock"``) and the publish
    lock file — the seams where ``filelock`` opens a file on Unix (via
    ``os.open``).
    """
    real_open = os.open

    def spy_open(path: str, *args: int) -> int:
        if path == target_path:
            raise PermissionError(errno.EACCES, "Permission denied")
        return real_open(path, *args)

    monkeypatch.setattr(os, "open", spy_open)


def fail_acquire_for(
    monkeypatch: pytest.MonkeyPatch,
    target_path: str,
    make_error: Callable[[str], Exception],
) -> None:
    """Make ``FileLock.acquire`` raise only for the lock file at ``target_path``.

    Every other ``FileLock.acquire`` call behaves normally, so the failure
    reaches one lock without blocking the rest of the acquisition.

    Args:
        monkeypatch: The fixture that patches ``FileLock.acquire`` for the test.
        target_path: The lock file whose acquisition fails.
        make_error: Builds the exception to raise, given the lock file's path.
    """
    real_acquire = FileLock.acquire

    def selective_failure(self: FileLock, *args: Any, **kwargs: Any) -> None:
        if self.lock_file == target_path:
            raise make_error(self.lock_file)
        real_acquire(self, *args, **kwargs)

    monkeypatch.setattr(FileLock, "acquire", selective_failure)


@contextmanager
def held_briefly(lock_path: str) -> Generator[None]:
    """Hold ``lock_path`` from another thread for ``TRANSIENT_HOLD_SECONDS`` after entry.

    The holder releases on its own; exiting the block joins its thread.

    Args:
        lock_path: The lock the rival thread takes.

    Yields:
        Once the rival thread holds the lock.

    Raises:
        RuntimeError: When the rival thread never signals that it holds the lock.
    """
    held = threading.Event()

    def hold() -> None:
        blocker = hold_lock(lock_path, "measure")
        held.set()
        time.sleep(TRANSIENT_HOLD_SECONDS)
        blocker.release()

    holder_thread = threading.Thread(target=hold)
    holder_thread.start()
    try:
        if not held.wait(timeout=HOLDER_READY_TIMEOUT_SECONDS):
            msg = f"holder thread did not take the lock within {HOLDER_READY_TIMEOUT_SECONDS}s"
            raise RuntimeError(msg)
        yield
    finally:
        holder_thread.join()


# ---------------------------------------------------------------------------
# acquire + holder metadata
# ---------------------------------------------------------------------------


def test_acquire_lock_when_free_does_stamp_compact_holder_json(lock_path: str):

    release = acquire_lock(lock_path, "compare")

    raw = Path(lock_path).read_text(encoding="utf-8")
    holder = json.loads(raw)
    assert_holder_record(holder)
    assert raw == json.dumps(holder, separators=(",", ":"))
    release()


def test_acquire_lock_when_parent_absent_does_create_leading_directories(tmp_path: Path):
    lock_path = str(tmp_path / "nested" / "deeper" / "gymrat.lock.json")

    release = acquire_lock(lock_path, "compare")

    assert Path(lock_path).exists()
    release()


# ---------------------------------------------------------------------------
# contention + diagnostics
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "stall_publish_lock",
    [
        pytest.param(False, id="publish-lock-free"),
        pytest.param(True, id="publish-lock-times-out"),
    ],
)
def test_acquire_lock_when_held_with_valid_json_does_report_holder_details(
    lock_path: str, stall_publish_lock: bool, monkeypatch: pytest.MonkeyPatch
):
    holder: dict[str, object] = {"pid": 99999, "command": "measure", "at": FIXED_HOLDER_AT}
    blocker = hold_lock(lock_path, holder=holder)
    if stall_publish_lock:
        fail_acquire_for(monkeypatch, publish_lock_file(lock_path), FileLockTimeout)

    try:
        with pytest.raises(LockContentionError) as caught:
            acquire_lock(lock_path, "compare")

        message = str(caught.value)
        assert "PID 99999" in message
        assert "measure" in message
        assert FIXED_HOLDER_AT in message
        assert caught.value.hint == LIVE_HOLDER_HINT
    finally:
        blocker.release()


@pytest.mark.parametrize(
    "content",
    [
        pytest.param(b"", id="empty"),
        pytest.param(b'{"pid":42,"comm', id="truncated-json"),
        pytest.param(b"\x80\x81\x82", id="non-utf8"),
    ],
)
def test_acquire_lock_when_held_with_unreadable_content_does_report_held_without_remove_advice(
    lock_path: str,
    content: bytes,
):
    Path(lock_path).parent.mkdir(parents=True, exist_ok=True)
    blocker = FileLock(os_lock_file(lock_path), timeout=0)
    blocker.acquire()
    Path(lock_path).write_bytes(content)

    try:
        with pytest.raises(LockContentionError) as caught:
            acquire_lock(lock_path, "compare")

        assert "held by another process" in str(caught.value).lower()
        assert caught.value.hint == LIVE_HOLDER_HINT
        full_text = str(caught.value) + (caught.value.hint or "")
        assert "remove" not in full_text.lower()
        assert "delete" not in full_text.lower()
    finally:
        blocker.release()


def test_acquire_lock_when_same_process_holds_lock_does_raise_gymrat_error(lock_path: str):
    release = acquire_lock(lock_path, "compare")

    try:
        with pytest.raises(GymratError):
            acquire_lock(lock_path, "measure")
    finally:
        release()


def test_acquire_lock_when_released_then_reacquired_does_succeed(lock_path: str):
    release = acquire_lock(lock_path, "compare")
    release()

    release2 = acquire_lock(lock_path, "measure")

    assert_holder_record(read_holder_json(lock_path), command="measure")
    release2()


def test_acquire_lock_when_hold_is_shorter_than_the_wait_budget_does_succeed(
    lock_path: str,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr("gymrat.session.lock._LOCK_ACQUIRE_TIMEOUT", TRANSIENT_WAIT_BUDGET_SECONDS)
    monkeypatch.setattr(
        "gymrat.session.lock._LOCK_ACQUIRE_POLL_INTERVAL", TRANSIENT_POLL_INTERVAL_SECONDS
    )

    with held_briefly(lock_path):
        release = acquire_lock(lock_path, "compare")

    assert_holder_record(read_holder_json(lock_path))
    release()


# ---------------------------------------------------------------------------
# release semantics
# ---------------------------------------------------------------------------


def test_release_when_called_twice_does_not_raise(lock_path: str):
    release = acquire_lock(lock_path, "compare")

    release()
    release()


def test_release_when_internal_error_does_warn_on_stderr(
    lock_path: str,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
):
    release = acquire_lock(lock_path, "compare")

    def failing_release(self: FileLock) -> None:
        msg = "disk went away"
        raise OSError(msg)

    monkeypatch.setattr(FileLock, "release", failing_release)

    release()

    assert "disk went away" in capsys.readouterr().err
    # Drop the flock for real so its file descriptor does not outlive the test.
    monkeypatch.undo()
    release()


def test_release_when_called_does_not_delete_lock_file(lock_path: str):
    release = acquire_lock(lock_path, "compare")
    holder_before = read_holder_json(lock_path)

    release()

    assert Path(lock_path).exists()
    assert read_holder_json(lock_path) == holder_before


# ---------------------------------------------------------------------------
# permission errors
# ---------------------------------------------------------------------------


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX hint")
@pytest.mark.parametrize(
    "lock_file",
    [
        pytest.param(os_lock_file, id="main-lock"),
        pytest.param(publish_lock_file, id="publish-lock"),
    ],
)
def test_acquire_lock_when_permission_error_posix_does_advise_removal(
    lock_path: str,
    lock_file: Callable[[str], str],
    monkeypatch: pytest.MonkeyPatch,
):
    Path(lock_path).parent.mkdir(parents=True, exist_ok=True)
    target_path = lock_file(lock_path)
    refuse_open(monkeypatch, target_path)

    with pytest.raises(GymratError) as caught:
        acquire_lock(lock_path, "compare")

    hint = caught.value.hint or ""
    assert not isinstance(caught.value, LockContentionError)
    assert "belongs to another user" in hint
    assert target_path in hint
    assert re.search("remove", hint, re.IGNORECASE)


def test_acquire_lock_when_permission_error_windows_does_advise_close_program(
    lock_path: str,
    monkeypatch: pytest.MonkeyPatch,
):
    Path(lock_path).parent.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr("sys.platform", "win32")
    os_lock_path = os_lock_file(lock_path)
    fail_acquire_for(
        monkeypatch, os_lock_path, lambda _: PermissionError(errno.EACCES, "Permission denied")
    )

    with pytest.raises(GymratError) as caught:
        acquire_lock(lock_path, "compare")

    hint = caught.value.hint or ""
    assert "locked by another process" in hint.lower()
    assert os_lock_path in hint
    assert "belongs to another user" not in hint.lower()


@pytest.mark.skipif(sys.platform == "win32", reason="fchmod not available on Windows")
def test_acquire_lock_when_acquired_does_chmod_lock_file_to_world_writable(lock_path: str):

    release = acquire_lock(lock_path, "compare")

    mode = Path(lock_path).stat().st_mode & 0o777
    assert mode == 0o666
    release()


# ---------------------------------------------------------------------------
# publish lock — serialized acquisition
# ---------------------------------------------------------------------------


def test_acquire_lock_when_publish_lock_times_out_does_still_acquire(
    lock_path: str,
    monkeypatch: pytest.MonkeyPatch,
):
    publish_path = publish_lock_file(lock_path)
    fail_acquire_for(monkeypatch, publish_path, FileLockTimeout)

    release = acquire_lock(lock_path, "compare")

    assert callable(release)
    assert_holder_record(read_holder_json(lock_path))
    release()


# ---------------------------------------------------------------------------
# is_held — advisory lock probe
# ---------------------------------------------------------------------------


@pytest.fixture
def take_rival(lock_path: str) -> Iterator[Callable[[], FileLock]]:
    """Take rival holds on the test's lock path, each released at teardown."""
    held: list[FileLock] = []

    def take() -> FileLock:
        lock = hold_lock(lock_path)
        held.append(lock)
        return lock

    yield take
    for lock in held:
        lock.release()


def _no_rival(_take: Callable[[], FileLock]) -> None:
    pass


def _rival_holding(take: Callable[[], FileLock]) -> None:
    take()


def _rival_released(take: Callable[[], FileLock]) -> None:
    take().release()


@pytest.mark.parametrize(
    ("arrange_rival", "expected"),
    [
        pytest.param(_no_rival, False, id="free"),
        pytest.param(_rival_holding, True, id="held"),
        pytest.param(_rival_released, False, id="released"),
    ],
)
def test_is_held_when_probed_does_answer_whether_a_rival_holds_the_lock(
    lock_path: str,
    take_rival: Callable[[], FileLock],
    arrange_rival: Callable[[Callable[[], FileLock]], None],
    expected: bool,
):
    Path(lock_path).parent.mkdir(parents=True, exist_ok=True)
    arrange_rival(take_rival)

    held = is_held(lock_path)

    assert held is expected


def test_is_held_when_called_from_holding_process_does_return_true(lock_path: str):
    release = acquire_lock(lock_path, "compare")

    try:
        result = is_held(lock_path)

        assert result is True
    finally:
        release()


def test_is_held_when_probed_does_leave_the_lock_held_and_holder_record_untouched(
    lock_path: str,
):
    holder: dict[str, object] = {"pid": 99999, "command": "measure", "at": FIXED_HOLDER_AT}
    blocker = hold_lock(lock_path, holder=holder)

    try:
        is_held(lock_path)

        with pytest.raises(LockContentionError):
            acquire_lock(lock_path, "compare")
        assert read_holder_json(lock_path) == holder
    finally:
        blocker.release()


def test_is_held_when_permission_error_does_return_false(
    lock_path: str,
    monkeypatch: pytest.MonkeyPatch,
):
    Path(lock_path).parent.mkdir(parents=True, exist_ok=True)
    refuse_open(monkeypatch, os_lock_file(lock_path))

    result = is_held(lock_path)

    assert result is False


def _record(content: bytes) -> Callable[[Path], None]:
    """Build an arrange step that writes ``content`` as the holder record."""

    def write(path: Path) -> None:
        path.write_bytes(content)

    return write


def _no_record(_path: Path) -> None:
    pass


def _directory_at(path: Path) -> None:
    path.mkdir()


# ---------------------------------------------------------------------------
# read_holder — holder record reporting
# ---------------------------------------------------------------------------


def test_read_holder_when_lock_acquired_does_return_pid_command_and_time(lock_path: str):
    release = acquire_lock(lock_path, "compare")

    try:
        holder = read_holder(lock_path)
    finally:
        release()

    assert holder is not None
    assert holder.pid == os.getpid()
    assert holder.command == "compare"
    assert HOLDER_AT_PATTERN.match(holder.at)


@pytest.mark.parametrize(
    "arrange",
    [
        pytest.param(_no_record, id="absent"),
        pytest.param(_directory_at, id="path-is-a-directory"),
        pytest.param(_record(b""), id="empty"),
        pytest.param(_record(b'{"pid":42,"comm'), id="truncated-json"),
        pytest.param(_record(b"\x80\x81\x82"), id="non-utf8"),
        pytest.param(_record(b'["pid", "command", "at"]'), id="not-an-object"),
        pytest.param(_record(b'{"pid":42,"command":"measure"}'), id="missing-at"),
        pytest.param(
            _record(f'{{"pid":"forty-two","command":"measure","at":"{FIXED_HOLDER_AT}"}}'.encode()),
            id="pid-not-an-integer",
        ),
        pytest.param(
            _record(f'{{"pid":true,"command":"measure","at":"{FIXED_HOLDER_AT}"}}'.encode()),
            id="pid-is-a-bool",
        ),
        pytest.param(
            _record(f'{{"pid":42,"command":7,"at":"{FIXED_HOLDER_AT}"}}'.encode()),
            id="command-not-a-string",
        ),
        pytest.param(
            _record(b'{"pid":42,"command":"measure","at":1767225600}'), id="at-not-a-string"
        ),
        pytest.param(
            _record(
                f'{{"pid":42,"command":"measure","at":"{FIXED_HOLDER_AT}","host":"box"}}'.encode()
            ),
            id="extra-field",
        ),
    ],
)
def test_read_holder_when_record_unreadable_does_return_none(
    lock_path: str, arrange: Callable[[Path], None]
):
    Path(lock_path).parent.mkdir(parents=True, exist_ok=True)
    arrange(Path(lock_path))

    holder = read_holder(lock_path)

    assert holder is None


def test_read_holder_when_holder_released_does_still_return_record(lock_path: str):
    record: dict[str, object] = {"pid": 99999, "command": "measure", "at": FIXED_HOLDER_AT}
    blocker = hold_lock(lock_path, holder=record)
    blocker.release()

    holder = read_holder(lock_path)

    assert holder == LockHolder(pid=99999, command="measure", at=FIXED_HOLDER_AT)


def test_read_holder_when_called_does_not_create_lock_files(lock_path: str):
    Path(lock_path).parent.mkdir(parents=True, exist_ok=True)
    record: dict[str, object] = {"pid": 99999, "command": "measure", "at": FIXED_HOLDER_AT}
    Path(lock_path).write_text(json.dumps(record), encoding="utf-8")

    read_holder(lock_path)

    assert not Path(os_lock_file(lock_path)).exists()
    assert not Path(publish_lock_file(lock_path)).exists()
