"""Behavioral tests for the single-flight repository lock.

Every scenario drives the public :func:`acquire_lock` and its release handle
through real :class:`filelock.FileLock` operations and patched system calls at
the exact seams a permission error or release failure would hit.
"""

import errno
import json
import os
import re
import shutil
import sys
import tempfile
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
    _os_lock_file,
    _publish_lock_file,
    acquire_lock,
    is_held,
    now_iso,
    read_holder,
)
from tests.conftest import hold_lock

# ---------------------------------------------------------------------------
# Constants and helpers
# ---------------------------------------------------------------------------

AT_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$")

LIVE_HOLDER_HINT = "Another gymrat run is active in this repo. Wait for it to finish."

FIXED_AT = "2026-01-01T00:00:00.000Z"

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

_temp_dirs: list[str] = []


def fresh_lock_path(*segments: str) -> str:
    """Return a lock path inside its own temp dir, so tests never share a file."""
    directory = tempfile.mkdtemp(prefix="lock-test-")
    _temp_dirs.append(directory)
    return str(Path(directory, *segments, "gymrat.lock.json"))


@pytest.fixture(autouse=True)
def _cleanup_temp_dirs() -> Iterator[None]:
    """Remove any temp dirs ``fresh_lock_path`` created during the test."""
    yield
    while _temp_dirs:
        shutil.rmtree(_temp_dirs.pop(), ignore_errors=True)


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
    assert AT_PATTERN.match(record["at"])


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


def time_out_publish_lock(monkeypatch: pytest.MonkeyPatch, publish_path: str) -> None:
    """Make ``FileLock.acquire`` raise ``Timeout`` only for the publish lock.

    Simulates a stalled publisher without blocking the winning acquisition.

    Args:
        monkeypatch: The fixture that patches ``FileLock.acquire`` for the test.
        publish_path: The publish lock file whose acquisition times out.
    """
    fail_acquire_for(monkeypatch, publish_path, FileLockTimeout)


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


def test_now_iso_when_called_does_render_millisecond_utc_timestamp():
    result = now_iso()

    assert AT_PATTERN.match(result)


def test_acquire_lock_when_free_does_stamp_compact_holder_json():
    lock_path = fresh_lock_path()

    release = acquire_lock(lock_path, "compare")

    raw = Path(lock_path).read_text(encoding="utf-8")
    holder = json.loads(raw)
    assert_holder_record(holder)
    assert raw == json.dumps(holder, separators=(",", ":"))
    release()


def test_acquire_lock_when_parent_absent_does_create_leading_directories():
    lock_path = fresh_lock_path("nested", "deeper")

    release = acquire_lock(lock_path, "compare")

    assert Path(lock_path).exists()
    release()


# ---------------------------------------------------------------------------
# contention + diagnostics
# ---------------------------------------------------------------------------


def test_acquire_lock_when_held_with_valid_json_does_report_holder_details():
    lock_path = fresh_lock_path()
    holder: dict[str, object] = {"pid": 99999, "command": "measure", "at": FIXED_AT}
    blocker = hold_lock(lock_path, holder=holder)

    try:
        with pytest.raises(GymratError) as caught:
            acquire_lock(lock_path, "compare")

        message = str(caught.value)
        assert "PID 99999" in message
        assert "measure" in message
        assert FIXED_AT in message
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
    content: bytes,
):
    lock_path = fresh_lock_path()
    Path(lock_path).parent.mkdir(parents=True, exist_ok=True)
    blocker = FileLock(_os_lock_file(lock_path), timeout=0)
    blocker.acquire()
    Path(lock_path).write_bytes(content)

    try:
        with pytest.raises(GymratError) as caught:
            acquire_lock(lock_path, "compare")

        assert caught.value.hint == LIVE_HOLDER_HINT
        full_text = str(caught.value) + (caught.value.hint or "")
        assert "remove" not in full_text.lower()
        assert "delete" not in full_text.lower()
    finally:
        blocker.release()


def test_acquire_lock_when_same_process_holds_lock_does_raise_gymrat_error():
    lock_path = fresh_lock_path()
    release = acquire_lock(lock_path, "compare")

    try:
        with pytest.raises(GymratError):
            acquire_lock(lock_path, "measure")
    finally:
        release()


def test_acquire_lock_when_released_then_reacquired_does_succeed():
    lock_path = fresh_lock_path()
    release = acquire_lock(lock_path, "compare")
    release()

    release2 = acquire_lock(lock_path, "measure")

    assert_holder_record(read_holder_json(lock_path), command="measure")
    release2()


def test_acquire_lock_when_hold_is_shorter_than_the_wait_budget_does_succeed(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr("gymrat.session.lock._LOCK_ACQUIRE_TIMEOUT", TRANSIENT_WAIT_BUDGET_SECONDS)
    monkeypatch.setattr(
        "gymrat.session.lock._LOCK_ACQUIRE_POLL_INTERVAL", TRANSIENT_POLL_INTERVAL_SECONDS
    )
    lock_path = fresh_lock_path()

    with held_briefly(lock_path):
        release = acquire_lock(lock_path, "compare")

    assert_holder_record(read_holder_json(lock_path))
    release()


# ---------------------------------------------------------------------------
# crash recovery
# ---------------------------------------------------------------------------


def test_acquire_lock_when_previous_holder_released_does_succeed():
    lock_path = fresh_lock_path()
    holder: dict[str, object] = {"pid": 99999, "command": "measure", "at": FIXED_AT}
    blocker = hold_lock(lock_path, holder=holder)
    blocker.release()

    release = acquire_lock(lock_path, "compare")

    assert callable(release)
    assert_holder_record(read_holder_json(lock_path))
    release()


# ---------------------------------------------------------------------------
# release semantics
# ---------------------------------------------------------------------------


def test_release_when_called_twice_does_not_raise():
    lock_path = fresh_lock_path()
    release = acquire_lock(lock_path, "compare")

    release()
    release()


def test_release_when_internal_error_does_warn_on_stderr(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
):
    lock_path = fresh_lock_path()
    release = acquire_lock(lock_path, "compare")

    def failing_release(self: FileLock) -> None:
        msg = "disk went away"
        raise OSError(msg)

    monkeypatch.setattr(FileLock, "release", failing_release)

    release()

    assert "disk went away" in capsys.readouterr().err


def test_release_when_called_does_not_delete_lock_file():
    lock_path = fresh_lock_path()
    release = acquire_lock(lock_path, "compare")
    holder_before = read_holder_json(lock_path)

    release()

    assert Path(lock_path).exists()
    assert Path(_os_lock_file(lock_path)).exists()
    assert read_holder_json(lock_path) == holder_before


# ---------------------------------------------------------------------------
# permission errors
# ---------------------------------------------------------------------------


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX hint")
@pytest.mark.parametrize(
    "lock_file",
    [
        pytest.param(_os_lock_file, id="main-lock"),
        pytest.param(_publish_lock_file, id="publish-lock"),
    ],
)
def test_acquire_lock_when_permission_error_posix_does_advise_removal(
    lock_file: Callable[[str], str],
    monkeypatch: pytest.MonkeyPatch,
):
    lock_path = fresh_lock_path()
    Path(lock_path).parent.mkdir(parents=True, exist_ok=True)
    target_path = lock_file(lock_path)
    refuse_open(monkeypatch, target_path)

    with pytest.raises(GymratError) as caught:
        acquire_lock(lock_path, "compare")

    hint = caught.value.hint or ""
    assert "belongs to another user" in hint
    assert target_path in hint
    assert re.search("remove", hint, re.IGNORECASE)


def test_acquire_lock_when_permission_error_windows_does_advise_close_program(
    monkeypatch: pytest.MonkeyPatch,
):
    lock_path = fresh_lock_path()
    Path(lock_path).parent.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr("sys.platform", "win32")
    os_lock_path = _os_lock_file(lock_path)
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
def test_acquire_lock_when_acquired_does_chmod_lock_file_to_world_writable():
    lock_path = fresh_lock_path()

    release = acquire_lock(lock_path, "compare")

    mode = Path(lock_path).stat().st_mode & 0o777
    assert mode == 0o666
    release()


# ---------------------------------------------------------------------------
# publish lock — serialized acquisition
# ---------------------------------------------------------------------------


def test_acquire_lock_when_publish_lock_times_out_does_still_acquire(
    monkeypatch: pytest.MonkeyPatch,
):
    lock_path = fresh_lock_path()
    publish_path = _publish_lock_file(lock_path)
    time_out_publish_lock(monkeypatch, publish_path)

    release = acquire_lock(lock_path, "compare")

    assert callable(release)
    assert_holder_record(read_holder_json(lock_path))
    release()


def test_acquire_lock_when_publish_lock_times_out_and_contended_does_still_report_holder(
    monkeypatch: pytest.MonkeyPatch,
):
    lock_path = fresh_lock_path()
    publish_path = _publish_lock_file(lock_path)
    holder: dict[str, object] = {"pid": 99999, "command": "measure", "at": FIXED_AT}
    blocker = hold_lock(lock_path, holder=holder)
    time_out_publish_lock(monkeypatch, publish_path)

    try:
        with pytest.raises(GymratError) as caught:
            acquire_lock(lock_path, "compare")

        message = str(caught.value)
        assert "PID 99999" in message
        assert "measure" in message
        assert FIXED_AT in message
        assert caught.value.hint == LIVE_HOLDER_HINT
    finally:
        blocker.release()


# ---------------------------------------------------------------------------
# is_held — advisory lock probe
# ---------------------------------------------------------------------------


def test_is_held_when_lock_active_does_return_true():
    lock_path = fresh_lock_path()
    blocker = hold_lock(lock_path)

    try:
        result = is_held(lock_path)

        assert result is True
    finally:
        blocker.release()


def test_is_held_when_lock_released_does_return_false():
    lock_path = fresh_lock_path()
    blocker = hold_lock(lock_path)
    blocker.release()

    result = is_held(lock_path)

    assert result is False


def test_is_held_when_lock_never_existed_does_return_false():
    lock_path = fresh_lock_path()
    Path(lock_path).parent.mkdir(parents=True, exist_ok=True)

    result = is_held(lock_path)

    assert result is False


def test_is_held_when_called_from_holding_process_does_return_true():
    lock_path = fresh_lock_path()
    release = acquire_lock(lock_path, "compare")

    try:
        result = is_held(lock_path)

        assert result is True
    finally:
        release()


def test_is_held_when_probed_does_preserve_lock_file():
    lock_path = fresh_lock_path()
    Path(lock_path).parent.mkdir(parents=True, exist_ok=True)
    os_lock_path = _os_lock_file(lock_path)
    lock = FileLock(os_lock_path, timeout=0)
    lock.acquire()

    try:
        is_held(lock_path)

        assert Path(os_lock_path).exists()
    finally:
        lock.release()


def test_is_held_when_probed_does_not_read_or_write_holder_record():
    lock_path = fresh_lock_path()
    holder: dict[str, object] = {"pid": 99999, "command": "measure", "at": FIXED_AT}
    blocker = hold_lock(lock_path, holder=holder)

    try:
        is_held(lock_path)

        assert read_holder_json(lock_path) == holder
    finally:
        blocker.release()


def test_is_held_when_permission_error_does_return_false(
    monkeypatch: pytest.MonkeyPatch,
):
    lock_path = fresh_lock_path()
    Path(lock_path).parent.mkdir(parents=True, exist_ok=True)
    refuse_open(monkeypatch, _os_lock_file(lock_path))

    result = is_held(lock_path)

    assert result is False


# ---------------------------------------------------------------------------
# read_holder — holder record reporting
# ---------------------------------------------------------------------------


def test_read_holder_when_lock_acquired_does_return_pid_command_and_time():
    lock_path = fresh_lock_path()
    release = acquire_lock(lock_path, "compare")

    try:
        holder = read_holder(lock_path)
    finally:
        release()

    assert holder is not None
    assert holder.pid == os.getpid()
    assert holder.command == "compare"
    assert AT_PATTERN.match(holder.at)


@pytest.mark.parametrize(
    "content",
    [
        pytest.param(None, id="absent"),
        pytest.param(b"", id="empty"),
        pytest.param(b'{"pid":42,"comm', id="truncated-json"),
        pytest.param(b"\x80\x81\x82", id="non-utf8"),
        pytest.param(b'["pid", "command", "at"]', id="not-an-object"),
        pytest.param(b'{"pid":42,"command":"measure"}', id="missing-at"),
        pytest.param(
            b'{"pid":"forty-two","command":"measure","at":"2026-01-01T00:00:00.000Z"}',
            id="pid-not-an-integer",
        ),
        pytest.param(
            b'{"pid":true,"command":"measure","at":"2026-01-01T00:00:00.000Z"}',
            id="pid-is-a-bool",
        ),
        pytest.param(
            b'{"pid":42,"command":7,"at":"2026-01-01T00:00:00.000Z"}',
            id="command-not-a-string",
        ),
        pytest.param(b'{"pid":42,"command":"measure","at":1767225600}', id="at-not-a-string"),
        pytest.param(
            b'{"pid":42,"command":"measure","at":"2026-01-01T00:00:00.000Z","host":"box"}',
            id="extra-field",
        ),
    ],
)
def test_read_holder_when_record_unreadable_does_return_none(content: bytes | None):
    lock_path = fresh_lock_path()
    Path(lock_path).parent.mkdir(parents=True, exist_ok=True)
    if content is not None:
        Path(lock_path).write_bytes(content)

    holder = read_holder(lock_path)

    assert holder is None


def test_read_holder_when_record_path_is_a_directory_does_return_none():
    lock_path = fresh_lock_path()
    Path(lock_path).mkdir(parents=True)

    holder = read_holder(lock_path)

    assert holder is None


def test_read_holder_when_holder_released_does_still_return_record():
    lock_path = fresh_lock_path()
    record: dict[str, object] = {"pid": 99999, "command": "measure", "at": FIXED_AT}
    blocker = hold_lock(lock_path, holder=record)
    blocker.release()

    holder = read_holder(lock_path)

    assert holder == LockHolder(pid=99999, command="measure", at=FIXED_AT)


def test_read_holder_when_called_does_not_create_lock_files():
    lock_path = fresh_lock_path()
    Path(lock_path).parent.mkdir(parents=True, exist_ok=True)
    record: dict[str, object] = {"pid": 99999, "command": "measure", "at": FIXED_AT}
    Path(lock_path).write_text(json.dumps(record), encoding="utf-8")

    read_holder(lock_path)

    assert not Path(_os_lock_file(lock_path)).exists()
    assert not Path(_publish_lock_file(lock_path)).exists()


# ---------------------------------------------------------------------------
# LockContentionError — contention is distinguishable from other failures
# ---------------------------------------------------------------------------


def test_acquire_lock_when_held_does_raise_lock_contention_error():
    lock_path = fresh_lock_path()
    blocker = hold_lock(lock_path)

    try:
        with pytest.raises(LockContentionError):
            acquire_lock(lock_path, "compare")
    finally:
        blocker.release()


def test_acquire_lock_when_held_with_unreadable_record_does_raise_contention_generically():
    lock_path = fresh_lock_path()
    Path(lock_path).parent.mkdir(parents=True, exist_ok=True)
    blocker = FileLock(_os_lock_file(lock_path), timeout=0)
    blocker.acquire()
    Path(lock_path).write_bytes(b"")

    try:
        with pytest.raises(LockContentionError) as caught:
            acquire_lock(lock_path, "compare")

        assert "held by another process" in str(caught.value).lower()
    finally:
        blocker.release()


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX os.open seam")
def test_acquire_lock_when_permission_error_does_not_raise_lock_contention_error(
    monkeypatch: pytest.MonkeyPatch,
):
    lock_path = fresh_lock_path()
    Path(lock_path).parent.mkdir(parents=True, exist_ok=True)
    refuse_open(monkeypatch, _os_lock_file(lock_path))

    with pytest.raises(GymratError) as caught:
        acquire_lock(lock_path, "compare")

    assert not isinstance(caught.value, LockContentionError)
