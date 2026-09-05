"""Behavioral tests for the single-flight repository lock.

Every scenario drives the public :func:`acquire_lock` and its release handle
through real :class:`filelock.FileLock` operations and patched system calls at
the exact seams a permission error or release failure would hit.
"""

import json
import os
import re
import shutil
import sys
import tempfile
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from filelock import FileLock
from filelock import Timeout as FileLockTimeout

from gymrat.errors import GymratError
from gymrat.session.lock import _os_lock_file, _publish_lock_file, acquire_lock, is_held
from tests.conftest import hold_lock

# ---------------------------------------------------------------------------
# Constants and helpers
# ---------------------------------------------------------------------------

AT_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$")

LIVE_HOLDER_HINT = "Another gymrat run is active in this repo. Wait for it to finish."

FIXED_AT = "2026-01-01T00:00:00.000Z"

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


def read_holder(lock_path: str) -> dict[str, object]:
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
            raise PermissionError(13, "Permission denied")
        return real_open(path, *args)

    monkeypatch.setattr(os, "open", spy_open)


def time_out_publish_lock(monkeypatch: pytest.MonkeyPatch, publish_path: str) -> None:
    """Make ``FileLock.acquire`` raise ``Timeout`` only for the publish lock.

    Every other ``FileLock.acquire`` call (the main lock) behaves normally,
    simulating a stalled publisher without blocking the winning acquisition.
    """
    real_acquire = FileLock.acquire

    def selective_timeout(self: FileLock, *args: Any, **kwargs: Any) -> None:
        if self.lock_file == publish_path:
            raise FileLockTimeout(self.lock_file)
        real_acquire(self, *args, **kwargs)

    monkeypatch.setattr(FileLock, "acquire", selective_timeout)


# ---------------------------------------------------------------------------
# acquire + holder metadata
# ---------------------------------------------------------------------------


def test_acquire_lock_when_free_does_return_release_and_stamp_holder_json():
    lock_path = fresh_lock_path()

    release = acquire_lock(lock_path, "compare")

    assert callable(release)
    assert_holder_record(read_holder(lock_path))
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

    assert_holder_record(read_holder(lock_path), command="measure")
    release2()


def test_acquire_lock_when_transient_contention_does_succeed_after_retry(
    monkeypatch: pytest.MonkeyPatch,
):
    """A lock held for microseconds releases before the next retry.

    An is_held probe holds the OS lock only while probing, so acquire_lock
    succeeds instead of raising.
    """
    lock_path = fresh_lock_path()
    monkeypatch.setattr("gymrat.session.lock.LOCK_ACQUIRE_POLL_MS", 1)
    monkeypatch.setattr("gymrat.session.lock.LOCK_ACQUIRE_RETRIES", 3)

    real_acquire = FileLock.acquire
    os_lock = _os_lock_file(lock_path)
    os_lock_calls: list[int] = []

    def transient_timeout(self: FileLock, *args: Any, **kwargs: Any) -> None:
        if self.lock_file == os_lock:
            os_lock_calls.append(1)
            if len(os_lock_calls) == 1:
                raise FileLockTimeout(self.lock_file)
        real_acquire(self, *args, **kwargs)

    monkeypatch.setattr(FileLock, "acquire", transient_timeout)

    release = acquire_lock(lock_path, "compare")

    assert callable(release)
    assert len(os_lock_calls) >= 2
    release()


# ---------------------------------------------------------------------------
# crash recovery
# ---------------------------------------------------------------------------


def test_acquire_lock_when_previous_holder_released_does_succeed():
    """Simulates kernel cleanup after crash.

    Stale holder JSON remains on disk but the advisory lock is free.
    """
    lock_path = fresh_lock_path()
    holder: dict[str, object] = {"pid": 99999, "command": "measure", "at": FIXED_AT}
    blocker = hold_lock(lock_path, holder=holder)
    blocker.release()

    release = acquire_lock(lock_path, "compare")

    assert callable(release)
    assert_holder_record(read_holder(lock_path))
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
    holder_before = read_holder(lock_path)

    release()

    assert Path(lock_path).exists()
    assert read_holder(lock_path) == holder_before


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

    with (
        patch.object(
            FileLock,
            "acquire",
            autospec=True,
            side_effect=PermissionError(13, "Permission denied"),
        ),
        pytest.raises(GymratError) as caught,
    ):
        acquire_lock(lock_path, "compare")

    os_lock_path = _os_lock_file(lock_path)
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
    """Simulates a stalled publisher via a publish-lock ``Timeout``.

    The main lock is free, so ``acquire_lock`` should fall through to a
    successful acquisition and still publish the holder record.
    """
    lock_path = fresh_lock_path()
    publish_path = _publish_lock_file(lock_path)
    time_out_publish_lock(monkeypatch, publish_path)

    release = acquire_lock(lock_path, "compare")

    assert callable(release)
    assert_holder_record(read_holder(lock_path))
    release()


def test_acquire_lock_when_publish_lock_times_out_and_contended_does_still_report_holder(
    monkeypatch: pytest.MonkeyPatch,
):
    """Loser reports diagnostics even when the publish lock is unobtainable.

    A stalled publisher must not prevent the contention error from including
    best-effort diagnostics from whatever the holder file contains.
    """
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
        result = is_held(Path(lock_path))

        assert result is True
    finally:
        blocker.release()


def test_is_held_when_lock_released_does_return_false():
    lock_path = fresh_lock_path()
    blocker = hold_lock(lock_path)
    blocker.release()

    result = is_held(Path(lock_path))

    assert result is False


def test_is_held_when_lock_never_existed_does_return_false():
    lock_path = fresh_lock_path()
    Path(lock_path).parent.mkdir(parents=True, exist_ok=True)

    result = is_held(Path(lock_path))

    assert result is False


def test_is_held_when_called_from_holding_process_does_return_true():
    """A probe from the same process that holds the lock still reports held."""
    lock_path = fresh_lock_path()
    release = acquire_lock(lock_path, "compare")

    try:
        result = is_held(Path(lock_path))

        assert result is True
    finally:
        release()


def test_is_held_when_probed_does_preserve_lock_file():
    """The probe must not unlink the OS lock file, even on Windows backends.

    The ``hold_lock`` helper builds its ``FileLock`` without
    ``preserve_lock_file``, so this test constructs its own lock to control
    the file's lifetime.
    """
    lock_path = fresh_lock_path()
    Path(lock_path).parent.mkdir(parents=True, exist_ok=True)
    os_lock_path = _os_lock_file(lock_path)
    lock = FileLock(os_lock_path, timeout=0)
    lock.acquire()

    try:
        is_held(Path(lock_path))

        assert Path(os_lock_path).exists()
    finally:
        lock.release()


def test_is_held_when_probed_does_not_read_or_write_holder_record():
    """The probe touches the OS lock file only, never the holder JSON."""
    lock_path = fresh_lock_path()
    holder: dict[str, object] = {"pid": 99999, "command": "measure", "at": FIXED_AT}
    blocker = hold_lock(lock_path, holder=holder)

    try:
        is_held(Path(lock_path))

        assert read_holder(lock_path) == holder
    finally:
        blocker.release()


def test_is_held_when_permission_error_does_return_false(
    monkeypatch: pytest.MonkeyPatch,
):
    lock_path = fresh_lock_path()
    Path(lock_path).parent.mkdir(parents=True, exist_ok=True)
    refuse_open(monkeypatch, _os_lock_file(lock_path))

    result = is_held(Path(lock_path))

    assert result is False
