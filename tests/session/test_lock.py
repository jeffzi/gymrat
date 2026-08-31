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
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import patch

import pytest
from filelock import FileLock

from gymrat.errors import GymratError
from gymrat.session.lock import _os_lock_file, acquire_lock
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


def refuse_open(monkeypatch: pytest.MonkeyPatch, lock_path: str) -> None:
    """Make every ``os.open`` of the OS lock file raise PermissionError.

    The OS lock lives at ``lock_path + ".lock"``; targeting that path matches
    the seam where ``filelock`` opens the file on Unix (via ``os.open``).
    """
    real_open = os.open
    os_lock_path = _os_lock_file(lock_path)

    def spy_open(path: str, *args: int) -> int:
        if path == os_lock_path:
            raise PermissionError(13, "Permission denied")
        return real_open(path, *args)

    monkeypatch.setattr(os, "open", spy_open)


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


# ---------------------------------------------------------------------------
# crash recovery
# ---------------------------------------------------------------------------


def test_acquire_lock_when_previous_holder_released_does_succeed():
    """After a FileLock is released, acquire_lock succeeds.

    Simulates kernel cleanup after crash: stale holder JSON remains on disk
    but the advisory lock is free.
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
def test_acquire_lock_when_permission_error_posix_does_advise_removal(
    monkeypatch: pytest.MonkeyPatch,
):
    lock_path = fresh_lock_path()
    Path(lock_path).parent.mkdir(parents=True, exist_ok=True)
    refuse_open(monkeypatch, lock_path)

    with pytest.raises(GymratError) as caught:
        acquire_lock(lock_path, "compare")

    os_lock_path = _os_lock_file(lock_path)
    hint = caught.value.hint or ""
    assert "belongs to another user" in hint
    assert os_lock_path in hint
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
