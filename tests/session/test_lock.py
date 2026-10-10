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
from typing import Any, Protocol
from unittest.mock import create_autospec

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

# The transient-contention test widens the acquisition budget to
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


class Acquire(Protocol):
    """Takes a lock the way :func:`acquire_lock` does, released at teardown."""

    def __call__(
        self, path: str, command: str, *, wait: float = ..., poll_interval: float = ...
    ) -> Callable[[], None]: ...


@pytest.fixture
def acquire() -> Iterator[Acquire]:
    """Acquire locks through ``acquire_lock``, releasing every hold at teardown."""
    releases: list[Callable[[], None]] = []

    def take(path: str, command: str, **wait: float) -> Callable[[], None]:
        release = acquire_lock(path, command, **wait)
        releases.append(release)
        return release

    yield take
    for release in releases:
        release()


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

    ``os.open`` is the seam where ``filelock`` opens a lock file on Unix.

    Args:
        monkeypatch: The fixture that patches ``os.open`` for the test.
        target_path: The file whose opening fails: the OS lock file
            (``lock_path + ".lock"``) or the publish lock file.
    """
    real_open = os.open

    def spy_open(path: str, *args: int) -> int:
        if path == target_path:
            raise PermissionError(errno.EACCES, "Permission denied")
        return real_open(path, *args)

    monkeypatch.setattr(os, "open", create_autospec(os.open, side_effect=spy_open))


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

    monkeypatch.setattr(
        FileLock, "acquire", create_autospec(FileLock.acquire, side_effect=selective_failure)
    )


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


def test_acquire_lock_when_free_does_stamp_compact_holder_json(lock_path: str, acquire: Acquire):
    acquire(lock_path, "compare")

    raw = Path(lock_path).read_text(encoding="utf-8")
    holder = json.loads(raw)
    assert_holder_record(holder)
    assert raw == json.dumps(holder, separators=(",", ":"))


def test_acquire_lock_when_parent_absent_does_create_leading_directories(
    tmp_path: Path, acquire: Acquire
):
    lock_path = str(tmp_path / "nested" / "deeper" / "gymrat.lock.json")

    acquire(lock_path, "compare")

    assert Path(lock_path).exists()


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


def test_acquire_lock_when_held_with_unreadable_content_does_report_held_without_remove_advice(
    lock_path: str,
):
    blocker = hold_lock(lock_path)
    Path(lock_path).write_bytes(b'{"pid":42,"comm')

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


def test_acquire_lock_when_released_then_reacquired_does_succeed(lock_path: str, acquire: Acquire):
    release = acquire(lock_path, "compare")
    release()

    acquire(lock_path, "measure")

    assert_holder_record(read_holder_json(lock_path), command="measure")


def test_acquire_lock_when_hold_is_shorter_than_the_wait_budget_does_succeed(
    lock_path: str,
    acquire: Acquire,
):
    with held_briefly(lock_path):
        acquire(
            lock_path,
            "compare",
            wait=TRANSIENT_WAIT_BUDGET_SECONDS,
            poll_interval=TRANSIENT_POLL_INTERVAL_SECONDS,
        )

    assert_holder_record(read_holder_json(lock_path))


# ---------------------------------------------------------------------------
# release semantics
# ---------------------------------------------------------------------------


def test_release_when_called_twice_does_not_raise(lock_path: str):
    release = acquire_lock(lock_path, "compare")
    release()

    release()


def test_release_when_internal_error_does_warn_on_stderr(
    lock_path: str,
    acquire: Acquire,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
):
    release = acquire(lock_path, "compare")

    def failing_release(self: FileLock) -> None:
        msg = "disk went away"
        raise OSError(msg)

    # The patch is scoped to the call so the teardown release drops the flock for real.
    with monkeypatch.context() as patched:
        patched.setattr(
            FileLock, "release", create_autospec(FileLock.release, side_effect=failing_release)
        )
        release()

    assert "disk went away" in capsys.readouterr().err


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


@pytest.mark.skipif(
    sys.platform == "win32", reason="a directory at the lock path is a PermissionError on Windows"
)
@pytest.mark.parametrize(
    "lock_file",
    [
        pytest.param(os_lock_file, id="main-lock"),
        pytest.param(publish_lock_file, id="publish-lock"),
    ],
)
def test_acquire_lock_when_lock_file_cannot_be_opened_does_raise_without_a_hint(
    lock_path: str, lock_file: Callable[[str], str]
):
    target_path = lock_file(lock_path)
    Path(target_path).mkdir(parents=True)

    with pytest.raises(GymratError) as caught:
        acquire_lock(lock_path, "compare")

    assert not isinstance(caught.value, LockContentionError)
    assert str(caught.value).startswith(f"Lock file {target_path} could not be opened: ")
    assert os.strerror(errno.EISDIR) in str(caught.value)
    assert caught.value.hint is None


@pytest.mark.skipif(sys.platform == "win32", reason="fchmod not available on Windows")
def test_acquire_lock_when_acquired_does_chmod_lock_file_to_world_writable(
    lock_path: str, acquire: Acquire
):
    acquire(lock_path, "compare")

    mode = Path(lock_path).stat().st_mode & 0o777
    assert mode == 0o666


# ---------------------------------------------------------------------------
# publish lock — serialized acquisition
# ---------------------------------------------------------------------------


def test_acquire_lock_when_publish_lock_times_out_does_still_acquire(
    lock_path: str,
    acquire: Acquire,
    monkeypatch: pytest.MonkeyPatch,
):
    publish_path = publish_lock_file(lock_path)
    fail_acquire_for(monkeypatch, publish_path, FileLockTimeout)

    acquire(lock_path, "compare")

    assert_holder_record(read_holder_json(lock_path))


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


def test_is_held_when_probed_does_leave_the_lock_undisturbed(
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


def test_is_held_when_permission_error_does_raise_the_error_acquire_lock_raises(
    lock_path: str,
    monkeypatch: pytest.MonkeyPatch,
):
    Path(lock_path).parent.mkdir(parents=True, exist_ok=True)
    # filelock opens the lock file through ``os.open`` only on Unix, so the
    # refusal is injected at ``FileLock.acquire`` to reach every platform.
    fail_acquire_for(
        monkeypatch,
        os_lock_file(lock_path),
        lambda _: PermissionError(errno.EACCES, "Permission denied"),
    )
    with pytest.raises(GymratError) as acquire_refused:
        acquire_lock(lock_path, "compare")

    with pytest.raises(GymratError) as caught:
        is_held(lock_path)

    expected = acquire_refused.value
    assert str(caught.value).startswith(
        f"Lock file {os_lock_file(lock_path)} could not be opened: "
    )
    assert (type(caught.value), str(caught.value), caught.value.hint) == (
        type(expected),
        str(expected),
        expected.hint,
    )


@pytest.mark.skipif(
    sys.platform == "win32", reason="a directory at the lock path is a PermissionError on Windows"
)
def test_is_held_when_lock_file_cannot_be_opened_does_raise_without_a_hint(lock_path: str):
    os_lock_path = os_lock_file(lock_path)
    Path(os_lock_path).mkdir(parents=True)

    with pytest.raises(GymratError) as caught:
        is_held(lock_path)

    assert not isinstance(caught.value, LockContentionError)
    assert str(caught.value).startswith(f"Lock file {os_lock_path} could not be opened: ")
    assert os.strerror(errno.EISDIR) in str(caught.value)
    assert caught.value.hint is None


# ---------------------------------------------------------------------------
# read_holder — holder record reporting
# ---------------------------------------------------------------------------


def test_read_holder_when_lock_acquired_does_return_pid_command_and_time(
    lock_path: str, acquire: Acquire
):
    acquire(lock_path, "compare")

    holder = read_holder(lock_path)

    assert holder is not None
    assert holder.pid == os.getpid()
    assert holder.command == "compare"
    assert HOLDER_AT_PATTERN.match(holder.at)


@pytest.mark.parametrize(
    "content",
    [
        pytest.param(
            f'{{"pid":true,"command":"measure","at":"{FIXED_HOLDER_AT}"}}'.encode(),
            id="pid-is-a-bool",
        ),
        pytest.param(
            f'{{"pid":42,"command":"measure","at":"{FIXED_HOLDER_AT}","host":"box"}}'.encode(),
            id="extra-field",
        ),
    ],
)
def test_read_holder_when_record_not_a_holder_does_return_none(lock_path: str, content: bytes):
    Path(lock_path).write_bytes(content)

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
