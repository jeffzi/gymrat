"""Behavioral tests for the single-flight repository lock.

Every scenario drives the public :func:`acquire_lock` and its release handle
through real :class:`filelock.FileLock` operations and patched system calls at
the exact seams a permission error or release failure would hit.
"""

import errno
import json
import os
import sys
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, Protocol
from unittest.mock import create_autospec

import pytest
from filelock import FileLock
from filelock import Timeout as FileLockTimeout

from gymrat.errors import GymratError
from gymrat.session.lock import (
    LockContentionError,
    acquire_lock,
    is_held,
)
from tests._lock import (
    FIXED_HOLDER_AT,
    hold_lock,
    os_lock_file,
    publish_lock_file,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def lock_path(tmp_path: Path) -> str:
    """A lock path inside the test's own temp dir, so tests never share a file."""
    return str(tmp_path / "gymrat.lock.json")


class Acquire(Protocol):
    """Takes a lock the way :func:`acquire_lock` does, released at teardown."""

    def __call__(self, path: str, command: str) -> Callable[[], None]: ...


@pytest.fixture
def acquire() -> Iterator[Acquire]:
    """Acquire locks through ``acquire_lock``, releasing every hold at teardown."""
    releases: list[Callable[[], None]] = []

    def take(path: str, command: str) -> Callable[[], None]:
        release = acquire_lock(path, command)
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
    assert record["pid"] == expected_pid
    assert record["command"] == command


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


# ---------------------------------------------------------------------------
# acquire + holder metadata
# ---------------------------------------------------------------------------


def test_acquire_lock_when_parent_absent_does_create_leading_directories(
    tmp_path: Path, acquire: Acquire
):
    lock_path = str(tmp_path / "nested" / "deeper" / "gymrat.lock.json")

    acquire(lock_path, "compare")

    assert Path(lock_path).exists()


# ---------------------------------------------------------------------------
# contention + diagnostics
# ---------------------------------------------------------------------------


def test_acquire_lock_when_held_with_valid_json_does_report_holder_details(lock_path: str):
    holder: dict[str, object] = {"pid": 99999, "command": "measure", "at": FIXED_HOLDER_AT}
    blocker = hold_lock(lock_path, holder=holder)

    try:
        with pytest.raises(LockContentionError) as caught:
            acquire_lock(lock_path, "compare")

        message = str(caught.value)
        assert "99999" in message
        assert "measure" in message
        assert FIXED_HOLDER_AT in message
    finally:
        blocker.release()


def test_acquire_lock_when_held_with_unreadable_content_does_report_the_lock_held(lock_path: str):
    blocker = hold_lock(lock_path)
    Path(lock_path).write_bytes(b'{"pid":42,"comm')

    try:
        with pytest.raises(LockContentionError) as caught:
            acquire_lock(lock_path, "compare")

        assert lock_path in str(caught.value)
    finally:
        blocker.release()


def test_acquire_lock_when_released_then_reacquired_does_succeed(lock_path: str, acquire: Acquire):
    release = acquire(lock_path, "compare")
    release()

    acquire(lock_path, "measure")

    assert_holder_record(read_holder_json(lock_path), command="measure")


# ---------------------------------------------------------------------------
# release semantics
# ---------------------------------------------------------------------------


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

    assert lock_path in capsys.readouterr().err


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
    target_path = lock_file(lock_path)
    refuse_open(monkeypatch, target_path)

    with pytest.raises(GymratError) as caught:
        acquire_lock(lock_path, "compare")

    hint = caught.value.hint or ""
    assert "belongs to another user" in hint
    assert target_path in hint


def test_acquire_lock_when_permission_error_windows_does_advise_close_program(
    lock_path: str,
    monkeypatch: pytest.MonkeyPatch,
):
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


def _acquire_for_compare(lock_path: str) -> object:
    return acquire_lock(lock_path, "compare")


@pytest.mark.skipif(
    sys.platform == "win32", reason="a directory at the lock path is a PermissionError on Windows"
)
@pytest.mark.parametrize(
    ("probe", "lock_file"),
    [
        pytest.param(_acquire_for_compare, os_lock_file, id="acquire-main-lock"),
        pytest.param(_acquire_for_compare, publish_lock_file, id="acquire-publish-lock"),
        pytest.param(is_held, os_lock_file, id="is-held"),
    ],
)
def test_lock_probe_when_lock_file_cannot_be_opened_does_raise_without_a_hint(
    lock_path: str, probe: Callable[[str], object], lock_file: Callable[[str], str]
):
    target_path = lock_file(lock_path)
    Path(target_path).mkdir(parents=True)

    with pytest.raises(GymratError) as caught:
        probe(lock_path)

    assert target_path in str(caught.value)
    assert str(caught.value.__cause__) in str(caught.value)
    assert caught.value.hint is None


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


@pytest.mark.parametrize(
    ("arrange_rival", "expected"),
    [
        pytest.param(_no_rival, False, id="free"),
        pytest.param(_rival_holding, True, id="held"),
    ],
)
def test_is_held_when_probed_does_answer_whether_a_rival_holds_the_lock(
    lock_path: str,
    take_rival: Callable[[], FileLock],
    arrange_rival: Callable[[Callable[[], FileLock]], None],
    expected: bool,
):
    arrange_rival(take_rival)

    held = is_held(lock_path)

    assert held is expected


def test_is_held_when_permission_error_does_raise_the_error_acquire_lock_raises(
    lock_path: str,
    monkeypatch: pytest.MonkeyPatch,
):
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
    assert (type(caught.value), str(caught.value), caught.value.hint) == (
        type(expected),
        str(expected),
        expected.hint,
    )
