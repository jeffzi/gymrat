"""Repository-lock helpers shared by the lock, supervise and CLI tests."""

import contextlib
import json
import os
from collections.abc import Generator
from pathlib import Path

import pytest
from filelock import FileLock

from gymrat.errors import GymratError
from gymrat.session.lock import is_held, now_iso
from gymrat.session.paths import lockfile_path, supervise_lockfile_path

#: A fixed, well-formed ``at`` stamp for a planted holder record; its exact
#: value is immaterial to every test that uses it.
FIXED_HOLDER_AT = "2026-01-01T00:00:00.000Z"

#: Every lock ``hold_lock`` acquired, keyed by its OS lock file, so
#: ``remove_lock_files`` can release one a test still holds before unlinking it.
_held_locks: dict[str, list[FileLock]] = {}


def os_lock_file(lock_path: str) -> str:
    """The OS lock file ``acquire_lock`` takes beside the holder record at ``lock_path``.

    The suffix is restated because the production helper is private; tests
    need the path to inject faults at it and to find it in the messages that
    name it.

    Args:
        lock_path: The holder-record path.

    Returns:
        The path of the sibling OS lock file.
    """
    return lock_path + ".lock"


def publish_lock_file(lock_path: str) -> str:
    """The publish lock file ``acquire_lock`` serializes holder writes through.

    Args:
        lock_path: The holder-record path.

    Returns:
        The path of the sibling publish lock file.
    """
    return os_lock_file(lock_path) + ".publish"


def hold_lock(
    lock_path: str, command: str = "measure", *, holder: dict[str, object] | None = None
) -> FileLock:
    """Acquire a real OS lock on ``lock_path`` and stamp it with holder JSON.

    Simulates another live process holding the repository lock, so a rival
    ``acquire_lock`` call sees contention. The OS lock lives on
    ``lock_path + ".lock"`` — the same layout ``acquire_lock`` uses — so the
    holder JSON at ``lock_path`` stays readable on Windows where ``LockFileEx``
    blocks reads through a separate handle.

    Args:
        lock_path: The holder-record path whose OS lock is acquired.
        command: The command named in the built holder record.
        holder: An exact holder record to stamp instead of building one from
            ``command`` and the current process.

    Returns:
        The acquired ``FileLock``, for the caller to release during teardown.
    """
    Path(lock_path).parent.mkdir(parents=True, exist_ok=True)
    lock = FileLock(os_lock_file(lock_path), timeout=0)
    lock.acquire()
    _held_locks.setdefault(os_lock_file(lock_path), []).append(lock)
    if holder is None:
        holder = {"pid": os.getpid(), "command": command, "at": now_iso()}
    Path(lock_path).write_text(json.dumps(holder), encoding="utf-8")
    return lock


def hold_supervise_lock(root: str) -> FileLock:
    """Hold the real supervise lock for ``root``, as a live supervised run does.

    Args:
        root: The repository whose supervise lock is held.

    Returns:
        The acquired ``FileLock``, for the caller to release during teardown.
    """
    return hold_lock(supervise_lockfile_path(root), "supervise")


@contextlib.contextmanager
def held_supervise_lock(root: str) -> Generator[None]:
    """Hold the real supervise lock for ``root`` until the block exits.

    Args:
        root: The repository whose supervise lock is held.

    Yields:
        Nothing; the lock is held while the block runs and released after it.
    """
    lock = hold_supervise_lock(root)
    try:
        yield
    finally:
        lock.release()


@contextlib.contextmanager
def block_lock(lock_path: str) -> Generator[str]:
    """Put a directory where ``lock_path``'s OS lock file belongs, so probing it fails.

    Args:
        lock_path: The holder-record path whose OS lock is blocked.

    Yields:
        The blocking directory's path, which the lock-probe error names.
    """
    blocker = Path(os_lock_file(lock_path))
    blocker.mkdir(parents=True)
    try:
        yield str(blocker)
    finally:
        blocker.rmdir()


def lock_probe_error(lock_path: str) -> str:
    """The error ``is_held`` raises for ``lock_path`` while its OS lock is blocked.

    Callers that only pass the lock-probe error through compare against this, so
    its wording stays pinned in the lock tests alone.

    Args:
        lock_path: The holder-record path whose OS lock ``block_lock`` blocks.

    Returns:
        The message of the ``GymratError`` the probe raises.
    """
    with pytest.raises(GymratError) as caught:
        is_held(lock_path)
    return str(caught.value)


def remove_lock_files(root: str) -> None:
    """Remove the repository and supervise lock files keyed to ``root``.

    The lock files live in the system temp directory, not under ``root``, and
    persist after release (filelock preserves the file), so removing ``root``
    leaves them behind. Each lock is three files: the holder record, the OS
    lock, and the publish lock.

    A lock ``hold_lock`` acquired that the test still holds is released first:
    Windows refuses to unlink a locked file, and the fixture releasing it may
    tear down after this cleanup runs.

    Args:
        root: The repository whose lock files are removed.
    """
    for lock in (lockfile_path(root), supervise_lockfile_path(root)):
        for leftover in (lock, os_lock_file(lock), publish_lock_file(lock)):
            for held in _held_locks.pop(leftover, []):
                held.release()
            Path(leftover).unlink(missing_ok=True)
