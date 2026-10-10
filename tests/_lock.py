"""Repository-lock helpers shared by the lock, supervise and CLI tests."""

import contextlib
import json
import os
import re
from collections.abc import Generator
from pathlib import Path

from filelock import FileLock

from gymrat.session.lock import now_iso
from gymrat.session.paths import lockfile_path, supervise_lockfile_path

#: The ISO-8601 shape a freshly published holder record stamps into ``at``.
HOLDER_AT_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$")

#: A fixed, well-formed ``at`` stamp for a planted holder record; its exact
#: value is immaterial to every test that uses it.
FIXED_HOLDER_AT = "2026-01-01T00:00:00.000Z"


def os_lock_file(lock_path: str) -> str:
    """The OS lock file ``acquire_lock`` takes beside the holder record at ``lock_path``.

    The suffix is stated here rather than imported, so the tests pin the
    on-disk layout instead of borrowing it from the code under test.

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


def remove_lock_files(root: str) -> None:
    """Remove the repository and supervise lock files keyed to ``root``.

    The lock files live in the system temp directory, not under ``root``, and
    persist after release (filelock preserves the file), so removing ``root``
    leaves them behind. Each lock is three files: the holder record, the OS
    lock, and the publish lock.

    Args:
        root: The repository whose lock files are removed.
    """
    for lock in (lockfile_path(root), supervise_lockfile_path(root)):
        for leftover in (lock, os_lock_file(lock), publish_lock_file(lock)):
            Path(leftover).unlink(missing_ok=True)
