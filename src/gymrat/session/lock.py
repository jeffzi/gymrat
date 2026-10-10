"""Single-flight repository lock via OS advisory locks (``filelock.FileLock``).

A run takes the lock by acquiring a ``flock``/``LockFileEx`` on a dedicated
``.lock`` file next to the holder metadata path, waiting at most
``_LOCK_ACQUIRE_TIMEOUT`` seconds (by default) for a rival to let go.  The holder record is
written to the original path as compact JSON, readable on every platform — including
Windows, where ``LockFileEx`` creates a mandatory byte-range lock that blocks
reads through a separate handle.

A sibling publish lock serializes the window between acquiring the main lock and
writing the holder record.  The ordering — publish lock, then main lock attempt,
then write or read, then publish release — ensures that no rival observes an
empty, truncated, or previous-holder record while a fresh holder is mid-write.
When the publish lock cannot be obtained within ``_PUBLISH_LOCK_TIMEOUT`` seconds
(a stalled publisher), acquisition proceeds without it: a winner still writes its
record, a loser still reports best-effort diagnostics from whatever the file
holds.

Contention is decided within the wait for the main lock: the loser reads the
winner's holder record for diagnostics without needing liveness probes.  Crash
recovery is automatic — the kernel releases the advisory lock when the holder
exits — so a stale lockfile never needs manual cleanup.
"""

import contextlib
import os
import sys
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import NoReturn

from filelock import FileLock, Timeout
from pydantic import BaseModel, ConfigDict

from gymrat.errors import GymratError
from gymrat.session.sidecar import read_sidecar
from gymrat.utils import warn_to_stderr

__all__ = [
    "LockContentionError",
    "LockHolder",
    "acquire_lock",
    "is_held",
    "now_iso",
    "read_holder",
]

type ReleaseLock = Callable[[], None]
"""Gives up an acquired lock. Calling it more than once is harmless."""

_LIVE_HOLDER_HINT = "Another gymrat run is active in this repo. Wait for it to finish."

_PUBLISH_LOCK_TIMEOUT: float = 2.0
"""Maximum seconds to wait for the publish lock before proceeding without it."""

_WORLD_WRITABLE_MODE = 0o666
"""Permissions applied to the holder record so any user can overwrite or remove it."""

_LOCK_ACQUIRE_TIMEOUT: float = 0.2
"""Seconds to wait for the main lock before declaring contention.

The wait rides out a transient hold, such as an :func:`is_held` probe, without
reporting a spurious contention error.
"""

_LOCK_ACQUIRE_POLL_INTERVAL: float = _LOCK_ACQUIRE_TIMEOUT / 2
"""Seconds between lock attempts, so a waiter gets a couple of checks within the
budget before contention is reported."""


class LockHolder(BaseModel):
    """The process a holder record names, as stamped by :func:`acquire_lock`.

    Attributes:
        pid: Process ID of the run that took the lock.
        command: Name of the command that took the lock.
        at: ISO-8601 timestamp of when the lock was taken.
    """

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    pid: int
    command: str
    at: str


def now_iso() -> str:
    """The current UTC time as ISO-8601 with millisecond precision and a ``Z`` suffix."""
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


class LockContentionError(GymratError):
    """Another run already holds the single-flight repository lock."""


def _os_lock_file(lock_path: str) -> str:
    return lock_path + ".lock"


def _publish_lock_file(lock_path: str) -> str:
    return _os_lock_file(lock_path) + ".publish"


def is_held(lock_path: str) -> bool:
    """Report whether another party holds the advisory lock at ``lock_path``.

    The probe acquires a non-blocking ``FileLock`` on the OS lock file
    (``<lock_path>.lock``) and releases immediately.  The probe never reads or
    writes the holder record and always preserves the lock file on disk.

    Args:
        lock_path: Path to the holder-record file whose sibling OS lock file
            is probed.

    Returns:
        ``True`` when acquisition fails because another party holds the lock;
        ``False`` when it succeeds or the lock file cannot be opened at all.
    """
    probe = FileLock(_os_lock_file(lock_path), timeout=0, preserve_lock_file=True)
    try:
        probe.acquire()
    except Timeout:
        return True
    except OSError:
        return False
    else:
        probe.release()
        return False


def read_holder(lock_path: str) -> LockHolder | None:
    """Report the holder record :func:`acquire_lock` stamped at ``lock_path``.

    Reporting only, never liveness: the record is read straight off disk without
    touching the OS lock or the publish lock, so a record left behind by an exited
    holder is still returned. Pair it with :func:`is_held` to tell a live holder
    from a stale record.

    Args:
        lock_path: Path to the holder-record file to read.

    Returns:
        The recorded holder, or ``None`` when the file is absent, empty,
        truncated, or otherwise not a holder record.
    """
    return read_sidecar(Path(lock_path), LockHolder)


def _acquire_publish_lock(pub_lock_path: str) -> FileLock:
    """Best-effort acquire of the publish lock.

    A timeout is not an error — the caller proceeds without the publish lock in
    that case, and reads ``is_locked`` off the returned lock to know whether it
    has one to release.

    Args:
        pub_lock_path: Path to the publish lock file to acquire.

    Returns:
        The publish lock, acquired when the wait succeeded.

    Raises:
        GymratError: When the publish lock file cannot be opened due to
            permissions.
    """
    pub_lock = FileLock(pub_lock_path, timeout=_PUBLISH_LOCK_TIMEOUT, preserve_lock_file=True)
    try:
        pub_lock.acquire()
    except Timeout:
        pass
    except PermissionError as error:
        _raise_permission_error(pub_lock_path, error)
    return pub_lock


def _acquire_os_lock(
    lock_path: str, os_lock_path: str, *, wait: float, poll_interval: float
) -> FileLock:
    """Acquire the main OS lock within ``wait`` seconds, or raise a diagnostic error.

    Args:
        lock_path: Path to the holder-record file, read for diagnostics when
            the wait runs out.
        os_lock_path: Path to the sibling OS lock file to acquire.
        wait: Seconds to wait for a rival to let go before declaring contention.
        poll_interval: Seconds between lock attempts during the wait.

    Returns:
        The acquired ``FileLock``.

    Raises:
        LockContentionError: When another holder keeps the lock for the whole wait.
        GymratError: When the OS lock file cannot be opened due to permissions.
    """
    lock = FileLock(
        os_lock_path,
        timeout=wait,
        poll_interval=poll_interval,
        preserve_lock_file=True,
    )
    try:
        lock.acquire()
    except Timeout:
        _raise_contention_error(lock_path)
    except PermissionError as error:
        _raise_permission_error(os_lock_path, error)
    return lock


def acquire_lock(
    lock_path: str,
    command: str,
    *,
    wait: float = _LOCK_ACQUIRE_TIMEOUT,
    poll_interval: float = _LOCK_ACQUIRE_POLL_INTERVAL,
) -> ReleaseLock:
    """Take the single-flight lock at ``lock_path`` on behalf of ``command``.

    Args:
        lock_path: Path to the holder-record file whose sibling OS lock file
            is acquired.
        command: Name of the command taking the lock, written into the holder
            record for diagnostics when a rival process finds it.
        wait: Seconds to wait for a rival to let go before declaring contention.
        poll_interval: Seconds between lock attempts during the wait.

    Returns:
        An idempotent zero-argument callable that releases the lock.

    Raises:
        LockContentionError: When another process (or the same process) already
            holds the lock.
        GymratError: When the lock file or its sibling publish lock file cannot
            be opened due to permissions.
    """
    Path(lock_path).parent.mkdir(parents=True, exist_ok=True)

    record = LockHolder(pid=os.getpid(), command=command, at=now_iso()).model_dump_json()

    os_lock_path = _os_lock_file(lock_path)
    pub_lock_path = _publish_lock_file(lock_path)

    pub_lock = _acquire_publish_lock(pub_lock_path)
    try:
        lock = _acquire_os_lock(lock_path, os_lock_path, wait=wait, poll_interval=poll_interval)

        holder = Path(lock_path)
        holder.write_text(record, encoding="utf-8")
        # Best-effort: some filesystems (e.g. FAT) ignore chmod entirely, and the
        # lock directory is typically per-user anyway, so a failed chmod here is not
        # actionable and must not block lock acquisition.
        with contextlib.suppress(OSError):
            holder.chmod(_WORLD_WRITABLE_MODE)
    finally:
        if pub_lock.is_locked:
            pub_lock.release()

    def release() -> None:
        try:
            lock.release()
        except Exception as error:  # noqa: BLE001 — intentional catch-all: release must never raise
            warn_to_stderr(f"Warning: failed to release lock at {lock_path}: {error!s}")

    return release


def _raise_contention_error(lock_path: str) -> NoReturn:
    """Read the holder record from a contended lock file and raise a diagnostic error.

    When the record is readable, the message names the holder's PID, command, and
    start time. When the content is empty, truncated, or not valid JSON, a generic
    "held by another process" message is used. In both cases the hint directs the
    caller to wait — never to remove the file, because the OS lock proves a holder
    is live.

    Args:
        lock_path: Path to the contended holder-record file to read.

    Raises:
        LockContentionError: Always — with holder details when available.
    """
    holder = read_holder(lock_path)
    message = (
        f"Lock at {lock_path} is held by another process."
        if holder is None
        else f"Lock held by PID {holder.pid} ({holder.command}, started {holder.at})"
    )

    raise LockContentionError(message, hint=_LIVE_HOLDER_HINT)


def _raise_permission_error(os_lock_path: str, error: OSError) -> NoReturn:
    """Reframe a permission failure into a ``GymratError`` with platform-gated hints."""
    if sys.platform == "win32":
        hint = (
            f"The file may be locked by another process. "
            f"Close any program using {os_lock_path}, then rerun."
        )
    else:
        hint = f"It belongs to another user. Remove {os_lock_path} yourself, then rerun."

    message = f"Lock file {os_lock_path} could not be opened: {error!s}"
    raise GymratError(message, hint=hint) from error
