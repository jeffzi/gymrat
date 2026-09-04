"""Single-flight repository lock via OS advisory locks (``filelock.FileLock``).

A run takes the lock by acquiring a non-blocking ``flock``/``LockFileEx`` on a
dedicated ``.lock`` file next to the holder metadata path.  The holder record is
written to the original path as plain JSON, readable on every platform — including
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

Contention is instant: the loser reads the winner's holder record for diagnostics
without needing liveness probes.  Crash recovery is automatic — the kernel
releases the advisory lock when the holder exits — so a stale lockfile never
needs manual cleanup.
"""

import contextlib
import json
import os
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import NoReturn

from filelock import FileLock, Timeout

from gymrat.errors import GymratError
from gymrat.session.clock import now_iso

__all__ = ["acquire_lock", "is_held"]

type ReleaseLock = Callable[[], None]
"""Gives up an acquired lock. Calling it more than once is harmless."""

_LIVE_HOLDER_HINT = "Another gymrat run is active in this repo. Wait for it to finish."

_PUBLISH_LOCK_TIMEOUT: float = 2.0
"""Maximum seconds to wait for the publish lock before proceeding without it."""

_WORLD_WRITABLE_MODE = 0o666
"""Permissions applied to the holder record so any user can overwrite or remove it."""

LOCK_ACQUIRE_POLL_MS: int = 100
"""Milliseconds between lock-acquisition retries."""

LOCK_ACQUIRE_RETRIES: int = 3
"""Total attempts before declaring contention."""


def _os_lock_file(lock_path: str) -> str:
    return lock_path + ".lock"


def _publish_lock_file(lock_path: str) -> str:
    return _os_lock_file(lock_path) + ".publish"


def _non_blocking_lock(os_lock_path: str) -> FileLock:
    """A ``FileLock`` that fails immediately on contention instead of blocking."""
    return FileLock(os_lock_path, timeout=0, preserve_lock_file=True)


def is_held(lock_path: Path) -> bool:
    """Report whether another party holds the advisory lock at ``lock_path``.

    The probe acquires a non-blocking ``FileLock`` on the OS lock file
    (``<lock_path>.lock``) and releases immediately.  If acquisition fails the
    lock is held; if it succeeds or the file cannot be opened at all the lock
    is not held.  The probe never reads or writes the holder record and always
    preserves the lock file on disk.
    """
    os_lock_path = _os_lock_file(str(lock_path))
    probe = _non_blocking_lock(os_lock_path)
    try:
        probe.acquire()
    except Timeout:
        return True
    except OSError:
        return False
    else:
        probe.release()
        return False


def _acquire_publish_lock(pub_lock_path: str) -> tuple[FileLock, bool]:
    """Best-effort acquire of the publish lock.

    Returns the lock object and whether acquisition succeeded. A timeout is not
    an error — the caller proceeds without the publish lock in that case.
    """
    pub_lock = FileLock(pub_lock_path, timeout=_PUBLISH_LOCK_TIMEOUT, preserve_lock_file=True)
    try:
        pub_lock.acquire()
    except Timeout:
        return pub_lock, False
    except PermissionError as error:
        _raise_permission_error(pub_lock_path, error)
    else:
        return pub_lock, True


def _acquire_os_lock(lock_path: str, os_lock_path: str) -> FileLock:
    """Acquire the main non-blocking OS lock, or raise a diagnostic error.

    Retries up to ``LOCK_ACQUIRE_RETRIES`` times with ``LOCK_ACQUIRE_POLL_MS``
    between attempts so a transient hold (e.g. an ``is_held`` probe) does not
    cause a spurious contention error.
    """
    lock = _non_blocking_lock(os_lock_path)
    last_attempt = LOCK_ACQUIRE_RETRIES - 1
    for attempt in range(LOCK_ACQUIRE_RETRIES):
        try:
            lock.acquire()
        except Timeout:
            if attempt == last_attempt:
                _raise_contention_error(lock_path)
            time.sleep(LOCK_ACQUIRE_POLL_MS / 1000)
        except PermissionError as error:
            _raise_permission_error(os_lock_path, error)
        else:
            return lock
    _raise_contention_error(lock_path)
    return lock  # unreachable — _raise_contention_error always raises


def acquire_lock(lock_path: str, command: str) -> ReleaseLock:
    """Take the single-flight lock at ``lock_path`` on behalf of ``command``.

    Returns a zero-argument callable that releases the lock. The release is
    idempotent: calling it more than once is harmless.

    Raises:
        GymratError: When another process (or the same process) already holds
            the lock, or when the lock file or its sibling publish lock file
            cannot be opened due to permissions.
    """
    Path(lock_path).parent.mkdir(parents=True, exist_ok=True)

    record = json.dumps({"pid": os.getpid(), "command": command, "at": now_iso()})

    os_lock_path = _os_lock_file(lock_path)
    pub_lock_path = _publish_lock_file(lock_path)

    pub_lock, has_pub_lock = _acquire_publish_lock(pub_lock_path)
    try:
        lock = _acquire_os_lock(lock_path, os_lock_path)

        holder = Path(lock_path)
        holder.write_text(record, encoding="utf-8")
        # Best-effort: some filesystems (e.g. FAT) ignore chmod entirely, and the
        # lock directory is typically per-user anyway, so a failed chmod here is not
        # actionable and must not block lock acquisition.
        with contextlib.suppress(OSError):
            holder.chmod(_WORLD_WRITABLE_MODE)
    finally:
        if has_pub_lock:
            pub_lock.release()

    def release() -> None:
        try:
            lock.release()
        except Exception as error:  # noqa: BLE001 — intentional catch-all: release must never raise
            text = f"Warning: failed to release lock at {lock_path}: {error!s}\n"
            sys.stderr.write(text)

    return release


def _raise_contention_error(lock_path: str) -> NoReturn:
    """Read the holder record from a contended lock file and raise a diagnostic error.

    When the record is readable, the message names the holder's PID, command, and
    start time. When the content is empty, truncated, or not valid JSON, a generic
    "held by another process" message is used. In both cases the hint directs the
    caller to wait — never to remove the file, because the OS lock proves a holder
    is live.
    """
    try:
        content = Path(lock_path).read_text(encoding="utf-8")
        holder = json.loads(content)
        pid = holder["pid"]
        command = holder["command"]
        at = holder["at"]
        message = f"Lock held by PID {pid} ({command}, started {at})"
    except (OSError, ValueError, KeyError, UnicodeDecodeError):
        message = f"Lock at {lock_path} is held by another process."

    raise GymratError(message, hint=_LIVE_HOLDER_HINT)


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
