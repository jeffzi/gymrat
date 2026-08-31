"""Single-flight repository lock via OS advisory locks (``filelock.FileLock``).

A run takes the lock by acquiring a non-blocking ``flock``/``LockFileEx`` on a
dedicated ``.lock`` file next to the holder metadata path.  The holder record is
written to the original path as plain JSON, readable on every platform — including
Windows, where ``LockFileEx`` creates a mandatory byte-range lock that blocks
reads through a separate handle.

Contention is instant: the loser reads the winner's holder record for diagnostics
without needing liveness probes.  Crash recovery is automatic — the kernel
releases the advisory lock when the holder exits — so a stale lockfile never
needs manual cleanup.
"""

import contextlib
import json
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import NoReturn

from filelock import FileLock, Timeout

from gymrat.errors import GymratError
from gymrat.session.clock import now_iso

__all__ = ["acquire_lock"]

type ReleaseLock = Callable[[], None]
"""Gives up an acquired lock. Calling it more than once is harmless."""

_LIVE_HOLDER_HINT = "Another gymrat run is active in this repo. Wait for it to finish."


def _os_lock_file(lock_path: str) -> str:
    """Derive the OS lock file path from the caller-visible metadata path."""
    return lock_path + ".lock"


def acquire_lock(lock_path: str, command: str) -> ReleaseLock:
    """Take the single-flight lock at ``lock_path`` on behalf of ``command``.

    Returns a zero-argument callable that releases the lock. The release is
    idempotent: calling it more than once is harmless.

    Raises:
        GymratError: When another process (or the same process) already holds
            the lock, or when the lock file cannot be opened due to permissions.
    """
    Path(lock_path).parent.mkdir(parents=True, exist_ok=True)

    pid = os.getpid()
    record = json.dumps({"pid": pid, "command": command, "at": now_iso()})

    os_lock = _os_lock_file(lock_path)
    lock = FileLock(os_lock, timeout=0, preserve_lock_file=True)

    try:
        lock.acquire()
    except Timeout:
        _raise_contention_error(lock_path)
    except PermissionError as error:
        _raise_permission_error(os_lock, error)

    holder = Path(lock_path)
    holder.write_text(record, encoding="utf-8")
    with contextlib.suppress(OSError):
        holder.chmod(0o666)

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
