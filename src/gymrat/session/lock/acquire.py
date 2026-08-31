"""Single-flight repository lock built on the atomicity of ``os.link``.

A run publishes its holder record by writing it to a scratch file and hard-linking
that file into place: the link is exclusive — it fails with ``EEXIST`` when the
path is taken — so exactly one racer wins. A lockfile no live process holds is
stolen silently through a claim link derived from the file's device/inode, so a
crashed run never needs manual cleanup. The one state no later run can clear on
its own — a takeover killed between claiming and completing a displacement — is
reported with both files to delete.
"""

import json
import os
import sys
from collections.abc import Callable
from pathlib import Path

from gymrat.errors import GymratError
from gymrat.session.clock import now_iso
from gymrat.session.lock.lockfile import (
    _LIVE_HOLDER_HINT,
    LockIdentity,
    _Absent,
    _try_remove_directory,
    _Unreadable,
    held_by_error,
    publish_lock_record,
    read_lockfile,
    still_names_file,
    unlink_if_same_file,
)
from gymrat.session.lock.steal import (
    _Acquired,
    _AttemptOutcome,
    _Blocked,
    _Retry,
    is_alive,
    steal_lock,
    wedged_takeover_error,
)

MAX_ACQUIRE_ATTEMPTS = 3


type ReleaseLock = Callable[[], None]
"""Gives up an acquired lock. Calling it more than once is harmless.

Only the lockfile this run published is removed. A lock taken over since belongs
to whoever holds it now, and is left where it stands.
"""


def attempt_acquire(lock_path: str, record: str) -> _AttemptOutcome:
    """Make one bid for the lock at ``lock_path``: publish, or judge what is there.

    Raises:
        GymratError: When the lock is held by a process that is still running, or
            when the lockfile belongs to another user.
    """
    published = publish_lock_record(lock_path, record)
    if published is not None:
        return _Acquired(published)

    state = read_lockfile(lock_path)
    if isinstance(state, _Absent):
        return _Retry()
    if isinstance(state, _Unreadable):
        if _try_remove_directory(lock_path):
            return _Retry()
        return steal_lock(lock_path, state.identity, record, None)
    if is_alive(state.holder.pid):
        raise held_by_error(state.holder)
    return steal_lock(lock_path, state.identity, record, state.holder.pid)


def _make_release(lock_path: str, identity: LockIdentity) -> ReleaseLock:
    """Build a release handle that removes only the file this run published."""

    def release() -> None:
        try:
            unlink_if_same_file(lock_path, identity)
        except OSError as error:
            text = f"Warning: failed to release lock at {lock_path}: {error!s}\n"
            sys.stderr.write(text)

    return release


def acquire_lock(lock_path: str, command: str) -> ReleaseLock:
    """Take the single-flight lock at ``lock_path`` on behalf of ``command``.

    The lockfile is published exclusively, so two processes racing for it cannot
    both win. A lockfile no live process holds is stolen silently — a crashed run
    must not need manual cleanup, whether it left a holder record behind or a file
    too damaged to read — and losing that steal re-enters acquisition, where the
    winner is either a live holder to report or a lock released again in the
    meantime.

    One exception: a run killed between claiming the right to displace a stale
    lockfile and completing that displacement leaves a state no later run can
    clear on its own. The thrown error names both files to delete.

    Raises:
        GymratError: When the lock is held by a process that is still running,
            when the lockfile belongs to another user, or when every attempt was
            refused by the claim of a takeover that never finished.
    """
    pid = os.getpid()
    record = json.dumps({"pid": pid, "command": command, "at": now_iso()})

    Path(lock_path).parent.mkdir(parents=True, exist_ok=True)

    wedge = None
    wedged_every_attempt = True

    for _ in range(MAX_ACQUIRE_ATTEMPTS):
        outcome = attempt_acquire(lock_path, record)
        if isinstance(outcome, _Acquired):
            return _make_release(lock_path, outcome.identity)

        blocked = outcome.wedge if isinstance(outcome, _Blocked) else None
        if blocked is None:
            wedged_every_attempt = False
        elif wedge is None:
            wedge = blocked
        elif wedge.identity != blocked.identity:
            wedged_every_attempt = False

    if wedged_every_attempt and wedge is not None and still_names_file(lock_path, wedge.identity):
        raise wedged_takeover_error(lock_path, wedge)

    message = f"Lock at {lock_path} was claimed by another process on every attempt."
    hint = f"{_LIVE_HOLDER_HINT} If no gymrat process is running, delete {lock_path}."
    raise GymratError(message, hint=hint)
