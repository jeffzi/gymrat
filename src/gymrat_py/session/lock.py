"""Single-flight repository lock built on the atomicity of ``os.link``.

A run publishes its holder record by writing it to a scratch file and hard-linking
that file into place: the link is exclusive — it fails with ``EEXIST`` when the
path is taken — so exactly one racer wins. A lockfile no live process holds is
stolen silently through a claim link derived from the file's device/inode, so a
crashed run never needs manual cleanup. The one state no later run can clear on
its own — a takeover killed between claiming and completing a displacement — is
reported with both files to delete.
"""

import errno
import json
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import NoReturn

from gymrat_py.errors import GymratError, message_of
from gymrat_py.session.clock import now_iso

# Largest pid a liveness probe can be asked about: signalling rejects anything
# past int32, and 0/negative address process groups rather than a process.
MAX_PID = 2_147_483_647

# How many times acquisition re-reads a lockfile it lost a race for.
MAX_ACQUIRE_ATTEMPTS = 3

# The remedy for a lock whose holder answered a liveness probe. Deleting the
# lockfile is deliberately not offered: the holder is provably alive.
_LIVE_HOLDER_HINT = "Another gymrat run is active in this repo. Wait for it to finish."


type ReleaseLock = Callable[[], None]
"""Gives up an acquired lock. Calling it more than once is harmless.

Only the lockfile this run published is removed. A lock taken over since belongs
to whoever holds it now, and is left where it stands.
"""


@dataclass(frozen=True)
class LockHolder:
    """The process a lockfile records as its holder."""

    pid: int
    command: str
    at: str


@dataclass(frozen=True)
class LockIdentity:
    """Which file a lockfile read came from, as the filesystem identifies it.

    A path says nothing about which file it names from one moment to the next, so
    a steal that only knows the lock path cannot tell the record it judged stale
    from whatever a rival published there since. Device and inode do tell them
    apart.
    """

    dev: int
    ino: int


def _parse_holder(value: object) -> LockHolder | None:
    """Return the holder ``value`` describes, or ``None`` when it is debris.

    A pid outside ``[1, MAX_PID]`` — a fraction, 0, a negative, or a number past
    int32 — names no process any signal could reach, so the record is debris.
    Unknown extra keys are tolerated: a lockfile carrying a field this version
    does not know is still a live holder, and rejecting it would steal the lock
    out from under whoever wrote it.
    """
    if not isinstance(value, dict):
        return None
    pid = value.get("pid")
    command = value.get("command")
    at = value.get("at")
    if isinstance(pid, bool) or not isinstance(pid, int) or pid < 1 or pid > MAX_PID:
        return None
    if not isinstance(command, str) or not isinstance(at, str):
        return None
    return LockHolder(pid=pid, command=command, at=at)


def _identity_of(fd: int) -> LockIdentity:
    """Which file an open descriptor refers to, whatever its path names by now."""
    info = os.fstat(fd)
    return LockIdentity(dev=info.st_dev, ino=info.st_ino)


def _read_all(fd: int) -> str:
    """Read the whole contents of ``fd`` as UTF-8 text."""
    chunks: list[bytes] = []
    while chunk := os.read(fd, 65536):
        chunks.append(chunk)
    return b"".join(chunks).decode("utf-8")


def _write_all(fd: int, data: bytes) -> None:
    """Write every byte of ``data`` to ``fd``."""
    view = memoryview(data)
    while view:
        view = view[os.write(fd, view) :]


def _force_unlink(path: str) -> None:
    """Remove ``path``, treating an already-absent file as success."""
    try:
        os.unlink(path)  # noqa: PTH108
    except FileNotFoundError:
        return


# What a lockfile says at the moment it was read.
@dataclass(frozen=True)
class _Absent:
    """No file stands at the lock path."""


@dataclass(frozen=True)
class _Held:
    """A parseable holder record and the file it was read from."""

    holder: LockHolder
    identity: LockIdentity


@dataclass(frozen=True)
class _Unreadable:
    """Debris at the lock path — unparseable or a foreign shape — and its file."""

    identity: LockIdentity


type _LockfileState = _Absent | _Held | _Unreadable


def read_lockfile(lock_path: str) -> _LockfileState:
    """Read what the lockfile at ``lock_path`` currently says, and which file said it.

    The identity is taken from the open descriptor the contents are read through,
    so the two describe one file even when the lock path is taken over mid-read.
    Publication is atomic, so no reader ever catches a holder mid-write: an
    unreadable file is debris — a run killed before publishing, or a foreign file
    at the lock path — not a lock somebody is in the middle of taking.

    Raises:
        GymratError: When the lockfile belongs to another user.
    """
    try:
        fd = os.open(lock_path, os.O_RDONLY)
    except FileNotFoundError:
        return _Absent()
    except PermissionError as error:
        # A lockfile another user owns is unreadable to every later run, so the
        # steal path can never reach it — the only way out is by hand.
        message = f"Lock file {lock_path} could not be read: {message_of(error)}"
        hint = f"It belongs to another user. Remove {lock_path} yourself, then rerun."
        raise GymratError(message, hint=hint) from error

    try:
        identity = _identity_of(fd)
        contents = _read_all(fd)
    finally:
        os.close(fd)

    try:
        parsed = json.loads(contents)
    except ValueError:
        return _Unreadable(identity)
    holder = _parse_holder(parsed)
    return _Held(holder, identity) if holder is not None else _Unreadable(identity)


def is_alive(pid: int) -> bool:
    """Whether a process with ``pid`` still exists.

    Signal ``0`` runs the kernel's permission and existence checks without
    delivering anything. Only ``ESRCH`` means no such process: ``EPERM`` says the
    process is there but owned by another user, which is still a live holder.
    """
    try:
        os.kill(pid, 0)
    except OSError as error:
        return error.errno != errno.ESRCH
    return True


def unlink_if_same_file(lock_path: str, identity: LockIdentity) -> None:
    """Delete ``lock_path`` only while it still names the file ``identity`` came from.

    A path is not a lock: the file a run published can be displaced by a takeover
    at any moment, and deleting whatever answers to the path would hand the next
    holder's lock away to nobody. A different file there — or none at all — is
    left exactly as found.
    """
    try:
        fd = os.open(lock_path, os.O_RDONLY)
    except FileNotFoundError:
        return

    try:
        current = _identity_of(fd)
    finally:
        os.close(fd)

    if current == identity:
        _force_unlink(lock_path)


def still_names_file(lock_path: str, identity: LockIdentity) -> bool:
    """Whether ``lock_path`` still names the file ``identity`` came from.

    The answer is a snapshot, true only for the instant it was taken — enough to
    word a remedy about the file standing at the path, never enough to decide a
    write against it. :func:`unlink_if_same_file` is the safe form for that.
    """
    try:
        info = os.stat(lock_path)  # noqa: PTH116
    except FileNotFoundError:
        return False
    return LockIdentity(dev=info.st_dev, ino=info.st_ino) == identity


def _held_by_error(holder: LockHolder) -> GymratError:
    """Report a lock whose holder answered a liveness probe."""
    message = f"Lock held by PID {holder.pid} ({holder.command}, started {holder.at})"
    return GymratError(message, hint=_LIVE_HOLDER_HINT)


def publish_lock_record(lock_path: str, record: str) -> LockIdentity | None:
    """Publish ``record`` at ``lock_path``, or ``None`` when someone got there first.

    The record is written to a scratch file beside the lock and only then linked
    into place, so the lock path never exists holding half a record: readers see a
    whole holder or nothing at all. The link is also the exclusive step — it fails
    with ``EEXIST`` when the path is taken — so exactly one racer publishes.

    The identity is taken from the descriptor the record was written through,
    which is the same file the link names, so it stays true no matter what takes
    the lock path over afterwards.
    """
    scratch_path = f"{lock_path}.{os.getpid()}.record"
    try:
        fd = os.open(scratch_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    except OSError as error:
        # A leftover another user owns at the scratch path — a sticky ``/tmp`` —
        # is reframed like every other auxiliary displacement wall, not thrown raw.
        _rethrow_displacement_failure(lock_path, error)
    try:
        _write_all(fd, record.encode("utf-8"))
        identity = _identity_of(fd)
    finally:
        os.close(fd)

    try:
        os.link(scratch_path, lock_path)
    except FileExistsError:
        return None
    else:
        return identity
    finally:
        _force_unlink(scratch_path)


def _rethrow_displacement_failure(lock_path: str, error: OSError) -> NoReturn:
    """Reframe a permission failure to displace a stale lockfile, else re-raise.

    ``EPERM`` and ``EACCES`` say the file belongs to another user — a sticky
    ``/tmp`` — which no retry can resolve, so they are framed for whoever has to
    clean up.
    """
    if isinstance(error, PermissionError):
        message = f"Stale lock file {lock_path} could not be removed: {message_of(error)}"
        hint = f"It belongs to another user. Remove {lock_path} yourself, then rerun."
        raise GymratError(message, hint=hint) from error
    raise error


# What asking for the right to displace a stale lockfile turned up. ``_ClaimGone``
# covers both ways the judged file can stop being the one at the lock path: it
# vanished before the claim, or the claim came back naming a different file.
@dataclass(frozen=True)
class _ClaimClaimed:
    """The judged file was claimed for displacement; its claim lives here."""

    claim_path: str


@dataclass(frozen=True)
class _ClaimBlocked:
    """The claim name is already taken; the blocking claim lives here."""

    claim_path: str


@dataclass(frozen=True)
class _ClaimGone:
    """The judged file is no longer the one at the lock path."""


type _ClaimOutcome = _ClaimClaimed | _ClaimBlocked | _ClaimGone


def claim_stale_lock(lock_path: str, identity: LockIdentity) -> _ClaimOutcome:
    """Claim the right to displace the file ``identity`` was read from.

    The claim is a second name for the lockfile itself, spelled out of that file's
    identity, which makes it both halves of a safe steal. It is exclusive: two
    racers reaching the same staleness verdict about one file ask for the same
    name, ``EEXIST`` tells the loser so. And it is proof: a racer whose claim
    turns out to name a different file is holding a lock somebody published after
    the verdict, so it drops the claim and leaves that lock where it stands.

    Raises:
        GymratError: When the lockfile belongs to another user.
    """
    claim_path = f"{lock_path}.{identity.dev}-{identity.ino}.claim"
    try:
        os.link(lock_path, claim_path)
    except FileExistsError:
        return _ClaimBlocked(claim_path)
    except FileNotFoundError:
        return _ClaimGone()
    except OSError as error:
        _rethrow_displacement_failure(lock_path, error)

    info = os.stat(claim_path)  # noqa: PTH116
    if LockIdentity(dev=info.st_dev, ino=info.st_ino) == identity:
        return _ClaimClaimed(claim_path)
    _force_unlink(claim_path)
    return _ClaimGone()


def displace_stale_lock(lock_path: str) -> bool:
    """Clear the claimed lockfile off the lock path. Returns whether it went away.

    Only the holder of the claim gets here, and nothing else may displace a
    claimed file, so the file moved is the one that was judged stale. It is moved
    rather than deleted so a run killed mid-steal leaves the lock path free rather
    than holding a record it never published.

    Raises:
        GymratError: When the lockfile belongs to another user.
    """
    aside_path = f"{lock_path}.{os.getpid()}.stale"
    try:
        # Raw os.rename is the atomic displacement seam the retry loop races on.
        os.rename(lock_path, aside_path)  # noqa: PTH104
    except FileNotFoundError:
        return False
    except OSError as error:
        _rethrow_displacement_failure(lock_path, error)
    _force_unlink(aside_path)
    return True


@dataclass(frozen=True)
class _WedgedTakeover:
    """A takeover that died holding its claim, leaving the lock impossible to steal.

    The claim outlives the run that made it, so every later steal of that same
    file is refused the claim name forever. ``holder_pid`` is the dead process the
    lockfile named, or ``None`` when the lockfile was too damaged to name one.
    """

    identity: LockIdentity
    claim_path: str
    holder_pid: int | None


# What one pass at the lock path turned up.
@dataclass(frozen=True)
class _Acquired:
    """The lock was published; the release handle keys on this identity."""

    identity: LockIdentity


@dataclass(frozen=True)
class _Retry:
    """The lock path shifted under the attempt; read it afresh."""


@dataclass(frozen=True)
class _Blocked:
    """A claim stood in the way of a steal."""

    wedge: _WedgedTakeover


type _AttemptOutcome = _Acquired | _Retry | _Blocked


def steal_lock(
    lock_path: str,
    identity: LockIdentity,
    record: str,
    holder_pid: int | None,
) -> _AttemptOutcome:
    """Take over a lockfile no live process holds."""
    claim = claim_stale_lock(lock_path, identity)
    if isinstance(claim, _ClaimGone):
        return _Retry()
    if isinstance(claim, _ClaimBlocked):
        return _Blocked(_WedgedTakeover(identity, claim.claim_path, holder_pid))
    try:
        published = (
            publish_lock_record(lock_path, record) if displace_stale_lock(lock_path) else None
        )
        return _Acquired(published) if published is not None else _Retry()
    finally:
        _force_unlink(claim.claim_path)


def _wedged_takeover_error(lock_path: str, wedge: _WedgedTakeover) -> GymratError:
    """Word the remedy for a lock a dead takeover left impossible to steal."""
    if wedge.holder_pid is None:
        nobody_holds = "No gymrat process holds this lock."
    else:
        nobody_holds = f"No gymrat process holds this lock (PID {wedge.holder_pid} is dead)."
    message = f"Lock at {lock_path} was left behind by a run that died while taking it over."
    hint = f"{nobody_holds} To unblock, delete {lock_path} and {wedge.claim_path}, then rerun."
    return GymratError(message, hint=hint)


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
        return steal_lock(lock_path, state.identity, record, None)
    if is_alive(state.holder.pid):
        raise _held_by_error(state.holder)
    return steal_lock(lock_path, state.identity, record, state.holder.pid)


def _make_release(lock_path: str, identity: LockIdentity) -> ReleaseLock:
    """Build a release handle that removes only the file this run published."""

    def release() -> None:
        try:
            unlink_if_same_file(lock_path, identity)
        except OSError as error:
            text = f"Warning: failed to release lock at {lock_path}: {message_of(error)}\n"
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

    wedge: _WedgedTakeover | None = None
    wedged_every_attempt = True

    for _ in range(MAX_ACQUIRE_ATTEMPTS):
        outcome = attempt_acquire(lock_path, record)
        if isinstance(outcome, _Acquired):
            return _make_release(lock_path, outcome.identity)

        # Only one file blocking every single attempt rules out the rival that
        # takes the lock, works, and releases it between two of our reads.
        blocked = outcome.wedge if isinstance(outcome, _Blocked) else None
        if blocked is None:
            wedged_every_attempt = False
        elif wedge is None:
            wedge = blocked
        elif wedge.identity != blocked.identity:
            wedged_every_attempt = False

    # The wedged remedy tells the caller to delete the lockfile, so it may only be
    # given while the wedged file is the one still standing at the lock path. A
    # lock that moved on belongs to whoever published it — a run that may well be
    # alive — and contention is what this run really met.
    if wedged_every_attempt and wedge is not None and still_names_file(lock_path, wedge.identity):
        raise _wedged_takeover_error(lock_path, wedge)

    message = f"Lock at {lock_path} was claimed by another process on every attempt."
    hint = f"{_LIVE_HOLDER_HINT} If no gymrat process is running, delete {lock_path}."
    raise GymratError(message, hint=hint)
