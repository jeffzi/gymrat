"""On-disk lockfile layer: reading, publishing, and identity tracking."""

import json
import os
import sys
from dataclasses import dataclass
from typing import NoReturn

from gymrat.errors import GymratError

MAX_PID = 2_147_483_647

_LIVE_HOLDER_HINT = "Another gymrat run is active in this repo. Wait for it to finish."


@dataclass(frozen=True, slots=True)
class LockHolder:
    """The process a lockfile records as its holder."""

    pid: int
    command: str
    at: str


@dataclass(frozen=True, slots=True)
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
        os.unlink(path)  # noqa: PTH108 -- low-level os call for atomicity guarantees pathlib cannot provide
    except FileNotFoundError:
        return


def _try_remove_directory(path: str) -> bool:
    """Remove ``path`` if it is an empty directory. Returns whether it was removed."""
    try:
        os.rmdir(path)  # noqa: PTH106 -- low-level os call for consistency with the lock module's atomicity seams
    except (NotADirectoryError, FileNotFoundError, OSError):
        return False
    return True


@dataclass(frozen=True, slots=True)
class _Absent:
    """No file stands at the lock path."""


@dataclass(frozen=True, slots=True)
class _Held:
    """A parseable holder record and the file it was read from."""

    holder: LockHolder
    identity: LockIdentity


@dataclass(frozen=True, slots=True)
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
        message = f"Lock file {lock_path} could not be read: {error!s}"
        hint = f"It belongs to another user. Remove {lock_path} yourself, then rerun."
        raise GymratError(message, hint=hint) from error

    try:
        identity = _identity_of(fd)
        try:
            contents = _read_all(fd)
        except (UnicodeDecodeError, IsADirectoryError):
            return _Unreadable(identity)
    finally:
        os.close(fd)

    try:
        parsed = json.loads(contents)
    except ValueError:
        return _Unreadable(identity)
    holder = _parse_holder(parsed)
    return _Held(holder, identity) if holder is not None else _Unreadable(identity)


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
        info = os.stat(lock_path)  # noqa: PTH116 -- low-level os call for atomicity guarantees pathlib cannot provide
    except FileNotFoundError:
        return False
    return LockIdentity(dev=info.st_dev, ino=info.st_ino) == identity


def held_by_error(holder: LockHolder) -> GymratError:
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
        rethrow_displacement_failure(lock_path, error)
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


def rethrow_displacement_failure(lock_path: str, error: OSError) -> NoReturn:
    """Reframe a permission failure to displace a stale lockfile, else re-raise.

    On POSIX, ``EPERM`` and ``EACCES`` say the file belongs to another user — a
    sticky ``/tmp`` — which no retry can resolve, so they are framed for whoever
    has to clean up. On Windows a ``PermissionError`` is a sharing violation
    (the file is locked by another process), not an ownership signal.
    """
    if isinstance(error, PermissionError):
        message = f"Stale lock file {lock_path} could not be removed: {error!s}"
        if sys.platform == "win32":
            hint = (
                f"The file may be locked by another process. "
                f"Close any program using {lock_path}, then rerun."
            )
        else:
            hint = f"It belongs to another user. Remove {lock_path} yourself, then rerun."
        raise GymratError(message, hint=hint) from error
    raise error
