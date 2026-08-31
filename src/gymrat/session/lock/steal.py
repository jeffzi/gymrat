"""Steal state machine: claiming and displacing stale lock files.

Includes process liveness probes that the steal path uses to decide whether
the holder is dead.
"""

import errno
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass

from gymrat.errors import GymratError
from gymrat.session.lock.lockfile import (
    LockIdentity,
    _force_unlink,
    publish_lock_record,
    rethrow_displacement_failure,
)


@dataclass(frozen=True, slots=True)
class _ClaimClaimed:
    """The judged file was claimed for displacement; its claim lives here."""

    claim_path: str


@dataclass(frozen=True, slots=True)
class _ClaimBlocked:
    """The claim name is already taken; the blocking claim lives here."""

    claim_path: str


@dataclass(frozen=True, slots=True)
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
        rethrow_displacement_failure(lock_path, error)

    info = os.stat(claim_path)  # noqa: PTH116 -- low-level os call for atomicity guarantees pathlib cannot provide
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
        os.rename(lock_path, aside_path)  # noqa: PTH104 -- low-level os call for atomicity guarantees pathlib cannot provide
    except FileNotFoundError:
        return False
    except OSError as error:
        rethrow_displacement_failure(lock_path, error)
    _force_unlink(aside_path)
    return True


@dataclass(frozen=True, slots=True)
class _WedgedTakeover:
    """A takeover that died holding its claim, leaving the lock impossible to steal.

    The claim outlives the run that made it, so every later steal of that same
    file is refused the claim name forever. ``holder_pid`` is the dead process the
    lockfile named, or ``None`` when the lockfile was too damaged to name one.
    """

    identity: LockIdentity
    claim_path: str
    holder_pid: int | None


@dataclass(frozen=True, slots=True)
class _Acquired:
    """The lock was published; the release handle keys on this identity."""

    identity: LockIdentity


@dataclass(frozen=True, slots=True)
class _Retry:
    """The lock path shifted under the attempt; read it afresh."""


@dataclass(frozen=True, slots=True)
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


def wedged_takeover_error(lock_path: str, wedge: _WedgedTakeover) -> GymratError:
    """Word the remedy for a lock a dead takeover left impossible to steal."""
    if wedge.holder_pid is None:
        nobody_holds = "No gymrat process holds this lock."
    else:
        nobody_holds = f"No gymrat process holds this lock (PID {wedge.holder_pid} is dead)."
    message = f"Lock at {lock_path} was left behind by a run that died while taking it over."
    hint = f"{nobody_holds} To unblock, delete {lock_path} and {wedge.claim_path}, then rerun."
    return GymratError(message, hint=hint)


def _is_alive_posix(pid: int) -> bool:
    """Signal 0 checks existence without delivering anything."""
    try:
        os.kill(pid, 0)
    except OSError as error:
        return error.errno == errno.EPERM
    return True


def _is_alive_windows(pid: int) -> bool:
    """Open the process handle to check existence.

    ``os.kill(pid, 0)`` cannot be used on Windows: ``signal.CTRL_C_EVENT`` is 0,
    so CPython dispatches through ``GenerateConsoleCtrlEvent`` instead of
    ``OpenProcess``, broadcasting Ctrl+C to the target's console process group.

    When ``OpenProcess`` returns NULL, ``ctypes.get_last_error`` distinguishes
    the interesting cases:

    - **ERROR_ACCESS_DENIED (5)**: the process exists but the caller lacks the
      right to open it — alive, never eligible for a steal.
    - **ERROR_INVALID_PARAMETER (87)**: no process with that PID exists — dead,
      proceed with the steal.

    Any other error code is treated as alive (fail-safe) to avoid stealing from
    a holder whose liveness cannot be determined.

    ``use_last_error=True`` on ``WinDLL`` makes CPython snapshot ``GetLastError``
    immediately after the foreign call, before any intervening Python-level
    Windows API calls can clobber it.  ``ctypes.get_last_error`` retrieves the
    snapshot reliably.
    """
    import ctypes  # noqa: PLC0415 — Windows-only, deferred to avoid top-level import on POSIX

    invalid_parameter = 87

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[attr-defined] — WinDLL is Windows-only, unavailable on other platforms
    kernel32.OpenProcess.restype = ctypes.c_void_p
    kernel32.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
    kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
    handle = kernel32.OpenProcess(0x1000, 0, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
    if handle:
        kernel32.CloseHandle(handle)
        return True
    last_error = ctypes.get_last_error()  # pyrefly: ignore  # Windows-only, unavailable on other platforms
    return last_error != invalid_parameter


is_alive: Callable[[int], bool] = _is_alive_windows if sys.platform == "win32" else _is_alive_posix
