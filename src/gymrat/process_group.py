"""Tree-kill a spawned child's whole process group, cross-platform.

A child spawned into its own session or process group (POSIX
``start_new_session=True``) can leave grandchildren running when it is torn
down. :func:`kill_process_group` signals the entire group so no descendant
leaks, and never raises into the caller: a group that is already gone is
silent, and any other failure surfaces as a :class:`RuntimeWarning` — unless
the caller opts into ``defer_refusal``, in which case a POSIX ``EPERM``
refusal is returned silently instead, and the caller must signal again after
reaping the leader.

On Windows, ``taskkill /T /F`` walks the parent-child tree rather than
signaling a process group.
"""

import errno
import os
import signal
import subprocess
import sys
import warnings

_TASKKILL_GONE = 128
"""``taskkill`` exit status meaning the process was already gone."""


def _warn(message: str) -> None:
    """Raise a :class:`RuntimeWarning` attributed to :func:`kill_process_group`'s caller."""
    warnings.warn(message, RuntimeWarning, stacklevel=3)


def kill_process_group(pid: int, *, defer_refusal: bool = False) -> bool:
    """Kill the whole process group led by ``pid``, never raising into the caller.

    On POSIX this signals the group with ``SIGKILL``, staying silent when the
    group is already gone and warning on any other failure. On Windows it
    delegates to ``taskkill /T /F``, staying silent when the process is already
    gone and warning on any other failure.

    macOS refuses the signal with ``EPERM`` while every member of the group is
    still exiting or is a zombie not yet reaped, which cannot be told apart from a
    genuine permission failure at that moment. Once the leader is reaped the
    same signal settles it: the group is gone (silent), a live descendant is
    killed, or the refusal repeats and is genuine (warned).

    Args:
        pid: The process ID leading the group to kill.
        defer_refusal: Return instead of warning when the POSIX group refuses
            the signal with ``EPERM``. The caller must then reap the leader and
            call again without this flag so a genuine failure still warns.

    Returns:
        ``True`` when ``defer_refusal`` held back an ``EPERM`` refusal, ``False``
        otherwise.
    """
    if sys.platform == "win32":
        try:
            argv = ["taskkill", "/F", "/T", "/PID", str(pid)]
            subprocess.run(argv, capture_output=True, check=True)  # noqa: S603 -- argv is a fixed list, not shell-injected
        except subprocess.CalledProcessError as error:
            if error.returncode != _TASKKILL_GONE:
                _warn(f"taskkill failed for pid {pid}: {error}")
        except OSError as os_error:
            _warn(f"taskkill unavailable while killing pid {pid}: {os_error}")
        return False
    try:
        os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    except OSError as error:
        if defer_refusal and error.errno == errno.EPERM:
            return True
        _warn(f"killpg failed for pid {pid}: {error}")
    return False
