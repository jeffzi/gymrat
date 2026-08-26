"""Tree-kill a spawned child's whole process group, cross-platform.

A child spawned into its own session or process group (POSIX
``start_new_session=True``) can leave grandchildren running when it is torn
down. :func:`kill_process_group` signals the entire group so no descendant
leaks, and never raises into the caller: a group that is already gone is
silent, and any other failure surfaces as a :class:`RuntimeWarning`.

On Windows, ``taskkill /T /F`` walks the parent-child tree rather than
signaling a process group.
"""

import os
import signal
import subprocess
import sys
import warnings

_TASKKILL_GONE = 128
"""``taskkill`` exit status meaning the process was already gone."""


def current_platform() -> str:
    """Report the platform lazily instead of caching it at spawn.

    A caller (and its tests) can then redirect the kill strategy to the
    Windows path after a POSIX spawn.
    """
    return sys.platform


def kill_process_group(pid: int) -> None:
    """Kill the whole process group led by ``pid``, never raising into the caller.

    On POSIX this signals the group with ``SIGKILL``, staying silent when the
    group is already gone and warning on any other failure. On Windows it
    delegates to ``taskkill /T /F``, staying silent when the process is already
    gone and warning on any other failure.
    """
    if current_platform() == "win32":
        try:
            argv = ["taskkill", "/F", "/T", "/PID", str(pid)]
            subprocess.run(argv, capture_output=True, check=True)  # noqa: S603 -- argv is a fixed list, not shell-injected
        except subprocess.CalledProcessError as error:
            if error.returncode != _TASKKILL_GONE:
                warnings.warn(
                    f"taskkill failed for pid {pid}: {error}",
                    RuntimeWarning,
                    stacklevel=2,
                )
        except FileNotFoundError:
            warnings.warn(
                f"taskkill unavailable while killing pid {pid}",
                RuntimeWarning,
                stacklevel=2,
            )
        return
    try:
        os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    except OSError as error:
        warnings.warn(
            f"killpg failed for pid {pid}: {error}",
            RuntimeWarning,
            stacklevel=2,
        )
