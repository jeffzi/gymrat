"""Running the ``gymrat`` CLI as a child process and stopping it with a signal.

The out-of-process signal tests share this harness: a child started with the
fault handler on, so a run that outlives its signal can dump every thread's
stack, and a shell script that records its pid before it blocks, so a test knows
when the run is mid-step and which process group to reap.
"""

import shlex
import signal
import subprocess
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from pathlib import Path

import pytest

from tests._cli import ENTRY, no_color_env
from tests._process_helpers import reaped

#: Seconds a signal test gives a pid file to appear and a signalled child to exit.
SETTLE_TIMEOUT_S = 30.0


def pid_recording_script(pid_path: Path | str, body: str) -> str:
    """Build a shell script that writes its own pid to ``pid_path``, then runs ``body``.

    Args:
        pid_path: Where the script records its pid; relative paths resolve
            against the directory the script runs in.
        body: The shell lines that follow, typically ones that block.

    Returns:
        The script's text.
    """
    return f"#!/bin/sh\necho $$ > {shlex.quote(str(pid_path))}\n{body}"


@contextmanager
def spawned_gymrat(argv: Sequence[str], cwd: str) -> Generator[subprocess.Popen[str]]:
    """Run ``gymrat`` with ``argv`` out of process in ``cwd``, reaping it on the way out.

    Args:
        argv: The command line after the program name.
        cwd: The directory the child runs in.

    Yields:
        The running child, with its stdout and stderr piped as text.
    """
    with reaped(
        subprocess.Popen(  # noqa: S603 -- argv is the gymrat entry point plus the test's own command
            [*ENTRY, *argv],
            cwd=cwd,
            env={**no_color_env(), "PYTHONFAULTHANDLER": "1"},
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    ) as proc:
        yield proc


def stop_by_signal(
    proc: subprocess.Popen[str], signal_number: int, timeout_s: float = SETTLE_TIMEOUT_S
) -> None:
    """Send ``signal_number`` to ``proc`` and wait for it to exit.

    The child runs with the fault handler on, so when it outlives the wait a
    ``SIGABRT`` makes it dump every thread's stack, which the failure carries:
    where a process that ignored its signal is parked.

    Args:
        proc: A child started by :func:`spawned_gymrat`.
        signal_number: The signal to send.
        timeout_s: Seconds to wait for the child to exit.
    """
    proc.send_signal(signal_number)
    try:
        proc.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        proc.send_signal(signal.SIGABRT)
        _, stacks = proc.communicate(timeout=30)
        pytest.fail(f"gymrat still running {timeout_s:g} s after the signal; threads:\n{stacks}")
