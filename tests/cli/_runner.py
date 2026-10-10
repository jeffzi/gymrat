"""The in-process CLI runner, and one whose stdout fails the way a closed pipe does.

This is test-support code, not a test module: it carries no test functions.
"""

import contextlib
import errno
import os
import sys
from collections.abc import Generator
from typing import Any, override

from typer.testing import CliRunner

from tests._streams import RaisingStream

runner = CliRunner()


def closed_stdout_error() -> OSError:
    """Build the error a POSIX stdout write raises once its reader has gone.

    Returns:
        A fresh ``BrokenPipeError`` carrying ``EPIPE``.
    """
    return BrokenPipeError(errno.EPIPE, "Broken pipe")


def disk_full_error() -> OSError:
    """Build the error a stdout write raises when the disk is full.

    Returns:
        A fresh ``OSError`` carrying ``ENOSPC`` and its system message.
    """
    return OSError(errno.ENOSPC, os.strerror(errno.ENOSPC))


class FailingStdoutRunner(CliRunner):
    """A ``CliRunner`` whose isolated ``sys.stdout`` fails every write with ``error``.

    The runner still captures stderr, so a test can check nothing was reported.

    Args:
        error: The exception every stdout write raises.
    """

    def __init__(self, error: OSError) -> None:
        super().__init__()
        self._error = error

    @override
    @contextlib.contextmanager
    def isolation(self, *args: Any, **kwargs: Any) -> Generator[Any]:
        with super().isolation(*args, **kwargs) as streams:
            # Keep the runner's wrapper alive: collecting it closes the buffer it captures into.
            captured_stdout = sys.stdout
            sys.stdout = RaisingStream(self._error)
            try:
                yield streams
            finally:
                sys.stdout = captured_stdout
