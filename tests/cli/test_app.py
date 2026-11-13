"""Tests for the root ``gymrat`` typer app's ``--version`` option.

These drive the assembled CLI through :class:`typer.testing.CliRunner`, so the
``--version`` eager option is exercised the way a shell would invoke it. How a
command's color flag replaces the root's is pinned in ``tests/cli/test_console.py``.
"""

import errno
import importlib.metadata
import os

from gymrat.cli.app import app
from gymrat.errors import TOOL_FAILURE_EXIT_CODE
from tests._rich import unwrap_panel
from tests.cli._runner import (
    FailingStdoutRunner,
    disk_full_error,
    runner,
)

# ---------------------------------------------------------------------------
# --version
# ---------------------------------------------------------------------------


def test_app_when_version_flag_does_print_package_version():
    result = runner.invoke(app, ["--version"])

    assert result.exit_code == 0
    assert importlib.metadata.version("gymrat") in result.stdout


def test_app_when_version_and_stdout_write_fails_with_disk_full_does_exit_two_with_error_on_stderr():
    result = FailingStdoutRunner(disk_full_error()).invoke(app, ["--version"])

    assert result.exit_code == TOOL_FAILURE_EXIT_CODE
    assert os.strerror(errno.ENOSPC) in unwrap_panel(result.stderr)
