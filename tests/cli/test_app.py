"""Tests for the root ``gymrat`` typer app: version, debug, dispatch.

These drive the assembled CLI through :class:`typer.testing.CliRunner`, so the
root callback, the ``--version`` eager option, the root ``--debug`` flag, and the
exit-2 error a repository command prints outside a repository are exercised the
way a shell would invoke them. How a command's color flag replaces the root's is
pinned in ``tests/cli/test_console.py``.
"""

import errno
import importlib.metadata
import os

import pytest

from gymrat.cli.app import app
from gymrat.errors import TOOL_FAILURE_EXIT_CODE
from tests._ansi import strip_ansi
from tests._rich import unwrap_panel
from tests.cli._runner import (
    FailingStdoutRunner,
    disk_full_error,
    runner,
)


def _joined(text: str) -> str:
    """``text`` ANSI-stripped with every whitespace run, line wraps included, as one space."""
    return " ".join(strip_ansi(text).split())


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


# ---------------------------------------------------------------------------
# --debug
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("_in_non_repo")
def test_app_when_debug_flag_does_show_traceback_on_error():
    result = runner.invoke(app, ["--debug", "status"])

    assert result.exit_code == TOOL_FAILURE_EXIT_CODE
    assert "Traceback" in result.output


# ---------------------------------------------------------------------------
# repository discovery failure
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("_in_non_repo")
def test_app_when_run_outside_a_repository_does_exit_two_naming_the_directory():
    cwd = os.getcwd()  # noqa: PTH109 -- the message quotes the str cwd repo discovery saw

    result = runner.invoke(app, ["status"])

    assert result.exit_code == TOOL_FAILURE_EXIT_CODE
    assert cwd in _joined(result.stderr)
