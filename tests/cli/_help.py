"""Shared ``--help`` capture for the CLI test files.

This is test-support code, not a test module: it carries no test functions or
pytest fixtures of its own.
"""

from typer.testing import CliRunner

from gymrat.cli.app import app
from tests._ansi import strip_ansi

_runner = CliRunner()


def help_output(*command: str) -> str:
    """Help text of ``gymrat *command`` rendered wide and ANSI-stripped.

    Ambient color splits a token across escape sequences and a narrow terminal
    wraps it across lines; both break a plain substring match. No arguments
    captures the root ``gymrat --help``.
    """
    result = _runner.invoke(app, [*command, "--help"], env={"COLUMNS": "200"})
    assert result.exit_code == 0
    return strip_ansi(result.stdout)
