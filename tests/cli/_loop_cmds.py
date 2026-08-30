"""Shared helpers for the loop-command test files.

Builders and stubs used by more than one ``test_loop_cmds*.py`` module.  This
is test-support code, not a test module: it carries no test functions or pytest
fixtures of its own.
"""

import re
from pathlib import Path

import tomli_w
from typer.testing import CliRunner

from tests._ansi import strip_ansi

__all__ = ["never_tty", "plain_lines", "runner", "strip_ansi", "write_config"]

runner = CliRunner()

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def plain_lines(text: str) -> list[str]:
    """The non-blank lines of ``text``, stripped of color and surrounding space."""
    return [_ANSI_RE.sub("", line).strip() for line in text.split("\n") if line.strip()]


def never_tty(_stream: object) -> bool:
    """Stand in for ``is_tty`` so the discard command takes its non-interactive path."""
    return False


def write_config(root: str, **extra: object) -> None:
    """Write the implicit ``gymrat.toml`` at the repository root."""
    payload: dict[str, object] = {"bench": "npm run bench", **extra}
    (Path(root) / "gymrat.toml").write_text(tomli_w.dumps(payload), encoding="utf-8")
