"""Shared helpers for the loop-command test files.

Builders and stubs used by more than one ``test_loop_cmds*.py`` module.  This
is test-support code, not a test module: it carries no test functions or pytest
fixtures of its own.
"""

from pathlib import Path

import tomli_w
from typer.testing import CliRunner

from tests._ansi import SGR_RE, strip_ansi

__all__ = [
    "always_tty",
    "make_discard_repo",
    "make_stop_repo",
    "never_tty",
    "plain_lines",
    "runner",
    "strip_ansi",
    "write_config",
]

runner = CliRunner()


def plain_lines(text: str) -> list[str]:
    """The non-blank lines of ``text``, stripped of color and surrounding space."""
    return [SGR_RE.sub("", line).strip() for line in text.split("\n") if line.strip()]


def always_tty(_stream: object) -> bool:
    """Stand in for ``is_tty`` so the discard command takes its interactive path."""
    return True


def never_tty(_stream: object) -> bool:
    """Stand in for ``is_tty`` so the discard command takes its non-interactive path."""
    return False


def make_discard_repo(repo: str) -> str:
    """Set up ``repo`` with an open session and one unsettled iteration to discard."""
    from gymrat.loop.start import start_session
    from gymrat.session import append_record, session_jsonl_path
    from tests.loop.iterate._fixtures import resolved_config
    from tests.session.records._fixtures import iteration_record

    start_session(repo, "main", resolved_config())
    append_record(session_jsonl_path(repo), iteration_record(seq=1))
    return repo


def make_stop_repo(repo: str) -> str:
    """Set up ``repo`` with a settled, configured session ready for the stop command."""
    from tests.loop.settle._fixtures import iteration, start_with
    from tests.session.records._fixtures import committed_keep

    start_with(repo, (iteration(1), committed_keep(1)))
    write_config(repo)
    return repo


def write_config(root: str, **extra: object) -> None:
    """Write the implicit ``gymrat.toml`` at the repository root."""
    payload: dict[str, object] = {"bench": "npm run bench", **extra}
    (Path(root) / "gymrat.toml").write_text(tomli_w.dumps(payload), encoding="utf-8")
