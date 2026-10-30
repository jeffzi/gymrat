"""Command-level tests for stop and ``start --format json``.

Each command is driven through :class:`typer.testing.CliRunner` against a
throwaway repository from the shared ``create_scratch_repo`` factory, so the
suite is order-independent and safe under ``pytest-xdist`` / ``pytest-randomly``.

The shared session guards are pinned at their owner, tests/session/test_store.py;
here each is driven once, through the one command that reaches it differently:
``measure --record`` (the only conditional caller of the finalized guard),
``status`` (the only direct caller of the no-session guard) and ``iterate`` (whose
refusal must land before it numbers the iteration). The closed-stdout table also
covers ``supervise``.
"""

import json
from collections.abc import Callable
from pathlib import Path

import pytest

from gymrat.cli.app import app
from gymrat.session.paths import session_jsonl_path
from gymrat.session.records import CommandRecord
from tests._ansi import SGR_RE
from tests._cli import stderr_text
from tests._config import resolved_config
from tests.cli._command_stubs import (
    force_render_mode,
    stub_config,
    stub_measure,
)
from tests.cli._runner import (
    FailingStdoutRunner,
    closed_stdout_error,
    runner,
)
from tests.cli._session import (
    close_session_with_one_keep,
    open_hooked_iterate_session,
    open_session,
    write_bench_config,
)
from tests.cli.commands.supervise._seams import install_seams
from tests.loop._settle import CHECKS
from tests.loop.iterate._fixtures import install_improved_samples
from tests.session.records._fixtures import (
    last_command_record,
    records_of_type,
)

# ---------------------------------------------------------------------------
# the stop command
# ---------------------------------------------------------------------------


def test_stop_command_when_message_blank_does_exit_two_naming_the_option(stop_repo: str):
    result = runner.invoke(app, ["stop", "-m", "   "])

    assert result.exit_code == 2
    assert "message" in stderr_text(result).lower()


# ---------------------------------------------------------------------------
# guards every session command shares
# ---------------------------------------------------------------------------


def test_measure_record_when_session_finalized_does_refuse_as_finalized(repo: str):
    close_session_with_one_keep(repo)
    write_bench_config(repo, checks=CHECKS)

    result = runner.invoke(app, ["measure", "main", "--bench", "sh bench.sh", "--record"])

    assert result.exit_code == 2
    cmd = last_command_record(repo)
    assert (cmd.name, cmd.exit_code, cmd.reason) == ("measure", 2, "finalized")


def test_status_when_no_session_and_no_config_does_exit_two_with_a_start_hint(repo: str):
    result = runner.invoke(app, ["status"])

    assert (result.exit_code, result.stdout) == (2, "")
    assert "gymrat start" in result.stderr
    assert not Path(session_jsonl_path(repo)).exists()


def test_session_command_when_no_color_does_strip_ansi_from_stderr_error(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("FORCE_COLOR", "1")
    write_bench_config(repo)

    result = runner.invoke(app, ["start", "--no-color", "--baseline", "banana"])

    assert result.exit_code == 2
    assert result.stderr.startswith("Error: ")
    assert not SGR_RE.search(result.stderr)


def _open_and_stub_measure(repo: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Open a session and replace the measurement engine and config resolution."""
    open_session(repo)
    stub_measure(monkeypatch)


def _supervise_live(_repo: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace every supervise seam and force the live dashboard."""
    install_seams(monkeypatch)
    force_render_mode(monkeypatch, "supervise", "live")


_MEASURE_MAIN = ["measure", "main", "--bench", "sh bench.sh"]


@pytest.mark.parametrize(
    ("argv", "arrange"),
    [
        pytest.param(
            [*_MEASURE_MAIN, "--record"], _open_and_stub_measure, id="measure-record-note"
        ),
        pytest.param(
            ["supervise", "optimize it", "--max-minutes", "10"], _supervise_live, id="supervise"
        ),
    ],
)
def test_command_when_stdout_reader_closed_does_exit_without_stderr(
    *,
    repo: str,
    monkeypatch: pytest.MonkeyPatch,
    argv: list[str],
    arrange: Callable[[str, pytest.MonkeyPatch], None],
):
    arrange(repo, monkeypatch)

    result = FailingStdoutRunner(closed_stdout_error()).invoke(app, argv)

    assert (result.exit_code, result.stderr) == (0, "")


def test_iterate_when_supervised_run_live_does_refuse_as_supervised_use_tool(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    open_hooked_iterate_session(repo, monkeypatch)
    install_improved_samples(monkeypatch, repo)

    result = runner.invoke(app, ["iterate"])

    assert (result.exit_code, result.stdout) == (2, "")
    commands = records_of_type(repo, CommandRecord)
    assert [(cmd.name, cmd.exit_code, cmd.reason, cmd.origin, cmd.seq) for cmd in commands] == [
        ("iterate", 2, "supervised-use-tool", "cli", None)
    ]


# ---------------------------------------------------------------------------
# start --format json
# ---------------------------------------------------------------------------


_RUNBOOK = ".claude/skills/ecstatic-bench/SKILL.md"


@pytest.mark.usefixtures("repo")
def test_start_command_when_format_json_and_fresh_does_emit_the_configured_runbook(
    monkeypatch: pytest.MonkeyPatch,
):
    stub_config(monkeypatch, "session", resolved_config(runbook=_RUNBOOK))

    result = runner.invoke(app, ["start", "--baseline", "main", "--format", "json"])

    assert result.exit_code == 0
    assert json.loads(result.stdout)["runbook"] == _RUNBOOK
