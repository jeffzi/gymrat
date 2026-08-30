"""Command-level tests for ``gymrat keep`` and ``gymrat discard``.

The command wiring is driven through :class:`typer.testing.CliRunner` against a
throwaway repository from the shared ``create_scratch_repo`` factory, so the
suite is order-independent and safe under ``pytest-xdist`` / ``pytest-randomly``.
The one boundary these tests mock is the checks command — the consumer's own
test suite, which no test here can run — replaced at ``gymrat.loop.settle.exec``
by the ``checks_pass`` / ``checks_fail`` recorders. Every git operation is real,
run against the ``gymrat.toml`` each test lays down at the repository root.
"""

import re
from pathlib import Path

import pytest
import tomli_w
from typer.testing import CliRunner

from gymrat.cli.app import app
from gymrat.session import KeepRecord, experiment_worktree_dir
from tests._ansi import SGR_RE
from tests.loop._settle import (
    CHECKS,
    checks_fail,
    checks_pass,
    edit_experiment,
    head_of,
    iteration,
    last_record_of,
    start_with,
    status_of,
)

runner = CliRunner()


def _write_config(root: str, **extra: object) -> None:
    """Write the ``gymrat.toml`` the settle commands read their checks gate from."""
    payload: dict[str, object] = {"bench": "npm run bench", **extra}
    (Path(root) / "gymrat.toml").write_text(tomli_w.dumps(payload), encoding="utf-8")


# ---------------------------------------------------------------------------
# the keep command
# ---------------------------------------------------------------------------


def test_keep_command_when_checks_pass_does_commit_and_print_the_short_commit(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    start_with(repo, (iteration(1),))
    edit_experiment(repo)
    checks_pass(monkeypatch)
    _write_config(repo, checks=CHECKS)

    result = runner.invoke(app, ["keep", "-m", "cache the regex"])

    assert result.exit_code == 0
    record = last_record_of(repo)
    assert isinstance(record, KeepRecord)
    assert record.status == "committed"
    assert head_of(experiment_worktree_dir(repo))[:7] in result.stdout


def test_keep_command_when_nothing_to_commit_does_exit_one_recording_the_block(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    start_with(repo, (iteration(1),))
    checks_pass(monkeypatch)
    _write_config(repo, checks=CHECKS)

    result = runner.invoke(app, ["keep"])

    assert result.exit_code == 1
    record = last_record_of(repo)
    assert isinstance(record, KeepRecord)
    assert record.status == "blocked"
    assert record.reason == "nothing-to-commit"


def test_keep_command_when_refusing_does_print_a_report_carrying_no_hint_label(repo: str):
    start_with(repo, (iteration(1),))
    _write_config(repo, checks=CHECKS)

    result = runner.invoke(app, ["keep"])

    assert result.exit_code == 1
    assert "Hint" not in result.stdout
    assert "gymrat keep" in result.stdout


@pytest.mark.parametrize(
    ("variable", "expect_ansi"),
    [
        pytest.param("FORCE_COLOR", True, id="force-color"),
        pytest.param("NO_COLOR", False, id="no-color"),
    ],
)
def test_keep_command_when_refusing_does_take_report_color_from_the_environment(
    repo: str, monkeypatch: pytest.MonkeyPatch, variable: str, expect_ansi: bool
):
    for name in ("FORCE_COLOR", "NO_COLOR"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(variable, "1")
    start_with(repo, (iteration(1),))
    _write_config(repo, checks=CHECKS)

    result = runner.invoke(app, ["keep"])

    assert bool(SGR_RE.search(result.stdout)) is expect_ansi


def test_keep_command_when_checks_fail_does_exit_one_recording_the_block(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    start_with(repo, (iteration(1),))
    edit_experiment(repo)
    checks_fail(monkeypatch)
    _write_config(repo, checks=CHECKS)

    result = runner.invoke(app, ["keep"])

    assert result.exit_code == 1
    record = last_record_of(repo)
    assert isinstance(record, KeepRecord)
    assert record.status == "blocked"
    assert record.reason == "checks-failed"


# ---------------------------------------------------------------------------
# the discard command
# ---------------------------------------------------------------------------


def test_discard_command_when_run_does_clean_the_worktree_and_record_the_discard(repo: str):
    start_with(repo, (iteration(1),))
    edit_experiment(repo)
    _write_config(repo)

    result = runner.invoke(app, ["discard"])

    assert result.exit_code == 0
    assert status_of(experiment_worktree_dir(repo)) == ""
    assert last_record_of(repo).type == "discard"
    assert re.search(r"discard", result.stdout, re.IGNORECASE)


# ---------------------------------------------------------------------------
# no open session
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("command", ["keep", "discard"])
def test_settle_command_when_no_session_does_exit_two_with_a_start_hint(repo: str, command: str):
    _write_config(repo, checks=CHECKS)

    result = runner.invoke(app, [command])

    assert result.exit_code == 2
    assert "gymrat start" in result.stderr
