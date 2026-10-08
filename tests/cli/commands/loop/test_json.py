"""JSON contract tests for keep, discard, stop, status, start, finalize, sync ``--format json``.

The budget key is pinned once, on ``status``: every other command writes its
document through ``write_budget_report``, whose budget branches its own tests
cover.
"""

import json
from collections.abc import Callable
from pathlib import Path

import pytest

from gymrat.cli.app import app
from gymrat.loop.start import start_session
from gymrat.session.paths import experiment_worktree_dir
from gymrat.session.records import (
    FinalizeRecord,
    SessionLogRecord,
    StopRecord,
)
from tests._config import resolved_config
from tests._git import head_of
from tests.cli._budget import install_budget
from tests.cli._session import (
    close_session_with_one_keep,
    make_discard_repo,
    open_session_with_one_keep,
    runner,
    stub_config,
    write_bench_config,
    write_settled_session,
)
from tests.loop._settle import (
    CHECKS,
    checks_pass,
    edit_experiment,
    start_with,
    unimproved,
)
from tests.session.records._fixtures import (
    AT,
    COMMIT,
    SESSION_ID,
    append_records,
    committed_keep,
    iteration_record,
)

# ---------------------------------------------------------------------------
# keep --format json
# ---------------------------------------------------------------------------


def test_keep_command_when_format_json_and_committed_does_emit_structured_json(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    start_with(repo, (iteration_record(seq=1),))
    edit_experiment(repo)
    checks_pass(monkeypatch)
    write_bench_config(repo, checks=CHECKS)

    result = runner.invoke(app, ["keep", "-m", "cache the regex", "--format", "json"])

    assert result.exit_code == 0
    assert json.loads(result.stdout) == {
        "status": "committed",
        "reason": None,
        "checks": {
            "configured": True,
            "passed": True,
            "stdout_bytes": None,
            "stderr_bytes": None,
        },
        "commit": head_of(experiment_worktree_dir(repo)),
        "message": "cache the regex",
    }


@pytest.mark.parametrize(
    ("flags", "expected_exit", "expected_fields"),
    [
        pytest.param(
            [],
            1,
            {"status": "blocked", "reason": "not-improved", "commit": None, "message": None},
            id="refused",
        ),
        pytest.param(
            ["--allow-unimproved"],
            0,
            {"status": "committed", "reason": None},
            id="overridden",
        ),
    ],
)
def test_keep_command_when_format_json_and_iteration_unimproved_does_keep_its_key_set(
    repo: str,
    monkeypatch: pytest.MonkeyPatch,
    flags: list[str],
    expected_exit: int,
    expected_fields: dict[str, str | None],
):
    start_with(repo, (unimproved(1, "no-signal"),))
    edit_experiment(repo)
    checks_pass(monkeypatch)
    write_bench_config(repo, checks=CHECKS)

    result = runner.invoke(app, ["keep", *flags, "--format", "json"])

    assert result.exit_code == expected_exit
    doc = json.loads(result.stdout)
    assert doc.keys() == {"status", "reason", "checks", "commit", "message"}
    assert {key: doc[key] for key in expected_fields} == expected_fields


# ---------------------------------------------------------------------------
# discard --format json
# ---------------------------------------------------------------------------


def _unmeasured_edit(repo: str) -> None:
    """Open a session with an edit in the experiment worktree and nothing measured."""
    start_with(repo)
    edit_experiment(repo)


@pytest.mark.parametrize(
    ("arrange", "expected_seq", "expected_measured"),
    [
        pytest.param(make_discard_repo, 1, True, id="measured"),
        pytest.param(_unmeasured_edit, None, False, id="unmeasured"),
    ],
)
def test_discard_command_when_format_json_does_emit_structured_json(
    *,
    repo: str,
    monkeypatch: pytest.MonkeyPatch,
    arrange: Callable[[str], object],
    expected_seq: int | None,
    expected_measured: bool,
):
    arrange(repo)
    monkeypatch.setattr("gymrat.loop.discard.now_ns", lambda: AT)

    result = runner.invoke(app, ["discard", "--force", "--format", "json"])

    assert result.exit_code == 0
    assert json.loads(result.stdout) == {
        "seq": expected_seq,
        "at": AT,
        "measured": expected_measured,
    }


# ---------------------------------------------------------------------------
# status --format json
# ---------------------------------------------------------------------------


_FINALIZE = FinalizeRecord(
    type="finalize",
    at=AT,
    branch=f"gymrat/{SESSION_ID}-final",
    commit=COMMIT,
    message="squash 1 kept iteration",
)


@pytest.mark.parametrize(
    ("trailing", "finalized", "stopped"),
    [
        pytest.param((), False, False, id="open"),
        pytest.param((_FINALIZE,), True, False, id="finalized"),
        pytest.param(
            (StopRecord(type="stop", at=AT, message="user requested stop"),),
            False,
            True,
            id="stopped",
        ),
    ],
)
def test_status_command_when_format_json_does_emit_structured_json_on_stdout(
    repo: str, trailing: tuple[SessionLogRecord, ...], finalized: bool, stopped: bool
):
    write_settled_session(repo, *trailing)

    result = runner.invoke(app, ["status", "--format", "json"])

    assert result.exit_code == 0
    doc = json.loads(result.stdout)
    assert doc["session_id"] == SESSION_ID
    assert doc["branch"] == f"gymrat/{SESSION_ID}"
    assert doc["baseline"]["ref"] == "main"
    assert doc["baseline"]["sha"] == "a" * 40
    assert doc["iteration_count"] == 1
    assert doc["keep_count"] == 1
    assert doc["discard_count"] == 0
    assert doc["unsettled"] is False
    assert (doc["finalized"], doc["stopped"]) == (finalized, stopped)
    assert "budget" not in doc


# ---------------------------------------------------------------------------
# budget key — JSON output
# ---------------------------------------------------------------------------


def test_status_command_when_format_json_and_budget_active_does_include_budget_object(
    status_repo: str, monkeypatch: pytest.MonkeyPatch
):
    install_budget(status_repo, monkeypatch)

    result = runner.invoke(app, ["status", "--format", "json"])

    assert result.exit_code == 0
    doc = json.loads(result.stdout)
    assert "budget" in doc
    assert doc["budget"]["cap_minutes"] == 30
    assert isinstance(doc["budget"]["remaining_seconds"], int)


# ---------------------------------------------------------------------------
# stop --format json
# ---------------------------------------------------------------------------


def test_stop_command_when_format_json_does_emit_structured_json_with_at_and_message(
    stop_repo: str,
):
    result = runner.invoke(app, ["stop", "-m", "user requested stop", "--format", "json"])

    assert result.exit_code == 0
    doc = json.loads(result.stdout)
    assert doc["message"] == "user requested stop"
    assert "at" in doc
    assert isinstance(doc["at"], int)


# ---------------------------------------------------------------------------
# start --format json
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "runbook",
    [
        pytest.param(None, id="no-runbook"),
        pytest.param(".claude/skills/ecstatic-bench/SKILL.md", id="runbook"),
    ],
)
def test_start_command_when_format_json_and_fresh_does_emit_structured_json(
    repo: str, monkeypatch: pytest.MonkeyPatch, runbook: str | None
):
    stub_config(monkeypatch, "session", resolved_config(runbook=runbook))

    result = runner.invoke(app, ["start", "--baseline", "main", "--format", "json"])

    assert result.exit_code == 0
    doc = json.loads(result.stdout)
    assert "session_id" in doc
    assert doc["branch"].startswith("gymrat/")
    assert doc["baseline"]["ref"] == "main"
    assert isinstance(doc["baseline"]["sha"], str)
    assert isinstance(doc["worktrees"], dict)
    assert doc["resumed"] is False
    assert doc["iteration_count"] == 0
    assert doc["keep_count"] == 0
    assert doc["runbook"] == runbook
    assert doc["archived"] is None


def test_start_command_when_format_json_and_resumed_does_set_resumed_true_with_counts(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    start_session(repo, "main", resolved_config())
    append_records(repo, iteration_record(seq=1))
    append_records(repo, committed_keep(1))
    stub_config(monkeypatch, "session", resolved_config())

    result = runner.invoke(app, ["start", "--baseline", "main", "--format", "json"])

    assert result.exit_code == 0
    doc = json.loads(result.stdout)
    assert doc["resumed"] is True
    assert doc["iteration_count"] == 1
    assert doc["keep_count"] == 1


def test_start_command_when_format_json_and_archived_does_include_archived_session_id(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    closed_id = close_session_with_one_keep(repo)
    stub_config(monkeypatch, "session", resolved_config())

    result = runner.invoke(app, ["start", "--baseline", "main", "--format", "json"])

    assert result.exit_code == 0
    doc = json.loads(result.stdout)
    assert doc["archived"]["session_id"] == closed_id
    assert isinstance(doc["archived"]["path"], str)
    assert doc["resumed"] is False


# ---------------------------------------------------------------------------
# finalize --format json
# ---------------------------------------------------------------------------


def test_finalize_command_when_format_json_does_emit_structured_json(
    repo: str,
):
    open_session_with_one_keep(repo)

    result = runner.invoke(app, ["finalize", "-m", "squash the tuning session", "--format", "json"])

    assert result.exit_code == 0
    doc = json.loads(result.stdout)
    assert doc["branch"].endswith("-final")
    assert isinstance(doc["commit"], str)
    assert len(doc["commit"]) == 40
    assert doc["message"] == "squash the tuning session"
    assert isinstance(doc["at"], int)


# ---------------------------------------------------------------------------
# sync --format json
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "files",
    [
        pytest.param(["extra.py"], id="files-synced"),
        pytest.param([], id="nothing-to-sync"),
    ],
)
def test_sync_command_when_format_json_does_emit_the_synced_files(sync_repo: str, files: list[str]):
    for name in files:
        Path(sync_repo, name).write_text("# new\n", encoding="utf-8")

    result = runner.invoke(app, ["sync", "--format", "json"])

    assert result.exit_code == 0
    assert json.loads(result.stdout)["files"] == files
