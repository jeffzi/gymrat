"""JSON contract tests for keep, discard, stop, status, start, finalize, sync ``--format json``.

The budget key is pinned once, on ``status``: every other command writes its
document through ``write_budget_report``, whose budget branches its own tests
cover.
"""

import json
from pathlib import Path
from unittest.mock import create_autospec

import pytest

from gymrat.cli.app import app
from gymrat.loop.discard import DiscardResult, discard_session
from gymrat.loop.keep import KeepResult, keep_session
from gymrat.loop.start import start_session
from gymrat.session.records import (
    DiscardRecord,
    FinalizeRecord,
    KeepChecks,
    KeepRecord,
    SessionLogRecord,
    StopRecord,
)
from tests._config import resolved_config
from tests.cli._budget import install_budget
from tests.cli._session import (
    close_session_with_one_keep,
    make_discard_repo,
    never_tty,
    open_session_with_one_keep,
    runner,
    stub_resolve_config,
    write_bench_config,
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
    session_record,
    write_session_log,
)

# ---------------------------------------------------------------------------
# keep --format json
# ---------------------------------------------------------------------------


def _make_committed_keep_result() -> KeepResult:
    """A ``KeepResult`` for a committed keep with checks passing."""
    record = KeepRecord(
        type="keep",
        seq=1,
        at=AT,
        status="committed",
        checks=KeepChecks(configured=True, passed=True, stdout_bytes=80, stderr_bytes=0),
        commit=COMMIT,
        message="cache the regex",
    )
    return KeepResult(record=record, report="committed keep report")


def _wire_keep(repo: str, monkeypatch: pytest.MonkeyPatch, keep_result: KeepResult) -> None:
    """Wire ``keep`` with a config resolver and a recording keep_session stub."""
    start_session(repo, "main", resolved_config())
    append_records(repo, iteration_record(seq=1))
    monkeypatch.setattr(
        "gymrat.cli.commands.loop.keep_session",
        create_autospec(keep_session, return_value=keep_result),
    )


def test_keep_command_when_format_json_and_committed_does_emit_structured_json(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    _wire_keep(repo, monkeypatch, _make_committed_keep_result())

    result = runner.invoke(app, ["keep", "--format", "json"])

    assert result.exit_code == 0
    doc = json.loads(result.stdout)
    assert doc["status"] == "committed"
    assert doc["commit"] == COMMIT
    assert doc["message"] == "cache the regex"
    assert doc["reason"] is None
    assert doc["checks"]["configured"] is True
    assert doc["checks"]["passed"] is True
    assert doc["checks"]["stdout_bytes"] == 80
    assert doc["checks"]["stderr_bytes"] == 0


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


def _make_discard_result() -> DiscardResult:
    """A ``DiscardResult`` for a measured discard of iteration 1."""
    return DiscardResult(
        record=DiscardRecord(type="discard", seq=1, at=AT),
        report="discarded iteration 1",
        at=AT,
    )


def _make_unmeasured_discard_result() -> DiscardResult:
    """A ``DiscardResult`` for an unmeasured revert (no record)."""
    return DiscardResult(
        record=None,
        report="reverted unmeasured changes",
        at=AT,
    )


def _wire_discard(monkeypatch: pytest.MonkeyPatch, discard_result: DiscardResult) -> None:
    """Wire ``discard`` with a recording discard_session stub and skip its TTY prompt."""
    monkeypatch.setattr(
        "gymrat.cli.commands.loop.discard_session",
        create_autospec(discard_session, return_value=discard_result),
    )
    monkeypatch.setattr("gymrat.cli.commands.loop.is_tty", never_tty)


@pytest.fixture
def discard_repo(repo: str) -> str:
    """A repository with an open session and one unsettled iteration to discard."""
    return make_discard_repo(repo)


@pytest.mark.parametrize(
    ("discard_result", "expected_seq", "expected_measured"),
    [
        pytest.param(_make_discard_result(), 1, True, id="measured"),
        pytest.param(_make_unmeasured_discard_result(), None, False, id="unmeasured"),
    ],
)
def test_discard_command_when_format_json_does_emit_structured_json(
    discard_repo: str,
    monkeypatch: pytest.MonkeyPatch,
    discard_result: DiscardResult,
    expected_seq: int | None,
    expected_measured: bool,
):
    _wire_discard(monkeypatch, discard_result)

    result = runner.invoke(app, ["discard", "--force", "--format", "json"])

    assert result.exit_code == 0
    doc = json.loads(result.stdout)
    assert doc["seq"] == expected_seq
    assert doc["at"] == AT
    assert isinstance(doc["at"], int)
    assert doc["measured"] is expected_measured


# ---------------------------------------------------------------------------
# status --format json
# ---------------------------------------------------------------------------


def _write_status_session(repo: str, *trailing_records: SessionLogRecord) -> None:
    """A configured session with one kept iteration, followed by ``trailing_records``."""
    write_session_log(
        repo, session_record(), (iteration_record(seq=1), committed_keep(1), *trailing_records)
    )
    write_bench_config(repo)


@pytest.fixture
def status_repo(repo: str) -> str:
    """A repository with a configured session and one kept iteration."""
    _write_status_session(repo)
    return repo


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
    _write_status_session(repo, *trailing)

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


def test_status_command_when_format_json_and_no_budget_does_omit_budget_key(
    status_repo: str,
):
    result = runner.invoke(app, ["status", "--format", "json"])

    assert result.exit_code == 0
    doc = json.loads(result.stdout)
    assert "budget" not in doc


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
    stub_resolve_config(monkeypatch, runbook=runbook)

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
    stub_resolve_config(monkeypatch)

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
    stub_resolve_config(monkeypatch)

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
