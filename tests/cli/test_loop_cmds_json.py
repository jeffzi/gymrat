"""JSON contract tests for keep, discard, stop, status, start, finalize, sync ``--format json``.

Budget-key tests verify that a live budget inserts a ``budget`` object into
every JSON document, and that no budget means no key.
"""

import json
from pathlib import Path

import pytest

from gymrat.cli.app import app
from gymrat.loop.settle import DiscardResult, KeepResult
from gymrat.loop.start import start_session
from gymrat.session import (
    DiscardRecord,
    FinalizeRecord,
    KeepChecks,
    KeepRecord,
    SessionLogRecord,
    SessionRecord,
    StopRecord,
    append_record,
    experiment_worktree_dir,
    read_records,
    session_jsonl_path,
)
from tests.cli._budget import install_budget
from tests.cli._session import (
    make_discard_repo,
    make_stop_repo,
    never_tty,
    runner,
    write_config,
)
from tests.loop.iterate._fixtures import resolved_config
from tests.loop.settle._fixtures import (
    CHECKS,
    checks_pass,
    edit_experiment,
    git,
    head_of,
    start_with,
    unimproved,
)
from tests.session.records._fixtures import (
    AT,
    COMMIT,
    SESSION_ID,
    committed_keep,
    iteration_record,
    session_record,
    write_session_log,
)

# ---------------------------------------------------------------------------
# keep --format json
# ---------------------------------------------------------------------------


class _KeepSessionRecorder:
    """A stand-in for ``keep_session`` that returns a fixed ``KeepResult``."""

    def __init__(self, result: KeepResult) -> None:
        self._result = result

    async def __call__(self, *_args: object, **_kwargs: object) -> KeepResult:
        return self._result


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


def _make_blocked_keep_result() -> KeepResult:
    """A ``KeepResult`` for a blocked keep where checks failed."""
    record = KeepRecord(
        type="keep",
        seq=1,
        at=AT,
        status="blocked",
        checks=KeepChecks(configured=True, passed=False, stdout_bytes=120, stderr_bytes=45),
        reason="checks-failed",
    )
    return KeepResult(record=record, report="blocked keep report")


def _wire_keep(repo: str, monkeypatch: pytest.MonkeyPatch, keep_result: KeepResult) -> None:
    """Wire ``keep`` with a config resolver and a recording keep_session stub."""
    start_session(repo, "main", resolved_config())
    append_record(session_jsonl_path(repo), iteration_record(seq=1))
    monkeypatch.setattr("gymrat.cli.loop_cmds.keep_session", _KeepSessionRecorder(keep_result))


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


def test_keep_command_when_format_json_and_blocked_does_emit_blocked_json_with_reason(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    _wire_keep(repo, monkeypatch, _make_blocked_keep_result())

    result = runner.invoke(app, ["keep", "--format", "json"])

    assert result.exit_code == 1
    doc = json.loads(result.stdout)
    assert doc["status"] == "blocked"
    assert doc["reason"] == "checks-failed"
    assert doc["commit"] is None
    assert doc["message"] is None


def test_keep_command_when_format_json_does_include_stable_key_names(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    _wire_keep(repo, monkeypatch, _make_committed_keep_result())

    result = runner.invoke(app, ["keep", "--format", "json"])

    assert result.exit_code == 0
    doc = json.loads(result.stdout)
    assert {"status", "reason", "checks", "commit", "message"} <= doc.keys()
    assert {"configured", "passed", "stdout_bytes", "stderr_bytes"} <= doc["checks"].keys()


@pytest.mark.parametrize(
    ("flags", "expected_exit", "expected_fields"),
    [
        pytest.param([], 1, {"status": "blocked", "reason": "not-improved"}, id="refused"),
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
    write_config(repo, checks=CHECKS)

    result = runner.invoke(app, ["keep", *flags, "--format", "json"])

    assert result.exit_code == expected_exit
    doc = json.loads(result.stdout)
    assert doc.keys() == {"status", "reason", "checks", "commit", "message"}
    assert {key: doc[key] for key in expected_fields} == expected_fields


def test_keep_command_when_format_text_does_produce_plain_report(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    _wire_keep(repo, monkeypatch, _make_committed_keep_result())

    result = runner.invoke(app, ["keep", "--format", "text"])

    assert result.exit_code == 0
    assert "committed keep report" in result.stdout


# ---------------------------------------------------------------------------
# discard --format json
# ---------------------------------------------------------------------------


class _DiscardSessionRecorder:
    """A stand-in for ``discard_session`` that returns a fixed ``DiscardResult``."""

    def __init__(self, result: DiscardResult) -> None:
        self._result = result

    def __call__(self, *_args: object, **_kwargs: object) -> DiscardResult:
        return self._result


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
        "gymrat.cli.loop_cmds.discard_session", _DiscardSessionRecorder(discard_result)
    )
    monkeypatch.setattr("gymrat.cli.loop_cmds.is_tty", never_tty)


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


def test_discard_command_when_format_text_does_produce_plain_report(
    discard_repo: str, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr("gymrat.cli.loop_cmds.is_tty", never_tty)

    result = runner.invoke(app, ["discard", "--force", "--format", "text"])

    assert result.exit_code == 0
    assert "discard" in result.stdout.lower()


# ---------------------------------------------------------------------------
# status --format json
# ---------------------------------------------------------------------------


def _write_status_session(repo: str, *trailing_records: SessionLogRecord) -> None:
    """A configured session with one kept iteration, followed by ``trailing_records``."""
    write_session_log(
        repo, session_record(), (iteration_record(seq=1), committed_keep(1), *trailing_records)
    )
    write_config(repo)


@pytest.fixture
def status_repo(repo: str) -> str:
    """A repository with a configured session and one kept iteration."""
    _write_status_session(repo)
    return repo


def test_status_command_when_format_json_does_emit_structured_json_on_stdout(status_repo: str):
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
    assert doc["finalized"] is False
    assert doc["stopped"] is False


def test_status_command_when_format_json_and_finalized_does_set_finalized_true(repo: str):
    _write_status_session(
        repo,
        FinalizeRecord(
            type="finalize",
            at=AT,
            branch=f"gymrat/{SESSION_ID}-final",
            commit=COMMIT,
            message="squash 1 kept iteration",
        ),
    )

    result = runner.invoke(app, ["status", "--format", "json"])

    assert result.exit_code == 0
    doc = json.loads(result.stdout)
    assert doc["finalized"] is True


def test_status_command_when_format_json_does_include_stable_key_names(status_repo: str):
    result = runner.invoke(app, ["status", "--format", "json"])

    assert result.exit_code == 0
    doc = json.loads(result.stdout)
    assert {
        "session_id",
        "branch",
        "baseline",
        "iteration_count",
        "keep_count",
        "discard_count",
        "unsettled",
        "finalized",
        "stopped",
    } <= doc.keys()
    assert {"ref", "sha"} <= doc["baseline"].keys()


def test_status_command_when_format_json_and_stop_record_does_set_stopped_true(repo: str):
    _write_status_session(repo, StopRecord(type="stop", at=AT, message="user requested stop"))

    result = runner.invoke(app, ["status", "--format", "json"])

    assert result.exit_code == 0
    doc = json.loads(result.stdout)
    assert doc["stopped"] is True


def test_status_command_when_format_text_does_produce_identical_output(status_repo: str):
    result_text = runner.invoke(app, ["status", "--format", "text"])
    result_default = runner.invoke(app, ["status"])

    assert result_text.exit_code == 0
    assert result_default.exit_code == 0
    assert result_text.stdout == result_default.stdout


# ---------------------------------------------------------------------------
# budget key — JSON output
# ---------------------------------------------------------------------------


def test_keep_command_when_format_json_and_budget_active_does_include_budget_object(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    _wire_keep(repo, monkeypatch, _make_committed_keep_result())
    install_budget(repo, monkeypatch)

    result = runner.invoke(app, ["keep", "--format", "json"])

    assert result.exit_code == 0
    doc = json.loads(result.stdout)
    assert "budget" in doc
    assert doc["budget"]["cap_minutes"] == 30
    assert isinstance(doc["budget"]["remaining_seconds"], int)


def test_keep_command_when_format_json_and_blocked_and_budget_active_does_include_budget_object(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    _wire_keep(repo, monkeypatch, _make_blocked_keep_result())
    install_budget(repo, monkeypatch)

    result = runner.invoke(app, ["keep", "--format", "json"])

    assert result.exit_code == 1
    doc = json.loads(result.stdout)
    assert "budget" in doc
    assert doc["budget"]["cap_minutes"] == 30
    assert isinstance(doc["budget"]["remaining_seconds"], int)


def test_keep_command_when_format_json_and_no_budget_does_omit_budget_key(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    _wire_keep(repo, monkeypatch, _make_committed_keep_result())

    result = runner.invoke(app, ["keep", "--format", "json"])

    assert result.exit_code == 0
    doc = json.loads(result.stdout)
    assert "budget" not in doc


def test_discard_command_when_format_json_and_budget_active_does_include_budget_object(
    discard_repo: str, monkeypatch: pytest.MonkeyPatch
):
    _wire_discard(monkeypatch, _make_discard_result())
    install_budget(discard_repo, monkeypatch)

    result = runner.invoke(app, ["discard", "--force", "--format", "json"])

    assert result.exit_code == 0
    doc = json.loads(result.stdout)
    assert "budget" in doc
    assert doc["budget"]["cap_minutes"] == 30
    assert isinstance(doc["budget"]["remaining_seconds"], int)


def test_discard_command_when_format_json_and_no_budget_does_omit_budget_key(
    discard_repo: str, monkeypatch: pytest.MonkeyPatch
):
    _wire_discard(monkeypatch, _make_discard_result())

    result = runner.invoke(app, ["discard", "--force", "--format", "json"])

    assert result.exit_code == 0
    doc = json.loads(result.stdout)
    assert "budget" not in doc


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


@pytest.fixture
def stop_repo(repo: str) -> str:
    """A repository with a settled, configured session ready for the stop command."""
    return make_stop_repo(repo)


def test_stop_command_when_format_json_does_emit_structured_json_with_at_and_message(
    stop_repo: str,
):
    result = runner.invoke(app, ["stop", "-m", "user requested stop", "--format", "json"])

    assert result.exit_code == 0
    doc = json.loads(result.stdout)
    assert doc["message"] == "user requested stop"
    assert "at" in doc
    assert isinstance(doc["at"], int)


def test_stop_command_when_format_json_and_budget_active_does_include_budget_object(
    stop_repo: str, monkeypatch: pytest.MonkeyPatch
):
    install_budget(stop_repo, monkeypatch)

    result = runner.invoke(app, ["stop", "-m", "done", "--format", "json"])

    assert result.exit_code == 0
    doc = json.loads(result.stdout)
    assert "budget" in doc
    assert doc["budget"]["cap_minutes"] == 30
    assert isinstance(doc["budget"]["remaining_seconds"], int)


def test_stop_command_when_format_json_and_no_budget_does_omit_budget_key(
    stop_repo: str,
):
    result = runner.invoke(app, ["stop", "-m", "done", "--format", "json"])

    assert result.exit_code == 0
    doc = json.loads(result.stdout)
    assert "budget" not in doc


# ---------------------------------------------------------------------------
# start --format json
# ---------------------------------------------------------------------------


def _stub_resolve_config(monkeypatch: pytest.MonkeyPatch, **overrides: object) -> object:
    """Pin what ``start`` reads by replacing its ``resolve_config`` with a fixed config."""
    config = resolved_config(**overrides)

    def fake(*_a: object, **_k: object) -> object:
        return config

    monkeypatch.setattr("gymrat.cli.session_cmds.resolve_config", fake)
    return config


def test_start_command_when_format_json_and_fresh_does_emit_structured_json(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    _stub_resolve_config(monkeypatch)

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
    assert doc["runbook"] is None
    assert doc["archived"] is None


def test_start_command_when_format_json_and_resumed_does_set_resumed_true_with_counts(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    start_session(repo, "main", resolved_config())
    append_record(session_jsonl_path(repo), iteration_record(seq=1))
    append_record(session_jsonl_path(repo), committed_keep(1))
    _stub_resolve_config(monkeypatch)

    result = runner.invoke(app, ["start", "--baseline", "main", "--format", "json"])

    assert result.exit_code == 0
    doc = json.loads(result.stdout)
    assert doc["resumed"] is True
    assert doc["iteration_count"] == 1
    assert doc["keep_count"] == 1


def test_start_command_when_format_json_and_archived_does_include_archived_session_id(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    header = _open_session_with_one_keep(repo)
    from gymrat.loop.finalize import finalize_session

    finalize_session(repo)
    closed_id = header.session_id
    _stub_resolve_config(monkeypatch)

    result = runner.invoke(app, ["start", "--baseline", "main", "--format", "json"])

    assert result.exit_code == 0
    doc = json.loads(result.stdout)
    assert doc["archived"]["session_id"] == closed_id
    assert isinstance(doc["archived"]["path"], str)
    assert doc["resumed"] is False


def test_start_command_when_format_json_and_runbook_configured_does_include_runbook(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    _stub_resolve_config(monkeypatch, runbook=".claude/skills/ecstatic-bench/SKILL.md")

    result = runner.invoke(app, ["start", "--baseline", "main", "--format", "json"])

    assert result.exit_code == 0
    doc = json.loads(result.stdout)
    assert doc["runbook"] == ".claude/skills/ecstatic-bench/SKILL.md"


def test_start_command_when_format_json_does_include_stable_key_names(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    _stub_resolve_config(monkeypatch)

    result = runner.invoke(app, ["start", "--baseline", "main", "--format", "json"])

    assert result.exit_code == 0
    doc = json.loads(result.stdout)
    assert {
        "session_id",
        "branch",
        "baseline",
        "worktrees",
        "resumed",
        "iteration_count",
        "keep_count",
        "runbook",
        "archived",
    } <= doc.keys()


def test_start_command_when_format_text_does_produce_same_output_as_default(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    _stub_resolve_config(monkeypatch)

    result = runner.invoke(app, ["start", "--baseline", "main", "--format", "text"])

    assert result.exit_code == 0
    assert "Started session" in result.stdout or "Resumed session" in result.stdout


# ---------------------------------------------------------------------------
# finalize --format json
# ---------------------------------------------------------------------------


def _open_session_with_one_keep(repo: str) -> SessionRecord:
    """Open a session, commit and log one kept iteration, and return the session header."""
    start_session(repo, "main", resolved_config())
    worktree = experiment_worktree_dir(repo)
    (Path(worktree) / "step.txt").write_text("cache the regex\n", encoding="utf-8")
    git(["add", "-A"], worktree)
    git(["commit", "-m", "cache the regex"], worktree)
    commit = head_of(worktree)
    append_record(session_jsonl_path(repo), iteration_record(seq=1))
    append_record(session_jsonl_path(repo), committed_keep(1, commit=commit))
    header = read_records(session_jsonl_path(repo))[0]
    assert isinstance(header, SessionRecord)
    return header


def test_finalize_command_when_format_json_does_emit_structured_json(
    repo: str,
):
    _open_session_with_one_keep(repo)

    result = runner.invoke(app, ["finalize", "--format", "json"])

    assert result.exit_code == 0
    doc = json.loads(result.stdout)
    assert "branch" in doc
    assert doc["branch"].endswith("-final")
    assert "commit" in doc
    assert isinstance(doc["commit"], str)
    assert len(doc["commit"]) == 40
    assert "message" in doc
    assert "at" in doc
    assert isinstance(doc["at"], int)


def test_finalize_command_when_format_json_and_message_given_does_include_message(
    repo: str,
):
    _open_session_with_one_keep(repo)

    result = runner.invoke(app, ["finalize", "-m", "squash the tuning session", "--format", "json"])

    assert result.exit_code == 0
    doc = json.loads(result.stdout)
    assert doc["message"] == "squash the tuning session"


def test_finalize_command_when_format_json_does_include_stable_key_names(
    repo: str,
):
    _open_session_with_one_keep(repo)

    result = runner.invoke(app, ["finalize", "--format", "json"])

    assert result.exit_code == 0
    doc = json.loads(result.stdout)
    assert {"branch", "commit", "message", "at"} <= doc.keys()


def test_finalize_command_when_format_text_does_produce_same_output_as_default(
    repo: str,
):
    _open_session_with_one_keep(repo)

    result = runner.invoke(app, ["finalize", "--format", "text"])

    assert result.exit_code == 0
    assert "Finalized" in result.stdout


# ---------------------------------------------------------------------------
# sync --format json
# ---------------------------------------------------------------------------


@pytest.fixture
def sync_repo(repo: str) -> str:
    """A repository with an open session, ready for sync tests."""
    start_session(repo, "main", resolved_config())
    return repo


def test_sync_command_when_format_json_and_files_synced_does_emit_files_array(
    sync_repo: str,
):
    root = Path(sync_repo)
    (root / "extra.py").write_text("# new\n", encoding="utf-8")

    result = runner.invoke(app, ["sync", "--format", "json"])

    assert result.exit_code == 0
    doc = json.loads(result.stdout)
    assert isinstance(doc["files"], list)
    assert len(doc["files"]) > 0
    assert "extra.py" in doc["files"]


def test_sync_command_when_format_json_and_nothing_to_sync_does_emit_empty_files_array(
    sync_repo: str,
):
    result = runner.invoke(app, ["sync", "--format", "json"])

    assert result.exit_code == 0
    doc = json.loads(result.stdout)
    assert doc["files"] == []


def test_sync_command_when_format_json_does_include_stable_key_names(
    sync_repo: str,
):
    result = runner.invoke(app, ["sync", "--format", "json"])

    assert result.exit_code == 0
    doc = json.loads(result.stdout)
    assert "files" in doc


def test_sync_command_when_format_text_does_produce_identical_output(
    sync_repo: str,
):
    result_text = runner.invoke(app, ["sync", "--format", "text"])
    result_default = runner.invoke(app, ["sync"])

    assert result_text.exit_code == 0
    assert result_default.exit_code == 0
    assert result_text.stdout == result_default.stdout


# ---------------------------------------------------------------------------
# budget key — start, finalize, sync JSON output
# ---------------------------------------------------------------------------


def test_start_command_when_format_json_and_budget_active_does_include_budget_object(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    _stub_resolve_config(monkeypatch)
    Path(repo, ".gymrat").mkdir(exist_ok=True)
    install_budget(repo, monkeypatch)

    result = runner.invoke(app, ["start", "--baseline", "main", "--format", "json"])

    assert result.exit_code == 0
    doc = json.loads(result.stdout)
    assert "budget" in doc
    assert doc["budget"]["cap_minutes"] == 30
    assert isinstance(doc["budget"]["remaining_seconds"], int)


def test_start_command_when_format_json_and_no_budget_does_omit_budget_key(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    _stub_resolve_config(monkeypatch)

    result = runner.invoke(app, ["start", "--baseline", "main", "--format", "json"])

    assert result.exit_code == 0
    doc = json.loads(result.stdout)
    assert "budget" not in doc


def test_finalize_command_when_format_json_and_budget_active_does_include_budget_object(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    _open_session_with_one_keep(repo)
    install_budget(repo, monkeypatch)

    result = runner.invoke(app, ["finalize", "--format", "json"])

    assert result.exit_code == 0
    doc = json.loads(result.stdout)
    assert "budget" in doc
    assert doc["budget"]["cap_minutes"] == 30
    assert isinstance(doc["budget"]["remaining_seconds"], int)


def test_finalize_command_when_format_json_and_no_budget_does_omit_budget_key(
    repo: str,
):
    _open_session_with_one_keep(repo)

    result = runner.invoke(app, ["finalize", "--format", "json"])

    assert result.exit_code == 0
    doc = json.loads(result.stdout)
    assert "budget" not in doc


def test_sync_command_when_format_json_and_budget_active_does_include_budget_object(
    sync_repo: str, monkeypatch: pytest.MonkeyPatch
):
    install_budget(sync_repo, monkeypatch)

    result = runner.invoke(app, ["sync", "--format", "json"])

    assert result.exit_code == 0
    doc = json.loads(result.stdout)
    assert "budget" in doc
    assert doc["budget"]["cap_minutes"] == 30
    assert isinstance(doc["budget"]["remaining_seconds"], int)


def test_sync_command_when_format_json_and_no_budget_does_omit_budget_key(
    sync_repo: str,
):
    result = runner.invoke(app, ["sync", "--format", "json"])

    assert result.exit_code == 0
    doc = json.loads(result.stdout)
    assert "budget" not in doc
