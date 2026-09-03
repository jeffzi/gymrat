"""JSON contract tests for keep, discard, and status ``--format json``."""

import json

import pytest

from gymrat.cli.app import app
from gymrat.loop.settle import DiscardResult, KeepResult
from gymrat.loop.start import start_session
from gymrat.session import (
    DiscardRecord,
    FinalizeRecord,
    KeepChecks,
    KeepRecord,
    append_record,
    session_jsonl_path,
)
from tests.cli._loop_cmds import make_discard_repo, never_tty, runner, write_config
from tests.loop.iterate._fixtures import resolved_config
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
    assert doc["checks"]["stdoutBytes"] == 80
    assert doc["checks"]["stderrBytes"] == 0


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
    assert {"configured", "passed", "stdoutBytes", "stderrBytes"} <= doc["checks"].keys()


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


@pytest.fixture
def status_repo(repo: str) -> str:
    """A repository with a configured session and one kept iteration."""
    write_session_log(repo, session_record(), (iteration_record(seq=1), committed_keep(1)))
    write_config(repo)
    return repo


def test_status_command_when_format_json_does_emit_structured_json_on_stdout(status_repo: str):
    result = runner.invoke(app, ["status", "--format", "json"])

    assert result.exit_code == 0
    doc = json.loads(result.stdout)
    assert doc["sessionId"] == SESSION_ID
    assert doc["branch"] == f"gymrat/{SESSION_ID}"
    assert doc["baseline"]["ref"] == "main"
    assert doc["baseline"]["sha"] == "a" * 40
    assert doc["iterationCount"] == 1
    assert doc["keepCount"] == 1
    assert doc["discardCount"] == 0
    assert doc["unsettled"] is False
    assert doc["finalized"] is False


def test_status_command_when_format_json_and_finalized_does_set_finalized_true(repo: str):
    write_session_log(
        repo,
        session_record(),
        (
            iteration_record(seq=1),
            committed_keep(1),
            FinalizeRecord(
                type="finalize",
                at=AT,
                branch=f"gymrat/{SESSION_ID}-final",
                commit=COMMIT,
                message="squash 1 kept iteration",
            ),
        ),
    )
    write_config(repo)

    result = runner.invoke(app, ["status", "--format", "json"])

    assert result.exit_code == 0
    doc = json.loads(result.stdout)
    assert doc["finalized"] is True


def test_status_command_when_format_json_does_include_stable_key_names(status_repo: str):
    result = runner.invoke(app, ["status", "--format", "json"])

    assert result.exit_code == 0
    doc = json.loads(result.stdout)
    assert {
        "sessionId",
        "branch",
        "baseline",
        "iterationCount",
        "keepCount",
        "discardCount",
        "unsettled",
        "finalized",
    } <= doc.keys()
    assert {"ref", "sha"} <= doc["baseline"].keys()


def test_status_command_when_format_text_does_produce_identical_output(status_repo: str):
    result_text = runner.invoke(app, ["status", "--format", "text"])
    result_default = runner.invoke(app, ["status"])

    assert result_text.exit_code == 0
    assert result_default.exit_code == 0
    assert result_text.stdout == result_default.stdout
