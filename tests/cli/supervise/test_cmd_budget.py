"""Budget lifecycle tests for the ``gymrat supervise`` command.

Shares its seam-installation harness with :mod:`tests.cli.supervise.test_cmd`,
which owns the ``CliRunner`` wiring these tests reuse.
"""

from datetime import datetime
from pathlib import Path

import pytest

from gymrat.errors import GymratError
from gymrat.loop.start import StartResult
from gymrat.session import append_record
from gymrat.session.budget import Budget, read_budget, write_budget
from gymrat.session.clock import now_iso, now_ms
from gymrat.session.paths import budget_path, session_jsonl_path
from gymrat.supervisor import SupervisionResult
from tests.cli.supervise._fixtures import baseline_record, make_supervision_result
from tests.cli.supervise.test_cmd import (
    _CAP_MINUTES,
    _CAP_MS,
    _install_seams,
    _make_start_result,
    _run,
)

# ---------------------------------------------------------------------------
# budget lifecycle
# ---------------------------------------------------------------------------


def test_supervise_when_run_does_write_budget_before_supervise(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    """Budget file must exist and be live before the agent's first turn."""
    seams = _install_seams(monkeypatch)
    seen_budgets: list[Budget | None] = []

    async def probing_supervise(*args: object, **kwargs: object) -> SupervisionResult:
        seams.record_supervise_call(args, kwargs)
        seen_budgets.append(read_budget(repo, now_ms=now_ms()))
        return make_supervision_result()

    monkeypatch.setattr("gymrat.cli.supervise.cmd.supervise", probing_supervise)

    result = _run("optimize it", "--max-minutes", "10")

    assert result.exit_code == 0
    assert len(seen_budgets) == 1
    assert seen_budgets[0] is not None


def test_supervise_when_run_does_write_budget_with_correct_deadline(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    _install_seams(monkeypatch)
    captured_budgets: list[Budget] = []

    def capturing_write(root: str, budget: Budget) -> None:
        captured_budgets.append(budget)

    monkeypatch.setattr("gymrat.cli.supervise.cmd.write_budget", capturing_write)

    result = _run("optimize it", "--max-minutes", str(_CAP_MINUTES))

    assert result.exit_code == 0
    assert len(captured_budgets) == 1
    budget = captured_budgets[0]
    assert budget.max_minutes == _CAP_MINUTES
    expected_deadline = budget.started_at_ms + _CAP_MS
    assert budget.deadline_ms == expected_deadline


def test_supervise_when_preflight_records_baseline_does_start_budget_no_earlier(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    """Budget started_at_ms must be >= the baseline record's at in epoch ms."""
    captured_budgets: list[Budget] = []

    def capturing_write(root: str, budget: Budget) -> None:
        captured_budgets.append(budget)

    seams = _install_seams(monkeypatch)
    baseline_at = now_iso()

    def fake_preflight_with_baseline(
        *,
        root: str,
        config: object,
        baseline_ref: object = None,
        max_minutes: float,
        force: bool,
    ) -> StartResult:
        record = baseline_record(at=baseline_at)
        append_record(session_jsonl_path(root), record)
        seams.preflight_calls.append(
            {
                "root": root,
                "config": config,
                "baseline_ref": baseline_ref,
                "max_minutes": max_minutes,
                "force": force,
            }
        )
        return _make_start_result(root)

    monkeypatch.setattr("gymrat.cli.supervise.cmd.run_preflight", fake_preflight_with_baseline)
    monkeypatch.setattr("gymrat.cli.supervise.cmd.write_budget", capturing_write)

    result = _run("optimize it", "--max-minutes", str(_CAP_MINUTES))

    assert result.exit_code == 0
    assert len(captured_budgets) == 1
    baseline_epoch_ms = int(datetime.fromisoformat(baseline_at).timestamp() * 1000)
    assert captured_budgets[0].started_at_ms >= baseline_epoch_ms


def test_supervise_when_run_completes_does_clear_budget(repo: str, monkeypatch: pytest.MonkeyPatch):
    _install_seams(monkeypatch)
    cleared: list[str] = []
    monkeypatch.setattr("gymrat.cli.supervise.cmd.clear_budget", cleared.append)

    result = _run("optimize it", "--max-minutes", "10")

    assert result.exit_code == 0
    assert len(cleared) == 1


def test_supervise_when_supervise_raises_does_still_clear_budget(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    _install_seams(monkeypatch, raises=GymratError("boom"))
    cleared: list[str] = []
    monkeypatch.setattr("gymrat.cli.supervise.cmd.clear_budget", cleared.append)

    result = _run("optimize it", "--max-minutes", "10")

    assert result.exit_code == 2
    assert len(cleared) >= 1


def test_supervise_when_run_does_clear_budget_before_stopping_reporter(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    """Budget must be cleared before the reporter stops."""
    seams = _install_seams(monkeypatch)
    budget_gone_at_stop: list[bool] = []

    def probing_stop() -> None:
        budget_gone_at_stop.append(not Path(budget_path(repo)).exists())

    seams.reporter_stop.side_effect = probing_stop

    result = _run("optimize it", "--max-minutes", "10")

    assert result.exit_code == 0
    assert budget_gone_at_stop == [True]


def test_supervise_when_run_does_register_budget_termination_cleanup(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    """A termination signal (SIGTERM, SIGINT) must clear the budget file."""
    seams = _install_seams(monkeypatch)

    _run("optimize it", "--max-minutes", "10")
    (registered,) = seams.install_cleanup.call_args_list[1].args
    write_budget(repo, Budget(started_at_ms=0.0, max_minutes=10, deadline_ms=600_000.0))
    registered()

    assert not Path(budget_path(repo)).exists()
