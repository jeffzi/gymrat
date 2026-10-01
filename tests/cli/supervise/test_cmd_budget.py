"""Budget lifecycle tests for the ``gymrat supervise`` command.

Shares its seam-installation harness with :mod:`tests.cli.supervise.test_cmd`,
which owns the ``CliRunner`` wiring these tests reuse.
"""

from collections.abc import Callable
from pathlib import Path

import pytest

from gymrat.clock import now_ms, now_ns
from gymrat.errors import GymratError
from gymrat.loop.start import StartResult
from gymrat.session.budget import Budget, clear_budget, read_budget, write_budget
from gymrat.session.paths import budget_path, session_jsonl_path
from gymrat.session.store import append_record
from gymrat.supervisor import SupervisionResult
from tests.cli.supervise._fixtures import baseline_record, make_supervision_result
from tests.cli.supervise.test_cmd import (
    _CAP_MINUTES,
    _CAP_MS,
    _err_text,
    _install_seams,
    _make_start_result,
    _run,
    _Seams,
)

# ---------------------------------------------------------------------------
# budget lifecycle
# ---------------------------------------------------------------------------


def test_supervise_when_run_does_write_budget_before_supervise(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
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
    captured_budgets: list[Budget] = []

    def capturing_write(root: str, budget: Budget) -> None:
        captured_budgets.append(budget)

    seams = _install_seams(monkeypatch)
    baseline_at = now_ns()

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
        seams.preflight_calls.append({
            "root": root,
            "config": config,
            "baseline_ref": baseline_ref,
            "max_minutes": max_minutes,
            "force": force,
        })
        return _make_start_result(root)

    monkeypatch.setattr("gymrat.cli.supervise.cmd.run_preflight", fake_preflight_with_baseline)
    monkeypatch.setattr("gymrat.cli.supervise.cmd.write_budget", capturing_write)

    result = _run("optimize it", "--max-minutes", str(_CAP_MINUTES))

    assert result.exit_code == 0
    assert len(captured_budgets) == 1
    baseline_epoch_ms = baseline_at // 1_000_000
    assert captured_budgets[0].started_at_ms >= baseline_epoch_ms


def _record_budget_release(
    monkeypatch: pytest.MonkeyPatch, seams: _Seams, *, clear_error: Exception | None = None
) -> tuple[list[str], list[Callable[[], None]]]:
    """Record the budget clear and every registered cleanup's uninstall, tagged by index.

    The run also registers the process-kill cleanup through the same seam; its
    uninstall is tagged too so a test can filter the log down to the budget one,
    identified after the run by :func:`_budget_uninstall_tag` rather than by its
    position in the registration order.
    """
    events: list[str] = []
    installed: list[Callable[[], None]] = []

    def fake_install(cleanup: Callable[[], None]) -> Callable[[], None]:
        tag = f"uninstall-{len(installed)}"
        installed.append(cleanup)
        return lambda: events.append(tag)

    def fake_clear(root: str) -> None:
        events.append("clear")
        if clear_error is not None:
            raise clear_error

    seams.install_cleanup.side_effect = fake_install
    monkeypatch.setattr("gymrat.cli.supervise.cmd.clear_budget", fake_clear)
    return events, installed


def _budget_uninstall_tag(
    repo: str, monkeypatch: pytest.MonkeyPatch, installed: list[Callable[[], None]]
) -> str:
    """The uninstall tag of whichever installed cleanup clears the budget file.

    Restores the real ``clear_budget`` so each candidate can be invoked against a
    probe budget file and identified by its effect, not its registration order.
    """
    monkeypatch.setattr("gymrat.cli.supervise.cmd.clear_budget", clear_budget)
    write_budget(repo, Budget(started_at_ms=0.0, max_minutes=10, deadline_ms=600_000.0))
    for index, cleanup in enumerate(installed):
        cleanup()
        if not Path(budget_path(repo)).exists():
            return f"uninstall-{index}"
    msg = "no installed cleanup removed the budget file"
    raise AssertionError(msg)


@pytest.mark.parametrize(
    ("raises", "expected_exit"),
    [
        pytest.param(None, 0, id="run-completes"),
        pytest.param(GymratError("boom"), 2, id="supervise-raises"),
    ],
)
def test_supervise_when_run_ends_does_clear_budget_then_uninstall_its_cleanup_once(
    repo: str, monkeypatch: pytest.MonkeyPatch, raises: Exception | None, expected_exit: int
):
    seams = _install_seams(monkeypatch, raises=raises)
    events, installed = _record_budget_release(monkeypatch, seams)

    result = _run("optimize it", "--max-minutes", "10")

    assert result.exit_code == expected_exit
    budget_tag = _budget_uninstall_tag(repo, monkeypatch, installed)
    assert [event for event in events if event in ("clear", budget_tag)] == ["clear", budget_tag]


def test_supervise_when_clear_budget_raises_does_still_uninstall_budget_cleanup(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    seams = _install_seams(monkeypatch)
    events, installed = _record_budget_release(
        monkeypatch, seams, clear_error=OSError("budget file locked")
    )

    _run("optimize it", "--max-minutes", "10")

    budget_tag = _budget_uninstall_tag(repo, monkeypatch, installed)
    assert [event for event in events if event in ("clear", budget_tag)] == ["clear", budget_tag]


def test_supervise_when_clear_budget_raises_does_exit_two_naming_the_error(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    seams = _install_seams(monkeypatch)
    _record_budget_release(monkeypatch, seams, clear_error=OSError("budget file locked"))

    result = _run("optimize it", "--max-minutes", "10")

    assert result.exit_code == 2
    assert "Error: budget file locked" in _err_text(result)


def test_supervise_when_run_does_clear_budget_before_stopping_reporter(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
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
    seams = _install_seams(monkeypatch)

    _run("optimize it", "--max-minutes", "10")
    write_budget(repo, Budget(started_at_ms=0.0, max_minutes=10, deadline_ms=600_000.0))

    for cleanup in seams.installed_cleanups():
        cleanup()

    assert not Path(budget_path(repo)).exists()
