"""Budget lifecycle tests for the ``gymrat supervise`` command.

Drives the command through the seam kit in :mod:`tests.cli.commands.supervise._seams`,
which the other ``gymrat supervise`` command tests share.
"""

from collections.abc import Callable
from unittest.mock import create_autospec

import pytest

from gymrat.cli.supervise.preflight import PreflightFlags, run_preflight
from gymrat.clock import now_ms
from gymrat.errors import GymratError
from gymrat.exec import kill_live_process_groups
from gymrat.loop.start import StartResult
from gymrat.session.budget import Budget, clear_budget, read_budget, write_budget
from tests._process_helpers import CleanupRegistry, track_cleanups
from tests.cli.commands.supervise._seams import (
    CAP_MINUTES,
    CAP_MS,
    err_text,
    install_seams,
    make_start_result,
    run,
)
from tests.session.records._fixtures import (
    append_records,
    baseline_record,
)

# ---------------------------------------------------------------------------
# budget lifecycle
# ---------------------------------------------------------------------------


def test_supervise_when_run_does_write_the_capped_budget_before_supervise(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    seams = install_seams(monkeypatch)
    seen_budgets: list[Budget | None] = []
    seams.supervise_hook = lambda _call: seen_budgets.append(read_budget(repo, now_ms=now_ms()))
    earliest_start_ms = now_ms()

    result = run("optimize it", "--max-minutes", str(CAP_MINUTES))

    latest_start_ms = now_ms()
    assert result.exit_code == 0
    (budget,) = seen_budgets
    assert budget is not None
    assert budget.max_minutes == CAP_MINUTES
    assert earliest_start_ms + CAP_MS <= budget.deadline_ms <= latest_start_ms + CAP_MS


def _capture_budget_writes(monkeypatch: pytest.MonkeyPatch) -> list[Budget]:
    """Replace the command's budget write with a recorder; return the budgets it receives."""
    captured_budgets: list[Budget] = []

    def capturing_write(root: str, budget: Budget) -> None:
        captured_budgets.append(budget)

    monkeypatch.setattr(
        "gymrat.cli.commands.supervise.write_budget",
        create_autospec(write_budget, side_effect=capturing_write),
    )
    return captured_budgets


def test_supervise_when_preflight_records_baseline_does_start_the_budget_once_it_is_recorded(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    install_seams(monkeypatch)
    captured_budgets = _capture_budget_writes(monkeypatch)
    clock_ms = [1_000_000]
    monkeypatch.setattr(
        "gymrat.cli.commands.supervise.now_ms",
        create_autospec(now_ms, side_effect=lambda: clock_ms[0]),
    )

    def fake_preflight_with_baseline(
        *, root: str, config: object, flags: PreflightFlags
    ) -> StartResult:
        # The baseline bench takes a minute, so a budget started before it ends a minute early.
        clock_ms[0] += 60_000
        append_records(root, baseline_record(at=clock_ms[0] * 1_000_000))
        return make_start_result(root)

    monkeypatch.setattr(
        "gymrat.cli.commands.supervise.run_preflight",
        create_autospec(run_preflight, side_effect=fake_preflight_with_baseline),
    )

    result = run("optimize it", "--max-minutes", str(CAP_MINUTES))

    assert result.exit_code == 0
    assert [budget.deadline_ms for budget in captured_budgets] == [1_060_000 + CAP_MS]


def _record_hooks_armed_at_clear(
    monkeypatch: pytest.MonkeyPatch, *, clear_error: Exception | None = None
) -> tuple[CleanupRegistry, list[list[Callable[[], None]]]]:
    """Track the run's cleanups and snapshot the budget hooks still armed at each budget clear.

    The process-kill cleanup is left out of each snapshot, so a snapshot holds
    only the budget's own termination hook while that hook is armed.

    Args:
        monkeypatch: The fixture the cleanup installer and budget clear are replaced through.
        clear_error: An error the budget clear raises after taking its snapshot.

    Returns:
        The registry of the run's armed cleanups, and one snapshot per budget clear.
    """
    registry = track_cleanups(monkeypatch, "gymrat.cli.commands.supervise")
    armed_at_clear: list[list[Callable[[], None]]] = []

    def fake_clear(root: str) -> None:
        armed_at_clear.append([
            live for live in registry.live() if live is not kill_live_process_groups
        ])
        if clear_error is not None:
            raise clear_error

    monkeypatch.setattr(
        "gymrat.cli.commands.supervise.clear_budget",
        create_autospec(clear_budget, side_effect=fake_clear),
    )
    return registry, armed_at_clear


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
    install_seams(monkeypatch, raises=raises)
    registry, armed_at_clear = _record_hooks_armed_at_clear(monkeypatch)

    result = run("optimize it", "--max-minutes", "10")

    assert result.exit_code == expected_exit
    assert [len(armed) for armed in armed_at_clear] == [1]
    assert registry.live() == []


def test_supervise_when_clear_budget_raises_does_still_uninstall_the_budget_cleanup(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    install_seams(monkeypatch)
    registry, armed_at_clear = _record_hooks_armed_at_clear(
        monkeypatch, clear_error=OSError("budget file locked")
    )

    result = run("optimize it", "--max-minutes", "10")

    assert result.exit_code == 2
    assert "Error: budget file locked" in err_text(result)
    assert [len(armed) for armed in armed_at_clear] == [1]
    assert registry.live() == []
