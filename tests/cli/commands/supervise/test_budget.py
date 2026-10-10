"""Budget lifecycle tests for the ``gymrat supervise`` command.

Drives the command through the seam kit in :mod:`tests.cli.commands.supervise._seams`,
which the other ``gymrat supervise`` command tests share.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import create_autospec

import pytest

from gymrat.cli.supervise.preflight import PreflightFlags, run_preflight
from gymrat.clock import now_ms
from gymrat.errors import GymratError
from gymrat.exec import kill_live_process_groups
from gymrat.loop.start import StartResult
from gymrat.session.budget import Budget, clear_budget, read_budget
from gymrat.session.paths import budget_path
from gymrat.supervisor.supervise import SupervisedSession
from tests._cli import err_text
from tests._process_helpers import CleanupRegistry, track_cleanups
from tests.cli.commands.supervise._seams import (
    CAP_MINUTES,
    CAP_MS,
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
    seen: list[tuple[Budget | None, object]] = []
    seams.supervise_hook = lambda call: seen.append((
        read_budget(repo, now_ms=now_ms()),
        call["context"],
    ))
    earliest_start_ms = now_ms()

    result = run("optimize it", "--max-minutes", str(CAP_MINUTES))

    latest_start_ms = now_ms()
    assert result.exit_code == 0
    ((budget, context),) = seen
    assert budget is not None
    assert isinstance(context, SupervisedSession)
    assert budget.max_minutes == CAP_MINUTES
    assert earliest_start_ms + CAP_MS <= budget.deadline_ms <= latest_start_ms + CAP_MS
    assert context.budget == budget


def test_supervise_when_preflight_records_baseline_does_start_the_budget_once_it_is_recorded(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    seams = install_seams(monkeypatch)
    clock_ms = [1_000_000]
    seen_budgets: list[Budget | None] = []
    seams.supervise_hook = lambda _call: seen_budgets.append(read_budget(repo, now_ms=clock_ms[0]))
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
    assert [budget.deadline_ms if budget else None for budget in seen_budgets] == [
        1_060_000 + CAP_MS
    ]


def _record_hooks_armed_at_clear(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[CleanupRegistry, list[list[Callable[[], None]]]]:
    """Track the run's cleanups and snapshot the budget hooks still armed at each budget clear.

    The process-kill cleanup is left out of each snapshot, so a snapshot holds
    only the budget's own termination hook while that hook is armed. Each clear
    then goes on to the real budget clear with the arguments the run passed.

    Args:
        monkeypatch: The fixture the cleanup installer and budget clear are replaced through.

    Returns:
        The registry of the run's armed cleanups, and one snapshot per budget clear.
    """
    registry = track_cleanups(monkeypatch, "gymrat.cli.commands.supervise")
    armed_at_clear: list[list[Callable[[], None]]] = []

    def snapshot_then_clear(*args: Any, **kwargs: Any) -> None:
        armed_at_clear.append([
            live for live in registry.live() if live is not kill_live_process_groups
        ])
        clear_budget(*args, **kwargs)

    monkeypatch.setattr(
        "gymrat.cli.commands.supervise.clear_budget",
        create_autospec(clear_budget, side_effect=snapshot_then_clear),
    )
    return registry, armed_at_clear


@pytest.mark.parametrize(
    ("raises", "expected_exit"),
    [
        pytest.param(None, 0, id="run-completes"),
        pytest.param(GymratError("boom"), 2, id="supervise-raises"),
    ],
)
def test_supervise_when_run_ends_does_clear_budget_then_uninstall_its_cleanups_once(
    repo: str, monkeypatch: pytest.MonkeyPatch, raises: Exception | None, expected_exit: int
):
    seams = install_seams(monkeypatch, raises=raises)
    registry, armed_at_clear = _record_hooks_armed_at_clear(monkeypatch)
    kill_armed_during_run: list[bool] = []
    seams.supervise_hook = lambda _call: kill_armed_during_run.append(
        kill_live_process_groups in registry.live()
    )

    result = run("optimize it", "--max-minutes", "10")

    assert result.exit_code == expected_exit
    assert kill_armed_during_run == [True]
    assert [len(armed) for armed in armed_at_clear] == [1]
    assert registry.live() == []


def _block_budget_removal(repo: str) -> None:
    """Swap the run's budget file for a directory, which the OS refuses to unlink."""
    budget_file = Path(budget_path(repo))
    budget_file.unlink()
    budget_file.mkdir()


def test_supervise_when_os_refuses_budget_removal_does_warn_through_the_reporter_and_end_as_usual(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    seams = install_seams(monkeypatch)
    registry, armed_at_clear = _record_hooks_armed_at_clear(monkeypatch)
    seams.supervise_hook = lambda _call: _block_budget_removal(repo)

    result = run("optimize it", "--max-minutes", "10")

    assert result.exit_code == 0, err_text(result)
    assert "  exit    nothing to settle" in result.stdout.splitlines()
    assert len(seams.exit_calls) == 1
    assert [len(armed) for armed in armed_at_clear] == [1]
    assert registry.live() == []
    assert registry.uninstall_counts() == [1, 1]
    assert [
        warning.startswith(f"warning: could not remove the budget file {budget_path(repo)}: ")
        for warning in seams.warnings
    ] == [True]
    assert "could not remove the budget file" not in err_text(result)
