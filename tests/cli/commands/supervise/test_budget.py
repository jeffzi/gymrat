"""Budget lifecycle tests for the ``gymrat supervise`` command.

Drives the command through the seam kit in :mod:`tests.cli.commands.supervise._seams`,
which the other ``gymrat supervise`` command tests share.
"""

from collections.abc import Callable
from pathlib import Path

import pytest

from gymrat.cli.supervise.preflight import PreflightFlags
from gymrat.clock import now_ms
from gymrat.errors import GymratError
from gymrat.loop.start import StartResult
from gymrat.session.budget import Budget, clear_budget, read_budget, write_budget
from gymrat.session.paths import budget_path
from gymrat.supervisor.supervise import SupervisionResult
from tests.cli.commands.supervise._seams import (
    CAP_MINUTES,
    CAP_MS,
    Seams,
    err_text,
    install_seams,
    make_start_result,
    patch_supervise,
    run,
)
from tests.cli.supervise._fixtures import make_supervision_result
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

    async def probing_supervise(*args: object, **kwargs: object) -> SupervisionResult:
        seams.record_supervise_call(args, kwargs)
        seen_budgets.append(read_budget(repo, now_ms=now_ms()))
        return make_supervision_result()

    patch_supervise(monkeypatch, probing_supervise)
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

    monkeypatch.setattr("gymrat.cli.commands.supervise.write_budget", capturing_write)
    return captured_budgets


def test_supervise_when_preflight_records_baseline_does_start_the_budget_once_it_is_recorded(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    install_seams(monkeypatch)
    captured_budgets = _capture_budget_writes(monkeypatch)
    clock_ms = [1_000_000.0]
    monkeypatch.setattr("gymrat.cli.commands.supervise.now_ms", lambda: clock_ms[0])

    def fake_preflight_with_baseline(
        *, root: str, config: object, flags: PreflightFlags
    ) -> StartResult:
        # The baseline bench takes a minute, so a budget started before it ends a minute early.
        clock_ms[0] += 60_000
        append_records(root, baseline_record(at=int(clock_ms[0]) * 1_000_000))
        return make_start_result(root)

    monkeypatch.setattr("gymrat.cli.commands.supervise.run_preflight", fake_preflight_with_baseline)

    result = run("optimize it", "--max-minutes", str(CAP_MINUTES))

    assert result.exit_code == 0
    assert [budget.deadline_ms for budget in captured_budgets] == [1_060_000.0 + CAP_MS]


def _record_budget_release(
    monkeypatch: pytest.MonkeyPatch, seams: Seams, *, clear_error: Exception | None = None
) -> tuple[list[str], list[Callable[[], None]]]:
    """Record the budget clear and every registered cleanup's uninstall, tagged by index.

    The run also registers the process-kill cleanup through the same seam; its
    uninstall is tagged too so a test can filter the log down to the budget one,
    identified after the run by :func:`_budget_uninstall_tag` rather than by its
    position in the registration order.

    Args:
        monkeypatch: The fixture the budget clear is replaced through.
        seams: The installed seams whose cleanup installer is redirected.
        clear_error: An error the budget clear raises after recording itself.

    Returns:
        The event log the clear and every uninstall append to, and the
        cleanups the run registered, in registration order.
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
    monkeypatch.setattr("gymrat.cli.commands.supervise.clear_budget", fake_clear)
    return events, installed


def _budget_uninstall_tag(
    repo: str, monkeypatch: pytest.MonkeyPatch, installed: list[Callable[[], None]]
) -> str:
    """The uninstall tag of whichever installed cleanup clears the budget file.

    Restores the real ``clear_budget`` so each candidate can be invoked against a
    probe budget file and identified by its effect, not its registration order.

    Args:
        repo: The repository whose budget file the probe writes.
        monkeypatch: The fixture the real budget clear is restored through.
        installed: The cleanups the run registered, in registration order.

    Returns:
        The ``uninstall-<index>`` tag of the cleanup that removed the budget file.

    Raises:
        AssertionError: No installed cleanup removed the budget file.
    """
    monkeypatch.setattr("gymrat.cli.commands.supervise.clear_budget", clear_budget)
    write_budget(repo, Budget(max_minutes=10, deadline_ms=600_000.0))
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
    seams = install_seams(monkeypatch, raises=raises)
    events, installed = _record_budget_release(monkeypatch, seams)

    result = run("optimize it", "--max-minutes", "10")

    assert result.exit_code == expected_exit
    budget_tag = _budget_uninstall_tag(repo, monkeypatch, installed)
    assert [event for event in events if event in ("clear", budget_tag)] == ["clear", budget_tag]


def test_supervise_when_clear_budget_raises_does_still_uninstall_the_budget_cleanup(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    seams = install_seams(monkeypatch)
    events, installed = _record_budget_release(
        monkeypatch, seams, clear_error=OSError("budget file locked")
    )

    result = run("optimize it", "--max-minutes", "10")

    assert result.exit_code == 2
    assert "Error: budget file locked" in err_text(result)
    budget_tag = _budget_uninstall_tag(repo, monkeypatch, installed)
    assert [event for event in events if event in ("clear", budget_tag)] == ["clear", budget_tag]
