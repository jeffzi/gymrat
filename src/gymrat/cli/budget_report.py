"""Session budget reporting for the CLI: report trailers and over-budget warnings.

The report writers read the live session budget here to append a time-left
trailer to the text report or a ``budget`` object to the JSON document, and the
benchmarking commands warn here when the last full measurement would outlast
what is left of the budget.
"""

from collections.abc import Callable

from gymrat import clock as _clock
from gymrat.cli.shared import write_stdout
from gymrat.errors import GymratError
from gymrat.eta import format_duration
from gymrat.report.json_doc import BudgetSummary
from gymrat.session.budget import (
    SIDES_PER_ITERATE,
    Budget,
    estimate_iterate_duration,
    format_budget_trailer,
    read_budget,
)
from gymrat.session.paths import repo_root, session_jsonl_path
from gymrat.session.store import read_records
from gymrat.warn import warn_to_stderr


def budget_summary_of(budget: Budget, current_ms: float) -> BudgetSummary:
    """The JSON ``budget`` field for an active budget at ``current_ms``.

    Args:
        budget: The active session budget.
        current_ms: The current time, in epoch milliseconds.

    Returns:
        The budget cap and the whole seconds left at ``current_ms``.
    """
    return BudgetSummary(
        cap_minutes=budget.max_minutes,
        remaining_seconds=int(budget.remaining_ms(current_ms) // 1000),
    )


def budget_snapshot(root: str) -> tuple[str, BudgetSummary | None]:
    """Read the live budget at *root* and return the text trailer and JSON summary.

    Args:
        root: The repository root whose session budget to read.

    Returns:
        The text trailer and JSON summary, or ``("", None)`` when no budget
        is active.
    """
    current = _clock.now_ms()
    budget = read_budget(root, now_ms=current)
    if budget is None:
        return "", None
    return "\n" + format_budget_trailer(budget, current), budget_summary_of(budget, current)


def write_budget_report(
    root: str,
    *,
    use_json: bool,
    render_json: Callable[[BudgetSummary | None], str],
    text_report: str,
) -> None:
    """Render the JSON or text report from a single budget read, then write it once.

    Args:
        root: The repository root whose session budget to read.
        use_json: When true, delegate to *render_json*; otherwise concatenate
            *text_report* with the budget trailer.
        render_json: Callable that turns an optional ``BudgetSummary`` into a
            complete JSON string.
        text_report: Pre-rendered text body used in plain-text mode.
    """
    trailer, summary = budget_snapshot(root)
    report = render_json(summary) if use_json else text_report + trailer
    write_stdout(report + "\n")


def _repo_root_or_none() -> str | None:
    """The current repository root, or ``None`` outside a git repo or on a read failure."""
    try:
        return repo_root()
    except (GymratError, OSError):
        return None


def budget_for_report() -> tuple[str, BudgetSummary | None]:
    """Read the live budget rooted at the current repository.

    Returns:
        The text trailer and JSON summary, or ``("", None)`` outside a git
        repository or when no budget is active.
    """
    root = _repo_root_or_none()
    if root is None:
        return "", None
    return budget_snapshot(root)


def warn_duration_over_budget(*, halve: bool) -> None:
    """Warn on stderr when the estimated duration would outlast the budget.

    Nothing is written when the budget or the estimate is unknown.

    Args:
        halve: When ``True``, check half the last iterate estimate — one side,
            the shape ``measure`` runs — and name that per-side figure on its
            own. When ``False``, check the full estimate — both sides, the
            shape ``compare`` runs — and lead with the full cost, keeping the
            per-side figure in parentheses so a per-side number that still
            fits does not read as if nothing were wrong.
    """
    root = _repo_root_or_none()
    if root is None:
        return
    current = _clock.now_ms()
    budget = read_budget(root, now_ms=current)
    if budget is None:
        return
    try:
        records = read_records(session_jsonl_path(root))
    except (GymratError, OSError):
        return
    estimate = estimate_iterate_duration(records)
    if estimate is None:
        return
    per_side_ms = estimate.duration_ms / SIDES_PER_ITERATE
    threshold_ms = per_side_ms if halve else estimate.duration_ms
    remaining = budget.remaining_ms(current)
    if threshold_ms > remaining:
        per_side = format_duration(per_side_ms)
        cost = (
            f"{per_side} per side"
            if halve
            else f"{format_duration(estimate.duration_ms)} ({per_side} per side)"
        )
        left = format_duration(remaining)
        warn_to_stderr(f"warning: {left} left; the last full measurement took at most {cost}")
