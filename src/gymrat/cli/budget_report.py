"""Report output for the CLI: the stdout report, its budget trailer, over-budget warnings.

The report writers read the live session budget here to append a time-left
trailer to the text report or a ``budget`` object to the JSON document, and the
benchmarking commands warn here when the last full measurement would outlast
what is left of the budget.
"""

import sys
from collections.abc import Callable
from dataclasses import replace
from typing import Protocol

from gymrat import clock as _clock
from gymrat.cli import console
from gymrat.cli.exit import write_stdout
from gymrat.cli.options import OutputFormat
from gymrat.cli.run_setup import SharedFlags
from gymrat.errors import GymratError
from gymrat.eta import MS_PER_SECOND, format_duration
from gymrat.report.json_doc import BudgetSummary
from gymrat.report.types import ReportOptions
from gymrat.session.budget import (
    SIDES_PER_ITERATE,
    estimate_iterate_duration,
    read_budget,
)
from gymrat.session.paths import repo_root, session_jsonl_path
from gymrat.session.store import read_records
from gymrat.warn import warn_to_stderr


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
    remaining_ms = budget.remaining_ms(current)
    summary = BudgetSummary(
        cap_minutes=budget.max_minutes,
        remaining_seconds=int(remaining_ms // MS_PER_SECOND),
    )
    return f"\n{format_duration(remaining_ms)} left of {budget.max_minutes:g}m", summary


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

    Raises:
        OSError: When the stdout write fails for any reason other than a closed pipe.
    """
    trailer, summary = budget_snapshot(root)
    report = render_json(summary) if use_json else text_report + trailer
    write_stdout(report + "\n")


class JsonRenderer[T](Protocol):
    """A JSON renderer for a report result type.

    Implementations omit the ``budget`` field from the document entirely when *budget*
    is ``None``, rather than serializing it as ``null``.
    """

    def __call__(self, result: T, /, *, budget: BudgetSummary | None = None) -> str:
        """Render *result* as a JSON document.

        Args:
            result: The report result to serialize.
            budget: The session budget summary to embed, or ``None`` to omit the field.

        Returns:
            The JSON document text.
        """
        ...


def wants_json(flags: SharedFlags) -> bool:
    """Whether ``flags.format`` selects the JSON document rather than the text report."""
    return flags.format == OutputFormat.json.value


def _repo_root_or_none() -> str | None:
    """The current repository root, or ``None`` outside a git repo or on a read failure."""
    try:
        return repo_root()
    except (GymratError, OSError):
        return None


def emit_report[T](
    result: T,
    flags: SharedFlags,
    render_opts: ReportOptions,
    *,
    text: Callable[[T, ReportOptions], str],
    json: JsonRenderer[T],
) -> None:
    """Render ``result`` per ``flags.format`` and write it to stdout with the live budget.

    The budget is read from the session at the current repository; outside a
    git repository, or with no budget active, the report carries none.

    The text renderer captures into an in-memory buffer that cannot see the real
    stdout, so a deferred color choice (``render_opts.color is None``) is
    resolved here against stdout's own TTY state through the shared precedence:
    a report printed to a terminal is styled, the same report piped away is
    plain. An explicit ``--color`` / ``--no-color`` veto in ``render_opts.color``
    is passed through untouched. The JSON document is never styled, so it ignores
    ``render_opts`` entirely.

    Args:
        result: The typed result to render.
        flags: CLI flags selecting ``text`` or ``json`` output format.
        render_opts: Styling options forwarded to the text renderer.
        text: Renders the text report, which the time-left trailer follows.
        json: Renders the JSON document, which embeds the budget summary.

    Raises:
        OSError: When the stdout write fails for any reason other than a closed pipe.
    """
    root = _repo_root_or_none()
    trailer, summary = budget_snapshot(root) if root is not None else ("", None)
    if wants_json(flags):
        write_stdout(json(result, budget=summary) + "\n")
        return
    color = console.resolve_stream_color(render_opts.color, sys.stdout)
    write_stdout(text(result, replace(render_opts, color=color)) + trailer + "\n")


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
