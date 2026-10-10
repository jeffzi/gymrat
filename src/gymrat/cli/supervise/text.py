"""Rich-free text the supervise reducer and its view layer both render.

The reducer folds the loop summary into its state as a plain string, so it must
stay clear of the Rich view layer; :mod:`gymrat.cli.supervise.frame` renders the
same summary with styles.  Both read :func:`loop_segments`, which keeps the two
from ever disagreeing about what the summary says.  Segments carry a style
*role* rather than a theme style, because the theme lives behind a Rich import.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from gymrat.report.format import format_percent_delta
from gymrat.utils import MISSING_DELTA, StyledSegment, pluralize

if TYPE_CHECKING:
    from gymrat.cli.supervise.types import Exiting
    from gymrat.session.schema import Outcome
    from gymrat.session.store import ReadSessionResult
    from gymrat.supervisor.exit_sequence import ExitPhase

NO_SESSION_TEXT = "no session yet"
"""Shown in place of the loop summary before any session data has been read."""

type LoopStyle = Literal["alert", "count", "done", "meta", "pending", "plain", "regressed"]
"""The style role of a loop-summary segment; ``"plain"`` means unstyled."""


def exit_phase_text(phase: ExitPhase | Exiting) -> str:
    """The exit-sequence phase line both dashboard modes show, without elapsed time.

    Args:
        phase: The phase the run-end exit sequence is in.

    Returns:
        ``settling…`` while settling, otherwise the lock wait naming the holder's
        PID, or ``PID unknown`` when the holder cannot be read.
    """
    if phase.kind == "settling":
        return "settling…"
    holder = "unknown" if phase.pid is None else str(phase.pid)
    return f"waiting for gymrat (PID {holder})"


def _iter_label(count: int, max_iterations: int | None) -> str:
    if max_iterations is not None:
        # The noun agrees with the cap, so the capped form stays plural at any count.
        return f"{count}/{max_iterations} iterations"
    return pluralize(count, "iteration")


def _outcome_role(outcome: Outcome) -> LoopStyle:
    if outcome == "improved":
        return "done"
    if outcome == "regressed":
        return "regressed"
    return "meta"


def _last_iteration_segments(
    delta_pct: float | None, outcome: Outcome, *, unsettled: bool
) -> list[StyledSegment[LoopStyle]]:
    delta = format_percent_delta(delta_pct, missing=MISSING_DELTA)
    role = _outcome_role(outcome)
    segments: list[StyledSegment[LoopStyle]] = [
        StyledSegment(" · last ", "plain"),
        StyledSegment(delta, role),
        StyledSegment(" ", "plain"),
        StyledSegment(outcome, role),
    ]
    if unsettled:
        segments.append(StyledSegment(", unsettled", "alert"))
    return segments


def loop_segments(
    session_result: ReadSessionResult | None, max_iterations: int | None
) -> tuple[StyledSegment[LoopStyle], ...]:
    """The iteration-progress summary, split into styled runs of text.

    Args:
        session_result: The latest session read, or ``None`` when the session
            file has not been created yet.
        max_iterations: The configured iteration cap, shown as ``N/M`` in the
            label.  ``None`` when uncapped.

    Returns:
        The segments to concatenate, in order: pending when no session exists, a
        baseline-only note, a finalized marker, or a kept/discarded/last-delta
        summary for an in-progress run.
    """
    if session_result is None:
        return (StyledSegment(NO_SESSION_TEXT, "pending"),)

    state = session_result.state

    if state.finalized is not None:
        return (
            StyledSegment(_iter_label(state.iteration_count, max_iterations), "count"),
            StyledSegment(" · finalized", "done"),
        )

    if state.iteration_count == 0:
        if session_result.has_baseline:
            return (StyledSegment("baseline recorded · no iterations yet", "plain"),)
        return (StyledSegment(NO_SESSION_TEXT, "pending"),)

    segments: list[StyledSegment[LoopStyle]] = [
        StyledSegment(_iter_label(state.iteration_count, max_iterations), "count"),
        StyledSegment(" · ", "plain"),
        StyledSegment(f"{state.keep_count} kept", "done"),
        StyledSegment(" · ", "plain"),
        StyledSegment(f"{state.discard_count} discarded", "meta"),
    ]
    last = state.last_iteration
    if last is not None:
        segments += _last_iteration_segments(
            last.primary.delta_pct, last.outcome, unsettled=state.unsettled
        )
    return tuple(segments)
