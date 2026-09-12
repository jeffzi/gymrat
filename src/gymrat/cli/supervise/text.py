"""Rich-free text the supervise reducer and its view layer both render.

The reducer folds the loop summary into its state as a plain string, so it must
stay clear of the Rich view layer; :mod:`gymrat.cli.supervise.frame` renders the
same summary with styles.  Both read :func:`loop_segments`, which keeps the two
from ever disagreeing about what the summary says.  Segments carry a style
*role* rather than a theme style, because the theme lives behind a Rich import.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, NamedTuple

from gymrat.model import Effect
from gymrat.report.format import format_delta

if TYPE_CHECKING:
    from gymrat.cli.supervise.state import ReadSessionResult

#: Shown in place of the loop summary before any session data has been read.
NO_SESSION_TEXT = "no session yet"

type LoopStyle = Literal["alert", "count", "done", "meta", "pending", "plain", "regressed"]
"""The style role of a loop-summary segment; ``"plain"`` means unstyled."""


class LoopSegment(NamedTuple):
    """One run of loop-summary text together with the role it renders under."""

    text: str
    style: LoopStyle


def format_cost(usd: float) -> str:
    """Format a USD amount as a two-decimal dollar string."""
    return f"${usd:.2f}"


def format_caps(max_minutes: float, max_usd: float | None) -> str:
    """Format "caps {minutes}m" alone, or with ", {cost}" appended when a spend cap is set."""
    caps_parts = [f"{max_minutes:g}m"]
    if max_usd is not None:
        caps_parts.append(format_cost(max_usd))
    return f"caps {', '.join(caps_parts)}"


def _iter_label(count: int, max_iterations: int | None) -> str:
    if max_iterations is not None:
        # The noun agrees with the cap, so the capped form stays plural at any count.
        return f"{count}/{max_iterations} iterations"
    return f"{count} iteration" if count == 1 else f"{count} iterations"


def _outcome_role(outcome: str) -> LoopStyle:
    if outcome in ("improved", "kept"):
        return "done"
    if outcome == "regressed":
        return "regressed"
    return "meta"


def _last_iteration_segments(
    delta_pct: float | None, outcome: str, *, unsettled: bool
) -> list[LoopSegment]:
    delta = "—" if delta_pct is None else format_delta(Effect(value=delta_pct, unit="percent"))
    role = _outcome_role(outcome)
    segments = [
        LoopSegment(" · last ", "plain"),
        LoopSegment(delta, role),
        LoopSegment(" ", "plain"),
        LoopSegment(outcome, role),
    ]
    if unsettled:
        segments.append(LoopSegment(", unsettled", "alert"))
    return segments


def loop_segments(
    session_result: ReadSessionResult | None, max_iterations: int | None
) -> tuple[LoopSegment, ...]:
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
        return (LoopSegment(NO_SESSION_TEXT, "pending"),)

    state = session_result.state

    if state.finalized is not None:
        return (
            LoopSegment(_iter_label(state.iteration_count, max_iterations), "count"),
            LoopSegment(" · finalized", "done"),
        )

    if state.iteration_count == 0:
        if session_result.has_baseline:
            return (LoopSegment("baseline recorded · no iterations yet", "plain"),)
        return (LoopSegment(NO_SESSION_TEXT, "pending"),)

    segments = [
        LoopSegment(_iter_label(state.iteration_count, max_iterations), "count"),
        LoopSegment(" · ", "plain"),
        LoopSegment(f"{state.keep_count} kept", "done"),
        LoopSegment(" · ", "plain"),
        LoopSegment(f"{state.discard_count} discarded", "meta"),
    ]
    last = state.last_iteration
    if last is not None:
        segments += _last_iteration_segments(
            last.primary.delta_pct, last.outcome, unsettled=state.unsettled
        )
    return tuple(segments)


def loop_plain_text(session_result: ReadSessionResult | None, max_iterations: int | None) -> str:
    """The unstyled text of the loop summary :func:`loop_segments` describes."""
    return "".join(segment.text for segment in loop_segments(session_result, max_iterations))
