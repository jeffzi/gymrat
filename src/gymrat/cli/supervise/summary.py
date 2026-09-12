"""Closing summary and format helpers for the supervise command output."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from rich.text import Text

from gymrat.cli.style import (
    GLYPH_ALERT,
    GLYPH_DONE,
    GLYPH_ERROR,
    STYLE_ALERT,
    STYLE_DONE,
    STYLE_META,
    STYLE_REGRESSED,
)
from gymrat.cli.supervise.frame import (
    build_best_text,
    build_loop_text,
    log_path_text,
)
from gymrat.cli.supervise.text import format_cost
from gymrat.eta import format_duration
from gymrat.supervisor.events import SUMMARY_MAX_CHARS

if TYPE_CHECKING:
    from gymrat.cli.supervise.state import ReadSessionResult
    from gymrat.config import Effort
    from gymrat.supervisor import SupervisionResult

_SUMMARY_LABEL_WIDTH = 6
"""The label column of the summary rows, wide enough for "agent"."""

_CAP_LABELS: dict[str, str] = {"wall-clock": "wall-clock cap", "spend-cap": "spend cap"}


def _build_outcome_text(result: SupervisionResult) -> Text:
    """The glyph-led headline: how the run ended, then its duration and cost."""
    text = Text()
    if _completed_on_its_own(result):
        text.append(f"{GLYPH_DONE} completed", style=STYLE_DONE)
    elif result.outcome.reason == "error":
        text.append(f"{GLYPH_ERROR} error", style=STYLE_REGRESSED)
    elif result.ended_by == "guard":
        reason = result.end_reason or "unknown"
        text.append(f"{GLYPH_ALERT} stopped by guard: {reason}", style=STYLE_ALERT)
    else:
        cap = _CAP_LABELS[result.ended_by]
        text.append(f"{GLYPH_ALERT} interrupted by {cap}", style=STYLE_ALERT)
    text.append(" · ", style=STYLE_META)
    text.append(format_duration(result.duration_ms))
    text.append(" · ", style=STYLE_META)
    text.append(format_cost(result.cost_usd))
    return text


def _row_prefix(label: str) -> str:
    """The two-space-indented, padded label lead-in shared by every summary row."""
    return f"  {label:<{_SUMMARY_LABEL_WIDTH}}  "


def _summary_row(label: str, content: Text) -> Text:
    row = Text(_row_prefix(label))
    row.append_text(content)
    return row


def _build_agent_row(final_text: str) -> Text:
    """Build the agent summary row, clipping long messages.

    When *final_text* exceeds ``SUMMARY_MAX_CHARS`` code points, it is truncated
    with an ellipsis and a note directing the user to the event log (whose path
    is printed on the next row).  Short messages render unchanged with
    continuation-line indentation preserved.

    Args:
        final_text: The agent's message to render, already resolved to the
            session's stop message or last text block by the caller.

    Returns:
        The styled ``Text`` row for the agent summary.
    """
    label = "agent"
    if len(final_text) > SUMMARY_MAX_CHARS:
        clipped = f"{final_text[:SUMMARY_MAX_CHARS]}… (full message in log)"
        return _summary_row(label, Text(clipped))
    indent = " " * len(_row_prefix(label))
    indented = final_text.replace("\n", f"\n{indent}")
    return _summary_row(label, Text(indented))


def _completed_on_its_own(result: SupervisionResult) -> bool:
    """Whether the session ended by itself, not by a cap trip or an error."""
    return result.ended_by == "session" and result.outcome.reason != "error"


def _resolve_agent_text(
    session_result: ReadSessionResult | None, final_text: str | None
) -> str | None:
    """The session's stop message, falling back to the agent's last text block."""
    stop_message = session_result.stop_message if session_result is not None else None
    return stop_message or final_text


@dataclass(frozen=True, slots=True)
class SessionLabels:
    """Optional model/effort labels shown in the closing summary."""

    model: str | None = None
    effort: Effort | None = None


_NO_LABELS = SessionLabels()


def build_summary(
    result: SupervisionResult,
    *,
    log_path: str,
    session_result: ReadSessionResult | None,
    final_text: str | None = None,
    labels: SessionLabels = _NO_LABELS,
) -> Text:
    """Build the closing summary ``gymrat supervise`` prints when a run ends.

    The headline states how the run ended; the rows below it reuse the
    dashboard's best and loop renderables, so the last thing printed reads like
    the frame it replaces, and end with where the event log landed.

    When the session ended on its own (not by a cap or error) and the agent
    produced text, an ``agent`` row appears after the headline showing the
    session's stop message when the log ends on one, otherwise the agent's
    last text block, with paragraph breaks preserved.

    ``labels.model`` and ``labels.effort`` appear as labelled rows when in force.

    Args:
        result: The supervision outcome whose ``ended_by`` drives the headline.
        log_path: Absolute path to the event log, printed as the final row.
        session_result: The latest session read, supplying best-iteration and
            loop-progress content.  ``None`` when the session file was never
            created.
        final_text: Override text for the agent row.  When ``None``, the
            function falls back to the session's stop message or last text
            block.
        labels: Model and effort labels to surface as extra rows.

    Returns:
        The assembled ``Text`` block for the closing summary.
    """
    rows = [_build_outcome_text(result)]
    if _completed_on_its_own(result) or result.ended_by == "guard":
        agent_text = _resolve_agent_text(session_result, final_text)
        if agent_text is not None:
            rows.append(_build_agent_row(agent_text))
    if labels.model is not None:
        rows.append(_summary_row("model", Text(labels.model)))
    if labels.effort is not None:
        rows.append(_summary_row("effort", Text(labels.effort)))
    best_text = build_best_text(session_result)
    if best_text is not None:
        rows.append(_summary_row("best", best_text))
    rows.append(_summary_row("loop", build_loop_text(session_result, None)))
    rows.append(_summary_row("log", log_path_text(log_path)))
    return Text("\n").join(rows)
