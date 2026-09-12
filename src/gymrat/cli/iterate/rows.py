"""Row rendering for the iterate progress checklist.

Each iteration phase (hook, prepare, passes, judge, confirm, record) is tracked
by a :class:`~gymrat.cli.iterate.state.NodeState`, a frozen value carrying the
phase's three verb forms, its timing, and its completion status. The functions
here turn one such row into a Rich renderable, using the spinner and progress
bar the renderer owns for that row.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rich.text import Text

from gymrat.cli.iterate.state import REGRESSED_NAME_CAP, JudgeDetail
from gymrat.cli.style import (
    GLYPH_ALERT,
    GLYPH_DONE,
    GLYPH_PENDING,
    STYLE_ALERT,
    STYLE_DONE,
    STYLE_LABEL,
    STYLE_META,
    STYLE_PENDING,
    STYLE_RUNNING,
    STYLE_TIMER_DONE,
    STYLE_TIMER_RUNNING,
    STYLE_VERB,
)
from gymrat.eta import format_duration
from gymrat.metric_name import format_inline, parse

if TYPE_CHECKING:
    from rich.console import RenderableType
    from rich.progress import Progress
    from rich.spinner import Spinner

    from gymrat.cli.iterate.state import NodeState


def render_row(
    node: NodeState,
    *,
    spinner: Spinner,
    bar: Progress | None,
    running_ms: float | None,
) -> RenderableType:
    """Render one checklist row in whichever state it is in.

    Args:
        node: The row to render.
        spinner: The renderer's persistent spinner for this row, updated in
            place so its animation stays continuous across frames.
        bar: The renderer's progress bar for this row, or ``None`` for a row
            that has none. A running row with a bar renders as that bar.
        running_ms: Elapsed milliseconds to show on a running row, or ``None``
            to show no timer.

    Returns:
        The renderable for the row.
    """
    match node.status:
        case "running" if bar is not None:
            return bar
        case "running":
            return render_running_row(node, spinner, running_ms)
        case "done":
            return render_done_row(node)
        case _:
            return render_idle_row(node)


def render_running_row(
    node: NodeState, spinner: Spinner, running_ms: float | None
) -> RenderableType:
    """Alert state shows a static glyph; normal state spins ``spinner``."""
    style = STYLE_ALERT if node.alert else STYLE_RUNNING
    text = Text()
    text.append(node.gerund, style=STYLE_VERB)
    if node.note:
        text.append(f" {node.note}", style=STYLE_META)
    if node.target:
        text.append(" · ", style=STYLE_META)
        text.append(node.target, style=STYLE_LABEL)
    if running_ms is not None:
        text.append(f" {format_duration(running_ms)}", style=STYLE_TIMER_RUNNING)
    if node.alert:
        return Text.assemble((f"{GLYPH_ALERT} ", style), text)
    spinner.update(text=text, style=style)
    return spinner


def render_done_row(node: NodeState) -> Text:
    """A :class:`JudgeDetail` detail is styled by the view; a string gets ``STYLE_META``."""
    text = Text()
    text.append(f"{_glyph(node)} ", style=STYLE_ALERT if node.alert else STYLE_DONE)
    text.append(node.past)
    if node.detail:
        if isinstance(node.detail, JudgeDetail):
            text.append(" ")
            text.append_text(build_judge_detail(node.detail))
        else:
            text.append(f" {node.detail}", style=STYLE_META)
    if node.elapsed_ms > 0:
        text.append(f" {format_duration(node.elapsed_ms)}", style=STYLE_TIMER_DONE)
    return text


def render_idle_row(node: NodeState) -> Text:
    """Render a not-yet-started phase: its glyph, noun, and optional hint."""
    text = Text()
    text.append(f"{_glyph(node)} {node.noun}", style=STYLE_PENDING)
    if node.hint:
        text.append(f" ({node.hint})", style=STYLE_PENDING)
    return text


def _glyph(node: NodeState) -> str:
    if node.alert:
        return GLYPH_ALERT
    match node.status:
        case "done":
            return GLYPH_DONE
        case _:
            return GLYPH_PENDING


# ---------------------------------------------------------------------------
# Judge detail builder
# ---------------------------------------------------------------------------


def build_judge_detail(detail: JudgeDetail) -> Text:
    """Build the rich Text detail for the judge's done row.

    Args:
        detail: The judge's verdict. At most :data:`REGRESSED_NAME_CAP`
            regressed names are spelled out; the rest are collapsed to ``"…"``.
            A ``None`` delta renders as ``"—"``.

    Returns:
        A styled ``Text`` for the judge row's detail.
    """
    delta_pct = detail.primary_delta_pct
    delta_str = f"{delta_pct:+.1f}%" if delta_pct is not None else "—"
    primary = f"{delta_str} on {detail.primary_metric}" if delta_pct is not None else delta_str
    regressed = detail.regressed_names

    text = Text()
    text.append(primary, style=STYLE_META)
    text.append(" · ", style=STYLE_META)
    if regressed:
        text.append(f"{len(regressed)} regressed: ", style=STYLE_META)
        for i, name in enumerate(regressed[:REGRESSED_NAME_CAP]):
            if i > 0:
                text.append(", ", style=STYLE_META)
            text.append_text(Text.from_markup(format_inline(parse(name))))
        if len(regressed) > REGRESSED_NAME_CAP:
            text.append(", …", style=STYLE_META)
    else:
        text.append("no gating regression", style=STYLE_META)
    return text
