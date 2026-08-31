"""Row state and rendering helpers for the iterate progress checklist.

Each iteration phase (hook, prepare, passes, judge, confirm, record) is tracked
by a :class:`NodeState` that carries the phase's three verb forms, its timing,
and its completion status. The rendering functions convert node states into Rich
renderables for the live display.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, NamedTuple

from rich.spinner import Spinner
from rich.text import Text

from gymrat.cli.style import (
    GLYPH_ALERT,
    GLYPH_DONE,
    GLYPH_PENDING,
    SPINNER_NAME,
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
from gymrat.plural import pluralize

if TYPE_CHECKING:
    from collections.abc import Sequence

    from rich.console import RenderableType
    from rich.progress import Progress, TaskID

    from gymrat.cli.progress import _ClockColumn

_REGRESSED_NAME_CAP = 3
"""How many regressed metric names the judge's done line spells out."""


@dataclass(slots=True)
class NodeState:
    """Status, wording, and timing of a single checklist row.

    The three verb forms are what the row's state reads as: ``noun`` while
    pending, ``gerund`` while running, ``past`` once done. ``hint`` is the dim
    explanation shown behind a pending row, ``note`` the dim context and
    ``target`` the in-flight target label shown while running, and ``detail``
    the dim outcome shown when done. A ``skipped`` row is dropped from the
    checklist entirely.
    """

    noun: str
    gerund: str
    past: str
    hint: str = ""
    note: str = ""
    target: str = ""
    detail: str | Text = ""
    status: Literal["pending", "running", "done", "skipped"] = "pending"
    start_ms: float = 0.0
    elapsed_ms: float = 0.0
    alert: bool = False
    bar: Progress | None = field(default=None, repr=False)
    spinner: Spinner | None = field(default=None, repr=False)

    @property
    def glyph(self) -> str:
        """The status glyph for this row — done, alert, or pending."""
        if self.alert:
            return GLYPH_ALERT
        match self.status:
            case "done":
                return GLYPH_DONE
            case _:
                return GLYPH_PENDING


# ---------------------------------------------------------------------------
# Row rendering
# ---------------------------------------------------------------------------


def render_row(node: NodeState, running_ms: float | None) -> RenderableType:
    """Dispatch to the appropriate row renderer based on node status."""
    match node.status:
        case "running":
            return render_running_row(node, running_ms)
        case "done":
            return render_done_row(node)
        case _:
            return render_idle_row(node)


def render_running_row(node: NodeState, running_ms: float | None) -> RenderableType:
    """Alert state shows a static glyph; normal state spins."""
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
    if node.spinner is None:
        node.spinner = Spinner(SPINNER_NAME, text=text, style=style)
    else:
        node.spinner.update(text=text, style=style)
    return node.spinner


def render_done_row(node: NodeState) -> Text:
    """A ``Text`` detail is appended unstyled; a plain string gets ``STYLE_META``."""
    text = Text()
    text.append(f"{node.glyph} ", style=STYLE_ALERT if node.alert else STYLE_DONE)
    text.append(node.past)
    if node.detail:
        if isinstance(node.detail, Text):
            text.append(" ")
            text.append_text(node.detail)
        else:
            text.append(f" {node.detail}", style=STYLE_META)
    if node.elapsed_ms > 0:
        text.append(f" {format_duration(node.elapsed_ms)}", style=STYLE_TIMER_DONE)
    return text


def render_idle_row(node: NodeState) -> Text:
    """Render a row that has not run yet: pending with its hint."""
    text = Text()
    text.append(f"{node.glyph} {node.noun}", style=STYLE_PENDING)
    if node.hint:
        text.append(f" ({node.hint})", style=STYLE_PENDING)
    return text


# ---------------------------------------------------------------------------
# Judge detail builder
# ---------------------------------------------------------------------------


def build_judge_detail(
    primary_metric: str,
    primary_delta_pct: float | None,
    regressed: Sequence[str],
) -> Text:
    """Build the rich Text detail for the judge's done row."""
    delta_str = f"{primary_delta_pct:+.1f}%" if primary_delta_pct is not None else "—"
    primary = f"{delta_str} on {primary_metric}" if primary_delta_pct is not None else delta_str

    detail = Text()
    detail.append(primary, style=STYLE_META)
    if regressed:
        detail.append(" · ", style=STYLE_META)
        detail.append(f"{len(regressed)} regressed: ", style=STYLE_META)
        for i, name in enumerate(regressed[:_REGRESSED_NAME_CAP]):
            if i > 0:
                detail.append(", ", style=STYLE_META)
            detail.append_text(Text.from_markup(format_inline(parse(name), color=True)))
        if len(regressed) > _REGRESSED_NAME_CAP:
            detail.append(", …", style=STYLE_META)
    if not regressed:
        detail.append(" · ", style=STYLE_META)
        detail.append("no gating regression", style=STYLE_META)
    return detail


@dataclass(slots=True)
class PhaseCounters:
    """Per-phase sampling counters shared between measure and confirm passes."""

    completed: int = 0
    finish_count: int = 0
    total_time_ms: float = 0.0
    start_ms: float = 0.0
    task_id: TaskID | None = None
    clock_col: _ClockColumn | None = None


class IterateNodes(NamedTuple):
    """Per-phase node states for a single iteration, plus the ordered checklist."""

    before_hook: NodeState
    prepare: NodeState
    passes: NodeState
    judge: NodeState
    confirm: NodeState
    record: NodeState
    all_nodes: tuple[NodeState, ...]


def build_nodes(
    primary_metric: str,
    metric_count: int,
    *,
    has_before_hook: bool,
    has_after_hook: bool,
) -> IterateNodes:
    """Build the per-phase node states and return them with the ordered tuple."""
    judge_hint = f"{primary_metric} primary"
    if metric_count > 0:
        judge_hint = f"{pluralize(metric_count, 'metric')} · {judge_hint}"
    before_hook = NodeState(noun="before hook", gerund="before hook", past="before hook")
    prepare = NodeState(noun="prepare", gerund="preparing", past="prepared")
    passes = NodeState(noun="passes", gerund="sampling", past="sampled")
    judge = NodeState(
        noun="judge",
        gerund="judging",
        past="judged",
        hint=judge_hint,
        note=judge_hint,
    )
    confirm = NodeState(
        noun="confirm",
        gerund="confirming",
        past="confirmed",
        hint="only if a gating metric regresses",
    )
    record = NodeState(
        noun="record",
        gerund="recording",
        past="recorded",
        hint="then after hook" if has_after_hook else "",
    )
    hook_rows = (before_hook,) if has_before_hook else ()
    all_nodes = (*hook_rows, prepare, passes, judge, confirm, record)
    return IterateNodes(before_hook, prepare, passes, judge, confirm, record, all_nodes)


def format_judge_plain(
    primary_delta_pct: float | None,
    regressed: Sequence[str],
    metric_count: int,
) -> str:
    """Format the judge result for plain (non-live) mode."""
    delta_str = f"{primary_delta_pct:+.1f}%" if primary_delta_pct is not None else "—"
    handoff: list[str] = []
    if regressed:
        shown = ", ".join(regressed[:_REGRESSED_NAME_CAP])
        if len(regressed) > _REGRESSED_NAME_CAP:
            shown += ", …"
        handoff = [f"{len(regressed)} regressed: {shown}"]
    non_regressed_count = metric_count - len(regressed)
    return " · ".join([delta_str, f"{non_regressed_count} improve/noise", *handoff])
