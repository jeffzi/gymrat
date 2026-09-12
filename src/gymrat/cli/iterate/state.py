"""Pure state model for the ``gymrat iterate`` checklist.

The renderer in :mod:`.progress` owns the terminal; this module owns the data it
paints. :func:`advance` folds one progress event into a new :class:`IterateState`
without touching a clock, a writer, or a Rich object — every timestamp a
transition needs comes from the event's own ``at_ms``. That keeps the checklist
reproducible and testable without a console.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Literal

from gymrat.eta import SamplingEta, format_duration
from gymrat.plural import pluralize
from gymrat.progress_events import (
    ConfirmFinished,
    ConfirmStarted,
    HookFinished,
    HookStarted,
    IterationRecorded,
    JudgeFinished,
    JudgeStarted,
    PassFinished,
    PassStarted,
    PrepareFinished,
    PrepareStarted,
    ProgressEvent,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

TARGETS_PER_ROUND = 2
"""Passes one sampling round runs: one against the baseline, one against the candidate."""

REGRESSED_NAME_CAP = 3
"""How many regressed metric names the judge's lines spell out."""


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class JudgeDetail:
    """The judge's verdict as data, for the view to render.

    Attributes:
        primary_metric: Name of the primary metric, shown beside the delta.
        primary_delta_pct: Percentage delta on the primary metric, or ``None``
            when no delta is available.
        regressed_names: Names of the metrics that regressed.
    """

    primary_metric: str
    primary_delta_pct: float | None
    regressed_names: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class NodeState:
    """Status, wording, and timing of a single checklist row.

    Attributes:
        noun: Row label shown while ``status`` is ``"pending"``.
        gerund: Row label shown while ``status`` is ``"running"``.
        past: Row label shown once ``status`` is ``"done"``.
        hint: Dim explanation shown behind a pending row, or ``""`` for none.
        note: Dim context shown beside a running row, or ``""`` for none.
        target: In-flight target label shown while running, or ``""`` for none.
        detail: Outcome shown when done, either plain text or a
            :class:`JudgeDetail` for the judge row, or ``""`` for none.
        status: The row's lifecycle state, driving which rendering the view
            picks. A ``"skipped"`` row is dropped from the checklist entirely.
        start_ms: Timestamp the row's current run began, or ``0.0`` before it
            has started.
        elapsed_ms: Milliseconds the row's completed run took, or ``0.0``
            before it is done.
        alert: Whether the row is flagged for attention, changing its glyph
            and style.
    """

    noun: str
    gerund: str
    past: str
    hint: str = ""
    note: str = ""
    target: str = ""
    detail: str | JudgeDetail = ""
    status: Literal["pending", "running", "done", "skipped"] = "pending"
    start_ms: float = 0.0
    elapsed_ms: float = 0.0
    alert: bool = False


@dataclass(frozen=True, slots=True)
class PhaseCounters:
    """Sampling progress of one phase — the measure passes or the confirm passes.

    Attributes:
        eta: Finished-pass samples and the remaining-time estimate they make.
        start_ms: Timestamp of the pass currently in flight.
    """

    eta: SamplingEta
    start_ms: float = 0.0


@dataclass(frozen=True, slots=True)
class IterateNodes:
    """Per-phase node states for a single iteration.

    Attributes:
        before_hook: The before-hook row, present in ``all_nodes`` only when the
            iteration runs one.
        prepare: The worktree preparation row.
        passes: The measure-pass row.
        judge: The verdict row.
        confirm: The confirmation-pass row.
        record: The outcome row.
        has_before_hook: Whether the before-hook row belongs in the checklist.
    """

    before_hook: NodeState
    prepare: NodeState
    passes: NodeState
    judge: NodeState
    confirm: NodeState
    record: NodeState
    has_before_hook: bool

    @property
    def all_nodes(self) -> tuple[NodeState, ...]:
        """The checklist rows in display order."""
        hook_rows = (self.before_hook,) if self.has_before_hook else ()
        return (*hook_rows, self.prepare, self.passes, self.judge, self.confirm, self.record)


@dataclass(frozen=True, slots=True)
class IterateState:
    """Everything the iterate checklist shows, as data.

    Attributes:
        nodes: The checklist rows.
        pass_phase: Sampling progress of the measure passes.
        confirm_phase: Sampling progress of the confirmation passes.
        prepare_current_start_ms: Start timestamp of the prepare step in flight,
            which the prepare row accumulates from.
        run_start_ms: Timestamp of the first event seen, or ``None`` before one
            arrives. Plain-mode timestamps are relative to it.
        total: Passes the iteration expects in all.
        primary_metric: Name of the metric the judge gates on.
        checks_cmd: The shell command the record row mentions, or ``None``.
    """

    nodes: IterateNodes
    pass_phase: PhaseCounters
    confirm_phase: PhaseCounters
    prepare_current_start_ms: float
    run_start_ms: float | None
    total: int
    primary_metric: str
    checks_cmd: str | None


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def build_nodes(
    primary_metric: str,
    metric_count: int,
    *,
    has_before_hook: bool,
    has_after_hook: bool,
) -> IterateNodes:
    """Build the per-phase node states an iteration starts from.

    Args:
        primary_metric: Name of the primary metric, shown in the judge node's
            hint.
        metric_count: Total number of evaluated metrics, shown in the judge
            hint alongside the primary.
        has_before_hook: Whether to include a before-hook node in the
            checklist.
        has_after_hook: Whether the record node's hint should mention a
            subsequent after hook.

    Returns:
        The assembled node collection, every row pending.
    """
    judge_hint = f"{primary_metric} primary"
    if metric_count > 0:
        judge_hint = f"{pluralize(metric_count, 'metric')} · {judge_hint}"
    return IterateNodes(
        before_hook=NodeState(noun="before hook", gerund="before hook", past="before hook"),
        prepare=NodeState(noun="prepare", gerund="preparing", past="prepared"),
        passes=NodeState(noun="passes", gerund="sampling", past="sampled"),
        judge=NodeState(
            noun="judge",
            gerund="judging",
            past="judged",
            hint=judge_hint,
            note=judge_hint,
        ),
        confirm=NodeState(
            noun="confirm",
            gerund="confirming",
            past="confirmed",
            hint="only if a gating metric regresses",
        ),
        record=NodeState(
            noun="record",
            gerund="recording",
            past="recorded",
            hint="then after hook" if has_after_hook else "",
        ),
        has_before_hook=has_before_hook,
    )


def initial_state(  # noqa: PLR0913 -- one parameter per iteration fact the state carries
    *,
    sample_count: int,
    metric_count: int,
    primary_metric: str,
    checks_cmd: str | None,
    has_before_hook: bool,
    has_after_hook: bool,
) -> IterateState:
    """Build the state an iteration starts from.

    Args:
        sample_count: Passes per target; the iteration total is this times
            :data:`TARGETS_PER_ROUND`.
        metric_count: Number of metrics the judge evaluates.
        primary_metric: Name of the metric the judge gates on.
        checks_cmd: The shell command the record row mentions, or ``None`` to
            omit the note.
        has_before_hook: Whether the checklist includes a before-hook row.
        has_after_hook: Whether the record row's hint mentions an after hook.

    Returns:
        A state with every row pending and no event applied yet.
    """
    total = sample_count * TARGETS_PER_ROUND
    return IterateState(
        nodes=build_nodes(
            primary_metric,
            metric_count,
            has_before_hook=has_before_hook,
            has_after_hook=has_after_hook,
        ),
        pass_phase=PhaseCounters(eta=SamplingEta.start(total)),
        confirm_phase=PhaseCounters(eta=SamplingEta.start(total)),
        prepare_current_start_ms=0.0,
        run_start_ms=None,
        total=total,
        primary_metric=primary_metric,
        checks_cmd=checks_cmd,
    )


# ---------------------------------------------------------------------------
# Transitions
# ---------------------------------------------------------------------------


def advance(state: IterateState, event: ProgressEvent) -> IterateState:
    """Fold one progress event into a new state.

    Args:
        state: The state to apply the event to; it is left unchanged.
        event: The event to apply. Its ``at_ms`` is the only clock a transition
            reads.

    Returns:
        The state after the event — an equal state when the event changes
        nothing, such as an after-stage hook the checklist does not show.

    Raises:
        KeyError: If no transition is registered for the event's type.
    """
    anchored = state if state.run_start_ms is not None else replace(state, run_start_ms=event.at_ms)
    return _TRANSITIONS[type(event)](anchored, event)


def _hook_started(state: IterateState, event: HookStarted) -> IterateState:
    if event.stage != "before":
        return state
    running = replace(state.nodes.before_hook, status="running", start_ms=event.at_ms)
    return replace(state, nodes=replace(state.nodes, before_hook=running))


def _hook_finished(state: IterateState, event: HookFinished) -> IterateState:
    if event.stage != "before":
        return state
    hook = state.nodes.before_hook
    done = replace(hook, status="done", elapsed_ms=event.at_ms - hook.start_ms)
    return replace(state, nodes=replace(state.nodes, before_hook=done))


def _prepare_started(state: IterateState, event: PrepareStarted) -> IterateState:
    running = replace(state.nodes.prepare, status="running", target=event.label)
    return replace(
        state,
        nodes=replace(state.nodes, prepare=running),
        prepare_current_start_ms=event.at_ms,
    )


def _prepare_finished(state: IterateState, event: PrepareFinished) -> IterateState:
    prepare = state.nodes.prepare
    elapsed_ms = prepare.elapsed_ms + (event.at_ms - state.prepare_current_start_ms)
    done = replace(prepare, status="done", target="", elapsed_ms=elapsed_ms)
    return replace(state, nodes=replace(state.nodes, prepare=done))


def _pass_started(state: IterateState, event: PassStarted) -> IterateState:
    if event.phase == "confirm":
        return replace(state, confirm_phase=replace(state.confirm_phase, start_ms=event.at_ms))
    running = replace(state.nodes.passes, status="running")
    return replace(
        state,
        nodes=replace(state.nodes, passes=running),
        pass_phase=replace(state.pass_phase, start_ms=event.at_ms),
    )


def _pass_finished(state: IterateState, event: PassFinished) -> IterateState:
    counters = state.confirm_phase if event.phase == "confirm" else state.pass_phase
    advanced = replace(counters, eta=counters.eta.advanced(event.at_ms - counters.start_ms))

    if event.phase == "confirm":
        return replace(state, confirm_phase=advanced)

    sampled = replace(state, pass_phase=advanced)
    if advanced.eta.completed < state.total:
        return sampled
    done = replace(
        state.nodes.passes,
        status="done",
        elapsed_ms=advanced.eta.total_time_ms,
        detail=f"{state.total} passes",
    )
    return replace(sampled, nodes=replace(sampled.nodes, passes=done))


def _judge_started(state: IterateState, event: JudgeStarted) -> IterateState:
    running = replace(state.nodes.judge, status="running", start_ms=event.at_ms)
    return replace(state, nodes=replace(state.nodes, judge=running))


def _judge_finished(state: IterateState, event: JudgeFinished) -> IterateState:
    judge = state.nodes.judge
    done = replace(
        judge,
        status="done",
        note="",
        elapsed_ms=event.at_ms - judge.start_ms if judge.start_ms > 0 else 0.0,
        detail=JudgeDetail(
            primary_metric=state.primary_metric,
            primary_delta_pct=event.primary_delta_pct,
            regressed_names=tuple(event.regressed),
        ),
    )
    confirm = (
        state.nodes.confirm if event.regressed else replace(state.nodes.confirm, status="skipped")
    )
    return replace(state, nodes=replace(state.nodes, judge=done, confirm=confirm))


def _confirm_started(state: IterateState, event: ConfirmStarted) -> IterateState:
    note = (
        "full suite"
        if event.filtered_metrics is None
        else pluralize(len(event.filtered_metrics), "metric")
    )
    running = replace(state.nodes.confirm, status="running", start_ms=event.at_ms, note=note)
    alerted = replace(state.nodes.judge, alert=True)
    return replace(state, nodes=replace(state.nodes, judge=alerted, confirm=running))


def _confirm_finished(state: IterateState, event: ConfirmFinished) -> IterateState:
    confirm = state.nodes.confirm
    done = replace(
        confirm,
        status="done",
        note="",
        elapsed_ms=event.at_ms - confirm.start_ms,
        detail=_confirm_detail(state, reproduced=event.reproduced),
    )
    settled = replace(state.nodes.judge, alert=False)
    return replace(state, nodes=replace(state.nodes, judge=settled, confirm=done))


def _iteration_recorded(state: IterateState, event: IterationRecorded) -> IterateState:
    done = replace(state.nodes.record, status="done", detail=_record_detail(state, event.outcome))
    return replace(state, nodes=replace(state.nodes, record=done))


_TRANSITIONS: dict[type[ProgressEvent], Callable[..., IterateState]] = {
    HookStarted: _hook_started,
    HookFinished: _hook_finished,
    PrepareStarted: _prepare_started,
    PrepareFinished: _prepare_finished,
    PassStarted: _pass_started,
    PassFinished: _pass_finished,
    JudgeStarted: _judge_started,
    JudgeFinished: _judge_finished,
    ConfirmStarted: _confirm_started,
    ConfirmFinished: _confirm_finished,
    IterationRecorded: _iteration_recorded,
}
"""Every event the checklist reacts to, mapped to the transition it runs."""


def _confirm_detail(state: IterateState, *, reproduced: bool) -> str:
    outcome = "regressions reproduced" if reproduced else "regressions not reproduced"
    return f"{state.confirm_phase.eta.completed}/{state.total} · {outcome}"


def _record_detail(state: IterateState, outcome: str) -> str:
    detail = f"{outcome} suggested"
    if state.checks_cmd is not None:
        detail += f" — checks ({state.checks_cmd}) run at gymrat keep"
    return detail


# ---------------------------------------------------------------------------
# Plain-mode lines
# ---------------------------------------------------------------------------


def plain_line(before: IterateState, after: IterateState, event: ProgressEvent) -> str | None:
    """Return the plain-mode milestone line an event prints, if any.

    Args:
        before: The state the event was applied to.
        after: The state :func:`advance` returned for the event.
        event: The event that was applied.

    Returns:
        The milestone line without its timestamp prefix, or ``None`` when the
        event is not a milestone.
    """
    match event:
        case PrepareFinished():
            elapsed_ms = event.at_ms - before.prepare_current_start_ms
            return f"prepare {event.label} done ({format_duration(elapsed_ms)})"
        case PassFinished(phase="measure") if after.pass_phase.eta.completed >= after.total:
            return f"passes done ({format_duration(after.pass_phase.eta.total_time_ms)})"
        case JudgeFinished():
            breakdown = format_judge_plain(
                event.primary_delta_pct, event.regressed, event.metric_count
            )
            return f"judge {breakdown}"
        case ConfirmFinished():
            return f"confirm {_confirm_detail(after, reproduced=event.reproduced)}"
        case IterationRecorded():
            return f"recorded {_record_detail(after, event.outcome)}"
        case _:
            return None


def format_judge_plain(
    primary_delta_pct: float | None,
    regressed: Sequence[str],
    metric_count: int,
) -> str:
    """Format the judge result for plain (non-live) mode.

    Args:
        primary_delta_pct: Percentage delta on the primary metric, or ``None``
            when no delta is available (renders as ``"—"``).
        regressed: Names of regressed metrics. At most
            :data:`REGRESSED_NAME_CAP` names are spelled out; the rest are
            collapsed to ``"…"``.
        metric_count: Total number of evaluated metrics, used to derive the
            non-regressed count.

    Returns:
        A ``" · "``-joined string of the delta, non-regressed count, and
        (when any regressed) their names.
    """
    delta_str = f"{primary_delta_pct:+.1f}%" if primary_delta_pct is not None else "—"
    handoff: list[str] = []
    if regressed:
        shown = ", ".join(regressed[:REGRESSED_NAME_CAP])
        if len(regressed) > REGRESSED_NAME_CAP:
            shown += ", …"
        handoff = [f"{len(regressed)} regressed: {shown}"]
    non_regressed_count = metric_count - len(regressed)
    return " · ".join([delta_str, f"{non_regressed_count} improve/noise", *handoff])
