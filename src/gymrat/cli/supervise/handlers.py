"""Event handlers for the supervise dashboard.

Each handler mutates the shared :class:`ReporterCtx` in response to one event
type, then emits a render cycle (live update or plain-text write).
``handle_event`` dispatches a :class:`SessionEvent` to the matching handler;
``render_live`` pushes a single explicit frame refresh.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, assert_never

from gymrat.cli.supervise.frame import (
    NO_SESSION_TEXT,
    build_frame,
    build_loop_text,
    format_cost,
)
from gymrat.cli.supervise.state import (
    Capped,
    Composing,
    FinishedTool,
    InFlight,
    NestedPhase,
    NestedTool,
    Responding,
    Thinking,
    TrackedTool,
    Waiting,
)
from gymrat.cli.supervise.summary import format_caps
from gymrat.supervisor.events import (
    CapEvent,
    CompactionEvent,
    FollowUpEvent,
    LaunchEvent,
    ModelPhaseEvent,
    SessionEvent,
    TextDeltaEvent,
    ThinkingUpdateEvent,
    ToolEndEvent,
    ToolProgressEvent,
    ToolStartEvent,
    TurnEndEvent,
    UsageUpdateEvent,
)

if TYPE_CHECKING:
    from gymrat.cli.supervise.state import ReporterCtx

logger = logging.getLogger(__name__)

_NS_PER_MS = 1_000_000


# ---------------------------------------------------------------------------
# Emit / refresh
# ---------------------------------------------------------------------------


def _try_read_session(ctx: ReporterCtx) -> None:
    try:
        ctx.session_result = ctx.read_session_fn()
    except Exception as exc:
        logger.exception("session read failed")
        ctx.warn_fn(f"session read failed: {exc}")
        ctx.session_result = None


def render_live(ctx: ReporterCtx) -> None:
    """Push one explicit frame refresh to the Live display.

    Called after every state mutation that should be visible in the dashboard.
    In plain mode or when no Live exists, this is a no-op.

    Args:
        ctx: The shared reporter context supplying the Live instance and the
            data used to build the frame.
    """
    if ctx.is_plain or ctx.live is None:
        return
    ctx.live.update(build_frame(ctx), refresh=True)


def _emit(ctx: ReporterCtx, plain_text: str) -> None:
    if ctx.is_plain:
        ctx.plain_write_fn(plain_text)
    else:
        render_live(ctx)


def _plain_loop_update(ctx: ReporterCtx) -> None:
    if not ctx.is_plain:
        return
    plain = build_loop_text(ctx.session_result, ctx.max_iterations).plain
    if plain not in {ctx.last_loop_text, NO_SESSION_TEXT}:
        ctx.last_loop_text = plain
        ctx.plain_write_fn(plain)


def _refresh_session(ctx: ReporterCtx) -> None:
    _try_read_session(ctx)
    _plain_loop_update(ctx)
    render_live(ctx)


def _next_liveness_after_tool_end(
    ctx: ReporterCtx, event: ToolEndEvent
) -> Capped | InFlight | Waiting:
    if isinstance(ctx.liveness, Capped):
        return ctx.liveness
    last_entry = next(reversed(ctx.in_flight_tools.items()), None)
    if last_entry is not None:
        tool_id, tracked = last_entry
        return InFlight(
            tool_use_id=tool_id,
            tool_name=tracked.tool_name,
            since=tracked.started_at,
            input_summary=tracked.input_summary,
        )
    ended_at_ms = event.at // _NS_PER_MS
    return Waiting(
        since=ended_at_ms,
        tool_name=event.tool_name,
        tool_ended_at=ended_at_ms,
        result=event.result,
    )


def _waiting_from_last_tool(ctx: ReporterCtx, timestamp: int) -> Waiting:
    last = ctx.finished_tools[-1] if ctx.finished_tools else None
    return Waiting(
        since=timestamp,
        tool_name=last.tool_name if last is not None else None,
        tool_ended_at=last.ended_at if last is not None else None,
        result=last.result if last is not None else None,
    )


# ---------------------------------------------------------------------------
# Event handlers
# ---------------------------------------------------------------------------


def _handle_cap_event(ctx: ReporterCtx, cap: CapEvent) -> None:
    ctx.liveness = Capped(cap_type=cap.cap, action=cap.action)
    _emit(ctx, f"cap {cap.cap} — {cap.action}")


def _handle_launch(ctx: ReporterCtx, event: LaunchEvent) -> None:
    ctx.launch_timestamp = event.at // _NS_PER_MS
    _try_read_session(ctx)
    _emit(ctx, format_caps(ctx.max_minutes, ctx.max_usd))


def _handle_usage_update(ctx: ReporterCtx, event: UsageUpdateEvent) -> None:
    ctx.cost_usd = event.cost_usd
    _emit(ctx, f"cost {format_cost(event.cost_usd)}")


def _handle_tool_start(ctx: ReporterCtx, event: ToolStartEvent) -> None:
    at_ms = event.at // _NS_PER_MS
    if event.parent_tool_use_id is not None:
        if event.parent_tool_use_id not in ctx.in_flight_tools:
            return
        ctx.nested[event.parent_tool_use_id] = NestedTool(
            tool_name=event.tool_name,
            input_summary=event.input_summary,
            since=at_ms,
        )
        ctx.nested_tool_ids[event.tool_use_id] = event.parent_tool_use_id
        return

    ctx.in_flight_tools[event.tool_use_id] = TrackedTool(
        tool_name=event.tool_name, started_at=at_ms, input_summary=event.input_summary
    )
    if not isinstance(ctx.liveness, Capped):
        ctx.liveness = InFlight(
            tool_use_id=event.tool_use_id,
            tool_name=event.tool_name,
            since=at_ms,
            input_summary=event.input_summary,
        )
    render_live(ctx)


def _handle_tool_end(ctx: ReporterCtx, event: ToolEndEvent) -> None:
    if event.parent_tool_use_id is not None:
        parent_id = ctx.nested_tool_ids.pop(event.tool_use_id, None)
        if parent_id is not None and isinstance(ctx.nested.get(parent_id), NestedTool):
            del ctx.nested[parent_id]
        return

    tracked = ctx.in_flight_tools.get(event.tool_use_id)
    if tracked is not None:
        tool_name = tracked.tool_name
        input_summary = tracked.input_summary
    else:
        tool_name = event.tool_name
        input_summary = ""
    should_refresh_session = tracked is None or tool_name == "Bash"

    ctx.in_flight_tools.pop(event.tool_use_id, None)
    ctx.nested.pop(event.tool_use_id, None)
    ctx.nested_tool_ids = {k: v for k, v in ctx.nested_tool_ids.items() if v != event.tool_use_id}

    ctx.finished_tools.append(
        FinishedTool(
            tool_name=tool_name,
            input_summary=input_summary,
            duration_ms=event.duration_ms,
            result=event.result,
            ended_at=event.at // _NS_PER_MS,
        )
    )

    if tracked is not None:
        ctx.liveness = _next_liveness_after_tool_end(ctx, event)

    if should_refresh_session:
        _refresh_session(ctx)
    else:
        render_live(ctx)


def _handle_thinking_update(ctx: ReporterCtx, event: ThinkingUpdateEvent) -> None:
    if event.parent_tool_use_id is not None:
        return
    if isinstance(ctx.liveness, (Capped, InFlight)):
        return
    if isinstance(ctx.liveness, Thinking):
        ctx.liveness = Thinking(since=ctx.liveness.since, estimated_tokens=event.estimated_tokens)
    else:
        ctx.liveness = Thinking(
            since=event.at // _NS_PER_MS, estimated_tokens=event.estimated_tokens
        )
    render_live(ctx)


def _handle_model_phase(ctx: ReporterCtx, event: ModelPhaseEvent) -> None:
    at_ms = event.at // _NS_PER_MS
    if event.parent_tool_use_id is not None:
        parent_id = event.parent_tool_use_id
        if parent_id not in ctx.in_flight_tools:
            return
        if event.phase == "turn_end":
            ctx.nested.pop(parent_id, None)
        elif not isinstance(ctx.nested.get(parent_id), NestedTool):
            tool_name = event.tool_name if event.phase == "tool_input" else None
            ctx.nested[parent_id] = NestedPhase(phase=event.phase, since=at_ms, tool_name=tool_name)
        return

    if isinstance(ctx.liveness, (Capped, InFlight)):
        return

    match event.phase:
        case "thinking":
            tokens = ctx.liveness.estimated_tokens if isinstance(ctx.liveness, Thinking) else 0
            ctx.liveness = Thinking(since=at_ms, estimated_tokens=tokens)
        case "responding":
            ctx.liveness = Responding(since=at_ms)
        case "tool_input":
            tool_name = event.tool_name if event.tool_name is not None else "unknown"
            ctx.liveness = Composing(tool_name=tool_name, since=at_ms)
        case "turn_end":
            ctx.liveness = _waiting_from_last_tool(ctx, at_ms)
    render_live(ctx)


_FOLLOW_UP_LABELS: dict[str, str] = {"replied": "replied", "waiting": "waiting for gymrat"}


def _handle_turn_end(ctx: ReporterCtx, event: TurnEndEvent) -> None:
    ctx.turn_count += 1
    if event.origin == "agent":
        ctx.last_agent_text = event.text
    if not isinstance(ctx.liveness, Capped):
        ctx.liveness = _waiting_from_last_tool(ctx, event.at // _NS_PER_MS)
    render_live(ctx)


def _handle_follow_up(ctx: ReporterCtx, event: FollowUpEvent) -> None:
    if event.action == "ended":
        label = f"ended {event.reason}" if event.reason else "ended"
    else:
        label = _FOLLOW_UP_LABELS.get(event.action, event.action)
    decision = f"turn {ctx.turn_count} ended · {label}"
    ctx.last_decision = decision
    _emit(ctx, decision)


def _handle_tool_event(
    ctx: ReporterCtx, event: ToolStartEvent | ToolEndEvent | ToolProgressEvent
) -> None:
    match event:
        case ToolStartEvent():
            _handle_tool_start(ctx, event)
        case ToolEndEvent():
            _handle_tool_end(ctx, event)
        case ToolProgressEvent():
            render_live(ctx)


def _handle_compaction(ctx: ReporterCtx) -> None:
    ctx.last_decision = "context compacted"
    _emit(ctx, ctx.last_decision)


def handle_event(ctx: ReporterCtx, event: SessionEvent) -> None:  # noqa: C901 -- flat match over the event union
    """Dispatch a session event to the matching handler.

    Args:
        ctx: The shared reporter context mutated by each handler.
        event: The incoming session event to process.
    """
    match event:
        case CapEvent():
            _handle_cap_event(ctx, event)
        case LaunchEvent():
            _handle_launch(ctx, event)
        case UsageUpdateEvent():
            _handle_usage_update(ctx, event)
        case ToolStartEvent() | ToolEndEvent() | ToolProgressEvent():
            _handle_tool_event(ctx, event)
        case ThinkingUpdateEvent():
            _handle_thinking_update(ctx, event)
        case ModelPhaseEvent():
            _handle_model_phase(ctx, event)
        case TextDeltaEvent():
            pass  # no dashboard rendering depends on streamed text deltas
        case TurnEndEvent():
            _handle_turn_end(ctx, event)
        case FollowUpEvent():
            _handle_follow_up(ctx, event)
        case CompactionEvent():
            _handle_compaction(ctx)
        case _:  # pragma: no cover - exhaustive over the event union
            assert_never(event)
