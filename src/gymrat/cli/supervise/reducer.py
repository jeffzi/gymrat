"""Pure state transitions for the supervise dashboard.

:class:`ReporterState` holds everything the dashboard renders; :func:`advance`
holds every rule for how an event changes it.  Nothing here reads a clock, opens
a file, or touches a terminal — the shell in :mod:`gymrat.cli.supervise.progress`
supplies the session read, then replaces its state with whatever ``advance``
returns.  That keeps the transitions testable without a terminal and makes every
frame a function of the state alone.

Keep this module clear of the Rich view layer: plain text it shares with the
frame lives in :mod:`gymrat.cli.supervise.text`, so importing the reducer never
drags a terminal library in.

Mapping-shaped fields are tuples of key/value pairs rather than dicts so the
state stays immutable.  They hold a handful of entries at most, so the linear
lookups below cost nothing measurable.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, assert_never

from gymrat.cli.supervise.state import (
    Capped,
    Composing,
    FinishedTool,
    InFlight,
    NestedPhase,
    NestedTool,
    Responding,
    Starting,
    Thinking,
    TrackedTool,
    Waiting,
)
from gymrat.cli.supervise.text import (
    NO_SESSION_TEXT,
    format_caps,
    format_cost,
    loop_plain_text,
)
from gymrat.supervisor.events import (
    CapEvent,
    CompactionEvent,
    FollowUpEvent,
    LaunchEvent,
    ModelPhaseEvent,
    TextDeltaEvent,
    ThinkingUpdateEvent,
    ToolEndEvent,
    ToolProgressEvent,
    ToolStartEvent,
    TurnEndEvent,
    UsageUpdateEvent,
)

if TYPE_CHECKING:
    from gymrat.cli.supervise.state import (
        Liveness,
        NestedActivity,
        ReadSessionResult,
    )
    from gymrat.config import Effort
    from gymrat.supervisor.events import SessionEvent

_NS_PER_MS = 1_000_000

_MAX_FINISHED_TOOLS = 3

_FOLLOW_UP_LABELS: dict[str, str] = {"replied": "replied", "waiting": "waiting for gymrat"}

#: Tool whose completion is worth a session re-read: only gymrat sub-commands
#: run through it, and they are what write session records.
_SESSION_WRITING_TOOL = "Bash"


def _ms(at_ns: int) -> int:
    """Convert an event's nanosecond timestamp to whole milliseconds."""
    return at_ns // _NS_PER_MS


@dataclass(frozen=True, slots=True, kw_only=True)
class ReporterState:
    """Everything the supervise dashboard renders, as of the last event.

    The run facts (``root`` through ``log_path``) are fixed when the reporter is
    built; the remaining fields are what :func:`advance` folds events into.

    Attributes:
        root: The working directory the supervised run executes in.
        max_minutes: The run's wall-clock cap, in minutes.
        max_usd: The run's cost cap in USD, or ``None`` when uncapped.
        max_iterations: The run's iteration cap, or ``None`` when uncapped.
        label: The run's display label.
        session_id: The supervised session's id.
        branch: The git branch the run is on.
        model: The model name the run is configured to use, or ``None`` when
            unset.
        effort: The configured reasoning effort, or ``None`` when unset.
        log_path: Path to the session log file.
        in_flight_tools: Top-level tools seen started but not yet ended, as
            ``(tool_use_id, TrackedTool)`` pairs.
        finished_tools: The last :data:`_MAX_FINISHED_TOOLS` top-level tools
            that have ended, oldest first.
        launch_timestamp: Milliseconds timestamp of the launch event, or
            ``None`` before the run has launched.
        cost_usd: The most recently reported running cost, or ``None`` before
            any usage update has arrived.
        session_result: The session state as of the last re-read, or ``None``
            before the first re-read.
        liveness: What the model or top-level tool is doing right now.
        last_loop_text: The most recently rendered loop-progress text.
        nested: What each running nested subagent tool is doing, as
            ``(parent_tool_use_id, NestedActivity)`` pairs keyed by the
            top-level tool-use id the subagent runs under.
        nested_tool_ids: Live nested tool-use ids, as
            ``(tool_use_id, parent_tool_use_id)`` pairs mapping a nested
            tool's own id to the top-level tool-use id it is nested under.
        last_agent_text: The text of the last turn-end event whose origin was
            the agent, or ``None`` before any such turn has ended.
        turn_count: The number of turns that have ended so far.
        last_decision: The most recent follow-up or compaction decision text,
            or ``None`` before either has occurred.
    """

    root: str
    max_minutes: float
    max_usd: float | None = None
    max_iterations: int | None = None
    label: str = ""
    session_id: str = ""
    branch: str = ""
    model: str | None = None
    effort: Effort | None = None
    log_path: str = ""

    in_flight_tools: tuple[tuple[str, TrackedTool], ...] = ()
    finished_tools: tuple[FinishedTool, ...] = ()
    launch_timestamp: int | None = None
    cost_usd: float | None = None
    session_result: ReadSessionResult | None = None
    liveness: Liveness = field(default_factory=Starting)
    last_loop_text: str = ""
    nested: tuple[tuple[str, NestedActivity], ...] = ()
    nested_tool_ids: tuple[tuple[str, str], ...] = ()
    last_agent_text: str | None = None
    turn_count: int = 0
    last_decision: str | None = None


# ---------------------------------------------------------------------------
# Tuple-mapping helpers
# ---------------------------------------------------------------------------


def _get[V](pairs: tuple[tuple[str, V], ...], key: str) -> V | None:
    return next((value for name, value in pairs if name == key), None)


def _set[V](pairs: tuple[tuple[str, V], ...], key: str, value: V) -> tuple[tuple[str, V], ...]:
    if _get(pairs, key) is None:
        return (*pairs, (key, value))
    # Overwriting keeps the original position, matching dict assignment: the
    # tool-end fallback relies on the last pair being the newest tool.
    return tuple((name, value if name == key else held) for name, held in pairs)


def _drop[V](pairs: tuple[tuple[str, V], ...], key: str) -> tuple[tuple[str, V], ...]:
    return tuple((name, value) for name, value in pairs if name != key)


# ---------------------------------------------------------------------------
# Session refresh
# ---------------------------------------------------------------------------


def wants_session_refresh(state: ReporterState, event: SessionEvent) -> bool:
    """Whether *event* should make the shell re-read the session log.

    Args:
        state: The state as of just before *event*.
        event: The incoming session event.

    Returns:
        ``True`` for a launch and for a top-level tool end that either ran Bash
        or carries a tool-use id the reporter never saw start.
    """
    match event:
        case LaunchEvent():
            return True
        case ToolEndEvent() if event.parent_tool_use_id is None:
            tracked = _get(state.in_flight_tools, event.tool_use_id)
            return tracked is None or tracked.tool_name == _SESSION_WRITING_TOOL
        case _:
            return False


# ---------------------------------------------------------------------------
# Liveness transitions
# ---------------------------------------------------------------------------


def _waiting_from_last_tool(finished: tuple[FinishedTool, ...], timestamp: int) -> Waiting:
    last = finished[-1] if finished else None
    return Waiting(
        since=timestamp,
        tool_name=last.tool_name if last is not None else None,
        tool_ended_at=last.ended_at if last is not None else None,
        result=last.result if last is not None else None,
    )


def _next_liveness_after_tool_end(
    liveness: Liveness,
    remaining: tuple[tuple[str, TrackedTool], ...],
    event: ToolEndEvent,
) -> Capped | InFlight | Waiting:
    if isinstance(liveness, Capped):
        return liveness
    if remaining:
        tool_id, tracked = remaining[-1]
        return InFlight(
            tool_use_id=tool_id,
            tool_name=tracked.tool_name,
            since=tracked.started_at,
            input_summary=tracked.input_summary,
        )
    ended_at_ms = _ms(event.at)
    return Waiting(
        since=ended_at_ms,
        tool_name=event.tool_name,
        tool_ended_at=ended_at_ms,
        result=event.result,
    )


def _phase_liveness(state: ReporterState, event: ModelPhaseEvent, at_ms: int) -> Liveness:
    match event.phase:
        case "thinking":
            tokens = state.liveness.estimated_tokens if isinstance(state.liveness, Thinking) else 0
            return Thinking(since=at_ms, estimated_tokens=tokens)
        case "responding":
            return Responding(since=at_ms)
        case "tool_input":
            tool_name = event.tool_name if event.tool_name is not None else "unknown"
            return Composing(tool_name=tool_name, since=at_ms)
        case "turn_end":
            return _waiting_from_last_tool(state.finished_tools, at_ms)
        case _:  # pragma: no cover - exhaustive over the phase literals
            assert_never(event.phase)


# ---------------------------------------------------------------------------
# Per-event transitions
# ---------------------------------------------------------------------------


def _tool_start(state: ReporterState, event: ToolStartEvent) -> ReporterState:
    at_ms = _ms(event.at)
    parent_id = event.parent_tool_use_id
    if parent_id is not None:
        if _get(state.in_flight_tools, parent_id) is None:
            return state
        nested = _set(
            state.nested,
            parent_id,
            NestedTool(tool_name=event.tool_name, input_summary=event.input_summary, since=at_ms),
        )
        return replace(
            state,
            nested=nested,
            nested_tool_ids=_set(state.nested_tool_ids, event.tool_use_id, parent_id),
        )

    in_flight = _set(
        state.in_flight_tools,
        event.tool_use_id,
        TrackedTool(tool_name=event.tool_name, started_at=at_ms, input_summary=event.input_summary),
    )
    liveness = state.liveness
    if not isinstance(liveness, Capped):
        liveness = InFlight(
            tool_use_id=event.tool_use_id,
            tool_name=event.tool_name,
            since=at_ms,
            input_summary=event.input_summary,
        )
    return replace(state, in_flight_tools=in_flight, liveness=liveness)


def _nested_tool_end(state: ReporterState, event: ToolEndEvent) -> ReporterState:
    parent_id = _get(state.nested_tool_ids, event.tool_use_id)
    if parent_id is None:
        return state
    nested = state.nested
    if isinstance(_get(nested, parent_id), NestedTool):
        nested = _drop(nested, parent_id)
    return replace(
        state, nested=nested, nested_tool_ids=_drop(state.nested_tool_ids, event.tool_use_id)
    )


def _next_loop_text(state: ReporterState, session_result: ReadSessionResult | None) -> str:
    plain = loop_plain_text(session_result, state.max_iterations)
    if plain in {state.last_loop_text, NO_SESSION_TEXT}:
        return state.last_loop_text
    return plain


def _tool_end(
    state: ReporterState, event: ToolEndEvent, session_result: ReadSessionResult | None
) -> ReporterState:
    if event.parent_tool_use_id is not None:
        return _nested_tool_end(state, event)

    tracked = _get(state.in_flight_tools, event.tool_use_id)
    in_flight = _drop(state.in_flight_tools, event.tool_use_id)
    finished = (
        *state.finished_tools,
        FinishedTool(
            tool_name=tracked.tool_name if tracked is not None else event.tool_name,
            input_summary=tracked.input_summary if tracked is not None else "",
            duration_ms=event.duration_ms,
            result=event.result,
            ended_at=_ms(event.at),
        ),
    )
    liveness = state.liveness
    if tracked is not None:
        liveness = _next_liveness_after_tool_end(state.liveness, in_flight, event)

    ended = replace(
        state,
        in_flight_tools=in_flight,
        nested=_drop(state.nested, event.tool_use_id),
        nested_tool_ids=tuple(
            (name, parent) for name, parent in state.nested_tool_ids if parent != event.tool_use_id
        ),
        finished_tools=finished[-_MAX_FINISHED_TOOLS:],
        liveness=liveness,
    )
    if not wants_session_refresh(state, event):
        return ended
    return replace(
        ended,
        session_result=session_result,
        last_loop_text=_next_loop_text(state, session_result),
    )


def _thinking_update(state: ReporterState, event: ThinkingUpdateEvent) -> ReporterState:
    if event.parent_tool_use_id is not None:
        return state
    if isinstance(state.liveness, (Capped, InFlight)):
        return state
    since = state.liveness.since if isinstance(state.liveness, Thinking) else _ms(event.at)
    return replace(state, liveness=Thinking(since=since, estimated_tokens=event.estimated_tokens))


def _nested_model_phase(
    state: ReporterState, event: ModelPhaseEvent, parent_id: str, at_ms: int
) -> ReporterState:
    if _get(state.in_flight_tools, parent_id) is None:
        return state
    if event.phase == "turn_end":
        return replace(state, nested=_drop(state.nested, parent_id))
    if isinstance(_get(state.nested, parent_id), NestedTool):
        return state
    tool_name = event.tool_name if event.phase == "tool_input" else None
    phase = NestedPhase(phase=event.phase, since=at_ms, tool_name=tool_name)
    return replace(state, nested=_set(state.nested, parent_id, phase))


def _model_phase(state: ReporterState, event: ModelPhaseEvent) -> ReporterState:
    at_ms = _ms(event.at)
    if event.parent_tool_use_id is not None:
        return _nested_model_phase(state, event, event.parent_tool_use_id, at_ms)
    if isinstance(state.liveness, (Capped, InFlight)):
        return state
    return replace(state, liveness=_phase_liveness(state, event, at_ms))


def _turn_end(state: ReporterState, event: TurnEndEvent) -> ReporterState:
    liveness = state.liveness
    if not isinstance(liveness, Capped):
        liveness = _waiting_from_last_tool(state.finished_tools, _ms(event.at))
    return replace(
        state,
        turn_count=state.turn_count + 1,
        last_agent_text=event.text if event.origin == "agent" else state.last_agent_text,
        liveness=liveness,
    )


def _follow_up_decision(state: ReporterState, event: FollowUpEvent) -> str:
    if event.action == "ended":
        label = f"ended {event.reason}" if event.reason else "ended"
    else:
        label = _FOLLOW_UP_LABELS.get(event.action, event.action)
    return f"turn {state.turn_count} ended · {label}"


# ---------------------------------------------------------------------------
# Reducer
# ---------------------------------------------------------------------------


def advance(  # noqa: C901 -- flat match over the event union
    state: ReporterState,
    event: SessionEvent,
    session: ReadSessionResult | None,
) -> ReporterState:
    """Fold *event* into *state* and return the resulting state.

    The function is pure: the same arguments always produce an equal result, and
    *state* is never mutated.  Every transition stamps itself from ``event.at``,
    so a transition that comes to need the wall clock must take the caller's
    reading rather than call a clock and break that purity.

    Args:
        state: The state as of just before *event*.
        event: The incoming session event.
        session: The session read the shell performed for this event, or ``None``
            when no refresh was wanted or the read failed.

    Returns:
        The state after *event*, sharing unchanged fields with *state* — and
        *state* itself when *event* changes nothing.
    """
    match event:
        case CapEvent():
            advanced = replace(state, liveness=Capped(cap_type=event.cap, action=event.action))
        case LaunchEvent():
            advanced = replace(state, launch_timestamp=_ms(event.at), session_result=session)
        case UsageUpdateEvent():
            advanced = replace(state, cost_usd=event.cost_usd)
        case ToolStartEvent():
            advanced = _tool_start(state, event)
        case ToolEndEvent():
            advanced = _tool_end(state, event, session)
        case ToolProgressEvent() | TextDeltaEvent():
            advanced = state
        case ThinkingUpdateEvent():
            advanced = _thinking_update(state, event)
        case ModelPhaseEvent():
            advanced = _model_phase(state, event)
        case TurnEndEvent():
            advanced = _turn_end(state, event)
        case FollowUpEvent():
            advanced = replace(state, last_decision=_follow_up_decision(state, event))
        case CompactionEvent():
            advanced = replace(state, last_decision="context compacted")
        case _:  # pragma: no cover - exhaustive over the event union
            assert_never(event)
    return advanced


def plain_line(before: ReporterState, after: ReporterState, event: SessionEvent) -> str | None:
    """The plain-mode milestone line *event* produces, if any.

    Args:
        before: The state as of just before *event*.
        after: The state :func:`advance` returned for *event*.
        event: The session event that was just folded in.

    Returns:
        The line to write, or ``None`` when the event is not a plain-mode
        milestone.
    """
    match event:
        case LaunchEvent():
            return format_caps(after.max_minutes, after.max_usd)
        case UsageUpdateEvent():
            return f"cost {format_cost(event.cost_usd)}"
        case CapEvent():
            return f"cap {event.cap} — {event.action}"
        case FollowUpEvent() | CompactionEvent():
            return after.last_decision
        case _:
            if after.last_loop_text != before.last_loop_text:
                return after.last_loop_text
            return None
