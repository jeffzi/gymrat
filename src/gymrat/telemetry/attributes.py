"""Record-to-attribute mapping for OpenTelemetry spans (pure dict, no OTel imports)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from gymrat.session.records import (
    CommandRecord,
    IterationRecord,
    SessionLogRecord,
    _SequencedEnvelope,
)
from gymrat.supervisor.events import CapEvent, CompactionEvent, FollowUpEvent, TurnEndEvent

if TYPE_CHECKING:
    from gymrat.supervisor.events import LaunchEvent, SessionEvent

type Attrs = dict[str, str | int | float | bool]
"""The flat attribute dict a span or span event carries."""

_SCALAR_TYPES = (str, int, float, bool)

# ---------------------------------------------------------------------------
# Named attribute and span-name constants
# ---------------------------------------------------------------------------

SESSION_SPAN = "gymrat.session"
RUN_SPAN = "gymrat.run"

SESSION_ID = "gymrat.session.id"
SESSION_BRANCH = "gymrat.session.branch"

COMMAND_NAME = "gymrat.command.name"
COMMAND_EXIT_CODE = "gymrat.command.exit_code"
COMMAND_DURATION_MS = "gymrat.command.duration_ms"
COMMAND_REASON = "gymrat.command.reason"
COMMAND_ARGS_PREFIX = "gymrat.command.args"

RUN_HEAD_SHA = "gymrat.run.head_sha"
RUN_MAX_MINUTES = "gymrat.run.max_minutes"
RUN_MAX_USD = "gymrat.run.max_usd"
RUN_EFFORT = "gymrat.run.effort"
RUN_COST_USD = "gymrat.run.cost_usd"
RUN_ENDED_BY = "gymrat.run.ended_by"
RUN_END_REASON = "gymrat.run.end_reason"
RUN_DURATION_MS = "gymrat.run.duration_ms"

TURN_SESSION_COST_USD = "gymrat.turn.session_cost_usd"
TURN_ORIGIN = "gymrat.turn.origin"
TURN_BUDGET_EXHAUSTED = "gymrat.turn.budget_exhausted"
FOLLOW_UP_ACTION = "gymrat.follow_up.action"
FOLLOW_UP_REASON = "gymrat.follow_up.reason"
CAP_NAME = "gymrat.cap.name"

GEN_AI_MODEL = "gen_ai.request.model"
GEN_AI_PROVIDER = "gen_ai.provider.name"

ITERATION_SEQ = "gymrat.iteration.seq"
ITERATION_OUTCOME = "gymrat.iteration.outcome"
ITERATION_DELTA_PCT = "gymrat.iteration.delta_pct"

EVENT_TURN_END = "gymrat.turn_end"
EVENT_FOLLOW_UP = "gymrat.follow_up"
EVENT_CAP = "gymrat.cap"
EVENT_COMPACTION = "gymrat.compaction"

SESSION_SPAN_KEY = "session"
"""The key the session span's deterministic id is derived from."""

# Record fields carried by the envelope, not mapped to a `gymrat.<type>.<field>` attribute.
_SKIPPED_FIELD_NAMES = frozenset({"at", "seq", "type"})


def _record_attr_name(record_type: str, field_name: str) -> str:
    """Build the ``gymrat.<type>.<field>`` attribute name."""
    return f"gymrat.{record_type}.{field_name}"


def run_span_key(launch_at: int) -> str:
    """The key a run span's deterministic id is derived from."""
    return f"run:{launch_at}"


def run_attributes(launch: LaunchEvent) -> Attrs:
    """Build the attributes a run span starts with.

    Args:
        launch: The launch event of the run; its spend cap, effort and model
            are left out when ``None``.

    Returns:
        The flat attribute dict for the run span.
    """
    attrs: Attrs = {
        SESSION_ID: launch.session_id,
        RUN_HEAD_SHA: launch.head_sha,
        RUN_MAX_MINUTES: launch.max_minutes,
        GEN_AI_PROVIDER: "anthropic",
    }
    if launch.max_usd is not None:
        attrs[RUN_MAX_USD] = launch.max_usd
    if launch.effort is not None:
        attrs[RUN_EFFORT] = launch.effort
    if launch.model is not None:
        attrs[GEN_AI_MODEL] = launch.model
    return attrs


def run_event(event: SessionEvent) -> tuple[str, Attrs] | None:
    """Map a supervisor event to the span event a run span mirrors it as.

    Args:
        event: The supervisor event.

    Returns:
        The span event's name and attributes, or ``None`` for an event the run
        span does not mirror.
    """
    if isinstance(event, TurnEndEvent):
        return EVENT_TURN_END, {
            TURN_SESSION_COST_USD: event.cost_usd,
            TURN_ORIGIN: event.origin,
            TURN_BUDGET_EXHAUSTED: event.budget_exhausted,
        }
    if isinstance(event, FollowUpEvent):
        attrs: Attrs = {FOLLOW_UP_ACTION: event.action}
        if event.reason is not None:
            attrs[FOLLOW_UP_REASON] = event.reason
        return EVENT_FOLLOW_UP, attrs
    if isinstance(event, CapEvent):
        return EVENT_CAP, {CAP_NAME: event.cap}
    if isinstance(event, CompactionEvent):
        return EVENT_COMPACTION, {}
    return None


def _add_seq(attrs: Attrs, record: _SequencedEnvelope) -> None:
    """Add the iteration sequence number, when the record carries one."""
    if record.seq is not None:
        attrs[ITERATION_SEQ] = record.seq


def command_attributes(record: CommandRecord, session_id: str) -> Attrs:
    """Map a ``CommandRecord`` to a flat attribute dict for a command span."""
    attrs: Attrs = {
        SESSION_ID: session_id,
        COMMAND_NAME: record.name,
        COMMAND_EXIT_CODE: record.exit_code,
        COMMAND_DURATION_MS: record.duration_ms,
    }
    if record.reason is not None:
        attrs[COMMAND_REASON] = record.reason
    _add_seq(attrs, record)
    for key, val in record.args.items():
        if isinstance(val, _SCALAR_TYPES):
            attrs[f"{COMMAND_ARGS_PREFIX}.{key}"] = val
    return attrs


def record_event(record: SessionLogRecord) -> tuple[str, Attrs]:
    """Map a non-command session log record to ``(event_name, attributes)``."""
    record_type: str = record.type
    name = f"gymrat.{record_type}"
    attrs: Attrs = {}

    if isinstance(record, _SequencedEnvelope):
        _add_seq(attrs, record)

    if isinstance(record, IterationRecord):
        attrs[ITERATION_OUTCOME] = record.outcome
        if record.primary.delta_pct is not None:
            attrs[ITERATION_DELTA_PCT] = record.primary.delta_pct
    else:
        _add_scalar_fields(attrs, record_type, record)

    return name, attrs


def _add_scalar_fields(attrs: Attrs, record_type: str, record: SessionLogRecord) -> None:
    """Add scalar top-level fields from a non-iteration record under ``gymrat.<type>.<field>``."""
    for field_name, value in record:
        if field_name in _SKIPPED_FIELD_NAMES:
            continue
        if isinstance(value, _SCALAR_TYPES):
            attrs[_record_attr_name(record_type, field_name)] = value
