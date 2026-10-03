"""Record-to-attribute mapping for OpenTelemetry spans (pure dict, no OTel imports)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from gymrat.errors import TOOL_FAILURE_EXIT_CODE
from gymrat.session.records import (
    CommandRecord,
    IterationRecord,
    SessionLogRecord,
    _SequencedEnvelope,
)
from gymrat.supervisor.events import CapEvent, CompactionEvent, FollowUpEvent, TurnEndEvent
from gymrat.telemetry.ids import parse_traceparent

if TYPE_CHECKING:
    from opentelemetry.trace import SpanContext

    from gymrat.supervisor.events import SessionEvent

type _Attrs = dict[str, str | int | float | bool]

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


def run_attributes(  # noqa: PLR0913 -- one parameter per launch fact the run span carries
    *,
    session_id: str,
    head_sha: str,
    max_minutes: float,
    max_usd: float | None,
    effort: str | None,
    model: str | None,
) -> _Attrs:
    """Build the attributes a run span starts with.

    Args:
        session_id: The session the run belongs to.
        head_sha: The HEAD commit the run launched from.
        max_minutes: The run's wall-clock cap.
        max_usd: The run's spend cap, left out when ``None``.
        effort: The agent effort level, left out when ``None``.
        model: The model name, left out when ``None``.

    Returns:
        The flat attribute dict for the run span.
    """
    attrs: _Attrs = {
        SESSION_ID: session_id,
        RUN_HEAD_SHA: head_sha,
        RUN_MAX_MINUTES: max_minutes,
        GEN_AI_PROVIDER: "anthropic",
    }
    if max_usd is not None:
        attrs[RUN_MAX_USD] = max_usd
    if effort is not None:
        attrs[RUN_EFFORT] = effort
    if model is not None:
        attrs[GEN_AI_MODEL] = model
    return attrs


def run_event(event: SessionEvent) -> tuple[str, _Attrs] | None:
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
        attrs: _Attrs = {FOLLOW_UP_ACTION: event.action}
        if event.reason is not None:
            attrs[FOLLOW_UP_REASON] = event.reason
        return EVENT_FOLLOW_UP, attrs
    if isinstance(event, CapEvent):
        return EVENT_CAP, {CAP_NAME: event.cap}
    if isinstance(event, CompactionEvent):
        return EVENT_COMPACTION, {}
    return None


@dataclass(frozen=True, slots=True)
class CommandSpanInputs:
    """Pre-computed span inputs shared by live and replay command-span emitters.

    Attributes:
        name: The span name.
        key: The key the span's deterministic id is derived from.
        attributes: The span's attributes.
        link: The span context the command was launched under, when it recorded one.
        status: The name of the OpenTelemetry status code the span ends with,
            or ``None`` to leave the status unset (a gate trip is not an error).
        status_description: The reason an ``ERROR`` status carries, when the
            record has one.
    """

    name: str
    key: str
    attributes: _Attrs
    link: SpanContext | None
    status: Literal["OK", "ERROR"] | None
    status_description: str | None


def command_span_inputs(
    record: CommandRecord, *, session_id: str, line_number: int
) -> CommandSpanInputs:
    """Build the span name, key, attributes, link, and status for a command record.

    Args:
        record: The command record the span stands for.
        session_id: The session the command ran in.
        line_number: The record's line in the session log, which keys the span id.

    Returns:
        The inputs both the live and the replayed command span are built from.
    """
    link = parse_traceparent(record.traceparent) if record.traceparent else None
    failed = record.exit_code == TOOL_FAILURE_EXIT_CODE
    return CommandSpanInputs(
        name=f"gymrat.command.{record.name}",
        key=f"command:{line_number}",
        attributes=command_attributes(record, session_id),
        link=link,
        status="OK" if record.exit_code == 0 else "ERROR" if failed else None,
        status_description=record.reason if failed else None,
    )


def _add_seq(attrs: _Attrs, record: _SequencedEnvelope) -> None:
    """Add the iteration sequence number, when the record carries one."""
    if record.seq is not None:
        attrs[ITERATION_SEQ] = record.seq


def command_attributes(record: CommandRecord, session_id: str) -> _Attrs:
    """Map a ``CommandRecord`` to a flat attribute dict for a command span."""
    attrs: _Attrs = {
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


def record_event(record: SessionLogRecord) -> tuple[str, _Attrs]:
    """Map a non-command session log record to ``(event_name, attributes)``."""
    record_type: str = record.type
    name = f"gymrat.{record_type}"
    attrs: _Attrs = {}

    if isinstance(record, _SequencedEnvelope):
        _add_seq(attrs, record)

    if isinstance(record, IterationRecord):
        attrs[ITERATION_OUTCOME] = record.outcome
        if record.primary.delta_pct is not None:
            attrs[ITERATION_DELTA_PCT] = record.primary.delta_pct
    else:
        _add_scalar_fields(attrs, record_type, record)

    return name, attrs


def _add_scalar_fields(attrs: _Attrs, record_type: str, record: SessionLogRecord) -> None:
    """Add scalar top-level fields from a non-iteration record under ``gymrat.<type>.<field>``."""
    for field_name, value in record:
        if field_name in _SKIPPED_FIELD_NAMES:
            continue
        if isinstance(value, _SCALAR_TYPES):
            attrs[_record_attr_name(record_type, field_name)] = value
