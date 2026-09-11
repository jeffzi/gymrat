"""Record-to-attribute mapping for OpenTelemetry spans (pure dict, no OTel imports)."""

from __future__ import annotations

import functools
import types
import typing
from dataclasses import dataclass
from typing import TYPE_CHECKING

from gymrat.session.records.models import (
    CommandRecord,
    IterationRecord,
    SessionLogRecord,
    SessionRecord,
    _SequencedEnvelope,
)
from gymrat.telemetry.ids import parse_traceparent

if TYPE_CHECKING:
    from opentelemetry.trace import SpanContext

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

TURN_COST_USD = "gymrat.turn.cost_usd"
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

# ---------------------------------------------------------------------------
# Namespace sets for all_attribute_names()
# ---------------------------------------------------------------------------

_SESSION_ATTRS = frozenset({SESSION_ID, SESSION_BRANCH})

_COMMAND_ATTRS = frozenset({
    COMMAND_NAME,
    COMMAND_EXIT_CODE,
    COMMAND_DURATION_MS,
    COMMAND_REASON,
    COMMAND_ARGS_PREFIX,
})

_RUN_ATTRS = frozenset({
    RUN_HEAD_SHA,
    RUN_MAX_MINUTES,
    RUN_MAX_USD,
    RUN_EFFORT,
    RUN_COST_USD,
    RUN_ENDED_BY,
    RUN_END_REASON,
    RUN_DURATION_MS,
})

_TURN_AND_EVENT_ATTRS = frozenset({
    TURN_COST_USD,
    TURN_ORIGIN,
    TURN_BUDGET_EXHAUSTED,
    FOLLOW_UP_ACTION,
    FOLLOW_UP_REASON,
    CAP_NAME,
})

_GEN_AI_ATTRS = frozenset({GEN_AI_MODEL, GEN_AI_PROVIDER})

_ITERATION_ATTRS = frozenset({ITERATION_SEQ, ITERATION_OUTCOME, ITERATION_DELTA_PCT})

# Record fields carried by the envelope, not mapped to a `gymrat.<type>.<field>` attribute.
_SKIPPED_FIELD_NAMES = frozenset({"at", "seq", "type"})


def _record_attr_name(record_type: str, field_name: str) -> str:
    """Build the ``gymrat.<type>.<field>`` attribute name."""
    return f"gymrat.{record_type}.{field_name}"


def _literal_value(annotation: object) -> str:
    """Extract the single string value from a ``Literal['...']`` annotation."""
    args = typing.get_args(annotation)
    return args[0]


def _is_scalar_or_literal(annotation: object) -> bool:
    """True when ``annotation`` is a bare scalar type or a ``Literal[...]``."""
    return annotation in _SCALAR_TYPES or typing.get_origin(annotation) is typing.Literal


def _is_scalar_type(annotation: object) -> bool:
    """True when ``annotation`` resolves to a scalar OTel attribute type."""
    if _is_scalar_or_literal(annotation):
        return True
    if typing.get_origin(annotation) in {typing.Union, types.UnionType}:
        return all(_is_scalar_or_none(arg) for arg in typing.get_args(annotation))
    return False


def _is_scalar_or_none(annotation: object) -> bool:
    """True when ``annotation`` is a scalar type, NoneType, a Literal, or Annotated wrapping one."""
    if _is_scalar_or_literal(annotation) or annotation is type(None):
        return True
    origin = typing.get_origin(annotation)
    if origin is typing.Annotated:
        inner = typing.get_args(annotation)
        return _is_scalar_or_literal(inner[0]) or inner[0] is type(None) if inner else False
    return False


@functools.cache
def all_attribute_names() -> frozenset[str]:
    """Return every attribute name the telemetry layer can emit.

    The set includes fixed constants for session, run, turn, follow-up, cap,
    and command spans, derived ``gymrat.<type>.<field>`` names from record
    models other than session, iteration, and command — those are covered
    by the fixed constants above and are excluded here — iteration names,
    ``gen_ai.*`` names, and the ``gymrat.command.args`` pattern placeholder.

    Returns:
        A frozenset of dotted attribute name strings.
    """
    record_derived: set[str] = set()
    for record_cls in typing.get_args(SessionLogRecord.__value__):
        if record_cls in (SessionRecord, IterationRecord, CommandRecord):
            continue
        record_type = _literal_value(record_cls.model_fields["type"].annotation)
        for field_name, field_info in record_cls.model_fields.items():
            if field_name in _SKIPPED_FIELD_NAMES:
                continue
            raw_annotation = field_info.annotation
            origin = typing.get_origin(raw_annotation)
            if origin is typing.Annotated:
                inner_args = typing.get_args(raw_annotation)
                raw_annotation = inner_args[0] if inner_args else raw_annotation
            if _is_scalar_type(raw_annotation):
                record_derived.add(_record_attr_name(record_type, field_name))

    return (
        _SESSION_ATTRS
        | _COMMAND_ATTRS
        | _RUN_ATTRS
        | _TURN_AND_EVENT_ATTRS
        | _GEN_AI_ATTRS
        | _ITERATION_ATTRS
        | frozenset(record_derived)
    )


@dataclass(frozen=True, slots=True)
class CommandSpanInputs:
    """Pre-computed span inputs shared by live and replay command-span emitters."""

    name: str
    key: str
    attributes: _Attrs
    link: SpanContext | None


def command_span_inputs(
    record: CommandRecord, *, session_id: str, line_number: int
) -> CommandSpanInputs:
    """Build the span name, key, attributes, and link for a command record."""
    link = parse_traceparent(record.traceparent) if record.traceparent else None
    return CommandSpanInputs(
        name=f"gymrat.command.{record.name}",
        key=f"command:{line_number}",
        attributes=command_attributes(record, session_id),
        link=link,
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
