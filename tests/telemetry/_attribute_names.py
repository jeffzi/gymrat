"""Enumerates every attribute name the telemetry layer can emit.

The README documents the attribute namespace as a table; the drift test compares
that table against the set built here.
"""

from __future__ import annotations

import functools
import types
import typing

from gymrat.session.records import (
    SESSION_LOG_MODELS,
    CommandRecord,
    IterationRecord,
    SessionRecord,
    wire_type,
)
from gymrat.telemetry.attributes import (
    _SCALAR_TYPES,
    _SKIPPED_FIELD_NAMES,
    CAP_NAME,
    COMMAND_ARGS_PREFIX,
    COMMAND_DURATION_MS,
    COMMAND_EXIT_CODE,
    COMMAND_NAME,
    COMMAND_REASON,
    FOLLOW_UP_ACTION,
    FOLLOW_UP_REASON,
    GEN_AI_MODEL,
    GEN_AI_PROVIDER,
    ITERATION_DELTA_PCT,
    ITERATION_OUTCOME,
    ITERATION_SEQ,
    RUN_COST_USD,
    RUN_DURATION_MS,
    RUN_EFFORT,
    RUN_END_REASON,
    RUN_ENDED_BY,
    RUN_HEAD_SHA,
    RUN_MAX_MINUTES,
    RUN_MAX_USD,
    SESSION_BRANCH,
    SESSION_ID,
    TURN_BUDGET_EXHAUSTED,
    TURN_ORIGIN,
    TURN_SESSION_COST_USD,
    _record_attr_name,
)

# The attribute names that are not derived from a record model's fields.
_FIXED_ATTRS = frozenset({
    SESSION_ID,
    SESSION_BRANCH,
    COMMAND_NAME,
    COMMAND_EXIT_CODE,
    COMMAND_DURATION_MS,
    COMMAND_REASON,
    COMMAND_ARGS_PREFIX,
    RUN_HEAD_SHA,
    RUN_MAX_MINUTES,
    RUN_MAX_USD,
    RUN_EFFORT,
    RUN_COST_USD,
    RUN_ENDED_BY,
    RUN_END_REASON,
    RUN_DURATION_MS,
    TURN_SESSION_COST_USD,
    TURN_ORIGIN,
    TURN_BUDGET_EXHAUSTED,
    FOLLOW_UP_ACTION,
    FOLLOW_UP_REASON,
    CAP_NAME,
    GEN_AI_MODEL,
    GEN_AI_PROVIDER,
    ITERATION_SEQ,
    ITERATION_OUTCOME,
    ITERATION_DELTA_PCT,
})


def _is_scalar_or_literal(annotation: object) -> bool:
    return annotation in _SCALAR_TYPES or typing.get_origin(annotation) is typing.Literal


def _is_scalar_or_none(annotation: object) -> bool:
    if typing.get_origin(annotation) is typing.Annotated:
        inner = typing.get_args(annotation)
        if inner:
            annotation = inner[0]
    return _is_scalar_or_literal(annotation) or annotation is type(None)


def _is_scalar_type(annotation: object) -> bool:
    if _is_scalar_or_literal(annotation):
        return True
    if typing.get_origin(annotation) in {typing.Union, types.UnionType}:
        return all(_is_scalar_or_none(arg) for arg in typing.get_args(annotation))
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
    for record_cls in SESSION_LOG_MODELS:
        if record_cls in (SessionRecord, IterationRecord, CommandRecord):
            continue
        record_type = wire_type(record_cls)
        for field_name, field_info in record_cls.model_fields.items():
            if field_name in _SKIPPED_FIELD_NAMES:
                continue
            if _is_scalar_type(field_info.annotation):
                record_derived.add(_record_attr_name(record_type, field_name))

    return _FIXED_ATTRS | frozenset(record_derived)
