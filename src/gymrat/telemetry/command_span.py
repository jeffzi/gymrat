"""The span a command record stands for, shared by the live and the replay emitters.

Both emitters start a command's span here, so a live command and its replay
carry the same name, id, attributes, link and status. All ``opentelemetry``
imports live inside functions, as in :mod:`gymrat.telemetry.provider`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from gymrat.errors import TOOL_FAILURE_EXIT_CODE
from gymrat.session.records import SCALAR_ATTRIBUTE_TYPES, add_iteration_seq
from gymrat.telemetry.provider import SESSION_ID, parse_traceparent, start_span

if TYPE_CHECKING:
    from opentelemetry.context import Context
    from opentelemetry.trace import Span, SpanContext

    from gymrat.session.records import CommandRecord
    from gymrat.telemetry.provider import Attrs

COMMAND_NAME = "gymrat.command.name"
COMMAND_EXIT_CODE = "gymrat.command.exit_code"
COMMAND_DURATION_MS = "gymrat.command.duration_ms"
COMMAND_REASON = "gymrat.command.reason"
COMMAND_ARGS_PREFIX = "gymrat.command.args"


def command_attributes(record: CommandRecord, session_id: str) -> Attrs:
    """Map a ``CommandRecord`` to the attributes of a command span.

    Args:
        record: The command record; its reason and iteration sequence number
            are left out when ``None``, and only its scalar args are kept.
        session_id: The session the command ran in.

    Returns:
        The flat attribute dict for the command span.
    """
    attrs: Attrs = {
        SESSION_ID: session_id,
        COMMAND_NAME: record.name,
        COMMAND_EXIT_CODE: record.exit_code,
        COMMAND_DURATION_MS: record.duration_ms,
    }
    if record.reason is not None:
        attrs[COMMAND_REASON] = record.reason
    add_iteration_seq(attrs, record)
    for key, val in record.args.items():
        if isinstance(val, SCALAR_ATTRIBUTE_TYPES):
            attrs[f"{COMMAND_ARGS_PREFIX}.{key}"] = val
    return attrs


def command_link(record: CommandRecord) -> SpanContext | None:
    """The span context a command was launched under, read from its recorded traceparent.

    Args:
        record: The command record.

    Returns:
        The remote span context, or ``None`` when the record carries no
        traceparent or a malformed one.
    """
    return parse_traceparent(record.traceparent) if record.traceparent else None


def start_command_span(
    record: CommandRecord,
    *,
    session_id: str,
    line_number: int,
    context: Context,
    start_time: int,
) -> Span:
    """Start the span a command record stands for.

    Shared by the live and the replay emitters, so both give a command the same
    name, id, attributes, link and status. The span links to the span context
    the command was launched under, when it recorded one. An exit code of 0
    ends ``OK`` and a tool failure ends ``ERROR`` with the record's reason; a
    gate trip is not an error and leaves the status unset.

    Args:
        record: The command record the span stands for.
        session_id: The session the command ran in, which keys the span id
            even when tracing was configured for another session.
        line_number: The record's line in the session log, which keys the span id.
        context: The parent context the span starts under.
        start_time: When the command started, in nanoseconds since the epoch.

    Returns:
        The started span, linked and with its status set.
    """
    from opentelemetry.trace import Link, Status, StatusCode  # noqa: PLC0415 -- optional extra

    link = command_link(record)
    span = start_span(
        f"gymrat.command.{record.name}",
        span_key=f"command:{line_number}",
        session_id=session_id,
        context=context,
        links=[Link(link)] if link is not None else None,
        attributes=command_attributes(record, session_id),
        start_time=start_time,
    )
    if record.exit_code == 0:
        span.set_status(Status(StatusCode.OK))
    elif record.exit_code == TOOL_FAILURE_EXIT_CODE:
        span.set_status(Status(StatusCode.ERROR, description=record.reason))
    return span
