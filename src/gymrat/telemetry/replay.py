"""Replay session and supervisor logs into OpenTelemetry spans.

All ``opentelemetry`` imports live inside the function so importing this module
never pulls the SDK into ``sys.modules``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from gymrat.errors import TOOL_FAILURE_EXIT_CODE, GymratError
from gymrat.session.records import CommandRecord, SessionRecord, decode_log_line, parse_record
from gymrat.supervisor.events import (
    LaunchEvent,
    TurnEndEvent,
    UsageUpdateEvent,
    event_from_wire,
)
from gymrat.telemetry.attributes import (
    RUN_COST_USD,
    RUN_SPAN,
    SESSION_SPAN,
    SESSION_SPAN_KEY,
    Attrs,
    record_event,
    run_attributes,
    run_event,
    run_span_key,
)
from gymrat.telemetry.ids import parse_traceparent
from gymrat.telemetry.provider import start_command_span, start_span
from gymrat.utils import NS_PER_MS

if TYPE_CHECKING:
    from opentelemetry.context import Context
    from opentelemetry.trace import Span

    from gymrat.session.records import SessionLogRecord
    from gymrat.supervisor.events import SessionEvent

logger = logging.getLogger(__name__)

type _EventTuple = tuple[str, Attrs, int]
type _NumberedRecord = tuple[int, SessionLogRecord]
type _RunSpan = tuple[_ParsedRun, Span]

_UNKNOWN_COMMAND_NAME = "unknown"


def replay_session(
    session_log: str,
    supervisor_logs: list[str],
) -> int:
    """Replay JSONL session and supervisor logs into deterministic OTel spans.

    Args:
        session_log: Path to the session JSONL log file.
        supervisor_logs: Paths to the supervisor JSONL log files.

    Returns:
        The total number of spans exported.
    """
    from opentelemetry import trace  # noqa: PLC0415

    numbered_records = _read_session_log(session_log)
    if not numbered_records:
        return 0

    _first_lineno, first_record = numbered_records[0]
    if not isinstance(first_record, SessionRecord):
        return 0

    session_id = first_record.session_id
    runs = [
        run
        for log_path in supervisor_logs
        if (run := _parse_one_supervisor_log(log_path, session_id)) is not None
    ]

    latest_at = max([
        *(rec.at for _lineno, rec in numbered_records),
        *(run.last_at for run in runs),
    ])

    session_span = start_span(
        SESSION_SPAN,
        span_key=SESSION_SPAN_KEY,
        start_time=first_record.at,
    )
    session_ctx = trace.set_span_in_context(session_span)

    run_spans = [(run, _create_run_span(run, session_ctx)) for run in runs]
    cmd_count = _create_command_spans(numbered_records, session_id, session_span, run_spans)

    session_span.end(end_time=latest_at)
    return 1 + len(run_spans) + cmd_count


def _add_events(span: Span, events: list[_EventTuple]) -> None:
    """Add each ``(name, attributes, timestamp)`` event tuple to *span*, in order."""
    for ev_name, ev_attrs, ev_at in events:
        span.add_event(ev_name, attributes=ev_attrs, timestamp=ev_at)


def _create_run_span(run: _ParsedRun, session_ctx: Context) -> Span:
    """Open and close the ``gymrat.run`` span of one parsed supervisor run."""
    run_span = start_span(
        RUN_SPAN,
        span_key=run_span_key(run.launch_at),
        start_time=run.launch_at,
        context=session_ctx,
        attributes=run.attributes,
    )

    _add_events(run_span, run.events)

    if run.cost_usd is not None:
        run_span.set_attribute(RUN_COST_USD, run.cost_usd)

    run_span.end(end_time=run.last_at)
    return run_span


def _create_command_spans(
    numbered_records: list[_NumberedRecord],
    session_id: str,
    session_span: Span,
    run_spans: list[_RunSpan],
) -> int:
    """Create ``gymrat.command.*`` spans and attach inter-command records as events.

    Records before the first command and after the last command are attached
    as events on the session span.  Records between two commands are attached
    to the later command's span.

    Args:
        numbered_records: The session's records with their log line numbers.
        session_id: The session the spans belong to.
        session_span: The span that records outside every command attach to.
        run_spans: The supervisor runs, each with its span.

    Returns:
        The number of command spans created.
    """
    pending_events: list[_EventTuple] = []
    cmd_count = 0
    seen_command = False

    for line_number, rec in numbered_records:
        if isinstance(rec, SessionRecord):
            continue

        if isinstance(rec, CommandRecord):
            if not seen_command:
                # Pre-command records go on the session span, not on the first command.
                _add_events(session_span, pending_events)
                pending_events.clear()
                seen_command = True
            cmd_count += 1
            _emit_command_span(
                rec, line_number, session_id, session_span, run_spans, pending_events
            )
        else:
            ev_name, ev_attrs = record_event(rec)
            pending_events.append((ev_name, ev_attrs, rec.at))

    _add_events(session_span, pending_events)
    return cmd_count


def _emit_command_span(  # noqa: PLR0913, PLR0917 — accepts the full replay context
    rec: CommandRecord,
    line_number: int,
    session_id: str,
    session_span: Span,
    run_spans: list[_RunSpan],
    pending_events: list[_EventTuple],
) -> None:
    """Create one command span, drain pending events onto it, and close it."""
    from opentelemetry import trace  # noqa: PLC0415

    cmd_span = start_command_span(
        rec,
        session_id=session_id,
        line_number=line_number,
        context=trace.set_span_in_context(_find_parent_run(rec, run_spans) or session_span),
        start_time=rec.at - rec.duration_ms * NS_PER_MS,
    )

    _add_events(cmd_span, pending_events)
    pending_events.clear()

    cmd_span.end(end_time=rec.at)


class _ParsedRun:
    """Data extracted from one supervisor log before span creation."""

    __slots__ = ("attributes", "cost_usd", "events", "last_at", "launch_at")

    def __init__(self, launch: LaunchEvent) -> None:
        self.launch_at = launch.at
        self.last_at = launch.at
        self.cost_usd: float | None = None
        self.attributes = run_attributes(launch)
        self.events: list[_EventTuple] = []


def _read_lines(path: str) -> list[str]:
    """Read *path* as UTF-8 text.

    Deliberately narrower than :meth:`str.splitlines`, which also breaks on
    U+0085, U+2028 and U+2029 — characters a JSON string may carry raw, so
    splitting there would tear one record into two invalid halves.
    ``read_text`` already folds CRLF and lone CR into a line feed.

    Args:
        path: The file to read.

    Returns:
        The file's lines split on line feeds, or ``[]`` when the file does not
        exist.
    """
    try:
        lines = Path(path).read_text(encoding="utf-8").split("\n")
    except FileNotFoundError:
        return []
    if lines[-1] == "":
        lines.pop()
    return lines


def _read_session_log(path: str) -> list[_NumberedRecord]:
    """Parse a session JSONL log into typed records paired with physical line numbers."""
    records: list[_NumberedRecord] = []
    for line_number, line in enumerate(_read_lines(path), 1):
        if not line.strip():
            continue
        record = _parse_session_log_line(path, line_number, line)
        if record is not None:
            records.append((line_number, record))
    return records


def _parse_session_log_line(path: str, line_number: int, line: str) -> SessionLogRecord | None:
    """Parse one JSONL line into a typed record, or None to skip it."""
    try:
        wire_value = decode_log_line(line)
    except ValueError:
        logger.warning("session log %s: skipping line %d (invalid JSON)", path, line_number)
        return None
    if not isinstance(wire_value, dict):
        return None
    try:
        return parse_record(wire_value)
    except GymratError:
        if wire_value.get("type") != "command":
            return None
        cmd_name = wire_value.get("name", _UNKNOWN_COMMAND_NAME)
        logger.warning(
            "session log %s: command %r fell back to model_construct",
            path,
            cmd_name,
        )
        return _construct_command(wire_value)


def _construct_command(data: dict[str, object]) -> CommandRecord:
    """Build a CommandRecord without running model validators."""
    filled = dict(data)
    filled.setdefault("args", {})
    filled.setdefault("name", _UNKNOWN_COMMAND_NAME)
    filled.setdefault("duration_ms", 0)
    filled.setdefault("exit_code", TOOL_FAILURE_EXIT_CODE)
    # pyrefly: ignore[bad-argument-type] -- dict[str, object] is the wire dict shape
    return CommandRecord.model_construct(**filled)


def _parse_one_supervisor_log(log_path: str, session_id: str) -> _ParsedRun | None:
    """Parse a single supervisor log, returning None if it doesn't match."""
    lines = _read_lines(log_path)
    if not lines:
        return None

    first_obj = _safe_json(lines[0])
    if first_obj is None:
        return None

    first_event = event_from_wire(first_obj)
    if not isinstance(first_event, LaunchEvent):
        return None
    if first_event.session_id != session_id:
        return None

    run = _ParsedRun(first_event)

    for line in lines[1:]:
        obj = _safe_json(line)
        if obj is None:
            continue
        event = event_from_wire(obj)
        if event is None:
            continue
        run.last_at = max(run.last_at, event.at)
        _collect_run_event(run, event)

    return run


def _collect_run_event(run: _ParsedRun, event: SessionEvent) -> None:
    """Record the cost an event reports and the span event it is mirrored as."""
    if isinstance(event, TurnEndEvent | UsageUpdateEvent):
        run.cost_usd = event.cost_usd
    mirrored = run_event(event)
    if mirrored is not None:
        name, attributes = mirrored
        run.events.append((name, attributes, event.at))


def _find_parent_run(rec: CommandRecord, run_spans: list[_RunSpan]) -> Span | None:
    """Find the span of the run a command ran under.

    Args:
        rec: The command record.
        run_spans: The supervisor runs, each with its span.

    Returns:
        The span of the run whose time range contains the command; failing
        that, the run span the command's recorded traceparent links to; else
        ``None``.
    """
    for run, span in run_spans:
        if run.launch_at <= rec.at <= run.last_at:
            return span
    link = parse_traceparent(rec.traceparent) if rec.traceparent else None
    if link is None:
        return None
    for _run, span in run_spans:
        if span.get_span_context().span_id == link.span_id:
            return span
    return None


def _safe_json(line: str) -> dict[str, object] | None:
    """Parse a JSON line into a dict, or None on failure."""
    try:
        wire_value = decode_log_line(line)
    except ValueError:
        return None
    return wire_value if isinstance(wire_value, dict) else None
