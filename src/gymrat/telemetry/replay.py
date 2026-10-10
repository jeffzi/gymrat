"""Replay session and supervisor logs into OpenTelemetry spans.

All ``opentelemetry`` imports live inside the function so importing this module
never pulls the SDK into ``sys.modules``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from gymrat.errors import GymratError
from gymrat.session.object_line import decode_object_line
from gymrat.session.records import CommandRecord, SessionRecord, decode_log_line, parse_record
from gymrat.session.store import complete_lines, first_line_json
from gymrat.supervisor.events import (
    LaunchEvent,
    TurnEndEvent,
    UsageUpdateEvent,
    event_from_wire,
)
from gymrat.telemetry.provider import (
    RUN_COST_USD,
    Attrs,
    parse_traceparent,
    record_event,
    run_event,
    start_command_span,
)
from gymrat.telemetry.run_spans import start_run_span, start_session_span
from gymrat.utils import NS_PER_MS, warn_to_stderr

if TYPE_CHECKING:
    from opentelemetry.trace import Span

    from gymrat.session.records import SessionLogRecord
    from gymrat.supervisor.events import SessionEvent

type _EventTuple = tuple[str, Attrs, int]
type _RunSpan = tuple[_ParsedRun, Span]


def replay_session(
    session_log: str,
    supervisor_logs: list[str],
) -> int:
    """Replay JSONL session and supervisor logs into deterministic OTel spans.

    A session-log line that cannot be parsed is skipped with a warning on
    stderr, and so is a supervisor log that cannot be read or whose launch line
    names this session but is not a valid launch event. A supervisor log whose
    first line is not a launch of this session is skipped silently.

    Args:
        session_log: Path to the session JSONL log file.
        supervisor_logs: Paths to candidate supervisor JSONL log files; only
            those whose launch event names the session are replayed.

    Returns:
        The total number of spans exported.
    """
    records = _read_session_log(session_log)
    if not records:
        return 0

    first_record = records[0]
    if not isinstance(first_record, SessionRecord):
        return 0

    session_id = first_record.session_id
    runs = [
        run
        for log_path in supervisor_logs
        if (run := _parse_one_supervisor_log(log_path, session_id)) is not None
    ]

    latest_at = max([
        *(rec.at for rec in records),
        *(run.last_at for run in runs),
    ])

    session_span = start_session_span(
        session_id, branch=first_record.branch, start_time=first_record.at
    )

    run_spans = [(run, _create_run_span(run, session_span)) for run in runs]
    cmd_count = _create_command_spans(records, session_id, session_span, run_spans)

    session_span.end(end_time=latest_at)
    return 1 + len(run_spans) + cmd_count


def _add_events(span: Span, events: list[_EventTuple]) -> None:
    """Add each ``(name, attributes, timestamp)`` event tuple to *span*, in order."""
    for ev_name, ev_attrs, ev_at in events:
        span.add_event(ev_name, attributes=ev_attrs, timestamp=ev_at)


def _create_run_span(run: _ParsedRun, session_span: Span) -> Span:
    """Open and close the ``gymrat.run`` span of one parsed supervisor run."""
    run_span = start_run_span(run.launch, parent=session_span, start_time=run.launch.at)

    _add_events(run_span, run.events)

    if run.cost_usd is not None:
        run_span.set_attribute(RUN_COST_USD, run.cost_usd)

    run_span.end(end_time=run.last_at)
    return run_span


def _create_command_spans(
    records: list[SessionLogRecord],
    session_id: str,
    session_span: Span,
    run_spans: list[_RunSpan],
) -> int:
    """Create ``gymrat.command.*`` spans and attach the session's records as events.

    Records before a command, including those before the first command, are
    attached to that command's span, matching live emission where a command's
    span carries the records it appended to the log. Records after the last
    command have no owning command and are attached to the session span. The
    session header is never an event.

    A command span is keyed by the command's 1-based position among the
    records, as live tracing keys it by the number of records the log holds
    once the command's own record is appended.

    Args:
        records: The session's records, in file order.
        session_id: The session the spans belong to.
        session_span: The span that records after the last command attach to.
        run_spans: The supervisor runs, each with its span.

    Returns:
        The number of command spans created.
    """
    pending_events: list[_EventTuple] = []
    cmd_count = 0

    for position, rec in enumerate(records, 1):
        if isinstance(rec, SessionRecord):
            continue

        if isinstance(rec, CommandRecord):
            cmd_count += 1
            _emit_command_span(rec, position, session_id, session_span, run_spans, pending_events)
        else:
            ev_name, ev_attrs = record_event(rec)
            pending_events.append((ev_name, ev_attrs, rec.at))

    _add_events(session_span, pending_events)
    return cmd_count


def _emit_command_span(  # noqa: PLR0913, PLR0917 — accepts the full replay context
    rec: CommandRecord,
    position: int,
    session_id: str,
    session_span: Span,
    run_spans: list[_RunSpan],
    pending_events: list[_EventTuple],
) -> None:
    """Create one command span, drain pending events onto it, and close it."""
    from opentelemetry import trace  # noqa: PLC0415 -- optional extra

    cmd_span = start_command_span(
        rec,
        session_id=session_id,
        line_number=position,
        context=trace.set_span_in_context(_find_parent_run(rec, run_spans) or session_span),
        start_time=rec.at - rec.duration_ms * NS_PER_MS,
    )

    _add_events(cmd_span, pending_events)
    pending_events.clear()

    cmd_span.end(end_time=rec.at)


@dataclass(slots=True)
class _ParsedRun:
    """Data extracted from one supervisor log before span creation."""

    launch: LaunchEvent
    last_at: int
    cost_usd: float | None = None
    events: list[_EventTuple] = field(default_factory=list)


def _read_bytes(path: str) -> bytes:
    """Read *path*'s raw content, or ``b""`` when the file does not exist."""
    try:
        return Path(path).read_bytes()
    except FileNotFoundError:
        return b""


def _read_lines(path: str) -> list[bytes]:
    """Read *path* as raw lines, leaving each one to be decoded on its own.

    Decoding the whole file at once would let one undecodable line, such as a
    final record torn mid-character by a crash, hide every other line. CRLF and
    lone CR are folded into a line feed, as text-mode reading would.

    Deliberately narrower than :meth:`str.splitlines`, which also breaks on
    U+0085, U+2028 and U+2029 — characters a JSON string may carry raw, so
    splitting there would tear one record into two invalid halves.

    Args:
        path: The file to read.

    Returns:
        The file's lines split on line feeds, or ``[]`` when the file does not
        exist.
    """
    content = _read_bytes(path)
    lines = content.replace(b"\r\n", b"\n").replace(b"\r", b"\n").split(b"\n")
    if lines[-1] == b"":
        lines.pop()
    return lines


def _read_session_log(path: str) -> list[SessionLogRecord]:
    """Parse a session JSONL log into typed records, reading lines as the store reads them.

    Blank lines and an unterminated final line are dropped, as
    :func:`~gymrat.session.store.read_records` drops them; a complete line that
    does not parse is skipped with a warning naming its line number.

    Args:
        path: The session log to read.

    Returns:
        The parsed records in file order, or ``[]`` when the log does not exist.
    """
    records: list[SessionLogRecord] = []
    for line_number, line in enumerate(complete_lines(_read_bytes(path)), 1):
        if not line.strip():
            continue
        record = _parse_session_log_line(path, line_number, line)
        if record is not None:
            records.append(record)
    return records


def _parse_session_log_line(path: str, line_number: int, line: bytes) -> SessionLogRecord | None:
    """Parse one JSONL line into a typed record, or None to skip it."""
    try:
        wire_value = decode_log_line(line.decode("utf-8"))
    except ValueError:
        warn_to_stderr(f"warning: session log {path}: skipping line {line_number} (invalid JSON)")
        return None
    if not isinstance(wire_value, dict):
        warn_to_stderr(f"warning: session log {path}: skipping line {line_number} (not a record)")
        return None
    try:
        return parse_record(wire_value)
    except GymratError:
        kind = wire_value.get("type", "untyped")
        warn_to_stderr(
            f"warning: session log {path}: skipping line {line_number} (invalid {kind} record)"
        )
        return None


def _parse_one_supervisor_log(log_path: str, session_id: str) -> _ParsedRun | None:
    """Parse one supervisor log when its launch event names the session.

    The typed launch event alone decides whether the log belongs to the
    session. A log that cannot be read, or whose launch line names the session
    but fails to parse as a launch event, is reported on stderr; any other
    mismatch is skipped silently.

    Args:
        log_path: The supervisor log to read.
        session_id: The session the launch event must name.

    Returns:
        The parsed run, or ``None`` when the log does not belong to the session.
    """
    try:
        first_obj = first_line_json(Path(log_path))
    except OSError as error:
        warn_to_stderr(f"warning: skipped {log_path}: {error.strerror or error}")
        return None
    if first_obj is None:
        return None

    launch = event_from_wire(first_obj)
    if not isinstance(launch, LaunchEvent):
        if first_obj.get("type") == "launch" and first_obj.get("session_id") == session_id:
            warn_to_stderr(f"warning: skipped {log_path}: its launch line is not a valid launch")
        return None
    if launch.session_id != session_id:
        return None

    run = _ParsedRun(launch=launch, last_at=launch.at)

    for line in _read_lines(log_path)[1:]:
        obj = decode_object_line(line)
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
        if run.launch.at <= rec.at <= run.last_at:
            return span
    link = parse_traceparent(rec.traceparent) if rec.traceparent else None
    if link is None:
        return None
    for _run, span in run_spans:
        if span.get_span_context().span_id == link.span_id:
            return span
    return None
