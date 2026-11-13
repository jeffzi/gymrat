"""Replay session and supervisor logs into OpenTelemetry spans.

All ``opentelemetry`` imports live inside the function so importing this module
never pulls the SDK into ``sys.modules``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from gymrat.errors import GymratError
from gymrat.session.records import (
    CommandRecord,
    SessionRecord,
    add_record_events,
    parse_record,
)
from gymrat.session.store import complete_lines, is_blank_line
from gymrat.supervisor.events import (
    LaunchEvent,
    TurnEndEvent,
    UsageUpdateEvent,
    event_from_wire,
)
from gymrat.telemetry.command_span import command_link, start_command_span
from gymrat.telemetry.provider import Attrs
from gymrat.telemetry.run_spans import RUN_COST_USD, run_event, start_run_span, start_session_span
from gymrat.utils import NS_PER_MS, loads_strict_json, os_error_reason, warn_to_stderr

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

    A session-log or supervisor-log line that cannot be parsed is skipped with
    a warning on stderr, and so is a supervisor log that cannot be read or whose
    launch line names this session but is not a valid launch event. A
    supervisor log whose first line is not a launch of this session is skipped
    silently.

    Args:
        session_log: Path to the session JSONL log file.
        supervisor_logs: Paths to candidate supervisor JSONL log files.

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


def _create_run_span(run: _ParsedRun, session_span: Span) -> Span:
    """Open and close the ``gymrat.run`` span of one parsed supervisor run."""
    run_span = start_run_span(run.launch, parent=session_span, start_time=run.launch.at)

    for ev_name, ev_attrs, ev_at in run.events:
        run_span.add_event(ev_name, attributes=ev_attrs, timestamp=ev_at)

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
    from opentelemetry import trace  # noqa: PLC0415 -- optional extra

    pending_records: list[SessionLogRecord] = []
    cmd_count = 0

    for position, rec in enumerate(records, 1):
        if isinstance(rec, SessionRecord):
            continue

        if isinstance(rec, CommandRecord):
            cmd_count += 1
            cmd_span = start_command_span(
                rec,
                session_id=session_id,
                position=position,
                context=trace.set_span_in_context(_find_parent_run(rec, run_spans) or session_span),
                start_time=rec.at - rec.duration_ms * NS_PER_MS,
            )
            add_record_events(cmd_span, pending_records)
            pending_records.clear()
            cmd_span.end(end_time=rec.at)
        else:
            pending_records.append(rec)

    add_record_events(session_span, pending_records)
    return cmd_count


@dataclass(slots=True)
class _ParsedRun:
    """Data extracted from one supervisor log before span creation."""

    launch: LaunchEvent
    last_at: int
    cost_usd: float | None = None
    events: list[_EventTuple] = field(default_factory=list)


def _first_line_object(path: Path) -> dict[str, object] | None:
    """Parse only the first line of ``path`` as a JSON object.

    Args:
        path: The JSONL file to peek at.

    Returns:
        The decoded object, or ``None`` when the file does not exist or its
        first line is not a JSON object, including a line that is not valid
        UTF-8 or holds a non-finite number.

    Raises:
        OSError: When the file exists but cannot be read, such as a directory
            or a file without read permission.
    """
    # Binary read of the first line alone: a text-mode readline decodes a whole
    # buffered chunk, so bytes on later lines could fail the read of a first
    # line that is itself valid.
    try:
        with path.open("rb") as handle:
            first_line = handle.readline()
    except FileNotFoundError:
        return None
    try:
        # UnicodeDecodeError is a ValueError, so a non-UTF-8 line lands here too.
        parsed = loads_strict_json(first_line.decode("utf-8"))
    except ValueError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _read_bytes(path: str) -> bytes:
    """Read *path*'s raw content, or ``b""`` when the file does not exist."""
    try:
        return Path(path).read_bytes()
    except FileNotFoundError:
        return b""


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
    for line_number, line in enumerate(complete_lines(_read_bytes(path)), start=1):
        if is_blank_line(line):
            continue
        record = _parse_session_log_line(path, line_number, line)
        if record is not None:
            records.append(record)
    return records


def _warn_skipped_line(log_kind: str, path: str, line_number: int, reason: str) -> None:
    warn_to_stderr(f"warning: {log_kind} log {path}: skipping line {line_number} ({reason})")


def _decode_log_line(
    log_kind: str, path: str, line_number: int, line: bytes, noun: str
) -> dict[str, object] | None:
    """Decode one JSONL line as an object, warning and returning None when it is not one."""
    try:
        wire_value = loads_strict_json(line.decode("utf-8"))
    except ValueError:
        _warn_skipped_line(log_kind, path, line_number, "invalid JSON")
        return None
    if not isinstance(wire_value, dict):
        _warn_skipped_line(log_kind, path, line_number, f"not {noun}")
        return None
    return wire_value


def _parse_session_log_line(path: str, line_number: int, line: bytes) -> SessionLogRecord | None:
    """Parse one JSONL line into a typed record, or None to skip it."""
    wire_value = _decode_log_line("session", path, line_number, line, "a record")
    if wire_value is None:
        return None
    try:
        return parse_record(wire_value)
    except GymratError:
        kind = wire_value.get("type", "untyped")
        _warn_skipped_line("session", path, line_number, f"invalid {kind} record")
        return None


def _parse_one_supervisor_log(log_path: str, session_id: str) -> _ParsedRun | None:
    """Parse one supervisor log when its launch event names the session.

    The typed launch event alone decides whether the log belongs to the
    session. A log that cannot be read, or whose launch line names the session
    but fails to parse as a launch event, is reported on stderr; any other
    mismatch is skipped silently. The lines after the launch are read as the
    session log's are: blank lines and an unterminated final line are dropped,
    and a complete line that matches no event is skipped with a warning naming
    its line number.

    Args:
        log_path: The supervisor log to read.
        session_id: The session the launch event must name.

    Returns:
        The parsed run, or ``None`` when the log does not belong to the session.
    """
    try:
        first_obj = _first_line_object(Path(log_path))
    except OSError as error:
        warn_to_stderr(f"warning: skipped {log_path}: {os_error_reason(error)}")
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

    for line_number, line in enumerate(complete_lines(_read_bytes(log_path)), start=1):
        if line_number == 1 or is_blank_line(line):
            continue
        event = _parse_supervisor_log_line(log_path, line_number, line)
        if event is None:
            continue
        run.last_at = max(run.last_at, event.at)
        _collect_run_event(run, event)

    return run


def _parse_supervisor_log_line(path: str, line_number: int, line: bytes) -> SessionEvent | None:
    """Parse one supervisor JSONL line into a typed event, or None to skip it."""
    wire_value = _decode_log_line("supervisor", path, line_number, line, "an event")
    if wire_value is None:
        return None
    event = event_from_wire(wire_value)
    if event is None:
        kind = wire_value.get("type", "untyped")
        _warn_skipped_line("supervisor", path, line_number, f"invalid {kind} event")
    return event


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
    link = command_link(rec)
    if link is None:
        return None
    for _run, span in run_spans:
        if span.get_span_context().span_id == link.span_id:
            return span
    return None
