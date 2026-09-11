"""Replay session and supervisor logs into OpenTelemetry spans.

All ``opentelemetry`` imports live inside the function so importing this module
never pulls the SDK into ``sys.modules``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from gymrat.errors import GymratError
from gymrat.session.records.models import CommandRecord, SessionRecord
from gymrat.session.records.parse import parse_record
from gymrat.supervisor.events import (
    CapEvent,
    CompactionEvent,
    FollowUpEvent,
    LaunchEvent,
    TurnEndEvent,
    UsageUpdateEvent,
    event_from_wire,
)
from gymrat.telemetry.attributes import (
    CAP_NAME,
    EVENT_CAP,
    EVENT_COMPACTION,
    EVENT_FOLLOW_UP,
    EVENT_TURN_END,
    FOLLOW_UP_ACTION,
    FOLLOW_UP_REASON,
    GEN_AI_MODEL,
    GEN_AI_PROVIDER,
    RUN_COST_USD,
    RUN_EFFORT,
    RUN_HEAD_SHA,
    RUN_MAX_MINUTES,
    RUN_MAX_USD,
    RUN_SPAN,
    SESSION_ID,
    SESSION_SPAN,
    TURN_BUDGET_EXHAUSTED,
    TURN_COST_USD,
    TURN_ORIGIN,
    command_span_inputs,
    record_event,
)
from gymrat.telemetry.ids import parse_traceparent
from gymrat.telemetry.provider import start_span

if TYPE_CHECKING:
    from opentelemetry.context import Context
    from opentelemetry.trace import Span

    from gymrat.session.records.models import SessionLogRecord
    from gymrat.supervisor.events import SessionEvent

logger = logging.getLogger(__name__)

type _EventTuple = tuple[str, dict[str, str | int | float | bool], int]
type _NumberedRecord = tuple[int, SessionLogRecord]

_UNKNOWN_COMMAND_NAME = "unknown"
_EXIT_CODE_ERROR = 2  # CLI exit-code convention: 2 = error
_NS_PER_MS = 1_000_000


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
    runs = _parse_supervisor_logs(supervisor_logs, session_id)

    latest_at = first_record.at
    for _lineno, rec in numbered_records:
        latest_at = max(latest_at, rec.at)
    for run in runs:
        latest_at = max(latest_at, run.last_at)

    session_span = start_span(
        SESSION_SPAN,
        span_key="session",
        start_time=first_record.at,
    )
    session_ctx = trace.set_span_in_context(session_span)

    run_infos = _create_run_spans(runs, session_ctx)
    cmd_count = _create_command_spans(numbered_records, session_id, session_span, run_infos)

    session_span.end(end_time=latest_at)
    return 1 + len(run_infos) + cmd_count


def _add_events(span: Span, events: list[_EventTuple]) -> None:
    """Add each ``(name, attributes, timestamp)`` event tuple to *span*, in order."""
    for ev_name, ev_attrs, ev_at in events:
        span.add_event(ev_name, attributes=ev_attrs, timestamp=ev_at)


def _create_run_spans(
    runs: list[_ParsedRun],
    session_ctx: Context,
) -> list[_RunInfo]:
    """Open and close one ``gymrat.run`` span per parsed supervisor run."""
    run_infos: list[_RunInfo] = []
    for run in runs:
        run_span = start_span(
            RUN_SPAN,
            span_key=f"run:{run.launch_at}",
            start_time=run.launch_at,
            context=session_ctx,
            attributes=run.attributes,
        )

        _add_events(run_span, run.events)

        if run.cost_usd is not None:
            run_span.set_attribute(RUN_COST_USD, run.cost_usd)

        run_span.end(end_time=run.last_at)
        run_infos.append(_RunInfo(run.launch_at, run.last_at, run_span))
    return run_infos


def _create_command_spans(
    numbered_records: list[_NumberedRecord],
    session_id: str,
    session_span: Span,
    run_infos: list[_RunInfo],
) -> int:
    """Create ``gymrat.command.*`` spans and attach inter-command records as events.

    Records before the first command and after the last command are attached
    as events on the session span.  Records between two commands are attached
    to the later command's span.

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
                rec, line_number, session_id, session_span, run_infos, pending_events
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
    run_infos: list[_RunInfo],
    pending_events: list[_EventTuple],
) -> None:
    """Create one command span, drain pending events onto it, and close it."""
    from opentelemetry import trace  # noqa: PLC0415
    from opentelemetry.trace import Link  # noqa: PLC0415

    parent_run = _find_parent_run(rec.at, run_infos)
    if parent_run is None and rec.traceparent:
        parent_run = _find_parent_by_traceparent(rec.traceparent, run_infos)
    parent_span = parent_run.span if parent_run is not None else session_span
    parent_ctx = trace.set_span_in_context(parent_span)

    inputs = command_span_inputs(rec, session_id=session_id, line_number=line_number)
    links = [Link(inputs.link)] if inputs.link is not None else None

    start_ns = rec.at - rec.duration_ms * _NS_PER_MS
    cmd_span = start_span(
        inputs.name,
        span_key=inputs.key,
        start_time=start_ns,
        context=parent_ctx,
        attributes=inputs.attributes,
        links=links,
    )

    _set_command_status(cmd_span, rec.exit_code)

    _add_events(cmd_span, pending_events)
    pending_events.clear()

    cmd_span.end(end_time=rec.at)


def _set_command_status(span: Span, exit_code: int) -> None:
    from opentelemetry.trace import StatusCode  # noqa: PLC0415

    if exit_code == 0:
        span.set_status(StatusCode.OK)
    elif exit_code == _EXIT_CODE_ERROR:
        span.set_status(StatusCode.ERROR)


class _RunInfo:
    """Time range and span handle for one supervisor run."""

    __slots__ = ("end", "span", "start")

    def __init__(self, start: int, end: int, span: Span) -> None:
        self.start = start
        self.end = end
        self.span = span


class _ParsedRun:
    """Data extracted from one supervisor log before span creation."""

    __slots__ = ("attributes", "cost_usd", "events", "last_at", "launch_at")

    def __init__(self, launch: LaunchEvent) -> None:
        self.launch_at = launch.at
        self.last_at = launch.at
        self.cost_usd: float | None = None

        attrs: dict[str, str | int | float | bool] = {
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
        self.attributes = attrs

        self.events: list[_EventTuple] = []


def _read_lines(path: str) -> list[str]:
    """Read *path* as UTF-8 text split into lines, or ``[]`` if it doesn't exist."""
    try:
        return Path(path).read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return []


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
        data = json.loads(line)
    except json.JSONDecodeError:
        logger.warning("session log %s: skipping line %d (invalid JSON)", path, line_number)
        return None
    if not isinstance(data, dict):
        return None
    try:
        return parse_record(data)
    except GymratError:
        if data.get("type") != "command":
            return None
        cmd_name = data.get("name", _UNKNOWN_COMMAND_NAME)
        logger.warning(
            "session log %s: command %r fell back to model_construct",
            path,
            cmd_name,
        )
        return _construct_command(data)


def _construct_command(data: dict[str, object]) -> CommandRecord:
    """Build a CommandRecord without running model validators."""
    filled = dict(data)
    filled.setdefault("reason", None)
    filled.setdefault("seq", None)
    filled.setdefault("traceparent", None)
    filled.setdefault("args", {})
    filled.setdefault("name", _UNKNOWN_COMMAND_NAME)
    filled.setdefault("duration_ms", 0)
    filled.setdefault("exit_code", _EXIT_CODE_ERROR)
    # pyrefly: ignore[bad-argument-type] -- dict[str, object] is the wire dict shape
    return CommandRecord.model_construct(**filled)


def _parse_supervisor_logs(
    supervisor_logs: list[str],
    session_id: str,
) -> list[_ParsedRun]:
    """Parse supervisor JSONL logs into _ParsedRun objects for matching sessions."""
    runs: list[_ParsedRun] = []

    for log_path in supervisor_logs:
        run = _parse_one_supervisor_log(log_path, session_id)
        if run is not None:
            runs.append(run)

    return runs


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
    """Classify a supervisor event and append it to a parsed run."""
    if isinstance(event, TurnEndEvent):
        run.cost_usd = event.cost_usd
        run.events.append((
            EVENT_TURN_END,
            {
                TURN_COST_USD: event.cost_usd,
                TURN_ORIGIN: event.origin,
                TURN_BUDGET_EXHAUSTED: event.budget_exhausted,
            },
            event.at,
        ))
    elif isinstance(event, UsageUpdateEvent):
        run.cost_usd = event.cost_usd
    elif isinstance(event, FollowUpEvent):
        attrs: dict[str, str | int | float | bool] = {FOLLOW_UP_ACTION: event.action}
        if event.reason is not None:
            attrs[FOLLOW_UP_REASON] = event.reason
        run.events.append((EVENT_FOLLOW_UP, attrs, event.at))
    elif isinstance(event, CapEvent):
        run.events.append((
            EVENT_CAP,
            {CAP_NAME: event.cap},
            event.at,
        ))
    elif isinstance(event, CompactionEvent):
        run.events.append((EVENT_COMPACTION, {}, event.at))


def _find_parent_run(at: int, run_infos: list[_RunInfo]) -> _RunInfo | None:
    """Find the run span whose time range contains ``at``."""
    for run in run_infos:
        if run.start <= at <= run.end:
            return run
    return None


def _find_parent_by_traceparent(traceparent: str, run_infos: list[_RunInfo]) -> _RunInfo | None:
    """Find the run span whose span ID matches the traceparent's span ID."""
    ctx = parse_traceparent(traceparent)
    if ctx is None:
        return None
    for run in run_infos:
        if run.span.get_span_context().span_id == ctx.span_id:
            return run
    return None


def _safe_json(line: str) -> dict[str, object] | None:
    """Parse a JSON line into a dict, or None on failure."""
    try:
        obj = json.loads(line)
    except (json.JSONDecodeError, ValueError):
        return None
    return obj if isinstance(obj, dict) else None
