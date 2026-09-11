"""Repository lock and command-trace recording.

The single-flight lock and the seam that appends a :class:`CommandRecord` after
every command that holds it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from gymrat.config import CliFlags
    from gymrat.session.records.models import SessionLogRecord
    from gymrat.session.schema import CommandReason

import typer

from gymrat.errors import GymratError
from gymrat.git import NotAGitRepositoryError
from gymrat.loop.iterate import LoopStopError
from gymrat.session import clock as _clock
from gymrat.session.lock import acquire_lock
from gymrat.session.paths import lockfile_path, repo_root, session_jsonl_path
from gymrat.session.records.models import CommandRecord
from gymrat.session.store import append_record, recover_torn_tail, session_header
from gymrat.warn import warn_to_stderr

GATE_EXIT_CODE = 1
TOOL_FAILURE_EXIT_CODE = 2


# ---------------------------------------------------------------------------
# Trace bookkeeping
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class CommandTrace:
    """Mutable bag the body of :func:`with_repo_lock` writes into.

    The seam reads ``gate``, ``reason``, and ``seq`` after the body settles to
    populate the :class:`CommandRecord` it appends.  The body sets them freely;
    they carry no meaning to the seam beyond the exit-code / reason mapping.
    """

    args: dict[str, object] = field(default_factory=dict)
    seq: int | None = None
    gate: bool = False
    reason: CommandReason | None = None


def config_trace_args(flags: CliFlags) -> dict[str, object]:
    """Trace ``args`` entries for the config-resolving flags every command shares.

    Every command that resolves a config carries the same six overrides into its
    trace's ``args``; this reads them off ``flags`` once, dropping ``None`` values
    so an override the caller left at its default never appears in the record.

    Args:
        flags: The CLI flags to read config overrides from.

    Returns:
        A dict of non-``None`` flag names to their values.
    """
    return {
        k: v
        for k, v in (
            ("bench", flags.bench),
            ("prepare", flags.prepare),
            ("adapter", flags.adapter),
            ("samples", flags.samples),
            ("timeout", flags.timeout),
            ("config", flags.config),
        )
        if v is not None
    }


# ---------------------------------------------------------------------------
# Lock lifecycle
# ---------------------------------------------------------------------------


def _resolve_exit(  # noqa: PLR0911 -- flat branch per exception type, each an exit-code mapping
    trace: CommandTrace,
    caught: BaseException | None,
) -> tuple[Literal[0, 1, 2], CommandReason | None]:
    """Map a body outcome to the ``(exit_code, reason)`` pair for the command record."""
    if caught is None:
        if trace.gate:
            return GATE_EXIT_CODE, trace.reason
        return 0, None

    if isinstance(caught, LoopStopError):
        reason: CommandReason = caught.reason or "stop-condition"
        return GATE_EXIT_CODE, reason

    if isinstance(caught, typer.Exit):
        code = caught.exit_code
        if code == TOOL_FAILURE_EXIT_CODE:
            return TOOL_FAILURE_EXIT_CODE, trace.reason or "error"
        # exit_code is int; callers only pass 0|1|2
        return min(code, TOOL_FAILURE_EXIT_CODE), trace.reason  # type: ignore[return-value]

    if isinstance(caught, GymratError):
        gym_reason: CommandReason = caught.reason or "error"
        return TOOL_FAILURE_EXIT_CODE, gym_reason

    return TOOL_FAILURE_EXIT_CODE, "error"


async def with_repo_lock[T](
    command: str,
    body: Callable[[CommandTrace], Awaitable[T]],
    *,
    args: dict[str, object] | None = None,
) -> T:
    """Hold the repository's single-flight lock for the length of ``body``.

    Inside a git repository the lock is acquired around ``body`` and released
    however it settles — including on exception, and always before the caller
    renders its report. Outside every git repository the answer is to run
    ``body`` with no lock at all; any other git failure exits without
    benchmarking rather than running unlocked.

    Holding the lock is what makes repairing the session log safe: a torn final
    line can only belong to a writer the previous run left dead, so the tail is
    dropped here — once per command, before ``body`` reads or appends anything.

    After the body settles the seam appends a :class:`CommandRecord` to the
    session log when one exists and is non-empty.  A failure to append warns to
    stderr and never masks the body's result or exception.

    Args:
        command: The command name recorded in the :class:`CommandRecord`.
        body: The async callable to run under the lock.
        args: Extra trace arguments to include in the command record.

    Returns:
        The value returned by ``body``.
    """
    trace = CommandTrace(args=args if args is not None else {})
    start = _clock.monotonic_ms()

    try:
        root = repo_root()
    except NotAGitRepositoryError:
        return await body(trace)
    except GymratError as error:
        from gymrat.cli.shared import exit_with_error  # noqa: PLC0415 -- avoids circular import

        exit_with_error(error)

    jsonl = session_jsonl_path(root)
    session_id, tracing_active = _maybe_configure_tracing(root, jsonl)
    start_ns = _clock.now_ns() if tracing_active else 0
    pre_body_lines = _count_lines(jsonl) if tracing_active else 0

    release = acquire_lock(lockfile_path(root), command)
    caught: BaseException | None = None
    result: T
    try:
        recover_torn_tail(jsonl)
        result = await body(trace)
    except BaseException as exc:
        caught = exc
        raise
    finally:
        elapsed_ms = int(_clock.monotonic_ms() - start)
        exit_code, reason = _resolve_exit(trace, caught)
        try:
            _record_command_outcome(
                root=root,
                command=command,
                trace=trace,
                exit_code=exit_code,
                reason=reason,
                elapsed_ms=elapsed_ms,
                tracing_active=tracing_active,
                session_id=session_id,
                start_ns=start_ns,
                pre_body_lines=pre_body_lines,
            )
        finally:
            release()
    return result


# ---------------------------------------------------------------------------
# Command-record persistence
# ---------------------------------------------------------------------------


def _record_command_outcome(  # noqa: PLR0913 -- all params are distinct concerns of the command outcome
    *,
    root: str,
    command: str,
    trace: CommandTrace,
    exit_code: Literal[0, 1, 2],
    reason: CommandReason | None,
    elapsed_ms: int,
    tracing_active: bool,
    session_id: str,
    start_ns: int,
    pre_body_lines: int,
) -> None:
    """Append the command record and, when tracing is active, emit its span."""
    _try_append_command_record(
        root=root,
        command=command,
        trace=trace,
        exit_code=exit_code,
        reason=reason,
        elapsed_ms=elapsed_ms,
    )
    if tracing_active:
        try:
            _emit_command_span(
                root=root,
                exit_code=exit_code,
                reason=reason,
                session_id=session_id,
                start_ns=start_ns,
                pre_body_lines=pre_body_lines,
            )
        except Exception as span_error:  # noqa: BLE001 -- must not mask the command outcome
            warn_to_stderr(f"failed to emit command span: {span_error}")


def _try_append_command_record(  # noqa: PLR0913 -- all six params are distinct concerns of the command record
    *,
    root: str,
    command: str,
    trace: CommandTrace,
    exit_code: Literal[0, 1, 2],
    reason: CommandReason | None,
    elapsed_ms: int,
) -> None:
    """Append a :class:`CommandRecord` when the session log exists and is non-empty."""
    jsonl = session_jsonl_path(root)
    if _jsonl_is_empty(jsonl):
        return

    try:
        record = CommandRecord(
            type="command",
            at=_clock.now_ns(),
            name=command,
            args=trace.args,
            exit_code=exit_code,
            reason=reason,
            duration_ms=elapsed_ms,
            traceparent=os.environ.get("GYMRAT_TRACEPARENT") or os.environ.get("TRACEPARENT"),
            seq=trace.seq,
        )
        append_record(jsonl, record)
    except Exception as error:  # noqa: BLE001 -- construction or IO failure must not mask the command outcome
        warn_to_stderr(f"failed to append command record: {error}")


# ---------------------------------------------------------------------------
# Telemetry tracing
# ---------------------------------------------------------------------------


def _maybe_configure_tracing(root: str, jsonl: str) -> tuple[str, bool]:
    if not os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "").strip():
        return "", False
    if _jsonl_is_empty(jsonl):
        return "", False
    header = session_header(root)
    if header is None:
        return "", False
    from gymrat.telemetry.provider import configure_tracing  # noqa: PLC0415

    active = configure_tracing(header.session_id)
    return header.session_id, active


def _jsonl_is_empty(jsonl_path: str) -> bool:
    """True when the session log doesn't exist or has no content yet."""
    try:
        return Path(jsonl_path).stat().st_size == 0
    except OSError:
        return True


def _count_lines(jsonl_path: str) -> int:
    """Count newline-terminated lines in ``jsonl_path``, 0 if absent."""
    try:
        data = Path(jsonl_path).read_bytes()
    except OSError:
        return 0
    return data.count(b"\n")


def _emit_command_span(  # noqa: PLR0913 -- keyword-only tracing context
    *,
    root: str,
    exit_code: Literal[0, 1, 2],
    reason: CommandReason | None,
    session_id: str,
    start_ns: int,
    pre_body_lines: int,
) -> None:
    """Create a retroactive command span with events, attributes, and links."""
    from opentelemetry.trace import (  # noqa: PLC0415
        Link,
        NonRecordingSpan,
        SpanContext,
        Status,
        StatusCode,
        TraceFlags,
        set_span_in_context,
    )

    from gymrat.telemetry.attributes import command_span_inputs, record_event  # noqa: PLC0415
    from gymrat.telemetry.ids import (  # noqa: PLC0415
        parse_traceparent,
        span_id_of,
        trace_id_of,
    )
    from gymrat.telemetry.provider import flush_tracing, start_span  # noqa: PLC0415

    jsonl = session_jsonl_path(root)
    records = _safe_read_records(jsonl)
    if not records:
        return

    cmd_record = records[-1]
    if not isinstance(cmd_record, CommandRecord):
        return

    inputs = command_span_inputs(cmd_record, session_id=session_id, line_number=len(records))

    parent_ctx = None
    gymrat_tp = os.environ.get("GYMRAT_TRACEPARENT")
    if gymrat_tp:
        parent_span_ctx = parse_traceparent(gymrat_tp)
        if parent_span_ctx is not None:
            parent_ctx = set_span_in_context(NonRecordingSpan(parent_span_ctx))

    if parent_ctx is None:
        session_span_ctx = SpanContext(
            trace_id=trace_id_of(session_id),
            span_id=span_id_of(session_id, "session"),
            is_remote=False,
            trace_flags=TraceFlags(TraceFlags.SAMPLED),
        )
        parent_ctx = set_span_in_context(NonRecordingSpan(session_span_ctx))

    links = [Link(inputs.link)] if inputs.link is not None else None

    with start_span(
        inputs.name,
        span_key=inputs.key,
        context=parent_ctx,
        links=links,
        attributes=inputs.attributes,
        start_time=start_ns,
    ) as span:
        # Outcome records appended by the body (between pre-body count and command record)
        for record in records[pre_body_lines:-1]:
            event_name, event_attrs = record_event(record)
            span.add_event(event_name, attributes=event_attrs, timestamp=record.at)

        if exit_code == 0:
            span.set_status(Status(StatusCode.OK))
        elif exit_code == TOOL_FAILURE_EXIT_CODE:
            span.set_status(Status(StatusCode.ERROR, description=reason))

    flush_tracing()


def _safe_read_records(jsonl_path: str) -> list[SessionLogRecord]:
    """Read records from the session log, returning empty on failure."""
    from gymrat.session.store import read_records  # noqa: PLC0415

    try:
        return read_records(jsonl_path)
    except GymratError:
        return []
