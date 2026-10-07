"""Tracing provider: lazy OTel setup, deterministic IDs, and record-to-attribute mapping.

All ``opentelemetry`` imports live inside functions so that importing the
module never pulls the SDK into ``sys.modules`` (CLAUDE.md: "Import the agent
SDK, and anything else with a startup cost, inside the function that needs it").
The attribute mapping builds plain dicts and needs no ``opentelemetry`` import
at all.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import os
import random
import warnings
from contextvars import ContextVar
from typing import TYPE_CHECKING, override

from gymrat.errors import TOOL_FAILURE_EXIT_CODE
from gymrat.session.records import (
    CommandRecord,
    IterationRecord,
    SessionLogRecord,
    _SequencedEnvelope,
)
from gymrat.supervisor.events import CapEvent, CompactionEvent, FollowUpEvent, TurnEndEvent
from gymrat.utils import ENDPOINT_ENV, otlp_endpoint

if TYPE_CHECKING:
    from collections.abc import Sequence

    from opentelemetry.context import Context
    from opentelemetry.sdk.trace import ReadableSpan, SpanProcessor, TracerProvider
    from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
    from opentelemetry.trace import Span, SpanContext, Tracer

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

_TRACEPARENT_HEADER = "traceparent"

# ---------------------------------------------------------------------------
# Deterministic ids and traceparent headers
# ---------------------------------------------------------------------------


def trace_id_of(session_id: str) -> int:
    """Derive a deterministic 128-bit trace ID from a session identifier.

    Args:
        session_id: The session identifier to hash.

    Returns:
        The first 128 bits of the identifier's SHA-256 digest, or ``1`` when
        that value is zero, since an all-zero trace ID is invalid in W3C Trace
        Context.
    """
    raw = int.from_bytes(hashlib.sha256(session_id.encode()).digest()[:16], "big")
    return raw or 1


def span_id_of(session_id: str, key: str) -> int:
    """Derive a deterministic 64-bit span ID from a session identifier and key.

    Args:
        session_id: The session identifier to hash.
        key: Distinguishes spans within the same session.

    Returns:
        The first 64 bits of the SHA-256 digest of the session identifier and
        key, or ``1`` when that value is zero, since an all-zero span ID is
        invalid in W3C Trace Context.
    """
    raw = int.from_bytes(hashlib.sha256((session_id + "\0" + key).encode()).digest()[:8], "big")
    return raw or 1


def existing_session_span(session_id: str) -> Span:
    """Reconstruct the session span an earlier process opened, as a parent for new spans.

    The session span's ids are deterministic, so a process that did not open it
    can still parent spans under it without holding the span itself.

    Args:
        session_id: The session whose span is reconstructed.

    Returns:
        A sampled, non-recording span carrying the session span's trace and
        span ids.
    """
    from opentelemetry.trace import NonRecordingSpan, SpanContext, TraceFlags  # noqa: PLC0415

    return NonRecordingSpan(
        SpanContext(
            trace_id=trace_id_of(session_id),
            span_id=span_id_of(session_id, SESSION_SPAN_KEY),
            is_remote=False,
            trace_flags=TraceFlags(TraceFlags.SAMPLED),
        )
    )


def format_traceparent(span: Span) -> str:
    """Format a W3C traceparent header from a span's context.

    Args:
        span: The span whose trace ID, span ID, and trace flags the header carries.

    Returns:
        The ``traceparent`` header value.

    Raises:
        ValueError: The span's context is invalid (zero trace ID or span ID), so
            it has no traceparent to carry.
    """
    from opentelemetry.trace import set_span_in_context  # noqa: PLC0415 -- optional extra
    from opentelemetry.trace.propagation.tracecontext import (  # noqa: PLC0415 -- optional extra
        TraceContextTextMapPropagator,
    )

    span_context = span.get_span_context()
    # The propagator only skips INVALID_SPAN_CONTEXT itself; any other invalid
    # context (e.g. a zero span ID) would be injected as a malformed header.
    if not span_context.is_valid:
        msg = f"span has no valid trace context: {span_context!r}"
        raise ValueError(msg)
    carrier: dict[str, str] = {}
    TraceContextTextMapPropagator().inject(carrier, context=set_span_in_context(span))
    return carrier[_TRACEPARENT_HEADER]


def parse_traceparent(value: str) -> SpanContext | None:
    """Parse a W3C traceparent header into a remote span context.

    Args:
        value: The ``traceparent`` header value.

    Returns:
        The remote span context the header describes, or ``None`` when the
        header is malformed.
    """
    from opentelemetry.trace import get_current_span  # noqa: PLC0415 -- optional extra
    from opentelemetry.trace.propagation.tracecontext import (  # noqa: PLC0415 -- optional extra
        TraceContextTextMapPropagator,
    )

    context = TraceContextTextMapPropagator().extract({_TRACEPARENT_HEADER: value})
    span_context = get_current_span(context).get_span_context()
    return span_context if span_context.is_valid else None


# ---------------------------------------------------------------------------
# Record-to-attribute mapping
# ---------------------------------------------------------------------------


def run_span_key(launch_at: int) -> str:
    """Build the key a run span's deterministic id is derived from.

    Args:
        launch_at: The run's launch timestamp, in nanoseconds since the Unix
            epoch.

    Returns:
        The span key, unique per launch.
    """
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
    _add_seq(attrs, record)
    for key, val in record.args.items():
        if isinstance(val, _SCALAR_TYPES):
            attrs[f"{COMMAND_ARGS_PREFIX}.{key}"] = val
    return attrs


def record_event(record: SessionLogRecord) -> tuple[str, Attrs]:
    """Map a non-command session log record to the span event it becomes.

    Any record with an iteration sequence number carries it. An iteration record
    adds its outcome and, when known, its primary delta; any other record adds
    its scalar top-level fields under ``gymrat.<type>.<field>``.

    Args:
        record: The session log record to map.

    Returns:
        The ``(event_name, attributes)`` pair, the name being ``gymrat.<type>``.
    """
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
            attrs[f"gymrat.{record_type}.{field_name}"] = value


# ---------------------------------------------------------------------------
# Tracer provider
# ---------------------------------------------------------------------------

_TRACES_ENDPOINT_ENV = "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"
_TRACES_PATH = "v1/traces"

_session_id: str = ""
_provider: TracerProvider | None = None
_tracer: Tracer | None = None
_export_failed: bool = False
_session_span_dropped: bool = False

_queued_span_id: ContextVar[int | None] = ContextVar("_queued_span_id", default=None)


def _traces_url(endpoint: str) -> str:
    """Build the URL spans are posted to, as the OTLP exporter does from the environment."""
    traces_endpoint = otlp_endpoint(os.environ.get(_TRACES_ENDPOINT_ENV))
    if traces_endpoint is not None:
        return traces_endpoint
    separator = "" if endpoint.endswith("/") else "/"
    return f"{endpoint}{separator}{_TRACES_PATH}"


def configure_tracing(
    session_id: str,
    *,
    span_processor: SpanProcessor | None = None,
    endpoint: str | None = None,
) -> bool:
    """Configure a TracerProvider for the given session.

    Module state is set only once every import has succeeded, so a call that
    returns ``False`` leaves the module as it found it.

    Args:
        session_id: The session identifier used to derive deterministic trace
            IDs.
        span_processor: The span processor to install; a default OTLP
            batch processor is used when omitted.
        endpoint: The OTLP endpoint to export to; ``None`` reads it from
            ``OTEL_EXPORTER_OTLP_ENDPOINT``.

    Returns:
        Whether a tracer provider is now active (``False`` when no endpoint
        is configured, or the SDK or, without *span_processor*, the OTLP
        exporter is missing).

    Raises:
        ValueError: When called a second time with a different *session_id*,
            or with a non-None *span_processor* when a provider is already
            configured (the processor would be silently discarded).
    """
    global _provider, _tracer, _session_id  # noqa: PLW0603 — module singleton

    if _provider is not None:
        if session_id != _session_id:
            msg = (
                f"configure_tracing already called with session_id={_session_id!r}; "
                f"cannot reconfigure with session_id={session_id!r}"
            )
            raise ValueError(msg)
        if span_processor is not None:
            msg = (
                f"configure_tracing already called for session_id={_session_id!r}; "
                "span_processor would be silently discarded"
            )
            raise ValueError(msg)
        return True

    endpoint = otlp_endpoint(endpoint if endpoint is not None else os.environ.get(ENDPOINT_ENV))
    if endpoint is None:
        return False

    try:
        from opentelemetry.sdk.resources import Resource  # noqa: PLC0415
        from opentelemetry.sdk.trace import TracerProvider as _TracerProvider  # noqa: PLC0415

        if span_processor is None:
            span_processor = _otlp_span_processor(endpoint)
    except ImportError:
        return False

    resource = Resource.create({
        "service.name": "gymrat",
        "service.version": importlib.metadata.version("gymrat"),
    })

    id_generator = _DeterministicIdGenerator(trace_id_of(session_id))
    # pyrefly: ignore[bad-argument-type] -- IdGenerator protocol mismatch
    provider = _TracerProvider(resource=resource, id_generator=id_generator)
    provider.add_span_processor(span_processor)

    _session_id = session_id
    _provider = provider
    _tracer = provider.get_tracer("gymrat")
    return True


def _otlp_span_processor(endpoint: str) -> SpanProcessor:
    """Build the batch processor that exports over OTLP HTTP.

    Args:
        endpoint: The OTLP base endpoint the traces URL is derived from.

    Returns:
        A batch processor whose failed exports set :func:`export_failed`.

    Raises:
        ImportError: When the OTLP exporter package is not installed.
    """
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import (  # noqa: PLC0415
        OTLPSpanExporter,
    )
    from opentelemetry.sdk.trace.export import BatchSpanProcessor  # noqa: PLC0415

    exporter = _failure_recording(OTLPSpanExporter(endpoint=_traces_url(endpoint)))
    return BatchSpanProcessor(exporter)


def start_span(
    name: str,
    *,
    span_key: str | None = None,
    session_id: str | None = None,
    **kwargs: object,
) -> Span:
    """Start a span through the module's tracer, which :func:`configure_tracing` must have set.

    Args:
        name: The span name.
        span_key: When given, derives a deterministic span ID from the
            session ID and this key.
        session_id: The session the deterministic span ID is derived from;
            ``None`` uses the session tracing was configured for.
        **kwargs: Additional keyword arguments forwarded to the tracer's
            ``start_span``.

    Returns:
        The started span.
    """
    global _session_span_dropped  # noqa: PLW0603 — module singleton
    token = None
    if span_key is not None:
        keyed_session = _session_id if session_id is None else session_id
        token = _queued_span_id.set(span_id_of(keyed_session, span_key))
    try:
        span = _tracer.start_span(name, **kwargs)  # type: ignore[arg-type]
    finally:
        if token is not None:
            _queued_span_id.reset(token)
    if span_key == SESSION_SPAN_KEY:
        _session_span_dropped = not span.is_recording()
    return span


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
    from opentelemetry.trace import Link, Status, StatusCode  # noqa: PLC0415

    link = parse_traceparent(record.traceparent) if record.traceparent else None
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


def flush_tracing() -> None:
    """Force-flush the provider if one exists; no-op otherwise.

    A flush that does not finish in time leaves exports whose outcome is
    unknown, so it counts as a failed export for :func:`export_failed`.
    """
    global _export_failed  # noqa: PLW0603 — module singleton
    if _provider is not None and not _provider.force_flush():
        _export_failed = True


def export_failed() -> bool:
    """Whether any span export of the default OTLP exporter has failed.

    ``BatchSpanProcessor`` only logs a failed export, so callers that must
    report one (``gymrat export``) check this after :func:`flush_tracing`.
    Command tracing is best-effort and never checks it.

    Returns:
        ``True`` once a batch was rejected, the collector was unreachable, or
        a flush timed out, since tracing was configured.
    """
    return _export_failed


def session_span_dropped() -> bool:
    """Whether the session span was started without recording.

    A disabled SDK (``OTEL_SDK_DISABLED``) or a sampler that drops the session
    trace hands out non-recording spans, which never reach the exporter, so
    callers that report an export (``gymrat export``) check this before
    claiming one.

    Returns:
        ``True`` once a span keyed :data:`SESSION_SPAN_KEY` was started
        non-recording, since tracing was configured.
    """
    return _session_span_dropped


def reset_tracing() -> None:
    """Shut down and clear the module singleton so the next configure starts fresh.

    Test-only seam: production code never calls this, since a process configures
    tracing for one session and keeps it until exit. Tests use it to isolate the
    module singleton between cases instead of reaching into the private
    attributes directly.

    The singleton is cleared before the shutdown runs, so a provider whose
    shutdown fails cannot leave stale state behind for the next configure.

    Raises:
        Exception: Whatever the provider's ``shutdown`` raises, propagated after
            the singleton has been cleared.
    """
    global _provider, _tracer, _session_id, _export_failed, _session_span_dropped  # noqa: PLW0603
    provider = _provider
    _provider = None
    _tracer = None
    _session_id = ""
    _export_failed = False
    _session_span_dropped = False
    if provider is not None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            provider.shutdown()


def _failure_recording(exporter: SpanExporter) -> SpanExporter:
    """Wrap ``exporter`` so that any failed export sets :func:`export_failed`."""
    from opentelemetry.sdk.trace.export import (  # noqa: PLC0415
        SpanExporter,
        SpanExportResult,
    )

    class _FailureRecordingExporter(SpanExporter):
        @override
        def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
            global _export_failed  # noqa: PLW0603 — module singleton
            result = exporter.export(spans)
            if result is not SpanExportResult.SUCCESS:
                _export_failed = True
            return result

        @override
        def shutdown(self) -> None:
            exporter.shutdown()

        @override
        def force_flush(self, timeout_millis: int = 30000) -> bool:
            return exporter.force_flush(timeout_millis)

    return _FailureRecordingExporter()


class _DeterministicIdGenerator:
    """OTel IdGenerator that produces deterministic trace IDs and queued span IDs."""

    def __init__(self, session_trace_id: int) -> None:
        self._session_trace_id = session_trace_id

    def generate_trace_id(self) -> int:
        return self._session_trace_id

    def generate_span_id(self) -> int:
        queued = _queued_span_id.get()
        if queued is not None:
            return queued
        return random.getrandbits(64) or 1

    def is_trace_id_random(self) -> bool:
        return False
