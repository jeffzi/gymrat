"""Tracing provider: lazy OTel setup, deterministic IDs, and the run span's attributes.

All ``opentelemetry`` imports live inside functions: ``opentelemetry`` ships
only with the ``otel`` extra, and the import-latency seam test keeps it out of
``sys.modules`` when this module is imported.
The attribute mapping builds plain dicts and needs no ``opentelemetry`` import
at all.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import random
import warnings
from contextvars import ContextVar
from typing import TYPE_CHECKING, override

from gymrat.utils import otlp_endpoint, otlp_endpoint_from_env

if TYPE_CHECKING:
    from collections.abc import Sequence

    from opentelemetry.context import Context
    from opentelemetry.sdk.trace import ReadableSpan, SpanProcessor, TracerProvider
    from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
    from opentelemetry.sdk.trace.id_generator import IdGenerator
    from opentelemetry.trace import Link, Span, SpanContext, Tracer

    from gymrat.supervisor.events import LaunchEvent

type Attrs = dict[str, str | int | float | bool]
"""The flat attribute dict a span or span event carries."""

# ---------------------------------------------------------------------------
# Named attribute and span-name constants
# ---------------------------------------------------------------------------

SESSION_SPAN = "gymrat.session"
RUN_SPAN = "gymrat.run"

SESSION_ID = "gymrat.session.id"
SESSION_BRANCH = "gymrat.session.branch"

RUN_HEAD_SHA = "gymrat.run.head_sha"
RUN_MAX_MINUTES = "gymrat.run.max_minutes"
RUN_MAX_USD = "gymrat.run.max_usd"
RUN_EFFORT = "gymrat.run.effort"
RUN_ENDED_BY = "gymrat.run.ended_by"
RUN_END_REASON = "gymrat.run.end_reason"
RUN_DURATION_MS = "gymrat.run.duration_ms"

GEN_AI_MODEL = "gen_ai.request.model"
GEN_AI_PROVIDER = "gen_ai.provider.name"

SESSION_SPAN_KEY = "session"
"""The key the session span's deterministic id is derived from."""

_TRACEPARENT_HEADER = "traceparent"

# ---------------------------------------------------------------------------
# Deterministic ids and traceparent headers
# ---------------------------------------------------------------------------


def id_from_digest(digest: bytes, width: int) -> int:
    """Read the leading ``width`` bytes of ``digest`` as a W3C Trace Context ID.

    Args:
        digest: The hash digest to read.
        width: How many leading bytes form the ID: 16 for a trace ID, 8 for a
            span ID.

    Returns:
        The leading bytes as a big-endian integer, or ``1`` when that value is
        zero, since an all-zero ID is invalid in W3C Trace Context.
    """
    return int.from_bytes(digest[:width], "big") or 1


def trace_id_of(session_id: str) -> int:
    """Derive a deterministic 128-bit trace ID from a session identifier.

    Args:
        session_id: The session identifier to hash.

    Returns:
        The first 128 bits of the identifier's SHA-256 digest, as read by
        :func:`id_from_digest`.
    """
    return id_from_digest(hashlib.sha256(session_id.encode()).digest(), 16)


def span_id_of(session_id: str, key: str) -> int:
    """Derive a deterministic 64-bit span ID from a session identifier and key.

    Args:
        session_id: The session identifier to hash.
        key: Distinguishes spans within the same session.

    Returns:
        The first 64 bits of the SHA-256 digest of the session identifier and
        key, as read by :func:`id_from_digest`.
    """
    return id_from_digest(hashlib.sha256((session_id + "\0" + key).encode()).digest(), 8)


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
# Run span attributes
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
    traces_endpoint = otlp_endpoint_from_env(_TRACES_ENDPOINT_ENV)
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

    endpoint = otlp_endpoint(endpoint) if endpoint is not None else otlp_endpoint_from_env()
    if endpoint is None:
        return False

    try:
        from opentelemetry.sdk.resources import Resource  # noqa: PLC0415 -- optional extra
        from opentelemetry.sdk.trace import (  # noqa: PLC0415 -- optional extra
            TracerProvider as _TracerProvider,
        )

        if span_processor is None:
            span_processor = _otlp_span_processor(endpoint)
    except ImportError:
        return False

    resource = Resource.create({
        "service.name": "gymrat",
        "service.version": importlib.metadata.version("gymrat"),
    })

    id_generator = _deterministic_id_generator(trace_id_of(session_id))
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
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import (  # noqa: PLC0415 -- optional extra
        OTLPSpanExporter,
    )
    from opentelemetry.sdk.trace.export import BatchSpanProcessor  # noqa: PLC0415 -- optional extra

    exporter = _failure_recording(OTLPSpanExporter(endpoint=_traces_url(endpoint)))
    return BatchSpanProcessor(exporter)


def start_span(  # noqa: PLR0913 -- the key and session, plus the tracer start_span keywords it forwards
    name: str,
    *,
    span_key: str | None = None,
    session_id: str | None = None,
    context: Context | None = None,
    links: Sequence[Link] | None = None,
    attributes: Attrs | None = None,
    start_time: int | None = None,
) -> Span:
    """Start a span through the module's tracer, which :func:`configure_tracing` must have set.

    Args:
        name: The span name.
        span_key: When given, derives a deterministic span ID from the
            session ID and this key.
        session_id: The session the deterministic span ID is derived from;
            ``None`` uses the session tracing was configured for.
        context: The parent context, or ``None`` for the current one.
        links: Spans this span links to, or ``None`` for none.
        attributes: The span's attributes, or ``None`` for none.
        start_time: The span's start, in epoch nanoseconds, or ``None`` for now.

    Returns:
        The started span.

    Raises:
        RuntimeError: When :func:`configure_tracing` has not set a tracer.
    """
    global _session_span_dropped  # noqa: PLW0603 — module singleton
    token = None
    if span_key is not None:
        keyed_session = _session_id if session_id is None else session_id
        token = _queued_span_id.set(span_id_of(keyed_session, span_key))
    try:
        if _tracer is None:
            msg = "configure_tracing must run before start_span"
            raise RuntimeError(msg)
        span = _tracer.start_span(
            name, context=context, links=links, attributes=attributes, start_time=start_time
        )
    finally:
        if token is not None:
            _queued_span_id.reset(token)
    if span_key == SESSION_SPAN_KEY:
        _session_span_dropped = not span.is_recording()
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
    global _provider, _tracer, _session_id, _export_failed, _session_span_dropped  # noqa: PLW0603 — module singleton
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
    from opentelemetry.sdk.trace.export import (  # noqa: PLC0415 -- optional extra
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


def _deterministic_id_generator(session_trace_id: int) -> IdGenerator:
    """Build the OTel ID generator that keys every span to the session.

    Trace IDs are never random, which the base class already reports.

    Args:
        session_trace_id: The trace ID every span of the session carries.

    Returns:
        A generator yielding ``session_trace_id`` for every trace and the queued
        span ID, or a random one when none is queued, for every span.
    """
    from opentelemetry.sdk.trace.id_generator import (  # noqa: PLC0415 -- optional extra
        IdGenerator as _IdGenerator,
    )

    class _DeterministicIdGenerator(_IdGenerator):
        @override
        def generate_trace_id(self) -> int:
            return session_trace_id

        @override
        def generate_span_id(self) -> int:
            queued = _queued_span_id.get()
            if queued is not None:
                return queued
            return random.getrandbits(64) or 1

    return _DeterministicIdGenerator()
