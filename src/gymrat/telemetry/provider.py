"""Tracing provider: lazy OTel setup with deterministic IDs.

All ``opentelemetry`` imports live inside functions so that importing the
module never pulls the SDK into ``sys.modules`` (CLAUDE.md: "Import the agent
SDK, and anything else with a startup cost, inside the function that needs it").
"""

from __future__ import annotations

import importlib.metadata
import os
import random
import warnings
from contextvars import ContextVar
from typing import TYPE_CHECKING, override

from gymrat.telemetry.endpoint import ENDPOINT_ENV, otlp_endpoint
from gymrat.telemetry.ids import span_id_of, trace_id_of

if TYPE_CHECKING:
    from collections.abc import Sequence

    from opentelemetry.sdk.trace import ReadableSpan, SpanProcessor, TracerProvider
    from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
    from opentelemetry.trace import Span, Tracer

_TRACES_ENDPOINT_ENV = "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"
_TRACES_PATH = "v1/traces"

_session_id: str = ""
_provider: TracerProvider | None = None
_tracer: Tracer | None = None
_export_failed: bool = False

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

    Args:
        session_id: The session identifier used to derive deterministic trace
            IDs.
        span_processor: The span processor to install; a default OTLP
            batch processor is used when omitted.
        endpoint: The OTLP endpoint to export to; ``None`` reads it from
            ``OTEL_EXPORTER_OTLP_ENDPOINT``.

    Returns:
        Whether a tracer provider is now active (``False`` when no endpoint
        is configured or the SDK is missing).

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
                f"span_processor would be silently discarded"
            )
            raise ValueError(msg)
        return True

    endpoint = otlp_endpoint(endpoint if endpoint is not None else os.environ.get(ENDPOINT_ENV))
    if endpoint is None:
        return False

    try:
        from opentelemetry.sdk.resources import Resource  # noqa: PLC0415
        from opentelemetry.sdk.trace import TracerProvider as _TracerProvider  # noqa: PLC0415
    except ImportError:
        return False

    _session_id = session_id

    resource = Resource.create({
        "service.name": "gymrat",
        "service.version": importlib.metadata.version("gymrat"),
    })

    id_generator = _DeterministicIdGenerator(trace_id_of(session_id))
    # pyrefly: ignore[bad-argument-type] -- IdGenerator protocol mismatch
    provider = _TracerProvider(resource=resource, id_generator=id_generator)

    if span_processor is not None:
        provider.add_span_processor(span_processor)
    else:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (  # noqa: PLC0415
            OTLPSpanExporter,
        )
        from opentelemetry.sdk.trace.export import BatchSpanProcessor  # noqa: PLC0415

        exporter = _failure_recording(OTLPSpanExporter(endpoint=_traces_url(endpoint)))
        provider.add_span_processor(BatchSpanProcessor(exporter))

    _provider = provider
    _tracer = provider.get_tracer("gymrat")
    return True


def start_span(name: str, *, span_key: str | None = None, **kwargs: object) -> Span:
    """Start a span through the module's tracer.

    Args:
        name: The span name.
        span_key: When given, derives a deterministic span ID from the
            session ID and this key.
        **kwargs: Additional keyword arguments forwarded to the tracer's
            ``start_span``.

    Returns:
        The started span, or ``INVALID_SPAN`` when no tracer is configured.
    """
    if _tracer is None:
        from opentelemetry.trace import INVALID_SPAN  # noqa: PLC0415

        return INVALID_SPAN

    token = None
    if span_key is not None:
        token = _queued_span_id.set(span_id_of(_session_id, span_key))
    try:
        return _tracer.start_span(name, **kwargs)  # type: ignore[arg-type]
    finally:
        if token is not None:
            _queued_span_id.reset(token)


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


def _reset_for_tests() -> None:
    """Shut down and clear the module singleton so the next configure starts fresh.

    The singleton is cleared before the shutdown runs, so a provider whose
    shutdown fails cannot leave stale state behind for the next configure.

    Raises:
        Exception: Whatever the provider's ``shutdown`` raises, propagated after
            the singleton has been cleared.
    """
    global _provider, _tracer, _session_id, _export_failed  # noqa: PLW0603
    provider = _provider
    _provider = None
    _tracer = None
    _session_id = ""
    _export_failed = False
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
            _queued_span_id.set(None)
            return queued
        return random.getrandbits(64) or 1

    def is_trace_id_random(self) -> bool:
        return False
