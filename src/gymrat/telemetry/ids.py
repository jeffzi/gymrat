from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from opentelemetry.trace import Span, SpanContext

_TRACEPARENT_HEADER = "traceparent"


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
