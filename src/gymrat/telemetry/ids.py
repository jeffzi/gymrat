from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from opentelemetry.trace import SpanContext


def trace_id_of(session_id: str) -> int:
    """Derive a deterministic 128-bit trace ID from a session identifier."""
    raw = int.from_bytes(hashlib.sha256(session_id.encode()).digest()[:16], "big")
    return raw or 1


def span_id_of(session_id: str, key: str) -> int:
    """Derive a deterministic 64-bit span ID from a session identifier and key."""
    raw = int.from_bytes(hashlib.sha256((session_id + "\0" + key).encode()).digest()[:8], "big")
    return raw or 1


def format_traceparent(span: object) -> str:
    """Format a W3C traceparent header from a span's context."""
    ctx = span.get_span_context()  # type: ignore[attr-defined]
    return f"00-{ctx.trace_id:032x}-{ctx.span_id:016x}-01"


_TRACE_HEX_LEN = 32
_SPAN_HEX_LEN = 16
_TRACEPARENT_PARTS = 4


def parse_traceparent(value: str) -> SpanContext | None:
    """Parse a W3C traceparent header into a SpanContext, or None if malformed."""
    parts = value.split("-")
    if len(parts) != _TRACEPARENT_PARTS:
        return None

    version, trace_hex, span_hex, _flags = parts
    if version != "00":
        return None
    if len(trace_hex) != _TRACE_HEX_LEN or len(span_hex) != _SPAN_HEX_LEN:
        return None

    try:
        trace_id = int(trace_hex, 16)
        span_id = int(span_hex, 16)
    except ValueError:
        return None

    if trace_id == 0 or span_id == 0:
        return None

    from opentelemetry.trace import SpanContext, TraceFlags  # noqa: PLC0415

    return SpanContext(
        trace_id=trace_id,
        span_id=span_id,
        is_remote=True,
        trace_flags=TraceFlags(TraceFlags.SAMPLED),
    )
