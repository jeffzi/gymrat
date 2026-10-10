"""The session span an earlier process opened, reconstructed as a parent for new spans."""

from __future__ import annotations

from typing import TYPE_CHECKING

from gymrat.telemetry.provider import SESSION_SPAN_KEY, span_id_of, trace_id_of

if TYPE_CHECKING:
    from opentelemetry.trace import Span


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
    from opentelemetry.trace import (  # noqa: PLC0415 -- optional extra
        NonRecordingSpan,
        SpanContext,
        TraceFlags,
    )

    return NonRecordingSpan(
        SpanContext(
            trace_id=trace_id_of(session_id),
            span_id=span_id_of(session_id, SESSION_SPAN_KEY),
            is_remote=False,
            trace_flags=TraceFlags(TraceFlags.SAMPLED),
        )
    )
