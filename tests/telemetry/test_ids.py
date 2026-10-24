"""Tests for deterministic trace/span ID generation and traceparent helpers."""

from __future__ import annotations

import pytest
from opentelemetry.trace import INVALID_SPAN, NonRecordingSpan, SpanContext, TraceFlags

from gymrat.telemetry.provider import (
    format_traceparent,
    id_from_digest,
    parse_traceparent,
    span_id_of,
    trace_id_of,
)

# ---------------------------------------------------------------------------
# trace_id_of
# ---------------------------------------------------------------------------


def test_trace_id_of_when_called_does_return_pinned_sha256_value() -> None:
    result = trace_id_of("test-session")

    assert result == 97386156705156847130924781873076287828


# ---------------------------------------------------------------------------
# id_from_digest
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("width", [pytest.param(16, id="trace-id"), pytest.param(8, id="span-id")])
def test_id_from_digest_when_leading_bytes_are_zero_does_return_one(width: int) -> None:
    # A zero ID is invalid in W3C Trace Context; the derivation returns 1 instead.
    digest = b"\x00" * width + b"\xff" * (32 - width)

    result = id_from_digest(digest, width)

    assert result == 1


# ---------------------------------------------------------------------------
# span_id_of
# ---------------------------------------------------------------------------


def test_span_id_of_when_called_does_return_pinned_sha256_value() -> None:
    result = span_id_of("test-session", "root")

    assert result == 3890457429828929257


# ---------------------------------------------------------------------------
# format_traceparent
# ---------------------------------------------------------------------------

_TRACE_ID = 0x0102030405060708090A0B0C0D0E0F10
_SPAN_ID = 0x1112131415161718


def test_format_traceparent_when_given_span_does_return_w3c_header() -> None:
    span = NonRecordingSpan(
        SpanContext(
            trace_id=_TRACE_ID,
            span_id=_SPAN_ID,
            is_remote=False,
            trace_flags=TraceFlags(TraceFlags.SAMPLED),
        )
    )

    result = format_traceparent(span)

    assert result == "00-0102030405060708090a0b0c0d0e0f10-1112131415161718-01"


@pytest.mark.parametrize(
    "span",
    [
        pytest.param(INVALID_SPAN, id="invalid-span"),
        pytest.param(
            NonRecordingSpan(
                SpanContext(
                    trace_id=_TRACE_ID,
                    span_id=0,
                    is_remote=False,
                    trace_flags=TraceFlags(TraceFlags.SAMPLED),
                )
            ),
            id="zero-span-id",
        ),
    ],
)
def test_format_traceparent_when_span_context_invalid_does_raise_value_error(
    span: NonRecordingSpan,
) -> None:
    with pytest.raises(ValueError, match="no valid trace context"):
        format_traceparent(span)


# ---------------------------------------------------------------------------
# parse_traceparent
# ---------------------------------------------------------------------------


def test_parse_traceparent_when_valid_does_return_remote_span_context() -> None:
    result = parse_traceparent("00-0102030405060708090a0b0c0d0e0f10-1112131415161718-01")

    assert isinstance(result, SpanContext)
    assert result.trace_id == _TRACE_ID
    assert result.span_id == _SPAN_ID
    assert result.is_remote is True
    assert result.trace_flags == TraceFlags.SAMPLED


def test_parse_traceparent_when_malformed_does_return_none() -> None:
    result = parse_traceparent("too-few-parts")

    assert result is None
