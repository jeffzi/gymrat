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


@pytest.mark.parametrize(
    ("trace_id", "span_id", "trace_flags", "expected"),
    [
        pytest.param(
            _TRACE_ID,
            _SPAN_ID,
            TraceFlags.SAMPLED,
            "00-0102030405060708090a0b0c0d0e0f10-1112131415161718-01",
            id="sampled",
        ),
        pytest.param(
            _TRACE_ID,
            _SPAN_ID,
            TraceFlags.DEFAULT,
            "00-0102030405060708090a0b0c0d0e0f10-1112131415161718-00",
            id="not-sampled",
        ),
        pytest.param(
            1,
            1,
            TraceFlags.SAMPLED,
            "00-00000000000000000000000000000001-0000000000000001-01",
            id="leading-zeros",
        ),
    ],
)
def test_format_traceparent_when_given_span_does_return_w3c_header(
    trace_id: int, span_id: int, trace_flags: TraceFlags, expected: str
) -> None:
    span = NonRecordingSpan(
        SpanContext(trace_id=trace_id, span_id=span_id, is_remote=False, trace_flags=trace_flags)
    )

    result = format_traceparent(span)

    assert result == expected


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


@pytest.mark.parametrize(
    ("header", "expected_flags"),
    [
        pytest.param(
            "00-0102030405060708090a0b0c0d0e0f10-1112131415161718-01",
            TraceFlags.SAMPLED,
            id="canonical",
        ),
        pytest.param(
            "01-0102030405060708090a0b0c0d0e0f10-1112131415161718-01",
            TraceFlags.SAMPLED,
            id="future-version",
        ),
        pytest.param(
            "01-0102030405060708090a0b0c0d0e0f10-1112131415161718-01-extra",
            TraceFlags.SAMPLED,
            id="future-version-extra-fields",
        ),
        pytest.param(
            " \t00-0102030405060708090a0b0c0d0e0f10-1112131415161718-01\t ",
            TraceFlags.SAMPLED,
            id="surrounding-whitespace",
        ),
        pytest.param(
            "00-0102030405060708090a0b0c0d0e0f10-1112131415161718-00",
            TraceFlags.DEFAULT,
            id="not-sampled",
        ),
    ],
)
def test_parse_traceparent_when_valid_does_return_remote_span_context(
    header: str, expected_flags: int
) -> None:
    result = parse_traceparent(header)

    assert isinstance(result, SpanContext)
    assert result.trace_id == 0x0102030405060708090A0B0C0D0E0F10
    assert result.span_id == 0x1112131415161718
    assert result.is_remote is True
    assert result.trace_flags == expected_flags


@pytest.mark.parametrize(
    "header",
    [
        pytest.param("00-abc-1112131415161718-01", id="short-trace-id"),
        pytest.param(
            "00-0102030405060708090a0b0c0d0e0f10ff-1112131415161718-01",
            id="long-trace-id",
        ),
        pytest.param("00-0102030405060708090a0b0c0d0e0f10-abc-01", id="short-span-id"),
        pytest.param(
            "00-0102030405060708090a0b0c0d0e0f10-1112131415161718ff-01",
            id="long-span-id",
        ),
        pytest.param("too-few-parts", id="one-part"),
        pytest.param("00-abc-01", id="three-parts"),
        pytest.param(
            "00-0102030405060708090a0b0c0d0e0f10-1112131415161718-01-extra",
            id="version-00-extra-fields",
        ),
        pytest.param(
            "ff-0102030405060708090a0b0c0d0e0f10-1112131415161718-01", id="forbidden-version"
        ),
        pytest.param("00-zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz-1112131415161718-01", id="non-hex-chars"),
        pytest.param("00-0102030405060708090A0B0C0D0E0F10-1112131415161718-01", id="uppercase-hex"),
        pytest.param(
            "00-00000000000000000000000000000000-1112131415161718-01", id="all-zero-trace-id"
        ),
        pytest.param(
            "00-0102030405060708090a0b0c0d0e0f10-0000000000000000-01", id="all-zero-span-id"
        ),
        pytest.param("", id="empty"),
    ],
)
def test_parse_traceparent_when_malformed_does_return_none(header: str) -> None:
    assert parse_traceparent(header) is None
