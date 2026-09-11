"""Tests for deterministic trace/span ID generation and traceparent helpers."""

from __future__ import annotations

import hashlib
from types import SimpleNamespace

import pytest
from opentelemetry.trace import SpanContext, TraceFlags

from gymrat.telemetry.ids import (
    format_traceparent,
    parse_traceparent,
    span_id_of,
    trace_id_of,
)

# ---------------------------------------------------------------------------
# trace_id_of
# ---------------------------------------------------------------------------


def test_trace_id_of_when_called_does_return_pinned_sha256_value() -> None:
    expected = int.from_bytes(hashlib.sha256(b"test-session").digest()[:16], "big")

    result = trace_id_of("test-session")

    assert result == expected
    assert result == 97386156705156847130924781873076287828


def test_trace_id_of_when_called_twice_does_return_same_value() -> None:
    first = trace_id_of("stable-id")

    second = trace_id_of("stable-id")

    assert first == second


def test_trace_id_of_when_result_would_be_zero_does_return_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A zero trace ID is invalid in W3C Trace Context; the function returns 1.

    class _FakeHash:
        def digest(self) -> bytes:
            return b"\x00" * 32

    monkeypatch.setattr(hashlib, "sha256", lambda _data: _FakeHash())  # pyrefly: ignore[implicit-any-lambda]

    result = trace_id_of("anything")

    assert result == 1


# ---------------------------------------------------------------------------
# span_id_of
# ---------------------------------------------------------------------------


def test_span_id_of_when_called_does_return_pinned_sha256_value() -> None:
    expected = int.from_bytes(hashlib.sha256(b"test-session\x00root").digest()[:8], "big")

    result = span_id_of("test-session", "root")

    assert result == expected
    assert result == 3890457429828929257


def test_span_id_of_when_called_twice_does_return_same_value() -> None:
    first = span_id_of("stable-id", "key")

    second = span_id_of("stable-id", "key")

    assert first == second


def test_span_id_of_when_result_would_be_zero_does_return_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeHash:
        def digest(self) -> bytes:
            return b"\x00" * 32

    monkeypatch.setattr(hashlib, "sha256", lambda _data: _FakeHash())  # pyrefly: ignore[implicit-any-lambda]

    result = span_id_of("anything", "key")

    assert result == 1


# ---------------------------------------------------------------------------
# format_traceparent
# ---------------------------------------------------------------------------


def test_format_traceparent_when_given_span_does_return_w3c_header() -> None:
    ctx = SimpleNamespace(
        trace_id=0x0102030405060708090A0B0C0D0E0F10,
        span_id=0x1112131415161718,
    )
    span = SimpleNamespace(get_span_context=lambda: ctx)

    result = format_traceparent(span)

    assert result == "00-0102030405060708090a0b0c0d0e0f10-1112131415161718-01"


def test_format_traceparent_when_ids_have_leading_zeros_does_zero_pad() -> None:
    ctx = SimpleNamespace(trace_id=1, span_id=1)
    span = SimpleNamespace(get_span_context=lambda: ctx)

    result = format_traceparent(span)

    assert result == "00-00000000000000000000000000000001-0000000000000001-01"


# ---------------------------------------------------------------------------
# parse_traceparent
# ---------------------------------------------------------------------------


def test_parse_traceparent_when_valid_does_return_remote_span_context() -> None:
    header = "00-0102030405060708090a0b0c0d0e0f10-1112131415161718-01"

    result = parse_traceparent(header)

    assert isinstance(result, SpanContext)
    assert result.trace_id == 0x0102030405060708090A0B0C0D0E0F10
    assert result.span_id == 0x1112131415161718
    assert result.is_remote is True
    assert result.trace_flags == TraceFlags.SAMPLED


@pytest.mark.parametrize(
    "header",
    [
        pytest.param("00-abc-1112131415161718-01", id="short-trace-id"),
        pytest.param(
            "00-0102030405060708090a0b0c0d0e0f10ff-1112131415161718-01",
            id="long-trace-id",
        ),
    ],
)
def test_parse_traceparent_when_trace_id_wrong_length_does_return_none(
    header: str,
) -> None:
    assert parse_traceparent(header) is None


@pytest.mark.parametrize(
    "header",
    [
        pytest.param("00-0102030405060708090a0b0c0d0e0f10-abc-01", id="short-span-id"),
        pytest.param(
            "00-0102030405060708090a0b0c0d0e0f10-1112131415161718ff-01",
            id="long-span-id",
        ),
    ],
)
def test_parse_traceparent_when_span_id_wrong_length_does_return_none(
    header: str,
) -> None:
    assert parse_traceparent(header) is None


@pytest.mark.parametrize(
    "header",
    [
        pytest.param("too-few-parts", id="one-part"),
        pytest.param("00-abc-01", id="three-parts"),
        pytest.param(
            "00-0102030405060708090a0b0c0d0e0f10-1112131415161718-01-extra", id="five-parts"
        ),
    ],
)
def test_parse_traceparent_when_wrong_part_count_does_return_none(
    header: str,
) -> None:
    assert parse_traceparent(header) is None


def test_parse_traceparent_when_wrong_version_does_return_none() -> None:
    assert parse_traceparent("01-0102030405060708090a0b0c0d0e0f10-1112131415161718-01") is None


def test_parse_traceparent_when_non_hex_chars_does_return_none() -> None:
    assert parse_traceparent("00-zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz-1112131415161718-01") is None


def test_parse_traceparent_when_all_zero_trace_id_does_return_none() -> None:
    assert parse_traceparent("00-00000000000000000000000000000000-1112131415161718-01") is None


def test_parse_traceparent_when_all_zero_span_id_does_return_none() -> None:
    assert parse_traceparent("00-0102030405060708090a0b0c0d0e0f10-0000000000000000-01") is None
