"""Tests for the tracing provider.

Covers configure, start_span and flush, the deterministic trace and span ID
derivation, and the traceparent helpers.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, override

import pytest
from opentelemetry.sdk.trace import SpanProcessor
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import INVALID_SPAN, SpanContext

if TYPE_CHECKING:
    from collections.abc import Callable

from gymrat.telemetry.provider import (
    configure_tracing,
    export_failed,
    flush_tracing,
    format_traceparent,
    id_from_digest,
    parse_traceparent,
    span_id_of,
    start_span,
    trace_id_of,
)
from gymrat.utils import ENDPOINT_ENV
from tests.telemetry._collector import otlp_collector
from tests.telemetry._fixtures import (
    arm_placeholder_endpoint,
    hide_otel_sdk,
    hide_otlp_exporter,
    memory_tracing,
)

_TRACES_ENDPOINT_ENV = "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"

SESSION = "test-session-provider"


# ---------------------------------------------------------------------------
# configure_tracing — environment gate
# ---------------------------------------------------------------------------


def test_configure_tracing_when_endpoint_unset_does_return_false():
    result = configure_tracing(SESSION)

    assert result is False


def test_configure_tracing_when_endpoint_given_does_export_to_it_over_the_environment(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv(ENDPOINT_ENV, "http://127.0.0.1:9")
    with otlp_collector() as collector:
        configure_tracing(SESSION, endpoint=collector.endpoint)
        with start_span("probe"):
            pass

        flush_tracing()

    assert [export.path for export in collector.received] == ["/v1/traces"]


def test_configure_tracing_when_endpoint_given_does_export_to_it_over_the_traces_variable(
    monkeypatch: pytest.MonkeyPatch,
):
    with otlp_collector() as collector:
        monkeypatch.setenv(_TRACES_ENDPOINT_ENV, f"{collector.endpoint}/custom/traces")
        configure_tracing(SESSION, endpoint=collector.endpoint)
        with start_span("probe"):
            pass

        flush_tracing()

    assert [export.path for export in collector.received] == ["/v1/traces"]


def test_configure_tracing_when_traces_endpoint_set_over_base_does_export_to_the_traces_endpoint(
    monkeypatch: pytest.MonkeyPatch,
):
    with otlp_collector() as collector:
        monkeypatch.setenv(ENDPOINT_ENV, "http://127.0.0.1:1")
        monkeypatch.setenv(_TRACES_ENDPOINT_ENV, f"{collector.endpoint}/custom/traces")
        configure_tracing(SESSION)
        with start_span("probe"):
            pass

        flush_tracing()

    assert [export.path for export in collector.received] == ["/custom/traces"]


# ---------------------------------------------------------------------------
# flush_tracing
# ---------------------------------------------------------------------------


class _StalledFlushSpanProcessor(SpanProcessor):
    """Span processor whose flush never finishes in time, as a hung exporter's would."""

    @override
    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return False


def test_flush_tracing_when_flush_does_not_finish_in_time_does_count_as_failed_export(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv(ENDPOINT_ENV, "http://collector:4318")
    configure_tracing(SESSION, span_processor=_StalledFlushSpanProcessor())

    flush_tracing()

    assert export_failed()


def test_flush_tracing_when_a_rejected_batch_precedes_accepted_ones_does_keep_the_failure(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("OTEL_BSP_MAX_EXPORT_BATCH_SIZE", "1")
    with otlp_collector(statuses=[400]) as collector:
        configure_tracing(SESSION, endpoint=collector.endpoint)
        for name in ("first", "second", "third"):
            with start_span(name):
                pass

        flush_tracing()

    assert (len(collector.received), export_failed()) == (3, True)


# ---------------------------------------------------------------------------
# configure_tracing — import gates
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "hide",
    [
        pytest.param(hide_otel_sdk, id="sdk-missing"),
        pytest.param(hide_otlp_exporter, id="exporter-missing"),
    ],
)
def test_configure_tracing_when_a_package_is_missing_does_return_false(
    monkeypatch: pytest.MonkeyPatch, hide: Callable[[pytest.MonkeyPatch], None]
):
    arm_placeholder_endpoint(monkeypatch)
    hide(monkeypatch)

    result = configure_tracing(SESSION)

    assert result is False


def test_configure_tracing_when_exporter_was_missing_does_configure_later_call_afresh(
    monkeypatch: pytest.MonkeyPatch,
):
    arm_placeholder_endpoint(monkeypatch)
    with monkeypatch.context() as hidden:
        hide_otlp_exporter(hidden)
        configure_tracing(SESSION)

    configured = configure_tracing(
        "session-after-missing-exporter",
        span_processor=SimpleSpanProcessor(InMemorySpanExporter()),
    )

    assert configured is True


# ---------------------------------------------------------------------------
# configure_tracing — provider creation
# ---------------------------------------------------------------------------


def test_configure_tracing_when_called_twice_does_reuse_provider():
    with memory_tracing(SESSION) as exporter:
        result = configure_tracing(SESSION)
        with start_span("probe"):
            pass

    assert result is True
    assert [span.name for span in exporter.get_finished_spans()] == ["probe"]


def test_configure_tracing_when_already_configured_with_span_processor_does_raise():
    with (
        memory_tracing(SESSION),
        pytest.raises(ValueError, match=SESSION),
    ):
        configure_tracing(
            SESSION,
            span_processor=SimpleSpanProcessor(InMemorySpanExporter()),
        )


def test_configure_tracing_when_called_with_different_session_id_does_raise():
    with memory_tracing(SESSION), pytest.raises(ValueError, match="other-session"):
        configure_tracing("other-session")


# ---------------------------------------------------------------------------
# Deterministic IDs via IdGenerator
# ---------------------------------------------------------------------------


def test_start_span_when_keyed_without_parent_does_derive_trace_and_span_ids_from_the_session():
    with memory_tracing(SESSION) as exporter, start_span("keyed-span", span_key="my-key"):
        pass

    finished = exporter.get_finished_spans()
    assert len(finished) == 1
    context = finished[0].context
    assert context is not None
    assert (context.trace_id, context.span_id) == (
        trace_id_of(SESSION),
        span_id_of(SESSION, "my-key"),
    )


def test_start_span_when_no_span_key_does_use_random_span_id():
    with memory_tracing(SESSION) as exporter:
        with start_span("random-span"):
            pass
        with start_span("random-span"):
            pass

    # pyrefly: ignore[missing-attribute] -- a finished span always carries its context
    span_ids = {span.context.span_id for span in exporter.get_finished_spans()}
    assert len(span_ids) == 2


# ---------------------------------------------------------------------------
# trace_id_of
# ---------------------------------------------------------------------------


def test_trace_id_of_when_called_does_return_pinned_sha256_value():
    result = trace_id_of("test-session")

    assert result == 97386156705156847130924781873076287828


# ---------------------------------------------------------------------------
# id_from_digest
# ---------------------------------------------------------------------------


def test_id_from_digest_when_leading_bytes_are_zero_does_return_one():
    # A zero ID is invalid in W3C Trace Context; the derivation returns 1 instead.
    width = 8
    digest = b"\x00" * width + b"\xff" * (32 - width)

    result = id_from_digest(digest, width)

    assert result == 1


# ---------------------------------------------------------------------------
# span_id_of
# ---------------------------------------------------------------------------


def test_span_id_of_when_called_does_return_pinned_sha256_value():
    result = span_id_of("test-session", "root")

    assert result == 3890457429828929257


# ---------------------------------------------------------------------------
# format_traceparent
# ---------------------------------------------------------------------------


def test_format_traceparent_when_span_context_invalid_does_raise_value_error():
    with pytest.raises(ValueError, match=re.escape(repr(INVALID_SPAN.get_span_context()))):
        format_traceparent(INVALID_SPAN)


# ---------------------------------------------------------------------------
# parse_traceparent
# ---------------------------------------------------------------------------


def test_parse_traceparent_when_valid_does_return_a_span_context():
    result = parse_traceparent("00-0102030405060708090a0b0c0d0e0f10-1112131415161718-01")

    assert isinstance(result, SpanContext)


def test_parse_traceparent_when_malformed_does_return_none():
    result = parse_traceparent("too-few-parts")

    assert result is None
