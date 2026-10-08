"""Tests for the tracing provider: configure, start_span, flush, reset."""

from __future__ import annotations

import importlib.metadata
from functools import partial
from typing import TYPE_CHECKING, override

import pytest
from opentelemetry.sdk.trace import SpanProcessor
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import NonRecordingSpan, SpanContext, TraceFlags, set_span_in_context

if TYPE_CHECKING:
    from collections.abc import Callable

from gymrat.telemetry.provider import (
    SESSION_SPAN,
    SESSION_SPAN_KEY,
    configure_tracing,
    export_failed,
    flush_tracing,
    reset_tracing,
    session_span_dropped,
    span_id_of,
    start_span,
    trace_id_of,
)
from tests.telemetry._collector import otlp_collector
from tests.telemetry._fixtures import hide_otel_sdk, hide_otlp_exporter, memory_tracing

_TRACES_ENDPOINT_ENV = "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"

SESSION = "test-session-provider"


# ---------------------------------------------------------------------------
# configure_tracing — environment gate
# ---------------------------------------------------------------------------


def _leave_endpoint_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    """Leave the endpoint as the test environment baseline has it: absent."""


def _set_endpoint(monkeypatch: pytest.MonkeyPatch, *, endpoint: str) -> None:
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", endpoint)


@pytest.mark.parametrize(
    "arrange",
    [
        pytest.param(_leave_endpoint_unset, id="unset"),
        pytest.param(partial(_set_endpoint, endpoint=""), id="empty"),
        pytest.param(partial(_set_endpoint, endpoint=" \t "), id="whitespace-only"),
    ],
)
def test_configure_tracing_when_endpoint_unset_or_blank_does_return_false(
    monkeypatch: pytest.MonkeyPatch, arrange: Callable[[pytest.MonkeyPatch], None]
):
    arrange(monkeypatch)

    result = configure_tracing(SESSION)

    assert result is False


def test_configure_tracing_when_endpoint_given_does_export_to_it_over_the_environment(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://127.0.0.1:9")
    with otlp_collector() as collector:
        configure_tracing(SESSION, endpoint=collector.endpoint)
        with start_span("probe"):
            pass

        flush_tracing()

    assert [(export.path, export.span_names) for export in collector.received] == [
        ("/v1/traces", ["probe"])
    ]


def _pad_endpoint(monkeypatch: pytest.MonkeyPatch, endpoint: str) -> None:
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", f"  {endpoint} ")


def _pad_traces_endpoint(monkeypatch: pytest.MonkeyPatch, endpoint: str) -> None:
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://127.0.0.1:1")
    monkeypatch.setenv(_TRACES_ENDPOINT_ENV, f"  {endpoint}/custom/traces ")


@pytest.mark.parametrize(
    ("arrange", "expected_path"),
    [
        pytest.param(_pad_endpoint, "/v1/traces", id="base-endpoint"),
        pytest.param(_pad_traces_endpoint, "/custom/traces", id="traces-endpoint-over-base"),
    ],
)
def test_configure_tracing_when_endpoint_variable_padded_does_export_to_it_trimmed(
    monkeypatch: pytest.MonkeyPatch,
    arrange: Callable[[pytest.MonkeyPatch, str], None],
    expected_path: str,
):
    with otlp_collector() as collector:
        arrange(monkeypatch, collector.endpoint)
        configure_tracing(SESSION)
        with start_span("probe"):
            pass

        flush_tracing()

    assert [(export.path, export.span_names) for export in collector.received] == [
        (expected_path, ["probe"])
    ]


class _StalledFlushSpanProcessor(SpanProcessor):
    """Span processor whose flush never finishes in time, as a hung exporter's would."""

    @override
    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return False


def test_flush_tracing_when_flush_does_not_finish_in_time_does_count_as_failed_export(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://collector:4318")
    configure_tracing(SESSION, span_processor=_StalledFlushSpanProcessor())

    flush_tracing()

    assert export_failed()


# ---------------------------------------------------------------------------
# configure_tracing — SDK import gate
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
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318")
    hide(monkeypatch)

    result = configure_tracing(SESSION)

    assert result is False


# ---------------------------------------------------------------------------
# configure_tracing — OTLP exporter import gate
# ---------------------------------------------------------------------------


def test_configure_tracing_when_exporter_was_missing_does_configure_later_call_afresh(
    monkeypatch: pytest.MonkeyPatch,
):

    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318")
    with monkeypatch.context() as hidden:
        hide_otlp_exporter(hidden)
        configure_tracing(SESSION)
    exporter = InMemorySpanExporter()

    configured = configure_tracing(
        "session-after-missing-exporter",
        span_processor=SimpleSpanProcessor(exporter),
    )
    with start_span("probe"):
        pass

    # pyrefly: ignore[missing-attribute] -- a finished span always carries its context
    trace_ids = [span.context.trace_id for span in exporter.get_finished_spans()]
    assert configured is True
    assert trace_ids == [trace_id_of("session-after-missing-exporter")]


# ---------------------------------------------------------------------------
# configure_tracing — provider creation
# ---------------------------------------------------------------------------


def test_configure_tracing_when_called_does_name_and_version_the_service():
    with memory_tracing(SESSION) as exporter:
        span = start_span("probe")
        span.__enter__()
        span.__exit__(None, None, None)

    finished = exporter.get_finished_spans()
    attributes = finished[0].resource.attributes
    assert (attributes["service.name"], attributes["service.version"]) == (
        "gymrat",
        importlib.metadata.version("gymrat"),
    )


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
        pytest.raises(ValueError, match="span_processor"),
    ):
        configure_tracing(
            SESSION,
            span_processor=SimpleSpanProcessor(InMemorySpanExporter()),
        )


def test_configure_tracing_when_called_with_different_session_id_does_raise():
    with memory_tracing(SESSION), pytest.raises(ValueError, match="session_id"):
        configure_tracing("other-session")


# ---------------------------------------------------------------------------
# Deterministic IDs via IdGenerator
# ---------------------------------------------------------------------------


def test_start_span_when_keyed_without_parent_does_derive_trace_and_span_ids_from_the_session():
    with memory_tracing(SESSION) as exporter:
        span = start_span("keyed-span", span_key="my-key")
        span.__enter__()
        span.__exit__(None, None, None)

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
# reset_tracing
# ---------------------------------------------------------------------------


class _ShutdownFailingSpanProcessor(SpanProcessor):
    """Span processor whose flushes time out and whose first ``shutdown`` raises.

    The first shutdown records what :func:`export_failed` and
    :func:`session_span_dropped` report while it runs, which tells whether the
    module state was cleared before the shutdown began.
    Later shutdowns are no-ops so the provider's ``atexit`` handler — left
    registered when the first shutdown raises — stays quiet at interpreter exit.
    """

    def __init__(self) -> None:
        self._shut_down = False
        self.flags_during_shutdown: tuple[bool, bool] | None = None

    @override
    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return False

    @override
    def shutdown(self) -> None:
        if self._shut_down:
            return
        self._shut_down = True
        self.flags_during_shutdown = (export_failed(), session_span_dropped())
        msg = "exporter shutdown failed"
        raise RuntimeError(msg)


def _configure_with_failing_shutdown(
    monkeypatch: pytest.MonkeyPatch,
) -> _ShutdownFailingSpanProcessor:
    """Configure tracing on a failing-shutdown processor, with export and session-drop flags set."""
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318")
    processor = _ShutdownFailingSpanProcessor()
    configure_tracing(SESSION, span_processor=processor)
    flush_tracing()
    # An unsampled parent makes the default parent-based sampler drop the session span.
    unsampled = NonRecordingSpan(
        SpanContext(trace_id=1, span_id=1, is_remote=True, trace_flags=TraceFlags(0))
    )
    start_span(SESSION_SPAN, span_key=SESSION_SPAN_KEY, context=set_span_in_context(unsampled))
    return processor


def test_reset_tracing_when_shutdown_raises_does_propagate_with_flags_cleared(
    monkeypatch: pytest.MonkeyPatch,
):
    processor = _configure_with_failing_shutdown(monkeypatch)

    with pytest.raises(RuntimeError, match="shutdown failed"):
        reset_tracing()

    assert processor.flags_during_shutdown == (False, False)


def test_configure_tracing_when_previous_reset_raised_does_configure_afresh(
    monkeypatch: pytest.MonkeyPatch,
):
    _configure_with_failing_shutdown(monkeypatch)
    with pytest.raises(RuntimeError, match="shutdown failed"):
        reset_tracing()

    result = configure_tracing(
        "session-after-failed-reset",
        span_processor=SimpleSpanProcessor(InMemorySpanExporter()),
    )

    assert result is True
