"""Tests for the tracing provider: configure, start_span, flush, reset."""

from __future__ import annotations

import contextlib
import importlib.metadata
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

    assert [(export.path, export.span_names) for export in collector.received] == [
        ("/v1/traces", ["probe"])
    ]


def _pad_endpoint(monkeypatch: pytest.MonkeyPatch, endpoint: str) -> None:
    monkeypatch.setenv(ENDPOINT_ENV, f"  {endpoint} ")


def _pad_traces_endpoint(monkeypatch: pytest.MonkeyPatch, endpoint: str) -> None:
    monkeypatch.setenv(ENDPOINT_ENV, "http://127.0.0.1:1")
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
    with memory_tracing(SESSION) as exporter, start_span("probe"):
        pass

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
# reset_tracing
# ---------------------------------------------------------------------------


class _ShutdownFailingSpanProcessor(SpanProcessor):
    """Span processor whose flushes time out and whose first ``shutdown`` raises.

    Later shutdowns are no-ops so the provider's ``atexit`` handler — left
    registered when the first shutdown raises — stays quiet at interpreter exit.
    """

    def __init__(self) -> None:
        self._shut_down = False

    @override
    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return False

    @override
    def shutdown(self) -> None:
        if self._shut_down:
            return
        self._shut_down = True
        msg = "exporter shutdown failed"
        raise RuntimeError(msg)


def _configure_with_failing_shutdown(monkeypatch: pytest.MonkeyPatch) -> None:
    """Configure tracing on a failing-shutdown processor, with export and session-drop flags set."""
    arm_placeholder_endpoint(monkeypatch)
    configure_tracing(SESSION, span_processor=_ShutdownFailingSpanProcessor())
    flush_tracing()
    # An unsampled parent makes the default parent-based sampler drop the session span.
    unsampled = NonRecordingSpan(
        SpanContext(trace_id=1, span_id=1, is_remote=True, trace_flags=TraceFlags(0))
    )
    start_span(SESSION_SPAN, span_key=SESSION_SPAN_KEY, context=set_span_in_context(unsampled))


def test_reset_tracing_when_shutdown_raises_does_propagate_with_flags_cleared(
    monkeypatch: pytest.MonkeyPatch,
):
    _configure_with_failing_shutdown(monkeypatch)

    with pytest.raises(RuntimeError, match="shutdown failed"):
        reset_tracing()

    assert (export_failed(), session_span_dropped()) == (False, False)


def test_configure_tracing_when_previous_reset_raised_does_configure_afresh(
    monkeypatch: pytest.MonkeyPatch,
):
    _configure_with_failing_shutdown(monkeypatch)
    with contextlib.suppress(RuntimeError):
        reset_tracing()

    result = configure_tracing(
        "session-after-failed-reset",
        span_processor=SimpleSpanProcessor(InMemorySpanExporter()),
    )

    assert result is True
