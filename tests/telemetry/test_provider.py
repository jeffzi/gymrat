"""Tests for the tracing provider: configure, start_span, flush, reset."""

from __future__ import annotations

import importlib.metadata
import sys
import warnings
from typing import TYPE_CHECKING, override

import pytest
from opentelemetry.sdk.trace import SpanProcessor

if TYPE_CHECKING:
    from collections.abc import Iterator

from gymrat.telemetry.ids import span_id_of, trace_id_of
from gymrat.telemetry.provider import (
    _reset_for_tests,
    configure_tracing,
    flush_tracing,
    start_span,
)
from tests.telemetry._fixtures import memory_tracing

SESSION = "test-session-provider"


def _reset_provider_quietly() -> None:
    """Call ``_reset_for_tests`` with OTel's deprecation warnings suppressed."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _reset_for_tests()


@pytest.fixture(autouse=True)
def _isolate_provider(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Ensure every test starts and ends with no provider and a clean environment."""
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    _reset_provider_quietly()
    yield
    _reset_provider_quietly()


# ---------------------------------------------------------------------------
# configure_tracing — environment gate
# ---------------------------------------------------------------------------


def test_configure_tracing_when_endpoint_unset_does_return_false(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)

    result = configure_tracing(SESSION)

    assert result is False


def test_configure_tracing_when_endpoint_blank_does_return_false(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "")

    result = configure_tracing(SESSION)

    assert result is False


# ---------------------------------------------------------------------------
# configure_tracing — SDK import gate
# ---------------------------------------------------------------------------


def test_configure_tracing_when_sdk_missing_does_return_false(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318")

    saved = {}
    sdk_keys = [
        k for k in sys.modules if k == "opentelemetry.sdk" or k.startswith("opentelemetry.sdk.")
    ]
    for key in sdk_keys:
        saved[key] = sys.modules[key]

    try:
        for key in sdk_keys:
            monkeypatch.setitem(sys.modules, key, None)
        monkeypatch.setitem(sys.modules, "opentelemetry.sdk", None)
        monkeypatch.setitem(sys.modules, "opentelemetry.sdk.trace", None)

        result = configure_tracing(SESSION)
    finally:
        for key, mod in saved.items():
            sys.modules[key] = mod

    assert result is False


# ---------------------------------------------------------------------------
# configure_tracing — provider creation
# ---------------------------------------------------------------------------


def test_configure_tracing_when_endpoint_set_does_return_true(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318")
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    result = configure_tracing(
        SESSION,
        span_processor=SimpleSpanProcessor(InMemorySpanExporter()),
    )

    assert result is True


def test_configure_tracing_when_called_does_set_resource_service_name(
    monkeypatch: pytest.MonkeyPatch,
):
    with memory_tracing(SESSION) as exporter:
        span = start_span("probe")
        span.__enter__()
        span.__exit__(None, None, None)

    finished = exporter.get_finished_spans()
    resource = finished[0].resource
    assert resource.attributes["service.name"] == "gymrat"


def test_configure_tracing_when_called_does_set_resource_service_version(
    monkeypatch: pytest.MonkeyPatch,
):
    expected_version = importlib.metadata.version("gymrat")

    with memory_tracing(SESSION) as exporter:
        span = start_span("probe")
        span.__enter__()
        span.__exit__(None, None, None)

    finished = exporter.get_finished_spans()
    resource = finished[0].resource
    assert resource.attributes["service.version"] == expected_version


def test_configure_tracing_when_called_twice_does_reuse_provider(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318")
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    configure_tracing(SESSION, span_processor=SimpleSpanProcessor(InMemorySpanExporter()))

    result = configure_tracing(SESSION)

    assert result is True


def test_configure_tracing_when_already_configured_with_span_processor_does_raise(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318")
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    configure_tracing(SESSION, span_processor=SimpleSpanProcessor(InMemorySpanExporter()))

    with pytest.raises(ValueError, match="span_processor"):
        configure_tracing(
            SESSION,
            span_processor=SimpleSpanProcessor(InMemorySpanExporter()),
        )


def test_configure_tracing_when_called_with_different_session_id_does_raise(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318")
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    configure_tracing(SESSION, span_processor=SimpleSpanProcessor(InMemorySpanExporter()))

    with pytest.raises(ValueError, match="session_id"):
        configure_tracing("other-session")


# ---------------------------------------------------------------------------
# Deterministic IDs via IdGenerator
# ---------------------------------------------------------------------------


def test_start_span_when_no_parent_does_use_session_trace_id():
    with memory_tracing(SESSION) as exporter:
        span = start_span("root-span")
        span.__enter__()
        span.__exit__(None, None, None)

    finished = exporter.get_finished_spans()
    assert len(finished) == 1
    assert finished[0].context.trace_id == trace_id_of(SESSION)  # pyrefly: ignore[missing-attribute]


def test_start_span_when_span_key_given_does_use_deterministic_span_id():
    with memory_tracing(SESSION) as exporter:
        span = start_span("keyed-span", span_key="my-key")
        span.__enter__()
        span.__exit__(None, None, None)

    finished = exporter.get_finished_spans()
    assert len(finished) == 1
    assert finished[0].context.span_id == span_id_of(SESSION, "my-key")  # pyrefly: ignore[missing-attribute]


def test_start_span_when_no_span_key_does_use_random_span_id():
    with memory_tracing(SESSION) as exporter:
        span = start_span("random-span")
        span.__enter__()
        span.__exit__(None, None, None)

    finished = exporter.get_finished_spans()
    assert len(finished) == 1
    assert finished[0].context.span_id != span_id_of(SESSION, "random-span")  # pyrefly: ignore[missing-attribute]


# ---------------------------------------------------------------------------
# flush_tracing
# ---------------------------------------------------------------------------


def test_flush_tracing_when_provider_exists_does_not_raise():
    with memory_tracing(SESSION):
        flush_tracing()


def test_flush_tracing_when_no_provider_does_not_raise():
    flush_tracing()


# ---------------------------------------------------------------------------
# _reset_for_tests
# ---------------------------------------------------------------------------


class _ShutdownFailingSpanProcessor(SpanProcessor):
    """Span processor whose first ``shutdown`` raises, mimicking a failing exporter.

    Later shutdowns are no-ops so the provider's ``atexit`` handler — left
    registered when the first shutdown raises — stays quiet at interpreter exit.
    """

    def __init__(self) -> None:
        self._shut_down = False

    @override
    def shutdown(self) -> None:
        if self._shut_down:
            return
        self._shut_down = True
        msg = "exporter shutdown failed"
        raise RuntimeError(msg)


def test_reset_for_tests_when_shutdown_raises_does_clear_provider(
    monkeypatch: pytest.MonkeyPatch,
):
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318")
    configure_tracing(SESSION, span_processor=_ShutdownFailingSpanProcessor())

    with pytest.raises(RuntimeError, match="shutdown failed"):
        _reset_provider_quietly()

    result = configure_tracing(
        "session-after-failed-reset",
        span_processor=SimpleSpanProcessor(InMemorySpanExporter()),
    )
    assert result is True


def test_reset_for_tests_when_called_does_allow_fresh_provider(
    monkeypatch: pytest.MonkeyPatch,
):
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318")
    configure_tracing(SESSION, span_processor=SimpleSpanProcessor(InMemorySpanExporter()))

    _reset_provider_quietly()

    exporter2 = InMemorySpanExporter()
    result = configure_tracing(
        "new-session",
        span_processor=SimpleSpanProcessor(exporter2),
    )
    assert result is True

    span = start_span("after-reset")
    span.__enter__()
    span.__exit__(None, None, None)

    finished = exporter2.get_finished_spans()
    assert len(finished) == 1
    assert finished[0].context.trace_id == trace_id_of("new-session")  # pyrefly: ignore[missing-attribute]
