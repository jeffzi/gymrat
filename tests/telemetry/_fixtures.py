"""Shared helpers for telemetry provider tests.

The ``memory_tracing`` context manager wires an in-memory exporter so tests
can inspect finished spans without a collector. ``isolate_tracing_provider``
is the autouse fixture every module that configures tracing registers by
importing it.
"""

from __future__ import annotations

import os
import sys
import warnings
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from collections.abc import Generator, Iterator

    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from gymrat.telemetry.provider import _reset_for_tests, configure_tracing


def reset_provider_quietly() -> None:
    """Call ``_reset_for_tests`` with OTel's deprecation warnings suppressed."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _reset_for_tests()


_OTLP_EXPORTER_PACKAGE = "opentelemetry.exporter.otlp.proto.http"


def hide_otlp_exporter(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the OTLP exporter package fail to import while the OpenTelemetry SDK still imports.

    Args:
        monkeypatch: Maps the package and its loaded submodules to ``None`` in
            ``sys.modules``, so importing any of them raises ``ImportError``.
    """
    hidden = [
        name
        for name in sys.modules
        if name == _OTLP_EXPORTER_PACKAGE or name.startswith(f"{_OTLP_EXPORTER_PACKAGE}.")
    ]
    for name in {_OTLP_EXPORTER_PACKAGE, f"{_OTLP_EXPORTER_PACKAGE}.trace_exporter", *hidden}:
        monkeypatch.setitem(sys.modules, name, None)


_OTEL_SDK_PACKAGE = "opentelemetry.sdk"


def hide_otel_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the OpenTelemetry SDK and every submodule of it fail to import.

    Args:
        monkeypatch: Maps the package and its loaded submodules to ``None`` in
            ``sys.modules``, so importing any of them raises ``ImportError``.
    """
    hidden = [
        name
        for name in sys.modules
        if name == _OTEL_SDK_PACKAGE or name.startswith(f"{_OTEL_SDK_PACKAGE}.")
    ]
    for name in {_OTEL_SDK_PACKAGE, f"{_OTEL_SDK_PACKAGE}.trace", *hidden}:
        monkeypatch.setitem(sys.modules, name, None)


@pytest.fixture(autouse=True)
def isolate_tracing_provider() -> Iterator[None]:
    """Start and end every test with no tracing provider."""
    reset_provider_quietly()
    yield
    reset_provider_quietly()


@contextmanager
def memory_tracing(session_id: str, *, buffered: bool = False) -> Generator[InMemorySpanExporter]:
    """Configure tracing with an in-memory exporter for testing.

    Args:
        session_id: The session the traced spans belong to.
        buffered: Hold finished spans in a batch processor whose schedule never
            fires during a test, so they reach the exporter only on a flush.
            The default exports each span the moment it ends.

    Yields:
        The exporter that collects the finished spans.

    Raises:
        AssertionError: When tracing could not be configured.
    """
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    exporter = InMemorySpanExporter()
    processor = (
        BatchSpanProcessor(exporter, schedule_delay_millis=60_000)
        if buffered
        else SimpleSpanProcessor(exporter)
    )
    os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = "http://localhost:4318"
    try:
        if not configure_tracing(session_id, span_processor=processor):
            msg = "tracing was not configured"
            raise AssertionError(msg)
        yield exporter
    finally:
        os.environ.pop("OTEL_EXPORTER_OTLP_ENDPOINT", None)
        _reset_for_tests()


def span_by_name(spans: tuple[Any, ...] | list[Any], name: str) -> Any:
    """Return the one span called ``name``.

    Args:
        spans: The finished spans an exporter collected.
        name: The span name to look up.

    Returns:
        The single span carrying that name.

    Raises:
        AssertionError: When no span, or more than one span, carries ``name``.
    """
    matches = [s for s in spans if s.name == name]
    assert len(matches) == 1, f"expected 1 span named {name!r}, got {len(matches)}"
    return matches[0]


def spans_by_prefix(spans: tuple[Any, ...] | list[Any], prefix: str) -> list[Any]:
    """Return the spans whose name starts with ``prefix``, in export order.

    Args:
        spans: The finished spans an exporter collected.
        prefix: The leading part of the span names to keep.

    Returns:
        The matching spans.
    """
    return [s for s in spans if s.name.startswith(prefix)]
