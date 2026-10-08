"""Shared helpers for telemetry provider tests.

The ``memory_tracing`` context manager wires an in-memory exporter so tests
can inspect finished spans without a collector; ``hide_otlp_exporter`` and
``hide_otel_sdk`` make the optional tracing packages fail to import.
"""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Generator

    import pytest

from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from gymrat.telemetry.provider import configure_tracing, reset_tracing


def _hide_package(monkeypatch: pytest.MonkeyPatch, package: str, submodule: str) -> None:
    # Map the package, the submodule the code under test imports, and every
    # loaded submodule to None in sys.modules, so importing any of them raises
    # ImportError.
    loaded = [name for name in sys.modules if name == package or name.startswith(f"{package}.")]
    for name in {package, f"{package}.{submodule}", *loaded}:
        monkeypatch.setitem(sys.modules, name, None)


def hide_otlp_exporter(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the OTLP exporter package fail to import while the OpenTelemetry SDK still imports.

    Args:
        monkeypatch: Maps the package and its loaded submodules to ``None`` in
            ``sys.modules``, so importing any of them raises ``ImportError``.
    """
    _hide_package(monkeypatch, "opentelemetry.exporter.otlp.proto.http", "trace_exporter")


def hide_otel_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the OpenTelemetry SDK and every submodule of it fail to import.

    Args:
        monkeypatch: Maps the package and its loaded submodules to ``None`` in
            ``sys.modules``, so importing any of them raises ``ImportError``.
    """
    _hide_package(monkeypatch, "opentelemetry.sdk", "trace")


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
        reset_tracing()


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
