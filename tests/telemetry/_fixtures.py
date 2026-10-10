"""Shared helpers for telemetry provider tests.

The ``memory_tracing`` context manager wires an in-memory exporter so tests
can inspect finished spans without a collector; ``hide_otlp_exporter`` and
``hide_otel_sdk`` make the optional tracing packages fail to import,
``arm_placeholder_endpoint`` sets an endpoint so tracing is asked for, and
``disable_otel_sdk`` turns the SDK off while that endpoint is set.
``traceparent`` and ``session_traceparent`` format the W3C header a command
inherits its parent span from.
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

from gymrat.telemetry.provider import configure_tracing, reset_tracing, span_id_of, trace_id_of
from gymrat.utils import ENDPOINT_ENV

#: An OTLP endpoint that turns tracing on; nothing is ever exported to it.
PLACEHOLDER_ENDPOINT = "http://localhost:4318"


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


def arm_placeholder_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    """Point ``OTEL_EXPORTER_OTLP_ENDPOINT`` at :data:`PLACEHOLDER_ENDPOINT` so tracing is asked for.

    Args:
        monkeypatch: Sets the endpoint variable for the rest of the test.
    """
    monkeypatch.setenv(ENDPOINT_ENV, PLACEHOLDER_ENDPOINT)


def disable_otel_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set ``OTEL_SDK_DISABLED`` while an OTLP endpoint is set, so tracing is asked for but off.

    Args:
        monkeypatch: Sets ``OTEL_SDK_DISABLED=true`` and points
            ``OTEL_EXPORTER_OTLP_ENDPOINT`` at :data:`PLACEHOLDER_ENDPOINT`.
    """
    monkeypatch.setenv("OTEL_SDK_DISABLED", "true")
    arm_placeholder_endpoint(monkeypatch)


@contextmanager
def memory_tracing(session_id: str, *, buffered: bool = False) -> Generator[InMemorySpanExporter]:
    """Configure tracing with an in-memory exporter for testing.

    For the duration of the block, ``OTEL_EXPORTER_OTLP_ENDPOINT`` is set to
    :data:`PLACEHOLDER_ENDPOINT`, because code that gates tracing on that
    variable (the command runner) must see tracing as on. The variable's
    previous value, or its absence, is restored on exit.

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
    previous_endpoint = os.environ.get(ENDPOINT_ENV)
    os.environ[ENDPOINT_ENV] = PLACEHOLDER_ENDPOINT
    try:
        if not configure_tracing(session_id, span_processor=processor):
            msg = "tracing was not configured"
            raise AssertionError(msg)
        yield exporter
    finally:
        if previous_endpoint is None:
            os.environ.pop(ENDPOINT_ENV, None)
        else:
            os.environ[ENDPOINT_ENV] = previous_endpoint
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


def traceparent(trace_id: int, span_id: int) -> str:
    """Format the W3C ``traceparent`` header naming a sampled span.

    Args:
        trace_id: The 128-bit trace id.
        span_id: The 64-bit span id.

    Returns:
        The header value.
    """
    return f"00-{trace_id:032x}-{span_id:016x}-01"


def session_traceparent(session_id: str, span_key: str) -> str:
    """Format the ``traceparent`` of the span keyed ``span_key`` in ``session_id``'s trace.

    Args:
        session_id: The session whose trace the span belongs to.
        span_key: The key the span id is derived from, such as ``"run:<at>"``.

    Returns:
        The header value.
    """
    return traceparent(trace_id_of(session_id), span_id_of(session_id, span_key))
