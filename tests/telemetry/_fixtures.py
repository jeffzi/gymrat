"""Shared helpers for telemetry provider tests.

The ``memory_tracing`` context manager wires an in-memory exporter so tests
can inspect finished spans without a collector. ``isolate_tracing_provider``
is the autouse fixture every module that configures tracing registers by
importing it.
"""

from __future__ import annotations

import os
import warnings
from contextlib import contextmanager
from typing import TYPE_CHECKING

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


@pytest.fixture(autouse=True)
def isolate_tracing_provider() -> Iterator[None]:
    """Start and end every test with no tracing provider."""
    reset_provider_quietly()
    yield
    reset_provider_quietly()


@contextmanager
def memory_tracing(session_id: str) -> Generator[InMemorySpanExporter]:
    """Configure tracing with an in-memory exporter for testing."""
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    exporter = InMemorySpanExporter()
    os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = "http://localhost:4318"
    try:
        if not configure_tracing(
            session_id,
            span_processor=SimpleSpanProcessor(exporter),
        ):
            msg = "tracing was not configured"
            raise AssertionError(msg)
        yield exporter
    finally:
        os.environ.pop("OTEL_EXPORTER_OTLP_ENDPOINT", None)
        _reset_for_tests()
