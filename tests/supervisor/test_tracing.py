"""Tests for the run-span observer that mirrors supervisor events onto OTel spans."""

from __future__ import annotations

import logging
import warnings

import pytest

from gymrat.supervisor.events import (
    CapEvent,
    CompactionEvent,
    FollowUpEvent,
    TurnEndEvent,
    UsageUpdateEvent,
)
from gymrat.supervisor.tracing import create_run_span_observer
from gymrat.telemetry.provider import _reset_for_tests, start_span
from tests.supervisor._fixtures import make_launch
from tests.telemetry._fixtures import memory_tracing

SESSION = "test-tracing-observer"


@pytest.fixture(autouse=True)
def _isolate_provider(monkeypatch: pytest.MonkeyPatch):
    """Start each test with a clean tracing provider."""
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _reset_for_tests()


# ---------------------------------------------------------------------------
# TurnEndEvent → gymrat.turn_end
# ---------------------------------------------------------------------------


def test_create_run_span_observer_when_turn_end_does_add_span_event():
    with memory_tracing(SESSION) as exporter:
        span = start_span("run")
        span.__enter__()
        observer = create_run_span_observer(span)
        event = TurnEndEvent(
            at=1_000_000_000, text="done", cost_usd=0.05, origin="agent", budget_exhausted=False
        )

        observer(event)

        span.__exit__(None, None, None)

    finished = exporter.get_finished_spans()
    span_events = [e for e in finished[0].events if e.name == "gymrat.turn_end"]
    assert len(span_events) == 1
    attrs = span_events[0].attributes
    assert attrs["gymrat.turn.cost_usd"] == pytest.approx(0.05)  # pyrefly: ignore[unsupported-operation]
    assert attrs["gymrat.turn.origin"] == "agent"  # pyrefly: ignore[unsupported-operation]
    assert attrs["gymrat.turn.budget_exhausted"] is False  # pyrefly: ignore[unsupported-operation]
    assert span_events[0].timestamp == 1_000_000_000


# ---------------------------------------------------------------------------
# FollowUpEvent → gymrat.follow_up
# ---------------------------------------------------------------------------


def test_create_run_span_observer_when_follow_up_with_reason_does_include_reason():
    with memory_tracing(SESSION) as exporter:
        span = start_span("run")
        span.__enter__()
        observer = create_run_span_observer(span)
        event = FollowUpEvent(at=2_000_000_000, action="replied", reason="user asked")

        observer(event)

        span.__exit__(None, None, None)

    finished = exporter.get_finished_spans()
    span_events = [e for e in finished[0].events if e.name == "gymrat.follow_up"]
    assert len(span_events) == 1
    attrs = span_events[0].attributes
    assert attrs["gymrat.follow_up.action"] == "replied"  # pyrefly: ignore[unsupported-operation]
    assert attrs["gymrat.follow_up.reason"] == "user asked"  # pyrefly: ignore[unsupported-operation]
    assert span_events[0].timestamp == 2_000_000_000


def test_create_run_span_observer_when_follow_up_without_reason_does_omit_reason():
    with memory_tracing(SESSION) as exporter:
        span = start_span("run")
        span.__enter__()
        observer = create_run_span_observer(span)
        event = FollowUpEvent(at=3_000_000_000, action="waiting")

        observer(event)

        span.__exit__(None, None, None)

    finished = exporter.get_finished_spans()
    span_events = [e for e in finished[0].events if e.name == "gymrat.follow_up"]
    assert len(span_events) == 1
    attrs = span_events[0].attributes
    assert attrs["gymrat.follow_up.action"] == "waiting"  # pyrefly: ignore[unsupported-operation]
    assert "gymrat.follow_up.reason" not in attrs  # pyrefly: ignore[not-iterable]


# ---------------------------------------------------------------------------
# CapEvent → gymrat.cap
# ---------------------------------------------------------------------------


def test_create_run_span_observer_when_cap_does_add_span_event():
    with memory_tracing(SESSION) as exporter:
        span = start_span("run")
        span.__enter__()
        observer = create_run_span_observer(span)
        event = CapEvent(at=4_000_000_000, cap="wall-clock")

        observer(event)

        span.__exit__(None, None, None)

    finished = exporter.get_finished_spans()
    span_events = [e for e in finished[0].events if e.name == "gymrat.cap"]
    assert len(span_events) == 1
    assert span_events[0].attributes["gymrat.cap.name"] == "wall-clock"  # pyrefly: ignore[unsupported-operation]
    assert span_events[0].timestamp == 4_000_000_000


# ---------------------------------------------------------------------------
# CompactionEvent → gymrat.compaction
# ---------------------------------------------------------------------------


def test_create_run_span_observer_when_compaction_does_add_span_event():
    with memory_tracing(SESSION) as exporter:
        span = start_span("run")
        span.__enter__()
        observer = create_run_span_observer(span)
        event = CompactionEvent(at=5_000_000_000)

        observer(event)

        span.__exit__(None, None, None)

    finished = exporter.get_finished_spans()
    span_events = [e for e in finished[0].events if e.name == "gymrat.compaction"]
    assert len(span_events) == 1
    assert span_events[0].timestamp == 5_000_000_000


# ---------------------------------------------------------------------------
# Ignored event types
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "event",
    [
        pytest.param(make_launch(at=6_000_000_000), id="launch"),
        pytest.param(UsageUpdateEvent(at=7_000_000_000, cost_usd=0.01), id="usage-update"),
    ],
)
def test_create_run_span_observer_when_irrelevant_event_does_not_add_span_event(event: object):
    with memory_tracing(SESSION) as exporter:
        span = start_span("run")
        span.__enter__()
        observer = create_run_span_observer(span)

        observer(event)  # pyrefly: ignore[bad-argument-type]

        span.__exit__(None, None, None)

    finished = exporter.get_finished_spans()
    assert finished[0].events == ()


# ---------------------------------------------------------------------------
# Error suppression
# ---------------------------------------------------------------------------

_TRACING_LOGGER = "gymrat.supervisor.tracing"


class _BrokenSpan:
    def add_event(self, *_args: object, **_kwargs):
        msg = "boom"
        raise RuntimeError(msg)


def _broken_span_turn_end_event() -> TurnEndEvent:
    return TurnEndEvent(
        at=8_000_000_000, text="done", cost_usd=0.05, origin="agent", budget_exhausted=False
    )


def test_create_run_span_observer_when_span_add_event_raises_does_not_propagate():
    observer = create_run_span_observer(_BrokenSpan())  # pyrefly: ignore[bad-argument-type]

    observer(_broken_span_turn_end_event())


def test_create_run_span_observer_when_mirror_fails_does_log_warning(
    caplog: pytest.LogCaptureFixture,
):
    observer = create_run_span_observer(_BrokenSpan())  # pyrefly: ignore[bad-argument-type]

    with caplog.at_level(logging.WARNING, logger=_TRACING_LOGGER):
        observer(_broken_span_turn_end_event())

    warning_records = [
        r for r in caplog.records if r.name == _TRACING_LOGGER and r.levelno == logging.WARNING
    ]
    assert len(warning_records) == 1
    assert "boom" in warning_records[0].message
