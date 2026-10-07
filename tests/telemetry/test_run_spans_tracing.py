"""Tests for the run-span observer that mirrors supervisor events onto OTel spans."""

from __future__ import annotations

import pytest

from gymrat.supervisor.events import (
    CapEvent,
    CompactionEvent,
    FollowUpEvent,
    ModelPhaseEvent,
    TextDeltaEvent,
    ThinkingUpdateEvent,
    ToolEndEvent,
    ToolStartEvent,
    TurnEndEvent,
    UsageUpdateEvent,
    combine_observers,
)
from gymrat.telemetry.provider import start_span
from gymrat.telemetry.run_spans import create_run_span_observer
from tests._logging import unhandled_logging
from tests.supervisor._fixtures import collecting_observer, make_launch
from tests.telemetry._fixtures import (
    isolate_tracing_provider as _isolate_tracing_provider,  # noqa: F401 -- registers the autouse fixture
)
from tests.telemetry._fixtures import memory_tracing

SESSION = "test-tracing-observer"


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
    assert dict(span_events[0].attributes or {}) == {
        "gymrat.turn.session_cost_usd": pytest.approx(0.05),
        "gymrat.turn.origin": "agent",
        "gymrat.turn.budget_exhausted": False,
    }
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
    assert dict(span_events[0].attributes or {}) == {
        "gymrat.follow_up.action": "replied",
        "gymrat.follow_up.reason": "user asked",
    }
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
    assert dict(span_events[0].attributes or {}) == {"gymrat.follow_up.action": "waiting"}


# ---------------------------------------------------------------------------
# CapEvent → gymrat.cap
# ---------------------------------------------------------------------------


def test_create_run_span_observer_when_cap_does_add_span_event():
    with memory_tracing(SESSION) as exporter:
        span = start_span("run")
        span.__enter__()
        observer = create_run_span_observer(span)
        event = CapEvent(at=4_000_000_000, cap="wall-clock", action="interrupting")

        observer(event)

        span.__exit__(None, None, None)

    finished = exporter.get_finished_spans()
    span_events = [e for e in finished[0].events if e.name == "gymrat.cap"]
    assert len(span_events) == 1
    assert dict(span_events[0].attributes or {}) == {"gymrat.cap.name": "wall-clock"}
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
    assert dict(span_events[0].attributes or {}) == {}
    assert span_events[0].timestamp == 5_000_000_000


# ---------------------------------------------------------------------------
# Ignored event types
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "event",
    [
        pytest.param(make_launch(at=6_000_000_000), id="launch"),
        pytest.param(UsageUpdateEvent(at=7_000_000_000, cost_usd=0.01), id="usage-update"),
        pytest.param(
            ThinkingUpdateEvent(at=7_000_000_000, estimated_tokens=10, delta=10),
            id="thinking-update",
        ),
        pytest.param(
            ToolStartEvent(
                at=7_000_000_000,
                tool_use_id="t1",
                tool_name="Read",
                input={},
                input_summary="Read x.py",
            ),
            id="tool-start",
        ),
        pytest.param(
            ToolEndEvent(
                at=7_000_000_000,
                tool_use_id="t1",
                tool_name="Read",
                duration_ms=5,
                result="ok",
                result_summary="ok",
            ),
            id="tool-end",
        ),
        pytest.param(TextDeltaEvent(at=7_000_000_000, chunk="hi"), id="text-delta"),
        pytest.param(ModelPhaseEvent(at=7_000_000_000, phase="thinking"), id="model-phase"),
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


class _BrokenSpan:
    def add_event(self, *_args: object, **_kwargs):
        msg = "boom"
        raise RuntimeError(msg)


def _broken_span_turn_end_event() -> TurnEndEvent:
    return TurnEndEvent(
        at=8_000_000_000, text="done", cost_usd=0.05, origin="agent", budget_exhausted=False
    )


def test_create_run_span_observer_when_mirror_fails_does_warn_once_and_leave_stderr_empty(
    capsys: pytest.CaptureFixture[str],
):
    observer = create_run_span_observer(_BrokenSpan())  # pyrefly: ignore[bad-argument-type]

    with unhandled_logging(), pytest.warns(RuntimeWarning, match="boom") as caught:
        observer(_broken_span_turn_end_event())

    assert (len(caught), capsys.readouterr().err) == (1, "")


def test_create_run_span_observer_when_mirror_fails_in_a_chain_does_warn_once_and_call_remaining():
    later = collecting_observer()
    chain = combine_observers(
        create_run_span_observer(_BrokenSpan()),  # pyrefly: ignore[bad-argument-type]
        later.observer,
    )
    event = _broken_span_turn_end_event()

    with pytest.warns(RuntimeWarning, match="boom") as caught:
        chain(event)

    assert (len(caught), later.events) == (1, [event])
