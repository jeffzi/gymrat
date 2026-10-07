"""Tests for the run-span observer that mirrors supervisor events onto OTel spans."""

from __future__ import annotations

import pytest

from gymrat.supervisor.events import (
    CapEvent,
    CompactionEvent,
    FollowUpEvent,
    ModelPhaseEvent,
    SessionEvent,
    TextDeltaEvent,
    ThinkingUpdateEvent,
    ToolEndEvent,
    ToolStartEvent,
    UsageUpdateEvent,
    combine_observers,
)
from gymrat.telemetry.provider import start_span
from gymrat.telemetry.run_spans import TracingState, create_run_span_observer, setup_tracing
from tests._logging import unhandled_logging
from tests.session.records._fixtures import SESSION_ID
from tests.supervisor._fixtures import (
    collecting_observer,
    make_launch,
    make_prompt,
    make_turn_end,
    noop_observer,
)
from tests.telemetry._fixtures import memory_tracing

SESSION = "test-tracing-observer"


# ---------------------------------------------------------------------------
# which events are mirrored onto the run span, and as what
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("event", "mirrored"),
    [
        pytest.param(
            make_turn_end(at=1_000_000_000, text="done", cost_usd=0.05),
            [
                (
                    "gymrat.turn_end",
                    {
                        "gymrat.turn.session_cost_usd": pytest.approx(0.05),
                        "gymrat.turn.origin": "agent",
                        "gymrat.turn.budget_exhausted": False,
                    },
                    1_000_000_000,
                )
            ],
            id="turn-end",
        ),
        pytest.param(
            FollowUpEvent(at=2_000_000_000, action="replied", reason="user asked"),
            [
                (
                    "gymrat.follow_up",
                    {"gymrat.follow_up.action": "replied", "gymrat.follow_up.reason": "user asked"},
                    2_000_000_000,
                )
            ],
            id="follow-up-with-reason",
        ),
        pytest.param(
            FollowUpEvent(at=3_000_000_000, action="waiting"),
            [("gymrat.follow_up", {"gymrat.follow_up.action": "waiting"}, 3_000_000_000)],
            id="follow-up-without-reason",
        ),
        pytest.param(
            CapEvent(at=4_000_000_000, cap="wall-clock", action="interrupting"),
            [("gymrat.cap", {"gymrat.cap.name": "wall-clock"}, 4_000_000_000)],
            id="cap",
        ),
        pytest.param(
            CompactionEvent(at=5_000_000_000),
            [("gymrat.compaction", {}, 5_000_000_000)],
            id="compaction",
        ),
        pytest.param(make_launch(at=6_000_000_000), [], id="launch-ignored"),
        pytest.param(
            UsageUpdateEvent(at=7_000_000_000, cost_usd=0.01), [], id="usage-update-ignored"
        ),
        pytest.param(
            ThinkingUpdateEvent(at=7_000_000_000, estimated_tokens=10, delta=10),
            [],
            id="thinking-update-ignored",
        ),
        pytest.param(
            ToolStartEvent(
                at=7_000_000_000,
                tool_use_id="t1",
                tool_name="Read",
                input={},
                input_summary="Read x.py",
            ),
            [],
            id="tool-start-ignored",
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
            [],
            id="tool-end-ignored",
        ),
        pytest.param(TextDeltaEvent(at=7_000_000_000, chunk="hi"), [], id="text-delta-ignored"),
        pytest.param(
            ModelPhaseEvent(at=7_000_000_000, phase="thinking"), [], id="model-phase-ignored"
        ),
    ],
)
def test_create_run_span_observer_when_event_observed_does_mirror_only_the_run_milestones(
    event: SessionEvent, mirrored: list[tuple[str, dict[str, object], int]]
):
    with memory_tracing(SESSION) as exporter:
        span = start_span("run")
        span.__enter__()
        observer = create_run_span_observer(span)

        observer(event)

        span.__exit__(None, None, None)

    finished = exporter.get_finished_spans()
    assert [(e.name, dict(e.attributes or {}), e.timestamp) for e in finished[0].events] == mirrored


# ---------------------------------------------------------------------------
# Error suppression
# ---------------------------------------------------------------------------


class _BrokenSpan:
    def add_event(self, *_args: object, **_kwargs):
        msg = "boom"
        raise RuntimeError(msg)


def test_create_run_span_observer_when_mirror_fails_in_a_chain_does_contain_the_failure(
    capsys: pytest.CaptureFixture[str],
):
    later = collecting_observer()
    chain = combine_observers(
        create_run_span_observer(_BrokenSpan()),  # pyrefly: ignore[bad-argument-type]
        later.observer,
    )
    event = make_turn_end(at=8_000_000_000, text="done", cost_usd=0.05)

    with unhandled_logging(), pytest.warns(RuntimeWarning, match="boom") as caught:
        chain(event)

    assert len(caught) == 1
    assert later.events == [event]
    assert capsys.readouterr().err == ""


# ---------------------------------------------------------------------------
# setup_tracing without a tracer
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "resumed",
    [pytest.param(False, id="opening-launch"), pytest.param(True, id="resumed-launch")],
)
def test_setup_tracing_when_sdk_disabled_and_endpoint_set_does_hold_no_span(
    monkeypatch: pytest.MonkeyPatch, resumed: bool
):
    monkeypatch.setenv("OTEL_SDK_DISABLED", "true")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318")
    prompt = make_prompt()
    observer = noop_observer()

    traced = setup_tracing(
        make_launch(at=1, head_sha="a" * 40, max_minutes=10, session_id=SESSION_ID),
        branch=f"gymrat/{SESSION_ID}",
        prompt=prompt,
        reporter_observer=observer,
        resumed=resumed,
    )

    assert traced == (prompt, observer, TracingState())
