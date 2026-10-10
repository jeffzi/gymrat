"""Tests for the run-span observer that mirrors supervisor events onto OTel spans."""

from __future__ import annotations

from typing import TYPE_CHECKING, NoReturn
from unittest.mock import create_autospec

import pytest
from opentelemetry.sdk.trace import Span as SdkSpan

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
)
from gymrat.telemetry.provider import start_span
from gymrat.telemetry.run_spans import (
    TracingState,
    create_run_span_observer,
    finalize_tracing,
    setup_tracing,
)
from gymrat.utils import ENDPOINT_ENV
from tests._logging import unhandled_logging
from tests.session.records._fixtures import SESSION_ID
from tests.supervisor._fixtures import make_launch, make_prompt, make_turn_end
from tests.telemetry._fixtures import disable_otel_sdk, memory_tracing

if TYPE_CHECKING:
    from collections.abc import Callable

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
    with memory_tracing(SESSION) as exporter, start_span("run") as span:
        observer = create_run_span_observer(span)

        observer(event)

    finished = exporter.get_finished_spans()
    assert [(e.name, dict(e.attributes or {}), e.timestamp) for e in finished[0].events] == mirrored


# ---------------------------------------------------------------------------
# failures reach the warn sink
# ---------------------------------------------------------------------------


class _BrokenSpan:
    def add_event(self, *_args: object, **_kwargs: object) -> NoReturn:
        msg = "boom"
        raise RuntimeError(msg)


def test_create_run_span_observer_when_mirror_fails_does_send_the_failure_to_the_sink(
    capsys: pytest.CaptureFixture[str],
):
    messages: list[str] = []
    # pyrefly: ignore[bad-argument-type] -- _BrokenSpan stands in for Span with add_event only
    observer = create_run_span_observer(_BrokenSpan(), warn=messages.append)
    event = make_turn_end(at=8_000_000_000, text="done", cost_usd=0.05)

    with unhandled_logging():
        observer(event)

    assert ["boom" in message for message in messages] == [True]
    assert capsys.readouterr().err == ""


def test_setup_tracing_when_run_span_mirror_fails_does_send_the_failure_to_the_sink(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        SdkSpan,
        "add_event",
        create_autospec(SdkSpan.add_event, side_effect=RuntimeError("mirror broke")),
    )
    messages: list[str] = []
    with memory_tracing(SESSION_ID):
        _, observer, state = setup_tracing(
            make_launch(at=1, head_sha="a" * 40, max_minutes=10, session_id=SESSION_ID),
            branch=f"gymrat/{SESSION_ID}",
            prompt=make_prompt(),
            warn=messages.append,
        )
        assert observer is not None

        with unhandled_logging():
            observer(make_turn_end(at=8_000_000_000, text="done", cost_usd=0.05))
        finalize_tracing(state, None)

    assert ["mirror broke" in message for message in messages] == [True]


# ---------------------------------------------------------------------------
# setup_tracing without a tracer
# ---------------------------------------------------------------------------


def _no_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    """Leave no OTLP endpoint set, so tracing is never asked for."""
    monkeypatch.delenv(ENDPOINT_ENV, raising=False)


@pytest.mark.parametrize(
    ("turn_tracing_off", "resumed"),
    [
        pytest.param(_no_endpoint, False, id="no-endpoint"),
        pytest.param(disable_otel_sdk, False, id="sdk-disabled-opening-launch"),
        pytest.param(disable_otel_sdk, True, id="sdk-disabled-resumed-launch"),
    ],
)
def test_setup_tracing_when_no_tracer_does_return_no_observer_and_hold_no_span(
    monkeypatch: pytest.MonkeyPatch,
    turn_tracing_off: Callable[[pytest.MonkeyPatch], None],
    resumed: bool,
):
    turn_tracing_off(monkeypatch)
    prompt = make_prompt()

    traced = setup_tracing(
        make_launch(at=1, head_sha="a" * 40, max_minutes=10, session_id=SESSION_ID),
        branch=f"gymrat/{SESSION_ID}",
        prompt=prompt,
        resumed=resumed,
    )

    assert traced == (prompt, None, TracingState())
