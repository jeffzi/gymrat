"""Tests for the run-span observer that mirrors supervisor events onto OTel spans."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from opentelemetry.trace import StatusCode

from gymrat.supervisor.driver import SessionOutcome
from gymrat.supervisor.events import (
    CapEvent,
    CompactionEvent,
    FollowUpEvent,
    SessionEvent,
    UsageUpdateEvent,
)
from gymrat.supervisor.supervise import SupervisionResult
from gymrat.telemetry.provider import SESSION_SPAN_KEY, span_id_of, start_span
from gymrat.telemetry.run_spans import (
    GEN_AI_PROVIDER,
    RUN_COST_USD,
    RUN_DURATION_MS,
    RUN_END_REASON,
    RUN_ENDED_BY,
    RUN_SPAN,
    SESSION_SPAN,
    TracingState,
    create_run_span_observer,
    finalize_tracing,
    run_attributes,
    run_span_key,
    setup_tracing,
)
from tests.session.records._fixtures import SESSION_ID
from tests.supervisor._fixtures import make_launch, make_prompt, make_turn_end
from tests.telemetry._fixtures import (
    disable_otel_sdk,
    memory_tracing,
    session_traceparent,
    span_by_name,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from gymrat.supervisor.events import LaunchEvent

SESSION = "test-tracing-observer"

_OUTCOME_KEYS = (RUN_COST_USD, RUN_ENDED_BY, RUN_END_REASON, RUN_DURATION_MS)


# ---------------------------------------------------------------------------
# run_attributes
# ---------------------------------------------------------------------------


_LAUNCH_RUN_ATTRIBUTES = {
    "gymrat.session.id": SESSION_ID,
    "gymrat.run.head_sha": "abc123def",
    "gymrat.run.max_minutes": 60.0,
}


@pytest.mark.parametrize(
    ("launch", "expected"),
    [
        pytest.param(
            make_launch(session_id=SESSION_ID, max_minutes=60.0),
            _LAUNCH_RUN_ATTRIBUTES,
            id="options-unset-are-left-off",
        ),
        pytest.param(
            make_launch(
                session_id=SESSION_ID,
                max_minutes=60.0,
                max_usd=5.0,
                effort="high",
                model="claude-sonnet-4-20250514",
            ),
            {
                **_LAUNCH_RUN_ATTRIBUTES,
                "gymrat.run.max_usd": 5.0,
                "gymrat.run.effort": "high",
                "gen_ai.request.model": "claude-sonnet-4-20250514",
            },
            id="options-set-are-recorded",
        ),
    ],
)
def test_run_attributes_when_called_does_map_the_launch_leaving_unset_options_off(
    launch: LaunchEvent, expected: dict[str, object]
):
    result = run_attributes(launch)

    assert {key: value for key, value in result.items() if key != GEN_AI_PROVIDER} == expected


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
        pytest.param(
            UsageUpdateEvent(at=7_000_000_000, cost_usd=0.01), [], id="usage-update-ignored"
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
# setup_tracing without a tracer
# ---------------------------------------------------------------------------


def _leave_environment_baseline(_monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the conftest baseline, which never sets an OTLP endpoint."""


@pytest.mark.parametrize(
    ("turn_tracing_off", "resumed"),
    [
        pytest.param(_leave_environment_baseline, False, id="no-endpoint"),
        pytest.param(disable_otel_sdk, False, id="sdk-disabled-opening-launch"),
        pytest.param(disable_otel_sdk, True, id="sdk-disabled-resumed-launch"),
    ],
)
def test_setup_tracing_when_no_tracer_does_leave_tracing_inactive(
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


# ---------------------------------------------------------------------------
# setup_tracing with a tracer
# ---------------------------------------------------------------------------


def test_setup_tracing_when_tracing_active_does_hand_the_prompt_the_run_span_traceparent():
    launch = make_launch(session_id=SESSION)

    with memory_tracing(SESSION):
        traced_prompt, _, state = setup_tracing(
            launch, branch="gymrat/x", prompt=make_prompt(), resumed=False
        )
        finalize_tracing(state, None)

    assert traced_prompt.traceparent == session_traceparent(SESSION, run_span_key(launch.at))


def test_setup_tracing_when_resumed_does_parent_the_run_span_on_the_existing_session_span():
    launch = make_launch(session_id=SESSION)

    with memory_tracing(SESSION) as exporter:
        _, _, state = setup_tracing(launch, branch="gymrat/x", prompt=make_prompt(), resumed=True)
        finalize_tracing(state, None)

    finished = exporter.get_finished_spans()
    assert [span.name for span in finished] == [RUN_SPAN]
    assert finished[0].parent is not None
    assert finished[0].parent.span_id == span_id_of(SESSION, SESSION_SPAN_KEY)


# ---------------------------------------------------------------------------
# finalize_tracing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("result", "outcome_attributes", "status"),
    [
        pytest.param(
            SupervisionResult(
                outcome=SessionOutcome(reason="completed", cost_usd=0.5),
                ended_by="session",
                duration_ms=1200,
            ),
            {RUN_COST_USD: 0.5, RUN_ENDED_BY: "session", RUN_DURATION_MS: 1200},
            StatusCode.UNSET,
            id="completed",
        ),
        pytest.param(
            SupervisionResult(
                outcome=SessionOutcome(reason="error", cost_usd=0.25, message="boom"),
                ended_by="wall-clock",
                duration_ms=900,
                end_reason="wall-clock",
            ),
            {
                RUN_COST_USD: 0.25,
                RUN_ENDED_BY: "wall-clock",
                RUN_END_REASON: "wall-clock",
                RUN_DURATION_MS: 900,
            },
            StatusCode.ERROR,
            id="error-with-end-reason",
        ),
        pytest.param(None, {}, StatusCode.UNSET, id="no-result"),
    ],
)
def test_finalize_tracing_when_run_ends_does_record_its_outcome_on_the_run_span_only(
    result: SupervisionResult | None,
    outcome_attributes: dict[str, object],
    status: StatusCode,
):
    launch = make_launch(session_id=SESSION)

    with memory_tracing(SESSION) as exporter:
        _, _, state = setup_tracing(launch, branch="gymrat/x", prompt=make_prompt(), resumed=False)
        finalize_tracing(state, result)

    finished = exporter.get_finished_spans()
    run, session = span_by_name(finished, RUN_SPAN), span_by_name(finished, SESSION_SPAN)
    run_attrs = dict(run.attributes or {})
    assert {key: run_attrs[key] for key in _OUTCOME_KEYS if key in run_attrs} == outcome_attributes
    assert (run.status.status_code, session.status.status_code) == (status, StatusCode.UNSET)
    assert not set(_OUTCOME_KEYS) & set(session.attributes or {})


def test_finalize_tracing_when_spans_are_buffered_does_flush_them_before_returning():
    launch = make_launch(session_id=SESSION)

    with memory_tracing(SESSION, buffered=True) as exporter:
        _, _, state = setup_tracing(launch, branch="gymrat/x", prompt=make_prompt(), resumed=False)
        finalize_tracing(state, None)

        flushed = sorted(span.name for span in exporter.get_finished_spans())

    assert flushed == sorted([RUN_SPAN, SESSION_SPAN])
