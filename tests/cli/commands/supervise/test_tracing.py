"""Tracing integration tests for the ``gymrat supervise`` command.

These tests verify session and run span export, attribute setting, observer
combination, and the untraced path. They share the seam infrastructure from
``_seams`` and add ``memory_tracing`` from the telemetry test fixtures.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

    from opentelemetry.trace import SpanContext

    from gymrat.supervisor.supervise import SupervisionResult

import pytest
from opentelemetry.trace import StatusCode

from gymrat.config import SuperviseConfig
from gymrat.errors import GymratError
from gymrat.supervisor.driver import SessionPrompt
from gymrat.supervisor.events import LaunchEvent
from gymrat.telemetry import provider
from gymrat.telemetry.provider import export_failed, span_id_of
from tests.cli._session import close_session_with_one_keep
from tests.cli.commands.supervise._seams import (
    CAP_MINUTES,
    command_config,
    err_text,
    install_seams,
    patch_supervise,
    record_stdout_writes,
    run,
)
from tests.cli.supervise._fixtures import (
    follow_up_event,
    make_supervision_result,
)
from tests.session.records._fixtures import SESSION_ID, session_header_of
from tests.telemetry._collector import otlp_collector
from tests.telemetry._fixtures import hide_otlp_exporter, memory_tracing, span_by_name


def _span_id(context: SpanContext | None) -> int | None:
    """The span id ``context`` carries, or ``None`` when the span has no such context."""
    return None if context is None else context.span_id


# ---------------------------------------------------------------------------
# span export — one session span parenting one run span, on every outcome
# ---------------------------------------------------------------------------

_OUTCOMES = [
    pytest.param(
        {},
        0,
        StatusCode.UNSET,
        {
            "gymrat.run.cost_usd": 0.05,
            "gymrat.run.ended_by": "session",
            "gymrat.run.duration_ms": 60_000,
        },
        id="completed",
    ),
    pytest.param(
        {
            "result": make_supervision_result(
                reason="error", duration_ms=5_000, cost_usd=0.03, end_reason="SDK failed"
            )
        },
        2,
        StatusCode.ERROR,
        {
            "gymrat.run.cost_usd": 0.03,
            "gymrat.run.ended_by": "session",
            "gymrat.run.duration_ms": 5_000,
            "gymrat.run.end_reason": "SDK failed",
        },
        id="error-outcome",
    ),
    pytest.param({"raises": GymratError("boom")}, 2, StatusCode.UNSET, {}, id="supervise-raises"),
]


@pytest.mark.parametrize(
    ("seam_kwargs", "exit_code", "run_status", "outcome_attributes"), _OUTCOMES
)
def test_supervise_when_tracing_enabled_does_export_one_quiet_session_span_parenting_one_run_span(
    *,
    repo: str,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    seam_kwargs: dict[str, Any],
    exit_code: int,
    run_status: StatusCode,
    outcome_attributes: dict[str, object],
):
    seams = install_seams(monkeypatch, **seam_kwargs)
    caplog.set_level("WARNING", logger="opentelemetry.sdk.trace")

    with memory_tracing(SESSION_ID) as exporter:
        result = run("optimize it", "--max-minutes", str(CAP_MINUTES))

    assert result.exit_code == exit_code
    spans = exporter.get_finished_spans()
    assert [s.name for s in spans] == ["gymrat.run", "gymrat.session"]
    run_span = span_by_name(spans, "gymrat.run")
    session_span = span_by_name(spans, "gymrat.session")
    launch = seams.supervise_calls[0]["launch"]
    assert isinstance(launch, LaunchEvent)
    assert session_span.parent is None
    assert session_span.context.span_id == span_id_of(SESSION_ID, "session")
    assert dict(session_span.attributes or {}) == {
        "gymrat.session.id": SESSION_ID,
        "gymrat.session.branch": f"gymrat/{SESSION_ID}",
    }
    assert (session_span.status.status_code, session_span.events) == (StatusCode.UNSET, ())
    assert run_span.parent.span_id == session_span.context.span_id
    assert run_span.context.span_id == span_id_of(SESSION_ID, f"run:{launch.at}")
    assert (run_span.status.status_code, run_span.events) == (run_status, ())
    assert dict(run_span.attributes or {}) == {
        "gymrat.session.id": SESSION_ID,
        "gymrat.run.head_sha": launch.head_sha,
        "gymrat.run.max_minutes": CAP_MINUTES,
        "gen_ai.provider.name": "anthropic",
        **outcome_attributes,
    }
    assert "Calling end() on an ended span." not in caplog.messages


def test_supervise_when_tracing_enabled_does_hand_supervise_the_run_span_context(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    seams = install_seams(monkeypatch)

    with memory_tracing(SESSION_ID):
        run("optimize it", "--max-minutes", str(CAP_MINUTES))

    call = seams.supervise_calls[0]
    launch = call["launch"]
    assert isinstance(launch, LaunchEvent)
    run_span_id = span_id_of(SESSION_ID, f"run:{launch.at}")
    prompt = call["prompt"]
    assert isinstance(prompt, SessionPrompt)
    assert prompt.traceparent is not None
    assert prompt.traceparent.startswith("00-")
    assert f"{run_span_id:016x}" in prompt.traceparent
    assert call["observer"] is not seams.observer


def test_supervise_when_tracing_enabled_does_still_hand_every_event_to_the_reporter(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    seams = install_seams(monkeypatch)
    event = follow_up_event(action="replied")

    async def emitting_supervise(*args: object, **kwargs: object) -> SupervisionResult:
        seams.record_supervise_call(args, kwargs)
        observer = kwargs["observer"]
        assert callable(observer)
        observer(event)
        return make_supervision_result()

    patch_supervise(monkeypatch, emitting_supervise)

    with memory_tracing(SESSION_ID):
        result = run("optimize it", "--max-minutes", str(CAP_MINUTES))

    assert (result.exit_code, seams.observed_events) == (0, [event])


def test_supervise_when_session_resumed_does_parent_both_run_spans_to_the_one_session_span(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    with memory_tracing(SESSION_ID) as exporter:
        install_seams(monkeypatch)
        opening = run("optimize it", "--max-minutes", str(CAP_MINUTES))
        install_seams(monkeypatch, resumed=True)
        resumed = run("optimize it", "--max-minutes", str(CAP_MINUTES), "--allow-dirty")

    spans = exporter.get_finished_spans()
    session_span_id = span_id_of(SESSION_ID, "session")
    assert (opening.exit_code, resumed.exit_code) == (0, 0), err_text(resumed)
    assert [_span_id(s.context) for s in spans if s.name == "gymrat.session"] == [session_span_id]
    assert [_span_id(s.parent) for s in spans if s.name == "gymrat.run"] == [
        session_span_id,
        session_span_id,
    ]


def test_supervise_when_tracing_enabled_does_flush_after_printing_the_summary(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    order: list[str] = []
    install_seams(monkeypatch)
    record_stdout_writes(monkeypatch, order, "summary")

    original_flush = None

    def tracking_flush() -> None:
        order.append("flush")
        if original_flush is not None:
            original_flush()

    with memory_tracing(SESSION_ID):
        original_flush = provider.flush_tracing
        monkeypatch.setattr(provider, "flush_tracing", tracking_flush)

        result = run("optimize it", "--max-minutes", str(CAP_MINUTES))

    assert result.exit_code == 0
    assert order.index("summary") < order.index("flush")


@pytest.mark.parametrize(
    ("extra_args", "supervise_config", "expected_attrs"),
    [
        pytest.param(("--max-usd", "5.0"), None, {"gymrat.run.max_usd": 5.0}, id="max-usd"),
        pytest.param(
            (),
            SuperviseConfig(model="opus", effort="high"),
            {"gen_ai.request.model": "opus", "gymrat.run.effort": "high"},
            id="model-and-effort",
        ),
    ],
)
def test_supervise_when_tracing_enabled_and_run_options_set_does_record_them_on_the_run_span(
    repo: str,
    monkeypatch: pytest.MonkeyPatch,
    extra_args: tuple[str, ...],
    supervise_config: SuperviseConfig | None,
    expected_attrs: dict[str, object],
):
    install_seams(monkeypatch, config=command_config(supervise=supervise_config))

    with memory_tracing(SESSION_ID) as exporter:
        result = run("optimize it", "--max-minutes", str(CAP_MINUTES), *extra_args)

    assert result.exit_code == 0
    run_span = span_by_name(exporter.get_finished_spans(), "gymrat.run")
    attributes = dict(run_span.attributes or {})
    assert {key: attributes.get(key) for key in expected_attrs} == expected_attrs


def test_supervise_when_collector_rejects_the_export_does_exit_zero(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    install_seams(monkeypatch)

    with otlp_collector(statuses=[400] * 8) as collector:
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", collector.endpoint)
        result = run("optimize it", "--max-minutes", str(CAP_MINUTES))

    assert (result.exit_code, export_failed(), "gymrat.run" in collector.span_names) == (
        0,
        True,
        True,
    )


def test_supervise_when_tracing_enabled_and_previous_session_finalized_does_trace_the_run_under_the_new_session(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    install_seams(monkeypatch, real_preflight=True)
    previous_id = close_session_with_one_keep(repo)

    with otlp_collector() as collector:
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", collector.endpoint)
        result = run("optimize it", "--max-minutes", str(CAP_MINUTES))

    new_id = session_header_of(repo).session_id
    traced = sorted(
        (span.name, span.attributes.get("gymrat.session.id"))
        for span in collector.spans
        if span.name in {"gymrat.run", "gymrat.command.supervise"}
    )
    assert result.exit_code == 0, err_text(result)
    assert new_id != previous_id
    assert traced == [("gymrat.command.supervise", new_id), ("gymrat.run", new_id)]


# ---------------------------------------------------------------------------
# an endpoint set but no tracer to be had — the run continues untraced
# ---------------------------------------------------------------------------


def _disable_sdk_with_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OTEL_SDK_DISABLED", "true")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318")


def _hide_exporter_with_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    hide_otlp_exporter(monkeypatch)
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318")


@pytest.mark.parametrize(
    "without_tracer",
    [
        pytest.param(_disable_sdk_with_endpoint, id="sdk-disabled"),
        pytest.param(_hide_exporter_with_endpoint, id="otlp-exporter-missing"),
    ],
)
def test_supervise_when_endpoint_set_but_no_tracer_available_does_run_untraced(
    repo: str,
    monkeypatch: pytest.MonkeyPatch,
    without_tracer: Callable[[pytest.MonkeyPatch], None],
):
    seams = install_seams(monkeypatch)
    without_tracer(monkeypatch)

    result = run("optimize it", "--max-minutes", str(CAP_MINUTES))

    assert result.exit_code == 0, err_text(result)
    (call,) = seams.supervise_calls
    prompt = call["prompt"]
    assert isinstance(prompt, SessionPrompt)
    assert (prompt.traceparent, call["observer"]) == (None, seams.observer)
