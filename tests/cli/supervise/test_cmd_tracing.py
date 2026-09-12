"""Tracing integration tests for the ``gymrat supervise`` command.

These tests verify session and run span export, attribute setting, observer
combination, and the disabled-tracing path. They share the seam infrastructure
from ``test_cmd`` and add ``memory_tracing`` from the telemetry test fixtures.
"""

from __future__ import annotations

import sys
import warnings
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterator

import pytest

from gymrat.cli.shared import write_and_flush
from gymrat.config import SuperviseConfig
from gymrat.errors import GymratError
from gymrat.supervisor import SessionPrompt, SupervisionResult
from tests.cli.supervise._fixtures import make_supervision_result
from tests.cli.supervise.test_cmd import (
    _CAP_MINUTES,
    _config,
    _install_seams,
    _run,
    _Seams,
)
from tests.session.records._fixtures import SESSION_ID


@pytest.fixture(autouse=True)
def _isolate_tracing_provider() -> Iterator[None]:
    """Reset the telemetry provider singleton between tests."""
    yield
    from gymrat.telemetry.provider import _reset_for_tests

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _reset_for_tests()


def _tracing_seams(
    monkeypatch: pytest.MonkeyPatch,
    *,
    result: SupervisionResult | None = None,
    raises: Exception | None = None,
    max_usd: str | None = None,
    effort: str | None = None,
    model: str | None = None,
) -> _Seams:
    """Install seams and configure tracing-relevant flags for the helper.

    Returns the seam recorder; the caller wraps the ``_run`` call in a
    ``memory_tracing`` context manager.
    """
    cfg_kwargs: dict[str, Any] = {}
    if model is not None:
        cfg_kwargs["supervise"] = SuperviseConfig(model=model)
    return _install_seams(
        monkeypatch,
        result=result,
        raises=raises,
        config=_config(**cfg_kwargs),
    )


def test_supervise_when_tracing_enabled_does_export_session_and_run_spans(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    from gymrat.telemetry.ids import span_id_of
    from tests.telemetry._fixtures import memory_tracing

    seams = _tracing_seams(monkeypatch)

    with memory_tracing(SESSION_ID) as exporter:
        result = _run("optimize it", "--max-minutes", str(_CAP_MINUTES))

    assert result.exit_code == 0
    spans = exporter.get_finished_spans()
    session_spans = [s for s in spans if s.name == "gymrat.session"]
    run_spans = [s for s in spans if s.name == "gymrat.run"]
    assert len(session_spans) == 1
    assert len(run_spans) == 1

    session_span = session_spans[0]
    assert session_span.context.span_id == span_id_of(SESSION_ID, "session")  # pyrefly: ignore[missing-attribute]
    assert session_span.attributes["gymrat.session.id"] == SESSION_ID  # pyrefly: ignore[unsupported-operation]
    assert session_span.attributes["gymrat.session.branch"] == f"gymrat/{SESSION_ID}"  # pyrefly: ignore[unsupported-operation]

    launch = seams.supervise_calls[0]["launch"]
    run_span = run_spans[0]
    run_key = f"run:{launch.at}"  # pyrefly: ignore[missing-attribute]
    assert run_span.context.span_id == span_id_of(SESSION_ID, run_key)  # pyrefly: ignore[missing-attribute]
    assert run_span.attributes["gymrat.session.id"] == SESSION_ID  # pyrefly: ignore[unsupported-operation]
    assert run_span.attributes["gymrat.run.head_sha"] == launch.head_sha  # pyrefly: ignore[missing-attribute,unsupported-operation]
    assert run_span.attributes["gymrat.run.max_minutes"] == _CAP_MINUTES  # pyrefly: ignore[unsupported-operation]
    assert run_span.attributes["gen_ai.provider.name"] == "anthropic"  # pyrefly: ignore[unsupported-operation]
    assert run_span.parent.span_id == session_span.context.span_id  # pyrefly: ignore[missing-attribute]


def test_supervise_when_tracing_enabled_does_set_run_end_attributes(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    from tests.telemetry._fixtures import memory_tracing

    sup_result = make_supervision_result(
        reason="completed", ended_by="session", duration_ms=60_000, cost_usd=0.05
    )
    _tracing_seams(monkeypatch, result=sup_result)

    with memory_tracing(SESSION_ID) as exporter:
        result = _run("optimize it", "--max-minutes", str(_CAP_MINUTES))

    assert result.exit_code == 0
    spans = exporter.get_finished_spans()
    run_span = next(s for s in spans if s.name == "gymrat.run")
    assert run_span.attributes["gymrat.run.cost_usd"] == sup_result.cost_usd  # pyrefly: ignore[unsupported-operation]
    assert run_span.attributes["gymrat.run.ended_by"] == sup_result.ended_by  # pyrefly: ignore[unsupported-operation]
    assert "gymrat.run.end_reason" not in run_span.attributes  # pyrefly: ignore[not-iterable]
    assert run_span.attributes["gymrat.run.duration_ms"] == sup_result.duration_ms  # pyrefly: ignore[unsupported-operation]


def test_supervise_when_tracing_enabled_and_outcome_error_does_set_error_status_on_run_span(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    from opentelemetry.trace import StatusCode

    from tests.telemetry._fixtures import memory_tracing

    sup_result = make_supervision_result(
        reason="error", duration_ms=5_000, cost_usd=0.03, end_reason="SDK failed"
    )
    _tracing_seams(monkeypatch, result=sup_result)

    with memory_tracing(SESSION_ID) as exporter:
        result = _run("optimize it", "--max-minutes", str(_CAP_MINUTES))

    assert result.exit_code == 2
    spans = exporter.get_finished_spans()
    run_span = next(s for s in spans if s.name == "gymrat.run")
    assert run_span.status.status_code == StatusCode.ERROR
    assert run_span.attributes["gymrat.run.end_reason"] == "SDK failed"  # pyrefly: ignore[unsupported-operation]


def test_supervise_when_tracing_enabled_and_outcome_not_error_does_leave_status_unset(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    from opentelemetry.trace import StatusCode

    from tests.telemetry._fixtures import memory_tracing

    _tracing_seams(monkeypatch)

    with memory_tracing(SESSION_ID) as exporter:
        result = _run("optimize it", "--max-minutes", str(_CAP_MINUTES))

    assert result.exit_code == 0
    spans = exporter.get_finished_spans()
    run_span = next(s for s in spans if s.name == "gymrat.run")
    assert run_span.status.status_code == StatusCode.UNSET


def test_supervise_when_tracing_enabled_does_end_spans_after_report_and_flush(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    from tests.telemetry._fixtures import memory_tracing

    order: list[str] = []
    _tracing_seams(monkeypatch)

    original_waf = write_and_flush

    def tracking_waf(stream: Any, data: str) -> None:
        if stream is sys.stdout:
            order.append("summary")
        original_waf(stream, data)

    monkeypatch.setattr("gymrat.cli.supervise.cmd.write_and_flush", tracking_waf)

    original_flush = None

    def tracking_flush() -> None:
        order.append("flush")
        if original_flush is not None:
            original_flush()

    with memory_tracing(SESSION_ID) as exporter:
        from gymrat.telemetry import provider as _prov

        original_flush = _prov.flush_tracing
        monkeypatch.setattr(_prov, "flush_tracing", tracking_flush)

        result = _run("optimize it", "--max-minutes", str(_CAP_MINUTES))

    assert result.exit_code == 0
    spans = exporter.get_finished_spans()
    assert any(s.name == "gymrat.session" for s in spans)
    assert any(s.name == "gymrat.run" for s in spans)
    assert "summary" in order
    assert "flush" in order
    assert order.index("summary") < order.index("flush")


def test_supervise_when_tracing_enabled_and_supervise_raises_does_still_flush(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    from tests.telemetry._fixtures import memory_tracing

    _tracing_seams(monkeypatch, raises=GymratError("boom"))

    with memory_tracing(SESSION_ID) as exporter:
        result = _run("optimize it", "--max-minutes", str(_CAP_MINUTES))

    assert result.exit_code == 2
    spans = exporter.get_finished_spans()
    assert any(s.name == "gymrat.session" for s in spans)
    assert any(s.name == "gymrat.run" for s in spans)


def test_supervise_when_tracing_enabled_does_set_traceparent_on_prompt(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    from gymrat.telemetry.ids import span_id_of
    from tests.telemetry._fixtures import memory_tracing

    seams = _tracing_seams(monkeypatch)

    with memory_tracing(SESSION_ID):
        result = _run("optimize it", "--max-minutes", str(_CAP_MINUTES))

    assert result.exit_code == 0
    prompt = seams.supervise_calls[0]["prompt"]
    assert isinstance(prompt, SessionPrompt)
    assert prompt.traceparent is not None
    assert prompt.traceparent.startswith("00-")

    launch = seams.supervise_calls[0]["launch"]
    run_key = f"run:{launch.at}"  # pyrefly: ignore[missing-attribute]
    expected_span_id = span_id_of(SESSION_ID, run_key)
    span_hex = f"{expected_span_id:016x}"
    assert span_hex in prompt.traceparent


def test_supervise_when_tracing_enabled_does_combine_observer_with_run_span_observer(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    from tests.telemetry._fixtures import memory_tracing

    seams = _tracing_seams(monkeypatch)

    with memory_tracing(SESSION_ID):
        result = _run("optimize it", "--max-minutes", str(_CAP_MINUTES))

    assert result.exit_code == 0
    observer = seams.supervise_calls[0]["observer"]
    assert observer is not seams.observer


def test_supervise_when_tracing_enabled_and_max_usd_given_does_set_run_attribute(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    from tests.telemetry._fixtures import memory_tracing

    _tracing_seams(monkeypatch)

    with memory_tracing(SESSION_ID) as exporter:
        result = _run("optimize it", "--max-minutes", str(_CAP_MINUTES), "--max-usd", "5.0")

    assert result.exit_code == 0
    spans = exporter.get_finished_spans()
    run_span = next(s for s in spans if s.name == "gymrat.run")
    assert run_span.attributes["gymrat.run.max_usd"] == 5.0  # pyrefly: ignore[unsupported-operation]


def test_supervise_when_tracing_enabled_and_no_max_usd_does_omit_attribute(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    from tests.telemetry._fixtures import memory_tracing

    _tracing_seams(monkeypatch)

    with memory_tracing(SESSION_ID) as exporter:
        result = _run("optimize it", "--max-minutes", str(_CAP_MINUTES))

    assert result.exit_code == 0
    spans = exporter.get_finished_spans()
    run_span = next(s for s in spans if s.name == "gymrat.run")
    assert "gymrat.run.max_usd" not in run_span.attributes  # pyrefly: ignore[not-iterable]


def test_supervise_when_tracing_enabled_and_model_set_does_set_provider_attributes(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    from tests.telemetry._fixtures import memory_tracing

    _tracing_seams(monkeypatch, model="opus")

    with memory_tracing(SESSION_ID) as exporter:
        result = _run("optimize it", "--max-minutes", str(_CAP_MINUTES))

    assert result.exit_code == 0
    spans = exporter.get_finished_spans()
    run_span = next(s for s in spans if s.name == "gymrat.run")
    assert run_span.attributes["gen_ai.request.model"] == "opus"  # pyrefly: ignore[unsupported-operation]
    assert run_span.attributes["gen_ai.provider.name"] == "anthropic"  # pyrefly: ignore[unsupported-operation]


def test_supervise_when_tracing_disabled_does_pass_reporter_observer_directly(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    seams = _tracing_seams(monkeypatch)

    result = _run("optimize it", "--max-minutes", str(_CAP_MINUTES))

    assert result.exit_code == 0
    assert seams.supervise_calls[0]["observer"] is seams.observer


def test_supervise_when_tracing_disabled_does_not_set_traceparent_on_prompt(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    seams = _tracing_seams(monkeypatch)

    result = _run("optimize it", "--max-minutes", str(_CAP_MINUTES))

    assert result.exit_code == 0
    prompt = seams.supervise_calls[0]["prompt"]
    assert isinstance(prompt, SessionPrompt)
    assert prompt.traceparent is None
