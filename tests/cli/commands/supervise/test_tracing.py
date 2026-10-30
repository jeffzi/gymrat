"""Tracing integration tests for the ``gymrat supervise`` command.

These tests verify session and run span export, observer combination, and that
every run component's warning reaches the reporter rather than stderr.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import create_autospec

if TYPE_CHECKING:
    from collections.abc import Callable

    from tests.cli.commands.supervise._seams import Seams

import pytest
from opentelemetry.sdk.trace import Span as SdkSpan

from gymrat.errors import GymratError
from gymrat.session.paths import budget_path
from gymrat.supervisor.driver import SessionPrompt
from gymrat.supervisor.events import create_event_log_writer
from tests._cli import combined_output
from tests.cli.commands.supervise._seams import (
    install_seams,
    run,
)
from tests.cli.supervise._fixtures import (
    follow_up_event,
    make_supervision_result,
)
from tests.session.records._fixtures import SESSION_ID
from tests.supervisor._fixtures import raising_observer
from tests.telemetry._fixtures import memory_tracing

# ---------------------------------------------------------------------------
# span export — one session span parenting one run span, on every outcome
# ---------------------------------------------------------------------------

_OUTCOMES = [
    pytest.param({}, id="completed"),
    pytest.param(
        {
            "result": make_supervision_result(
                reason="error", duration_ms=5_000, cost_usd=0.03, end_reason="SDK failed"
            )
        },
        id="error-outcome",
    ),
    pytest.param({"raises": GymratError("boom")}, id="supervise-raises"),
]


@pytest.mark.parametrize("seam_kwargs", _OUTCOMES)
def test_supervise_when_tracing_enabled_does_export_the_session_and_run_spans_once(
    *,
    repo: str,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    seam_kwargs: dict[str, Any],
):
    install_seams(monkeypatch, **seam_kwargs)
    caplog.set_level("WARNING", logger="opentelemetry.sdk.trace")

    with memory_tracing(SESSION_ID) as exporter:
        run()

    assert [s.name for s in exporter.get_finished_spans()] == ["gymrat.run", "gymrat.session"]
    assert [r for r in caplog.records if r.name == "opentelemetry.sdk.trace"] == []


def test_supervise_when_tracing_enabled_does_hand_supervise_the_run_span_context(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    seams = install_seams(monkeypatch)

    with memory_tracing(SESSION_ID):
        run()

    prompt = seams.supervise_calls[0]["prompt"]
    assert isinstance(prompt, SessionPrompt)
    assert prompt.traceparent is not None


# ---------------------------------------------------------------------------
# every warning a run component raises reaches the reporter, never raw stderr
# ---------------------------------------------------------------------------

_COMPONENT_WARNING = "component failure"


def _supervisor_warns(_monkeypatch: pytest.MonkeyPatch, seams: Seams, _repo: str) -> str:
    seams.supervise_hook = lambda call: call["warn"](_COMPONENT_WARNING)
    return _COMPONENT_WARNING


def _reporter_observer_raises(_monkeypatch: pytest.MonkeyPatch, seams: Seams, _repo: str) -> str:
    seams.observer = raising_observer(_COMPONENT_WARNING)
    seams.supervise_hook = lambda call: call["observer"](follow_up_event(action="replied"))
    return _COMPONENT_WARNING


def _event_log_writer_raises(monkeypatch: pytest.MonkeyPatch, seams: Seams, _repo: str) -> str:
    monkeypatch.setattr(
        "gymrat.cli.commands.supervise.create_event_log_writer",
        create_autospec(create_event_log_writer, return_value=raising_observer(_COMPONENT_WARNING)),
    )
    seams.exit_hook = lambda call: call["log"](follow_up_event(action="replied"))
    return _COMPONENT_WARNING


def _run_span_mirror_fails(monkeypatch: pytest.MonkeyPatch, seams: Seams, _repo: str) -> str:
    monkeypatch.setattr(
        SdkSpan,
        "add_event",
        create_autospec(SdkSpan.add_event, side_effect=RuntimeError(_COMPONENT_WARNING)),
    )
    seams.supervise_hook = lambda call: call["observer"](follow_up_event(action="replied"))
    return _COMPONENT_WARNING


def _budget_clear_refused(_monkeypatch: pytest.MonkeyPatch, seams: Seams, repo: str) -> str:
    def swap_budget_file_for_a_directory(_call: dict[str, Any]) -> None:
        # The OS refuses to unlink a directory, so the run-end budget clear warns.
        budget_file = Path(budget_path(repo))
        budget_file.unlink()
        budget_file.mkdir()

    seams.supervise_hook = swap_budget_file_for_a_directory
    return budget_path(repo)


@pytest.mark.parametrize(
    "make_component_warn",
    [
        pytest.param(_supervisor_warns, id="supervisor"),
        pytest.param(_reporter_observer_raises, id="reporter-observer"),
        pytest.param(_event_log_writer_raises, id="event-log-writer"),
        pytest.param(_run_span_mirror_fails, id="run-span-mirror"),
        pytest.param(_budget_clear_refused, id="budget-clear"),
    ],
)
def test_supervise_when_a_run_component_warns_does_route_it_through_the_reporter_not_stderr(
    repo: str,
    monkeypatch: pytest.MonkeyPatch,
    make_component_warn: Callable[[pytest.MonkeyPatch, Seams, str], str],
):
    seams = install_seams(monkeypatch)
    fragment = make_component_warn(monkeypatch, seams, repo)

    with memory_tracing(SESSION_ID):
        result = run()

    assert result.exit_code == 0, combined_output(result)
    assert [fragment in warning for warning in seams.warnings] == [True]
    assert fragment not in combined_output(result)
