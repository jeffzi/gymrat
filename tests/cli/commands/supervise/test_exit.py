"""Run-end tests for the ``gymrat supervise`` command: the exit sequence and exit codes.

Every return of ``supervise()`` hands the session to the exit sequence before the
reporter stops and the closing summary prints; the command's exit code then folds
the driver outcome, the exit sequence's error, and how the run ended. Shares its
seam-installation harness with :mod:`tests.cli.commands.supervise.test_supervise`, whose fake
exit sequence records each call and lets a test act from inside it.
"""

import asyncio
import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from gymrat.cli.supervise.types import ReadSessionResult
from gymrat.errors import GymratError
from gymrat.session.paths import budget_path
from gymrat.session.records import CommandRecord
from gymrat.supervisor.events import create_event_log_writer, event_from_wire
from gymrat.supervisor.exit_sequence import ExitPhase, ExitReport, ExitStep, run_exit_sequence
from gymrat.supervisor.supervise import EndedBy, SupervisionResult, supervise
from gymrat.telemetry import run_spans
from gymrat.telemetry.run_spans import TracingState
from tests.cli.commands.supervise._seams import (
    CAP_MINUTES,
    CAP_MS,
    Seams,
    err_text,
    install_seams,
    record_stdout_writes,
    run,
)
from tests.cli.supervise._fixtures import (
    follow_up_event,
    make_supervision_result,
    session_state_three_iterations,
)
from tests.session.records._fixtures import (
    records_of_type,
)
from tests.supervisor._mock_driver import CostStep, create_mock_driver

_EXIT_ERROR = "finalize failed: disk full"


# ---------------------------------------------------------------------------
# --no-finalize
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("flag_args", "expected_finalize"),
    [
        pytest.param((), True, id="default"),
        pytest.param(("--no-finalize",), False, id="no-finalize"),
    ],
)
def test_supervise_when_run_ends_does_pass_the_finalize_choice_to_the_exit_sequence(
    repo: str,
    monkeypatch: pytest.MonkeyPatch,
    flag_args: tuple[str, ...],
    expected_finalize: bool,
):
    seams = install_seams(monkeypatch)

    result = run("optimize it", "--max-minutes", "10", *flag_args)

    assert result.exit_code == 0
    assert [call["finalize"] for call in seams.exit_calls] == [expected_finalize]


# ---------------------------------------------------------------------------
# exit sequence wiring
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "supervision",
    [
        pytest.param(make_supervision_result(), id="session"),
        pytest.param(make_supervision_result(reason="error", message="lost"), id="error-outcome"),
        pytest.param(
            make_supervision_result(reason="interrupted", ended_by="wall-clock"),
            id="interrupted",
        ),
    ],
)
def test_supervise_when_supervise_returns_does_run_the_exit_sequence_over_the_session_and_its_end(
    repo: str, monkeypatch: pytest.MonkeyPatch, supervision: SupervisionResult
):
    seams = install_seams(monkeypatch, result=supervision)

    run("optimize it", "--max-minutes", "10")

    (call,) = seams.exit_calls
    assert call["context"] == seams.supervise_calls[0]["context"]
    assert call["ended_by"] == supervision.ended_by


def test_supervise_when_supervise_raises_does_fail_before_the_exit_sequence(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    seams = install_seams(monkeypatch, raises=GymratError("config broken"))

    result = run("optimize it", "--max-minutes", "10")

    assert result.exit_code == 2
    assert "config broken" in result.stderr
    assert seams.exit_calls == []
    seams.reporter_stop.assert_called()


def test_supervise_when_exit_sequence_reports_a_phase_does_show_it_on_the_reporter(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    seams = install_seams(monkeypatch)
    phase = ExitPhase(kind="settling", pid=None)
    seams.exit_hook = lambda call: call["progress"](phase)

    run("optimize it", "--max-minutes", "10")

    assert seams.exit_phases == [phase]


def test_supervise_when_exit_sequence_warns_does_route_it_through_the_reporter_not_stderr(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    seams = install_seams(monkeypatch)
    hint = "no checks command is configured"
    seams.exit_hook = lambda call: call["warn"](hint)

    result = run("optimize it", "--max-minutes", "10")

    assert seams.warnings == [hint]
    assert hint not in result.stderr


def test_supervise_when_exit_sequence_logs_an_event_does_append_it_to_the_run_log(
    repo: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    seams = install_seams(monkeypatch)
    log_path = tmp_path / "supervisor.jsonl"
    run_event = follow_up_event(action="replied")
    event = follow_up_event(action="ended", reason="nothing to settle")

    def log_after_the_run(call: dict[str, Any]) -> None:
        create_event_log_writer(log_path)(run_event)
        call["log"](event)

    seams.exit_hook = log_after_the_run

    run("optimize it", "--max-minutes", "10", "--log", str(log_path))

    logged = [event_from_wire(json.loads(line)) for line in log_path.read_text().splitlines()]
    assert logged[-2:] == [run_event, event]


def _untraced(_monkeypatch: pytest.MonkeyPatch, seams: Seams) -> list[object]:
    """Leave tracing off, so supervise is handed the reporter's own observer."""
    return seams.observed_events


def _traced_with_its_own_observer(monkeypatch: pytest.MonkeyPatch, _seams: Seams) -> list[object]:
    """Make tracing hand supervise an observer of its own, returning what that observer sees."""
    run_observed: list[object] = []

    def tracing_with_its_own_observer(
        _launch: object, *, prompt: object, **_kwargs: object
    ) -> tuple[object, ...]:
        return prompt, run_observed.append, TracingState()

    monkeypatch.setattr(run_spans, "setup_tracing", tracing_with_its_own_observer)
    return run_observed


@pytest.mark.parametrize(
    "observer_supervise_used",
    [
        pytest.param(_untraced, id="untraced"),
        pytest.param(_traced_with_its_own_observer, id="traced"),
    ],
)
def test_supervise_when_exit_sequence_logs_an_event_does_hand_it_to_the_observer_supervise_used(
    repo: str,
    monkeypatch: pytest.MonkeyPatch,
    observer_supervise_used: Callable[[pytest.MonkeyPatch, Seams], list[object]],
):
    seams = install_seams(monkeypatch)
    observed = observer_supervise_used(monkeypatch, seams)
    event = follow_up_event(action="ended", reason="nothing to settle")
    seams.exit_hook = lambda call: call["log"](event)

    run("optimize it", "--max-minutes", "10")

    assert observed == [event]


def test_supervise_when_run_ends_does_run_the_exit_sequence_in_the_supervisor_loop(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    seams = install_seams(monkeypatch)
    loops: list[asyncio.AbstractEventLoop] = []
    record_supervise_call = seams.record_supervise_call

    def recording_supervise(args: tuple[object, ...], kwargs: dict[str, object]) -> None:
        loops.append(asyncio.get_running_loop())
        record_supervise_call(args, kwargs)

    seams.record_supervise_call = recording_supervise
    seams.exit_hook = lambda _call: loops.append(asyncio.get_running_loop())

    run("optimize it", "--max-minutes", "10")

    supervise_loop, exit_loop = loops
    assert exit_loop is supervise_loop


def test_supervise_when_run_completes_does_log_one_supervise_command_record_per_stage(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    seams = install_seams(monkeypatch, real_preflight=True)
    seams.create_driver.return_value = create_mock_driver([CostStep(cost_usd=0.01)])
    monkeypatch.setattr("gymrat.cli.commands.supervise.supervise", supervise)
    monkeypatch.setattr("gymrat.cli.commands.supervise.run_exit_sequence", run_exit_sequence)

    result = run("optimize it", "--max-minutes", str(CAP_MINUTES))

    supervise_records = [
        (record.name, record.args)
        for record in records_of_type(repo, CommandRecord)
        if record.name == "supervise"
    ]
    assert result.exit_code == 0, err_text(result)
    assert supervise_records == [
        ("supervise", {"stage": "preflight"}),
        ("supervise", {"stage": "exit"}),
    ]


# ---------------------------------------------------------------------------
# ordering around the exit sequence
# ---------------------------------------------------------------------------


def test_supervise_when_run_ends_does_exit_sequence_then_stop_reporter_then_print_summary(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    # Without stop-before-print ordering the summary appends to the still-open
    # status row, corrupting the output.
    order: list[str] = []
    seams = install_seams(monkeypatch)
    seams.exit_hook = lambda _call: order.append("exit")
    seams.reporter_stop.side_effect = lambda: order.append("stop")
    record_stdout_writes(monkeypatch, order, "write")

    result = run("optimize it", "--max-minutes", "10")

    assert result.exit_code == 0
    assert list(dict.fromkeys(order)) == ["exit", "stop", "write"]


def test_supervise_when_exit_sequence_runs_does_find_the_budget_file_already_removed(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    seams = install_seams(monkeypatch)
    budget_present: list[bool] = []
    seams.exit_hook = lambda _call: budget_present.append(Path(budget_path(repo)).exists())

    run("optimize it", "--max-minutes", "10")

    assert budget_present == [False]


# ---------------------------------------------------------------------------
# closing summary after the exit sequence
# ---------------------------------------------------------------------------


def test_supervise_when_exit_sequence_reports_steps_does_print_them_as_exit_rows(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    seams = install_seams(monkeypatch)
    seams.exit_report = ExitReport(
        steps=(
            ExitStep(kind="settled", text="settled: kept iteration 1 (checks passed)"),
            ExitStep(kind="nothing", text="session left open (--no-finalize)"),
        )
    )

    result = run("optimize it", "--max-minutes", "10", "--no-finalize")

    assert result.exit_code == 0
    exit_rows = [line for line in result.stdout.splitlines() if line.startswith("  exit ")]
    assert exit_rows == [
        "  exit    settled: kept iteration 1 (checks passed)",
        "  exit    session left open (--no-finalize)",
    ]


@pytest.mark.parametrize(
    ("exit_report", "expected_exit"),
    [
        pytest.param(ExitReport(steps=()), 0, id="clean"),
        pytest.param(ExitReport(steps=(), error=_EXIT_ERROR), 2, id="error-without-event"),
    ],
)
def test_supervise_when_exit_sequence_changes_the_session_does_summarize_the_session_it_left(
    repo: str, monkeypatch: pytest.MonkeyPatch, exit_report: ExitReport, expected_exit: int
):
    seams = install_seams(monkeypatch)
    seams.exit_report = exit_report
    settled = ReadSessionResult(
        state=session_state_three_iterations(-4.2, "improved", seq=3), has_baseline=True
    )

    def settle(_call: dict[str, Any]) -> None:
        seams.session_result = settled

    seams.exit_hook = settle

    result = run("optimize it", "--max-minutes", "10")

    assert result.exit_code == expected_exit
    assert "  loop    3 iterations · 2 kept · 1 discarded · last -4.2% improved" in result.stdout


# ---------------------------------------------------------------------------
# exit codes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "ended_by",
    [pytest.param("session", id="session"), pytest.param("wall-clock", id="wall-clock")],
)
def test_supervise_when_exit_sequence_errors_does_exit_two_with_the_error_only_in_the_summary(
    repo: str, monkeypatch: pytest.MonkeyPatch, ended_by: EndedBy
):
    seams = install_seams(
        monkeypatch, result=make_supervision_result(reason="interrupted", ended_by=ended_by)
    )
    seams.exit_report = ExitReport(steps=(), error=_EXIT_ERROR)

    result = run("optimize it", "--max-minutes", "10")

    assert result.exit_code == 2
    assert f"  exit    error: {_EXIT_ERROR}" in result.stdout.splitlines()
    assert _EXIT_ERROR not in result.stderr


@pytest.mark.parametrize(
    "exit_report",
    [
        pytest.param(ExitReport(steps=()), id="exit-sequence-clean"),
        pytest.param(ExitReport(steps=(), error=_EXIT_ERROR), id="exit-sequence-errored"),
    ],
)
def test_supervise_when_outcome_error_with_message_does_surface_the_driver_message(
    repo: str, monkeypatch: pytest.MonkeyPatch, exit_report: ExitReport
):
    seams = install_seams(
        monkeypatch,
        result=make_supervision_result(
            reason="error", duration_ms=5_000, cost_usd=0.03, message="SDK connection lost"
        ),
    )
    seams.exit_report = exit_report

    result = run("optimize it", "--max-minutes", "10")

    assert result.exit_code == 2
    assert result.stdout.splitlines()[0] == "✗ error · 5s · $0.03"
    assert "SDK connection lost" in result.stderr
    assert _EXIT_ERROR not in result.stderr


@pytest.mark.parametrize(
    "message",
    [pytest.param(None, id="omitted"), pytest.param("", id="empty-string")],
)
def test_supervise_when_outcome_error_without_message_does_exit_two_quietly(
    repo: str, monkeypatch: pytest.MonkeyPatch, message: str | None
):
    install_seams(
        monkeypatch,
        result=make_supervision_result(
            reason="error", duration_ms=5_000, cost_usd=0.01, message=message
        ),
    )

    result = run("optimize it", "--max-minutes", "10")

    assert result.exit_code == 2
    assert not re.search(r"\berror\b", result.stderr, re.IGNORECASE)


@pytest.mark.parametrize(
    ("supervision", "max_minutes", "expected_exit", "headline"),
    [
        pytest.param(
            make_supervision_result(
                reason="interrupted", ended_by="wall-clock", duration_ms=CAP_MS, cost_usd=1.0
            ),
            str(CAP_MINUTES),
            1,
            "! interrupted by wall-clock cap · 10m 0s · $1.00",
            id="wall-clock-cap",
        ),
        pytest.param(
            make_supervision_result(
                reason="interrupted",
                ended_by="guard",
                duration_ms=30_000,
                cost_usd=0.10,
                end_reason="safety limit reached",
            ),
            "10",
            1,
            "! stopped by guard: safety limit reached · 30s · $0.10",
            id="guard",
        ),
        pytest.param(
            make_supervision_result(
                reason="interrupted",
                ended_by="stop-condition",
                end_reason="max iterations (2 of 2)",
            ),
            "10",
            0,
            "✓ stopped: max iterations (2 of 2) · 1m 0s · $0.05",
            id="stop-condition",
        ),
        pytest.param(
            make_supervision_result(reason="interrupted", ended_by="spend-cap", end_reason=""),
            "10",
            1,
            "! interrupted by spend cap · 1m 0s · $0.05",
            id="spend-cap",
        ),
        pytest.param(
            make_supervision_result(
                reason="interrupted",
                ended_by="hook-failure",
                end_reason="after hook failed on iteration 2: exit 1 (stdout 80 B, stderr 5 B)",
            ),
            "10",
            1,
            "! stopped: after hook failed on iteration 2: exit 1 (stdout 80 B, stderr 5 B)"
            " · 1m 0s · $0.05",
            id="hook-failure",
        ),
    ],
)
def test_supervise_when_run_ended_by_a_condition_does_exit_with_its_code_and_headline(
    *,
    repo: str,
    monkeypatch: pytest.MonkeyPatch,
    supervision: SupervisionResult,
    max_minutes: str,
    expected_exit: int,
    headline: str,
):
    install_seams(monkeypatch, result=supervision)

    result = run("optimize it", "--max-minutes", max_minutes)

    assert result.exit_code == expected_exit
    assert result.stdout.splitlines()[0] == headline
