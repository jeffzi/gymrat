"""Run-end tests for the ``gymrat supervise`` command: the exit sequence and exit codes.

Every return of ``supervise()`` hands the session to the exit sequence before the
reporter stops and the closing summary prints; the command's exit code then folds
the driver outcome, the exit sequence's error, and how the run ended. The
seam-installation harness lives in :mod:`tests.cli.commands.supervise._seams`, whose fake
exit sequence records each call and lets a test act from inside it.
"""

import asyncio
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import create_autospec

import pytest

from gymrat.errors import GymratError
from gymrat.session.store import ReadSessionResult
from gymrat.supervisor.events import event_from_wire
from gymrat.supervisor.exit_sequence import ExitReport
from gymrat.supervisor.supervise import SupervisionResult
from gymrat.telemetry import run_spans
from gymrat.telemetry.run_spans import TracingState
from tests.cli.commands.supervise._seams import (
    Seams,
    install_seams,
    run,
)
from tests.cli.supervise._fixtures import (
    follow_up_event,
    make_supervision_result,
    session_state_three_iterations,
)

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

    result = run(*flag_args)

    assert result.exit_code == 0
    assert [call["finalize"] for call in seams.exit_calls] == [expected_finalize]


# ---------------------------------------------------------------------------
# exit sequence wiring
# ---------------------------------------------------------------------------


def test_supervise_when_supervise_raises_does_fail_before_the_exit_sequence(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    seams = install_seams(monkeypatch, raises=GymratError("config broken"))

    result = run()

    assert result.exit_code == 2
    assert "config broken" in result.stderr
    assert seams.exit_calls == []


def test_supervise_when_exit_sequence_logs_an_event_does_append_it_to_the_run_log(
    repo: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    seams = install_seams(monkeypatch)
    log_path = tmp_path / "supervisor.jsonl"
    event = follow_up_event(action="ended", reason="nothing to settle")
    seams.exit_hook = lambda call: call["log"](event)

    run("--log", str(log_path))

    logged = [event_from_wire(json.loads(line)) for line in log_path.read_text().splitlines()]
    assert logged[-1:] == [event]


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

    monkeypatch.setattr(
        run_spans,
        "setup_tracing",
        create_autospec(run_spans.setup_tracing, side_effect=tracing_with_its_own_observer),
    )
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

    run()

    assert observed == [event]


def test_supervise_when_run_ends_does_run_the_exit_sequence_in_the_supervisor_loop(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    seams = install_seams(monkeypatch)
    loops: list[asyncio.AbstractEventLoop] = []
    seams.supervise_hook = lambda _call: loops.append(asyncio.get_running_loop())
    seams.exit_hook = lambda _call: loops.append(asyncio.get_running_loop())

    run()

    supervise_loop, exit_loop = loops
    assert exit_loop is supervise_loop


# ---------------------------------------------------------------------------
# closing summary after the exit sequence
# ---------------------------------------------------------------------------


def test_supervise_when_exit_sequence_changes_the_session_does_summarize_the_session_it_left(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    seams = install_seams(monkeypatch)
    settled = ReadSessionResult(
        state=session_state_three_iterations(-4.2, "improved", seq=3), has_baseline=True
    )

    def settle(_call: dict[str, Any]) -> None:
        seams.session_result = settled

    seams.exit_hook = settle

    result = run()

    assert result.exit_code == 0
    assert "3 iterations" in result.stdout


# ---------------------------------------------------------------------------
# exit codes
# ---------------------------------------------------------------------------


def test_supervise_when_exit_sequence_errors_does_exit_two_without_writing_the_error_to_stderr(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    seams = install_seams(
        monkeypatch, result=make_supervision_result(reason="interrupted", ended_by="wall-clock")
    )
    seams.exit_report = ExitReport(steps=(), error=_EXIT_ERROR)

    result = run()

    assert result.exit_code == 2
    assert _EXIT_ERROR not in result.stderr


def test_supervise_when_driver_and_exit_sequence_both_error_does_surface_the_driver_message_on_stderr(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    seams = install_seams(
        monkeypatch,
        result=make_supervision_result(reason="error", message="SDK connection lost"),
    )
    seams.exit_report = ExitReport(steps=(), error=_EXIT_ERROR)

    result = run()

    assert result.exit_code == 2
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

    result = run()

    assert result.exit_code == 2
    assert len(result.stderr.splitlines()) == 1


@pytest.mark.parametrize(
    ("supervision", "expected_exit"),
    [
        pytest.param(
            make_supervision_result(reason="interrupted", ended_by="wall-clock"),
            1,
            id="wall-clock-cap",
        ),
        pytest.param(
            make_supervision_result(
                reason="interrupted",
                ended_by="stop-condition",
                end_reason="max iterations (2 of 2)",
            ),
            0,
            id="stop-condition",
        ),
    ],
)
def test_supervise_when_run_ended_by_a_condition_does_exit_with_its_code(
    repo: str, monkeypatch: pytest.MonkeyPatch, supervision: SupervisionResult, expected_exit: int
):
    install_seams(monkeypatch, result=supervision)

    result = run()

    assert result.exit_code == expected_exit
