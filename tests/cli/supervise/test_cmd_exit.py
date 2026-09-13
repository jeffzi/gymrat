"""Run-end tests for the ``gymrat supervise`` command: the exit sequence and exit codes.

Every return of ``supervise()`` hands the session to the exit sequence before the
reporter stops and the closing summary prints; the command's exit code then folds
the driver outcome, the exit sequence's error, and how the run ended. Shares its
seam-installation harness with :mod:`tests.cli.supervise.test_cmd`, whose fake
exit sequence records each call and lets a test act from inside it.
"""

import asyncio
import contextlib
import json
import re
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from gymrat.cli.shared import write_and_flush
from gymrat.cli.supervise import span_lifecycle
from gymrat.cli.supervise.progress import ReadSessionResult
from gymrat.cli.supervise.span_lifecycle import TracingState
from gymrat.errors import GymratError
from gymrat.exec import ExecOptions
from gymrat.exec import exec as run_exec
from gymrat.session.paths import budget_path
from gymrat.supervisor import SupervisionResult, create_event_log_writer, event_from_wire
from gymrat.supervisor.exit_sequence import ExitPhase, ExitReport, ExitStep
from gymrat.supervisor.supervise import EndedBy
from tests._process_helpers import is_alive, wait_until_dead
from tests.cli.supervise._fixtures import (
    follow_up_event,
    make_supervision_result,
    session_state_three_iterations,
)
from tests.cli.supervise.test_cmd import _CAP_MINUTES, _CAP_MS, _err_text, _install_seams, _run

_EXIT_ERROR = "finalize failed: disk full"

# How long the signal-cleanup test waits for the killed child to disappear.
_KILL_SETTLE_S = 3.0


# ---------------------------------------------------------------------------
# --no-finalize
# ---------------------------------------------------------------------------


def test_supervise_help_when_rendered_does_describe_the_no_finalize_flag(repo: str):
    result = _run("--help")

    flat = re.sub(r"[│╭╮╰╯─\s]+", " ", _err_text(result))
    assert result.exit_code == 0
    assert "--no-finalize" in flat
    assert "leave the session open instead of finalizing it on exit" in flat


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
    seams = _install_seams(monkeypatch)

    result = _run("optimize it", "--max-minutes", "10", *flag_args)

    assert result.exit_code == 0
    assert [call["finalize"] for call in seams.exit_calls] == [expected_finalize]


# ---------------------------------------------------------------------------
# exit sequence wiring
# ---------------------------------------------------------------------------


_INTERRUPTED_ENDED_BY: tuple[EndedBy, ...] = (
    "wall-clock",
    "spend-cap",
    "guard",
    "stop-condition",
    "hook-failure",
)


@pytest.mark.parametrize(
    "supervision",
    [
        pytest.param(make_supervision_result(), id="session"),
        pytest.param(make_supervision_result(reason="error", message="lost"), id="error-outcome"),
        *(
            pytest.param(
                make_supervision_result(reason="interrupted", ended_by=ended_by), id=ended_by
            )
            for ended_by in _INTERRUPTED_ENDED_BY
        ),
    ],
)
def test_supervise_when_supervise_returns_does_run_the_exit_sequence_over_the_session_and_its_end(
    repo: str, monkeypatch: pytest.MonkeyPatch, supervision: SupervisionResult
):
    seams = _install_seams(monkeypatch, result=supervision)

    _run("optimize it", "--max-minutes", "10")

    (call,) = seams.exit_calls
    assert call["context"] == seams.supervise_calls[0]["context"]
    assert call["ended_by"] == supervision.ended_by


def test_supervise_when_supervise_raises_does_not_run_the_exit_sequence(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    seams = _install_seams(monkeypatch, raises=GymratError("config broken"))

    result = _run("optimize it", "--max-minutes", "10")

    assert result.exit_code == 2
    assert seams.exit_calls == []


def test_supervise_when_exit_sequence_reports_a_phase_does_show_it_on_the_reporter(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    seams = _install_seams(monkeypatch)
    phase = ExitPhase(kind="settling", pid=None)
    seams.exit_hook = lambda call: call["progress"](phase)

    _run("optimize it", "--max-minutes", "10")

    assert seams.exit_phases == [phase]


def test_supervise_when_exit_sequence_warns_does_route_it_through_the_reporter_not_stderr(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    seams = _install_seams(monkeypatch)
    hint = "no checks command is configured"
    seams.exit_hook = lambda call: call["warn"](hint)

    result = _run("optimize it", "--max-minutes", "10")

    assert seams.warnings == [hint]
    assert hint not in result.stderr


def test_supervise_when_exit_sequence_logs_an_event_does_write_it_to_the_log_and_the_observer(
    repo: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    seams = _install_seams(monkeypatch)
    log_path = tmp_path / "supervisor.jsonl"
    run_event = follow_up_event(action="replied")
    event = follow_up_event(action="ended", reason="nothing to settle")

    def log_after_the_run(call: dict[str, Any]) -> None:
        create_event_log_writer(log_path)(run_event)
        call["log"](event)

    seams.exit_hook = log_after_the_run

    _run("optimize it", "--max-minutes", "10", "--log", str(log_path))

    logged = [event_from_wire(json.loads(line)) for line in log_path.read_text().splitlines()]
    assert logged[-2:] == [run_event, event]
    assert event in seams.observed_events


def test_supervise_when_exit_sequence_logs_an_event_does_hand_it_to_the_observer_supervise_used(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    seams = _install_seams(monkeypatch)
    run_observed: list[object] = []

    def tracing_with_its_own_observer(*, prompt: object, **_kwargs: object) -> tuple[object, ...]:
        return prompt, run_observed.append, TracingState()

    monkeypatch.setattr(span_lifecycle, "setup_tracing", tracing_with_its_own_observer)
    event = follow_up_event(action="ended", reason="nothing to settle")
    seams.exit_hook = lambda call: call["log"](event)

    _run("optimize it", "--max-minutes", "10")

    assert run_observed == [event]


def test_supervise_when_run_ends_does_run_the_exit_sequence_in_the_supervisor_loop(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    seams = _install_seams(monkeypatch)
    loops: list[asyncio.AbstractEventLoop] = []
    seams.reporter_start.side_effect = lambda: loops.append(asyncio.get_running_loop())
    seams.exit_hook = lambda _call: loops.append(asyncio.get_running_loop())

    _run("optimize it", "--max-minutes", "10")

    start_loop, exit_loop = loops
    assert exit_loop is start_loop


# ---------------------------------------------------------------------------
# ordering around the exit sequence
# ---------------------------------------------------------------------------


def test_supervise_when_run_ends_does_exit_sequence_then_stop_reporter_then_print_summary(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    order: list[str] = []
    seams = _install_seams(monkeypatch)
    seams.exit_hook = lambda _call: order.append("exit")
    seams.reporter_stop.side_effect = lambda: order.append("stop")

    def tracking_write(stream: Any, data: str) -> None:
        if stream is sys.stdout:
            order.append("write")
        write_and_flush(stream, data)

    monkeypatch.setattr("gymrat.cli.supervise.cmd.write_and_flush", tracking_write)

    result = _run("optimize it", "--max-minutes", "10")

    assert result.exit_code == 0
    assert list(dict.fromkeys(order)) == ["exit", "stop", "write"]


def test_supervise_when_exit_sequence_runs_does_find_the_budget_file_already_removed(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    seams = _install_seams(monkeypatch)
    budget_present: list[bool] = []
    seams.exit_hook = lambda _call: budget_present.append(Path(budget_path(repo)).exists())

    _run("optimize it", "--max-minutes", "10")

    assert budget_present == [False]


# ---------------------------------------------------------------------------
# closing summary after the exit sequence
# ---------------------------------------------------------------------------


def test_supervise_when_exit_sequence_reports_steps_does_print_them_as_exit_rows(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    seams = _install_seams(monkeypatch)
    seams.exit_report = ExitReport(
        steps=(
            ExitStep(kind="settled", text="settled: kept iteration 1 (checks passed)"),
            ExitStep(kind="nothing", text="session left open (--no-finalize)"),
        )
    )

    result = _run("optimize it", "--max-minutes", "10", "--no-finalize")

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
    seams = _install_seams(monkeypatch)
    seams.exit_report = exit_report
    settled = ReadSessionResult(
        state=session_state_three_iterations(-4.2, "improved", seq=3), has_baseline=True
    )

    def settle(_call: dict[str, Any]) -> None:
        seams.session_result = settled

    seams.exit_hook = settle

    result = _run("optimize it", "--max-minutes", "10")

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
    seams = _install_seams(
        monkeypatch, result=make_supervision_result(reason="interrupted", ended_by=ended_by)
    )
    seams.exit_report = ExitReport(steps=(), error=_EXIT_ERROR)

    result = _run("optimize it", "--max-minutes", "10")

    assert result.exit_code == 2
    assert f"  exit    error: {_EXIT_ERROR}" in result.stdout.splitlines()
    assert _EXIT_ERROR not in result.stderr


def test_supervise_when_driver_and_exit_sequence_both_error_does_exit_two_with_driver_message(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    seams = _install_seams(
        monkeypatch,
        result=make_supervision_result(reason="error", message="SDK connection lost"),
    )
    seams.exit_report = ExitReport(steps=(), error=_EXIT_ERROR)

    result = _run("optimize it", "--max-minutes", "10")

    assert result.exit_code == 2
    assert "SDK connection lost" in result.stderr


def test_supervise_when_a_cap_ended_the_session_does_exit_one_naming_the_cap(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    _install_seams(
        monkeypatch,
        result=make_supervision_result(
            reason="interrupted", ended_by="wall-clock", duration_ms=_CAP_MS, cost_usd=1.0
        ),
    )

    result = _run("optimize it", "--max-minutes", str(_CAP_MINUTES))

    assert result.exit_code == 1
    assert result.stdout.splitlines()[0] == "! interrupted by wall-clock cap · 10m 0s · $1.00"


def test_supervise_when_supervise_raises_does_exit_two_with_message_on_stderr(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    _install_seams(monkeypatch, raises=GymratError("config broken"))

    result = _run("optimize it", "--max-minutes", "10")

    assert result.exit_code == 2
    assert "config broken" in result.stderr


def test_supervise_when_outcome_error_with_message_does_exit_two_and_surface_it(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    _install_seams(
        monkeypatch,
        result=make_supervision_result(
            reason="error", duration_ms=5_000, cost_usd=0.03, message="SDK connection lost"
        ),
    )

    result = _run("optimize it", "--max-minutes", "10")

    assert result.exit_code == 2
    assert result.stdout.splitlines()[0] == "✗ error · 5s · $0.03"
    assert "SDK connection lost" in result.stderr


@pytest.mark.parametrize(
    "message",
    [pytest.param(None, id="omitted"), pytest.param("", id="empty-string")],
)
def test_supervise_when_outcome_error_without_message_does_exit_two_quietly(
    repo: str, monkeypatch: pytest.MonkeyPatch, message: str | None
):
    _install_seams(
        monkeypatch,
        result=make_supervision_result(
            reason="error", duration_ms=5_000, cost_usd=0.01, message=message
        ),
    )

    result = _run("optimize it", "--max-minutes", "10")

    assert result.exit_code == 2
    assert not re.search(r"\berror\b", result.stderr, re.IGNORECASE)


def test_supervise_when_guard_ended_does_exit_one_with_guard_headline(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    _install_seams(
        monkeypatch,
        result=make_supervision_result(
            reason="interrupted",
            ended_by="guard",
            duration_ms=30_000,
            cost_usd=0.10,
            end_reason="safety limit reached",
        ),
    )

    result = _run("optimize it", "--max-minutes", "10")

    assert result.exit_code == 1
    headline = result.stdout.splitlines()[0]
    assert "stopped by guard" in headline
    assert "safety limit reached" in headline


@pytest.mark.parametrize(
    ("ended_by", "end_reason", "expected_exit"),
    [
        pytest.param("stop-condition", "max iterations (2 of 2)", 0, id="stop-condition"),
        pytest.param("spend-cap", "", 1, id="spend-cap"),
        pytest.param(
            "hook-failure",
            "after hook failed on iteration 2: exit 1 (stdout 80 B, stderr 5 B)",
            1,
            id="hook-failure",
        ),
    ],
)
def test_supervise_when_run_ended_by_a_condition_does_exit_with_its_code(
    repo: str,
    monkeypatch: pytest.MonkeyPatch,
    ended_by: EndedBy,
    end_reason: str,
    expected_exit: int,
):
    _install_seams(
        monkeypatch,
        result=make_supervision_result(
            reason="interrupted", ended_by=ended_by, end_reason=end_reason
        ),
    )

    result = _run("optimize it", "--max-minutes", "10")

    assert result.exit_code == expected_exit


# ---------------------------------------------------------------------------
# signal safety during the exit sequence
# ---------------------------------------------------------------------------


def _read_pid(pid_path: Path) -> int | None:
    """The pid the shell wrote to ``pid_path``, or ``None`` until the line is complete."""
    try:
        raw = pid_path.read_text()
    except FileNotFoundError:
        return None
    return int(raw) if raw.endswith("\n") else None


async def _wait_for_pid(pid_path: Path) -> int:
    """Poll ``pid_path`` until the shell has written a complete pid line, then return it.

    Args:
        pid_path: Path the shell writes the child's pid to.

    Returns:
        The pid written by the shell.

    Raises:
        TimeoutError: If no complete pid line appears within _KILL_SETTLE_S seconds.
    """
    loop = asyncio.get_running_loop()
    deadline = loop.time() + _KILL_SETTLE_S
    while (pid := _read_pid(pid_path)) is None:
        if loop.time() > deadline:
            message = f"pid never appeared at {pid_path}"
            raise TimeoutError(message)
        await asyncio.sleep(0.025)
    return pid


async def _child_survives_cleanups(cleanups: list[Callable[[], None]], cwd: Path) -> bool:
    """Spawn a live child through ``exec``, run every cleanup, and report whether it survived.

    Survival is judged after a bounded wait and before the ``exec`` task is
    cancelled, because that cancel kills the process group itself and would
    hide a cleanup that did nothing.

    Args:
        cleanups: Callbacks to run against the live child, in order.
        cwd: Directory the ``exec`` shell runs in and writes ``child.pid`` to.

    Returns:
        True when the child is still alive after the cleanups and the bounded
        wait, False when it was killed.
    """
    task = asyncio.create_task(
        run_exec("sleep 30 & echo $! > child.pid; wait", ExecOptions(cwd=str(cwd)))
    )
    try:
        pid = await _wait_for_pid(cwd / "child.pid")
        for cleanup in cleanups:
            cleanup()
        with contextlib.suppress(AssertionError):
            await wait_until_dead(pid, timeout_s=_KILL_SETTLE_S)
        return is_alive(pid)
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX-only shell and process groups")
def test_supervise_when_a_signal_arrives_does_kill_the_live_process_groups_it_spawned(
    repo: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    seams = _install_seams(monkeypatch)
    _run("optimize it", "--max-minutes", "10")
    cleanups = [call.args[0] for call in seams.install_cleanup.call_args_list]

    survived = asyncio.run(_child_survives_cleanups(cleanups, tmp_path))

    assert survived is False
