"""Iterate command tests: basic execution, progress renderer wiring, format flags, and budget.

Budget tests verify that a live budget appends a time-left line to text output
and inserts a ``budget`` key in JSON output, including on stop-condition exits.
"""

import json
import os
import shlex
import signal
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, override
from unittest.mock import Mock, create_autospec

import pytest
from rich.console import Console
from syrupy.assertion import SnapshotAssertion

from gymrat.cli.app import app
from gymrat.cli.console import stderr_console
from gymrat.cli.iterate.progress import IterateRenderer
from gymrat.loop.iterate.run import IterateOptions, IterateResult, iterate_session
from gymrat.progress_events import (
    JudgeStarted,
    PassFinished,
    PassStarted,
    PrepareFinished,
    PrepareStarted,
    ProgressEvent,
)
from gymrat.session.paths import progress_path
from gymrat.session.progress_file import SidecarWriter
from gymrat.session.records import CommandRecord
from tests._ansi import (
    strip_ansi,
    stripped_lines,
)
from tests._clock import install_monotonic_clock
from tests._process_helpers import wait_for_pid_file_blocking
from tests._rich import Clock, console_output, frame_text, screen_lines, sealed_console
from tests.cli._budget import set_origin
from tests.cli._session import (
    FailingStdoutRunner,
    closed_stdout_error,
    force_render_mode,
    leave_as_is,
    runner,
    write_bench_config,
)
from tests.cli._signalled_cli import (
    SETTLE_TIMEOUT_S,
    pid_recording_script,
    spawned_gymrat,
    stop_by_signal,
)
from tests.loop._settle import start_with
from tests.loop.iterate._fixtures import (
    MALFORMED_LINE_WARNING,
    OUTLASTING_ITERATION_MS,
    CollectSamplesRecorder,
    bench_malformed_once,
    install_collect_samples,
    regressed_run,
    stub_improved_samples,
    stub_runs,
    write_iterate_session,
)
from tests.session._budget import (
    install_budget,
    install_budget_with_ten_minutes_left,
    install_tight_budget,
)
from tests.session.records._fixtures import (
    iteration_record,
    last_command_record,
    records_of_type,
    settled_history,
)

# ---------------------------------------------------------------------------
# the iterate command
# ---------------------------------------------------------------------------


def test_iterate_command_when_run_does_report_the_measured_iteration_with_its_trace(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    write_iterate_session(repo)
    mock = install_collect_samples(monkeypatch)
    stub_improved_samples(mock, repo)

    result = runner.invoke(app, ["iterate", "--bench", "npm run bench", "--samples", "10"])

    assert result.exit_code == 0
    lines = stripped_lines(result.stdout, keep_blank=False)
    assert lines[0] == "iteration 1 · experiment vs baseline · 10 paired samples"
    assert lines[-1] == "gymrat keep"
    non_command = records_of_type(repo, CommandRecord, matching=False)
    assert len(non_command) == 2
    cmd = last_command_record(repo)
    assert (cmd.name, cmd.seq, cmd.exit_code, cmd.reason) == ("iterate", 1, 0, None)
    assert (cmd.args["bench"], cmd.args["samples"]) == ("npm run bench", 10)


def test_iterate_command_when_progress_sidecar_cannot_be_removed_does_still_succeed_with_a_warning(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    write_iterate_session(repo)
    mock = install_collect_samples(monkeypatch)
    stub_improved_samples(mock, repo)
    sidecar = progress_path(repo)
    original_unlink = os.unlink

    def failing_unlink(path: str | os.PathLike[str], *, dir_fd: int | None = None) -> None:
        if str(path) == sidecar:
            raise PermissionError(13, "The process cannot access the file", sidecar)
        original_unlink(path, dir_fd=dir_fd)

    monkeypatch.setattr(os, "unlink", create_autospec(os.unlink, side_effect=failing_unlink))

    result = runner.invoke(app, ["iterate", "--bench", "npm run bench"])

    assert result.exit_code == 0
    assert f"could not remove the progress sidecar {sidecar}" in result.stderr


# ---------------------------------------------------------------------------
# the iterate command — progress renderer wiring
# ---------------------------------------------------------------------------


_SUBSCRIBER_EVENTS: tuple[ProgressEvent, ...] = (
    PrepareStarted(label="baseline", at_ms=0),
    PrepareFinished(label="baseline", at_ms=1000),
    JudgeStarted(at_ms=2000),
)


def _emitting_iterate_session(events: Sequence[ProgressEvent] = _SUBSCRIBER_EVENTS) -> Mock:
    """A stand-in for ``iterate_session`` that reports ``events`` and succeeds."""

    async def emit(
        root: str,
        config: object,
        options: IterateOptions | None = None,
        *,
        color: bool | None = None,
    ) -> IterateResult:
        assert options is not None
        assert options.on_progress is not None
        for event in events:
            options.on_progress(event)
        return _make_iterate_result()

    return create_autospec(iterate_session, side_effect=emit)


#: Rows of the terminal the live-mode tests render the dashboard into.
_LIVE_SCREEN_HEIGHT = 40


def _install_live_console(monkeypatch: pytest.MonkeyPatch) -> Console:
    """Force live mode onto a sealed console the command renders into; return the console."""
    console = sealed_console(height=_LIVE_SCREEN_HEIGHT, get_time=Clock(0.0))
    force_render_mode(monkeypatch, "loop", "live")
    monkeypatch.setattr(
        "gymrat.cli.commands.loop.stderr_console",
        create_autospec(stderr_console, return_value=console),
    )
    return console


def _capture_renderers(monkeypatch: pytest.MonkeyPatch) -> list[IterateRenderer]:
    """Keep the real ``IterateRenderer``, collecting each one the command builds."""
    built: list[IterateRenderer] = []

    class _CapturedRenderer(IterateRenderer):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            built.append(self)

    monkeypatch.setattr("gymrat.cli.commands.loop.IterateRenderer", _CapturedRenderer)
    return built


def _install_iterate_session(monkeypatch: pytest.MonkeyPatch, session: object) -> None:
    """Replace ``iterate_session`` in the loop command module with ``session``."""
    monkeypatch.setattr("gymrat.cli.commands.loop.iterate_session", session)


def _make_iterate_result() -> IterateResult:
    """A minimal ``IterateResult`` with a dummy report and record."""
    return IterateResult(
        record=iteration_record(seq=1),
        report="iteration 1 · experiment vs baseline · 10 paired samples\ngymrat keep",
    )


#: A run up to the judge, with one measured pass the bar counts against the sample total.
_DASHBOARD_EVENTS: tuple[ProgressEvent, ...] = (
    PrepareStarted(label="baseline", at_ms=0),
    PrepareFinished(label="baseline", at_ms=1000),
    PassStarted(round=1, total_rounds=10, target_count=2, label="experiment", at_ms=1000),
    PassFinished(round=1, total_rounds=10, target_count=2, label="experiment", at_ms=1500),
    JudgeStarted(at_ms=2000),
)


def test_iterate_command_when_live_does_render_the_session_dashboard_timing_the_judge_in_seconds(
    repo: str, monkeypatch: pytest.MonkeyPatch, snapshot: SnapshotAssertion
):
    write_iterate_session(repo)
    _install_live_console(monkeypatch)
    renderers = _capture_renderers(monkeypatch)
    _install_iterate_session(monkeypatch, _emitting_iterate_session(_DASHBOARD_EVENTS))
    # Three seconds past the judge's start, read through the renderer's seconds clock.
    install_monotonic_clock(monkeypatch, 5000.0)

    runner.invoke(app, ["iterate", "--bench", "npm run bench"])

    (renderer,) = renderers
    assert frame_text(renderer.frame()) == snapshot


def test_iterate_command_when_verbose_does_keep_the_live_display(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    write_iterate_session(repo)
    _install_live_console(monkeypatch)
    renderers = _capture_renderers(monkeypatch)
    mounted: list[bool] = []

    async def note_transient(*_args: object, **_kwargs: object) -> IterateResult:
        (renderer,) = renderers
        if renderer.live is not None:
            mounted.append(renderer.live.transient)
        return _make_iterate_result()

    _install_iterate_session(
        monkeypatch, create_autospec(iterate_session, side_effect=note_transient)
    )

    result = runner.invoke(app, ["iterate", "--bench", "npm run bench", "--verbose"])

    assert (result.exit_code, mounted) == (0, [False])


def test_iterate_command_when_error_does_stop_the_live_display(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    write_iterate_session(repo)
    _install_live_console(monkeypatch)
    renderers = _capture_renderers(monkeypatch)
    _install_iterate_session(
        monkeypatch, create_autospec(iterate_session, side_effect=RuntimeError("bench exploded"))
    )

    result = runner.invoke(app, ["iterate", "--bench", "npm run bench"])

    (renderer,) = renderers
    assert (result.exit_code, renderer.live) == (2, None)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX-only shell and signals")
def test_iterate_command_when_terminated_mid_pass_does_remove_the_progress_sidecar(
    repo: str, tmp_path: Path, reap_groups: list[int]
):
    script = tmp_path / "bench.sh"
    script.write_text(
        pid_recording_script(tmp_path / "bench.pid", "exec sleep 120\n"), encoding="utf-8"
    )
    start_with(repo)
    write_bench_config(repo, bench=f"sh {shlex.quote(str(script))}")

    with spawned_gymrat(["iterate"], repo) as proc:
        # The pass-start event writes the sidecar before the bench runs.
        reap_groups.append(wait_for_pid_file_blocking(tmp_path / "bench.pid", SETTLE_TIMEOUT_S))

        stop_by_signal(proc, signal.SIGTERM)

    assert (proc.returncode, Path(progress_path(repo)).exists()) == (143, False)


# ---------------------------------------------------------------------------
# the iterate command — a failing progress subscriber
# ---------------------------------------------------------------------------


class _FailingSidecar:
    """A stand-in progress sidecar writer raising the next queued ``OSError`` per event."""

    def __init__(self, messages: Sequence[str]) -> None:
        self._errors = (OSError(message) for message in messages)

    def __call__(self, _event: ProgressEvent) -> None:
        raise next(self._errors)


def _install_spied_renderer(monkeypatch: pytest.MonkeyPatch) -> list[ProgressEvent]:
    """Replace ``IterateRenderer`` with the real renderer, spied on ``report``; return the events."""
    events: list[ProgressEvent] = []

    class _SpiedRenderer(IterateRenderer):
        @override
        def report(self, event: ProgressEvent) -> None:
            events.append(event)
            super().report(event)

    monkeypatch.setattr("gymrat.cli.commands.loop.IterateRenderer", _SpiedRenderer)
    return events


def _wire_failing_subscriber(
    repo: str, monkeypatch: pytest.MonkeyPatch, messages: Sequence[str]
) -> list[ProgressEvent]:
    """Wire ``iterate`` with a sidecar raising ``messages`` and a spied real renderer.

    Args:
        repo: The repository to open the session in.
        monkeypatch: Installs the renderer, session and sidecar doubles.
        messages: The ``OSError`` message the sidecar raises for each event, in order.

    Returns:
        The events the renderer received.
    """
    write_iterate_session(repo)
    events = _install_spied_renderer(monkeypatch)
    _install_iterate_session(monkeypatch, _emitting_iterate_session())
    monkeypatch.setattr(
        "gymrat.cli.commands.loop.SidecarWriter",
        create_autospec(SidecarWriter, return_value=_FailingSidecar(messages)),
    )
    return events


def _warning_lines(stderr: str) -> list[str]:
    """The stderr lines that open with ``warning: ``, stripped of color."""
    return [line for line in strip_ansi(stderr).splitlines() if line.startswith("warning: ")]


_THREE_DISK_FULL = ("disk full", "disk full", "disk full")


def test_iterate_command_when_debug_and_subscriber_raises_does_follow_the_warning_with_traceback(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    _wire_failing_subscriber(repo, monkeypatch, _THREE_DISK_FULL)

    result = runner.invoke(app, ["iterate", "--bench", "npm run bench", "--debug"])

    lines = strip_ansi(result.stderr).splitlines()
    warning_at = lines.index("warning: disk full")
    assert lines[warning_at + 1] == "Traceback (most recent call last):"
    assert _warning_lines(result.stderr) == ["warning: disk full"]


@pytest.mark.parametrize(
    ("messages", "warnings"),
    [
        pytest.param(_THREE_DISK_FULL, ["warning: disk full"], id="same-failure"),
        pytest.param(
            ("disk full", "permission denied", "disk full"),
            ["warning: disk full", "warning: permission denied"],
            id="different-failures",
        ),
    ],
)
def test_iterate_command_when_subscriber_raises_does_warn_once_per_distinct_failure_without_failing_the_run(
    repo: str, monkeypatch: pytest.MonkeyPatch, messages: tuple[str, ...], warnings: list[str]
):
    events = _wire_failing_subscriber(repo, monkeypatch, messages)

    result = runner.invoke(app, ["iterate", "--bench", "npm run bench"])

    assert _warning_lines(result.stderr) == warnings
    assert "Traceback" not in result.stderr
    assert events == list(_SUBSCRIBER_EVENTS)
    assert result.exit_code == 0
    assert stripped_lines(result.stdout, keep_blank=False) == [
        "iteration 1 · experiment vs baseline · 10 paired samples",
        "gymrat keep",
    ]
    assert last_command_record(repo).exit_code == 0


# ---------------------------------------------------------------------------
# the iterate command — a bench line the adapter cannot read
# ---------------------------------------------------------------------------


def test_iterate_command_when_plain_and_adapter_warns_does_print_it_once_on_stderr(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    write_iterate_session(repo)
    bench_malformed_once(monkeypatch)
    force_render_mode(monkeypatch, "loop", "plain")

    result = runner.invoke(app, ["iterate", "--bench", "npm run bench"])

    lines = strip_ansi(result.stderr).splitlines()
    assert [line for line in lines if "METRIC" in line] == [MALFORMED_LINE_WARNING]


def test_iterate_command_when_live_and_adapter_warns_does_leave_the_warning_on_screen(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    write_iterate_session(repo)
    bench_malformed_once(monkeypatch)
    console = _install_live_console(monkeypatch)

    result = runner.invoke(app, ["iterate", "--bench", "npm run bench"])

    screen = screen_lines(console_output(console), height=_LIVE_SCREEN_HEIGHT)
    assert result.exit_code == 0
    assert screen == [MALFORMED_LINE_WARNING]


# ---------------------------------------------------------------------------
# iterate --format json
# ---------------------------------------------------------------------------


def _improved_run(repo: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub the bench to read the experiment 10% faster and 20% leaner than the baseline."""
    stub_improved_samples(install_collect_samples(monkeypatch), repo)


def _confirmed_regression(repo: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub the bench to read a 10% regression on the run and again on its confirming rerun."""
    stub_runs(install_collect_samples(monkeypatch), repo, [regressed_run(), regressed_run()])


@pytest.mark.parametrize(
    ("bench", "expected"),
    [
        pytest.param(
            _improved_run,
            # The geomean of a 0.9x time and a 0.8x size: sqrt(0.72) - 1.
            {"outcome": "improved", "delta_pct": -15.147186, "confirm": None},
            id="no-rerun",
        ),
        pytest.param(
            _confirmed_regression,
            {
                "outcome": "regressed",
                "delta_pct": 10.0,
                "confirm": {"ran": True, "filtered": ["total_ms", "alloc_bytes"], "absent": None},
            },
            id="confirmed-on-rerun",
        ),
    ],
)
def test_iterate_command_when_format_json_does_emit_structured_json_on_stdout(
    repo: str,
    monkeypatch: pytest.MonkeyPatch,
    bench: Callable[[str, pytest.MonkeyPatch], None],
    expected: dict[str, object],
):
    write_iterate_session(repo)
    bench(repo, monkeypatch)

    result = runner.invoke(app, ["iterate", "--bench", "npm run bench", "--format", "json"])

    assert result.exit_code == 0
    doc = json.loads(result.stdout)
    assert doc.keys() == {"seq", "outcome", "primary", "metrics", "confirm"}
    assert doc["metrics"].keys() == {"total_ms", "alloc_bytes"}
    assert (doc["seq"], doc["outcome"], doc["primary"], doc["confirm"]) == (
        1,
        expected["outcome"],
        {"kind": "geomean", "delta_pct": pytest.approx(expected["delta_pct"])},
        expected["confirm"],
    )


def _at_iteration_cap(repo: str) -> None:
    """A settled session that already reached its configured cap of one iteration."""
    write_iterate_session(repo, settled_history())
    write_bench_config(repo, stop={"max_iterations": 1})


def _tool_run_with_ten_minutes_left(repo: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Run from the tool under a live 30-minute budget with ten minutes left on a frozen clock."""
    install_budget_with_ten_minutes_left(repo, monkeypatch)
    set_origin(monkeypatch, "tool")


@pytest.mark.parametrize(
    ("arrange", "budget_key"),
    [
        pytest.param(leave_as_is, {}, id="no-budget"),
        pytest.param(
            _tool_run_with_ten_minutes_left,
            {"budget": {"cap_minutes": 30, "remaining_seconds": 600}},
            id="under-budget",
        ),
    ],
)
def test_iterate_command_when_format_json_and_stop_condition_does_emit_stop_document(
    repo: str,
    monkeypatch: pytest.MonkeyPatch,
    improved_samples_mock: CollectSamplesRecorder,
    arrange: Callable[[str, pytest.MonkeyPatch], None],
    budget_key: dict[str, object],
):
    _at_iteration_cap(repo)
    arrange(repo, monkeypatch)

    result = runner.invoke(app, ["iterate", "--bench", "npm run bench", "--format", "json"])

    assert result.exit_code == 1
    assert json.loads(result.stdout) == {
        "stopped": True,
        "reason": "Stop condition met: max iterations (1 of 1)",
        **budget_key,
    }
    assert improved_samples_mock.call_count == 0


def test_iterate_command_when_stop_and_format_json_and_stdout_closed_does_exit_one(
    repo: str, improved_samples_mock: CollectSamplesRecorder
):
    _at_iteration_cap(repo)

    result = FailingStdoutRunner(closed_stdout_error()).invoke(
        app, ["iterate", "--bench", "npm run bench", "--format", "json"]
    )

    assert (result.exit_code, result.stderr) == (1, "")


# ---------------------------------------------------------------------------
# the iterate command — refused while a supervised run is live
# ---------------------------------------------------------------------------


@pytest.fixture
def improved_samples_mock(repo: str, monkeypatch: pytest.MonkeyPatch) -> CollectSamplesRecorder:
    """The stubbed bench, answering every sampling call with an improved run."""
    mock = install_collect_samples(monkeypatch)
    stub_improved_samples(mock, repo)
    return mock


@pytest.fixture
def supervised_repo(
    repo: str, monkeypatch: pytest.MonkeyPatch, improved_samples_mock: CollectSamplesRecorder
) -> str:
    """An open session with a before hook under a live budget, run from the shell."""
    header = write_iterate_session(repo)
    # The before hook runs in the experiment worktree, so it must exist for the hook to fire.
    Path(header.worktrees.experiment).mkdir()
    marker = Path(repo, "hook-ran")
    write_bench_config(repo, hooks={"before": f"touch '{marker}'"})
    install_budget(repo, monkeypatch)
    return repo


def test_iterate_command_when_tool_hosted_under_live_budget_does_run_the_before_hook(
    supervised_repo: str, monkeypatch: pytest.MonkeyPatch
):
    set_origin(monkeypatch, "tool")

    runner.invoke(app, ["iterate", "--bench", "npm run bench"])

    assert Path(supervised_repo, "hook-ran").exists()


def _unsettled(repo: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """An open session whose last iteration was never kept or discarded."""
    write_iterate_session(repo, (iteration_record(seq=1),))
    write_bench_config(repo)
    install_budget(repo, monkeypatch)


def _stop_condition_met(repo: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """A session at its iteration cap, under a live budget."""
    _at_iteration_cap(repo)
    install_budget(repo, monkeypatch)


def _budget_exceeded(repo: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """A settled session whose last iteration outlasts the 5 minutes the budget has left."""
    write_iterate_session(repo, settled_history(duration_ms=OUTLASTING_ITERATION_MS))
    write_bench_config(repo)
    install_tight_budget(repo, monkeypatch)


#: Every readiness setup, the ``(exit_code, reason, seq)`` the tool-hosted run
#: records for it, and the phrases of the refusal it prints.
READINESS = [
    pytest.param(
        _unsettled, (2, "unsettled", 1), ("Iteration 1 has not been settled",), id="unsettled"
    ),
    pytest.param(
        _stop_condition_met,
        (1, "stop-condition", 1),
        ("Stop condition met: max iterations (1 of 1)", "left of 30m"),
        id="stop-condition",
    ),
    pytest.param(
        _budget_exceeded,
        (1, "budget-exceeded", 1),
        ("the cap would cut this one off",),
        id="budget-exceeded",
    ),
]


@pytest.mark.parametrize(("setup", "expected", "phrases"), READINESS)
def test_iterate_command_when_tool_hosted_in_unready_state_does_refuse_with_its_own_reason(
    *,
    repo: str,
    monkeypatch: pytest.MonkeyPatch,
    setup: Callable[[str, pytest.MonkeyPatch], None],
    expected: tuple[int, str, int],
    phrases: tuple[str, ...],
    improved_samples_mock: CollectSamplesRecorder,
):
    setup(repo, monkeypatch)
    set_origin(monkeypatch, "tool")

    result = runner.invoke(app, ["iterate"])

    cmd = last_command_record(repo)
    assert (result.exit_code, cmd.reason, cmd.seq) == expected
    stderr = " ".join(stripped_lines(result.stderr, keep_blank=False))
    assert [phrase for phrase in phrases if phrase not in stderr] == []
    assert improved_samples_mock.call_count == 0


@pytest.mark.parametrize(
    "setup", [pytest.param(param.values[0], id=param.id) for param in READINESS]
)
def test_iterate_command_when_supervised_run_live_does_refuse_before_every_readiness_check(
    repo: str,
    monkeypatch: pytest.MonkeyPatch,
    setup: Callable[[str, pytest.MonkeyPatch], None],
    improved_samples_mock: CollectSamplesRecorder,
):
    setup(repo, monkeypatch)

    result = runner.invoke(app, ["iterate"])

    assert result.exit_code == 2
    assert last_command_record(repo).reason == "supervised-use-tool"
    assert improved_samples_mock.call_count == 0
