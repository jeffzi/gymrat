"""Iterate command tests: basic execution, progress renderer wiring, format flags, and budget.

Budget tests verify that a live budget appends a time-left line to text output
and inserts a ``budget`` key in JSON output, including on stop-condition exits.
"""

import json
import os
import re
import shlex
import signal
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, override
from unittest.mock import create_autospec

import pytest
from rich.console import Console
from syrupy.assertion import SnapshotAssertion

from gymrat.cli.app import app
from gymrat.cli.iterate.progress import IterateRenderer
from gymrat.loop.iterate.run import IterateOptions, IterateResult, LoopStopError, iterate_session
from gymrat.progress_events import (
    JudgeStarted,
    PassFinished,
    PassStarted,
    PrepareFinished,
    PrepareStarted,
    ProgressEvent,
)
from gymrat.session.paths import progress_path
from gymrat.session.records import CommandRecord, Confirm, PairedSamples
from tests._ansi import (
    strip_ansi,
    stripped_lines,
)
from tests._cli import ENTRY, no_color_env
from tests._process_helpers import wait_for_pid_file_blocking
from tests._rich import Clock, console_output, frame_text, screen_lines, sealed_console
from tests.cli._budget import (
    SUPERVISED_HINT,
    install_budget,
    install_tight_budget,
    set_origin,
)
from tests.cli._session import (
    FailingStdoutRunner,
    closed_stdout_error,
    last_command_record,
    runner,
    write_bench_config,
)
from tests.loop._settle import start_with
from tests.loop.iterate._fixtures import (
    MALFORMED_LINE_WARNING,
    CollectSamplesRecorder,
    baseline_rounds,
    bench_malformed_once,
    improved_rounds,
    install_collect_samples,
    iterate_session_header,
    stub_samples,
)
from tests.session.records._fixtures import (
    committed_keep,
    iteration_record,
    records_of_type,
    write_session_log,
)

# ---------------------------------------------------------------------------
# the iterate command
# ---------------------------------------------------------------------------


def test_iterate_command_when_run_does_measure_the_repo_report_on_stdout_and_record_the_trace(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    write_session_log(repo, iterate_session_header(repo))
    mock = install_collect_samples(monkeypatch)
    stub_samples(mock, repo, improved_rounds(), baseline_rounds())

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


def test_iterate_command_when_progress_sidecar_cannot_be_removed_does_warn_and_keep_the_exit_code(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    write_session_log(repo, iterate_session_header(repo))
    mock = install_collect_samples(monkeypatch)
    stub_samples(mock, repo, improved_rounds(), baseline_rounds())
    sidecar = progress_path(repo)
    original_unlink = os.unlink

    def failing_unlink(path: str | os.PathLike[str], *args: object, **kwargs: object) -> None:
        if str(path) == sidecar:
            raise PermissionError(13, "The process cannot access the file", sidecar)
        original_unlink(path, *args, **kwargs)  # type: ignore[arg-type]  # forwards whatever Path.unlink passed

    monkeypatch.setattr(os, "unlink", failing_unlink)

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


class _EmittingIterateSession:
    """A stand-in for ``iterate_session`` that reports ``events`` and succeeds."""

    def __init__(self, events: Sequence[ProgressEvent] = _SUBSCRIBER_EVENTS) -> None:
        self._events = events

    async def __call__(
        self,
        root: str,
        config: object,
        options: IterateOptions | None = None,
        *,
        color: bool | None = None,
    ) -> IterateResult:
        assert options is not None
        assert options.on_progress is not None
        for event in self._events:
            options.on_progress(event)
        return _make_iterate_result()


#: Rows of the terminal the live-mode tests render the dashboard into.
_LIVE_SCREEN_HEIGHT = 40


def _install_live_console(monkeypatch: pytest.MonkeyPatch) -> Console:
    """Force live mode onto a sealed console the command renders into; return the console."""
    console = sealed_console(height=_LIVE_SCREEN_HEIGHT, get_time=Clock(0.0))

    def fake_stderr_console(**_kwargs: object) -> Console:
        return console

    monkeypatch.setattr("gymrat.cli.commands.loop.resolve_render_mode", lambda: "live")
    monkeypatch.setattr("gymrat.cli.commands.loop.stderr_console", fake_stderr_console)
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


def _wire_successful_iterate(repo: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Open a fresh session and stub ``iterate_session`` to succeed."""
    write_session_log(repo, iterate_session_header(repo))
    _install_iterate_session(
        monkeypatch, create_autospec(iterate_session, return_value=_make_iterate_result())
    )


def _wire_stopping_iterate(repo: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Open a fresh session and stub ``iterate_session`` to hit the stop condition."""
    write_session_log(repo, iterate_session_header(repo))
    _install_iterate_session(
        monkeypatch,
        create_autospec(iterate_session, side_effect=LoopStopError("max iterations (3) reached")),
    )


#: A run up to the judge, with one measured pass the bar counts against the sample total.
_DASHBOARD_EVENTS: tuple[ProgressEvent, ...] = (
    PrepareStarted(label="baseline", at_ms=0),
    PrepareFinished(label="baseline", at_ms=1000),
    PassStarted(round=1, total_rounds=10, target_count=2, label="experiment", at_ms=1000),
    PassFinished(round=1, total_rounds=10, target_count=2, label="experiment", at_ms=1500),
    JudgeStarted(at_ms=2000),
)


def test_iterate_command_when_live_does_show_the_session_and_time_the_judge_in_seconds(
    repo: str, monkeypatch: pytest.MonkeyPatch, snapshot: SnapshotAssertion
):
    write_session_log(repo, iterate_session_header(repo))
    _install_live_console(monkeypatch)
    renderers = _capture_renderers(monkeypatch)
    _install_iterate_session(monkeypatch, _EmittingIterateSession(_DASHBOARD_EVENTS))
    # Three seconds past the judge's start, read through the renderer's seconds clock.
    monkeypatch.setattr("gymrat.clock.monotonic_ms", lambda: 5000.0)

    runner.invoke(app, ["iterate", "--bench", "npm run bench"])

    (renderer,) = renderers
    assert frame_text(renderer.frame()) == snapshot


@pytest.mark.parametrize(
    ("flags", "transient"),
    [
        pytest.param(["--verbose"], False, id="verbose"),
        pytest.param([], True, id="quiet"),
    ],
)
def test_iterate_command_when_live_does_keep_the_display_only_under_verbose(
    repo: str, monkeypatch: pytest.MonkeyPatch, flags: list[str], transient: bool
):
    write_session_log(repo, iterate_session_header(repo))
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

    result = runner.invoke(app, ["iterate", "--bench", "npm run bench", *flags])

    assert (result.exit_code, mounted) == (0, [transient])


def test_iterate_command_when_error_does_stop_the_live_display(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    write_session_log(repo, iterate_session_header(repo))
    _install_live_console(monkeypatch)
    renderers = _capture_renderers(monkeypatch)
    _install_iterate_session(
        monkeypatch, create_autospec(iterate_session, side_effect=RuntimeError("bench exploded"))
    )

    result = runner.invoke(app, ["iterate", "--bench", "npm run bench"])

    (renderer,) = renderers
    assert (result.exit_code, renderer.live) == (2, None)


_SETTLE_TIMEOUT_S = 30.0
"""Budget for the pid-file wait and the exit of the out-of-process iterate test."""

_BLOCKING_BENCH = """#!/bin/sh
echo $$ > "{directory}/bench.pid"
exec sleep 120
"""
"""A bench that records its pid and never finishes, so ``iterate`` is mid-pass when signalled."""


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX-only shell and signals")
def test_iterate_command_when_terminated_mid_pass_does_remove_the_progress_sidecar(
    repo: str, tmp_path: Path, reap_groups: list[int]
):
    script = tmp_path / "bench.sh"
    script.write_text(_BLOCKING_BENCH.format(directory=tmp_path), encoding="utf-8")
    start_with(repo)
    write_bench_config(repo, bench=f"sh {shlex.quote(str(script))}")
    proc = subprocess.Popen(  # noqa: S603 -- argv is the gymrat entry point plus a fixed command
        [*ENTRY, "iterate"],
        cwd=repo,
        env=no_color_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        # The pass-start event writes the sidecar before the bench runs.
        reap_groups.append(wait_for_pid_file_blocking(tmp_path / "bench.pid", _SETTLE_TIMEOUT_S))

        proc.send_signal(signal.SIGTERM)
        proc.communicate(timeout=_SETTLE_TIMEOUT_S)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.communicate()

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
    write_session_log(repo, iterate_session_header(repo))
    events = _install_spied_renderer(monkeypatch)
    monkeypatch.setattr("gymrat.cli.commands.loop.iterate_session", _EmittingIterateSession())
    sidecar = _FailingSidecar(messages)

    def sidecar_writer(_root: str) -> _FailingSidecar:
        return sidecar

    monkeypatch.setattr("gymrat.cli.commands.loop.SidecarWriter", sidecar_writer)
    return events


def _warning_lines(stderr: str) -> list[str]:
    """The stderr lines that open with ``warning: ``, stripped of color."""
    return [line for line in strip_ansi(stderr).splitlines() if line.startswith("warning: ")]


_THREE_DISK_FULL = ("disk full", "disk full", "disk full")


def test_iterate_command_when_subscriber_raises_different_failures_does_warn_once_per_failure(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    _wire_failing_subscriber(repo, monkeypatch, ("disk full", "permission denied", "disk full"))

    result = runner.invoke(app, ["iterate", "--bench", "npm run bench"])

    assert _warning_lines(result.stderr) == ["warning: disk full", "warning: permission denied"]


@pytest.mark.parametrize(
    "argv",
    [
        pytest.param(["--debug", "iterate", "--bench", "npm run bench"], id="root-flag"),
        pytest.param(["iterate", "--bench", "npm run bench", "--debug"], id="command-flag"),
    ],
)
def test_iterate_command_when_debug_and_subscriber_raises_does_follow_the_warning_with_traceback(
    repo: str, monkeypatch: pytest.MonkeyPatch, argv: list[str]
):
    _wire_failing_subscriber(repo, monkeypatch, _THREE_DISK_FULL)

    result = runner.invoke(app, argv)

    lines = strip_ansi(result.stderr).splitlines()
    warning_at = lines.index("warning: disk full")
    assert lines[warning_at + 1] == "Traceback (most recent call last):"
    assert _warning_lines(result.stderr) == ["warning: disk full"]


def test_iterate_command_when_subscriber_raises_does_warn_without_traceback_and_keep_the_run(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    events = _wire_failing_subscriber(repo, monkeypatch, _THREE_DISK_FULL)

    result = runner.invoke(app, ["iterate", "--bench", "npm run bench"])

    assert _warning_lines(result.stderr) == ["warning: disk full"]
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
    write_session_log(repo, iterate_session_header(repo))
    bench_malformed_once(monkeypatch)
    monkeypatch.setattr("gymrat.cli.commands.loop.resolve_render_mode", lambda: "plain")

    result = runner.invoke(app, ["iterate", "--bench", "npm run bench"])

    lines = strip_ansi(result.stderr).splitlines()
    assert [line for line in lines if "METRIC" in line] == [MALFORMED_LINE_WARNING]


def test_iterate_command_when_live_and_adapter_warns_does_leave_the_warning_on_screen(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    write_session_log(repo, iterate_session_header(repo))
    bench_malformed_once(monkeypatch)
    console = _install_live_console(monkeypatch)

    result = runner.invoke(app, ["iterate", "--bench", "npm run bench"])

    screen = screen_lines(console_output(console), height=_LIVE_SCREEN_HEIGHT)
    assert result.exit_code == 0
    assert screen == [MALFORMED_LINE_WARNING]


# ---------------------------------------------------------------------------
# iterate --format json
# ---------------------------------------------------------------------------


def test_iterate_command_when_format_json_does_emit_structured_json_on_stdout(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    _wire_successful_iterate(repo, monkeypatch)

    result = runner.invoke(app, ["iterate", "--bench", "npm run bench", "--format", "json"])

    assert result.exit_code == 0
    doc = json.loads(result.stdout)
    assert doc["seq"] == 1
    assert doc["outcome"] == "improved"
    assert doc["primary"]["kind"] == "geomean"
    assert doc["primary"]["delta_pct"] == pytest.approx(-7.2)
    assert "metrics" in doc
    assert doc["confirm"] is None


def test_iterate_command_when_format_json_does_include_confirm_when_rerun_happened(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    confirm = Confirm(
        ran=True,
        filtered=("total_ms",),
        samples=PairedSamples(
            experiment=({"total_ms": 14100},),
            baseline=({"total_ms": 15200},),
        ),
        absent=None,
    )
    record = iteration_record(seq=1, confirm=confirm)
    iterate_result = IterateResult(record=record, report="confirmed iteration report")
    write_session_log(repo, iterate_session_header(repo))
    _install_iterate_session(
        monkeypatch, create_autospec(iterate_session, return_value=iterate_result)
    )

    result = runner.invoke(app, ["iterate", "--bench", "npm run bench", "--format", "json"])

    assert result.exit_code == 0
    doc = json.loads(result.stdout)
    assert doc["confirm"] is not None
    assert doc["confirm"]["ran"] is True
    assert doc["confirm"]["filtered"] == ["total_ms"]


def test_iterate_command_when_format_json_and_stop_condition_does_emit_stop_document(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    _wire_stopping_iterate(repo, monkeypatch)

    result = runner.invoke(app, ["iterate", "--bench", "npm run bench", "--format", "json"])

    assert result.exit_code == 1
    doc = json.loads(result.stdout)
    assert doc["stopped"] is True
    assert "max iterations" in doc["reason"]


def test_iterate_command_when_stop_and_format_json_and_stdout_closed_does_exit_one(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    _wire_stopping_iterate(repo, monkeypatch)

    result = FailingStdoutRunner(closed_stdout_error()).invoke(
        app, ["iterate", "--bench", "npm run bench", "--format", "json"]
    )

    assert (result.exit_code, result.stderr) == (1, "")


def test_iterate_command_when_stdout_reader_closed_does_exit_zero_without_stderr(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    _wire_successful_iterate(repo, monkeypatch)

    result = FailingStdoutRunner(closed_stdout_error()).invoke(
        app, ["iterate", "--bench", "npm run bench"]
    )

    assert (result.exit_code, result.stderr) == (0, "")


# ---------------------------------------------------------------------------
# the iterate command — budget line and JSON key
# ---------------------------------------------------------------------------


def test_iterate_command_when_stop_condition_and_budget_active_does_include_time_left_in_stderr(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    _wire_stopping_iterate(repo, monkeypatch)
    install_budget(repo, monkeypatch)
    set_origin(monkeypatch, "tool")

    result = runner.invoke(app, ["iterate", "--bench", "npm run bench"])

    assert result.exit_code == 1
    assert re.search(r"left of 30m", result.stderr)


def test_iterate_command_when_stop_and_format_json_and_budget_active_does_include_budget_key(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    _wire_stopping_iterate(repo, monkeypatch)
    install_budget(repo, monkeypatch)
    set_origin(monkeypatch, "tool")

    result = runner.invoke(app, ["iterate", "--bench", "npm run bench", "--format", "json"])

    assert result.exit_code == 1
    doc = json.loads(result.stdout)
    assert "budget" in doc
    assert doc["budget"]["cap_minutes"] == 30
    assert isinstance(doc["budget"]["remaining_seconds"], int)


# ---------------------------------------------------------------------------
# the iterate command — refused while a supervised run is live
# ---------------------------------------------------------------------------


SUPERVISED_MESSAGE = "a supervised run is live; use the iterate tool"


@pytest.fixture
def improved_samples_mock(repo: str, monkeypatch: pytest.MonkeyPatch) -> CollectSamplesRecorder:
    """The stubbed bench, answering every sampling call with an improved run."""
    mock = install_collect_samples(monkeypatch)
    stub_samples(mock, repo, improved_rounds(), baseline_rounds())
    return mock


@pytest.fixture
def supervised_repo(
    repo: str, monkeypatch: pytest.MonkeyPatch, improved_samples_mock: CollectSamplesRecorder
) -> str:
    """An open session with a before hook under a live budget, run from the shell."""
    header = iterate_session_header(repo)
    write_session_log(repo, header)
    # The before hook runs in the experiment worktree, so it must exist for the hook to fire.
    Path(header.worktrees.experiment).mkdir()
    marker = Path(repo, "hook-ran")
    write_bench_config(repo, hooks={"before": f"touch '{marker}'"})
    install_budget(repo, monkeypatch)
    return repo


@pytest.mark.parametrize("output_format", ["text", "json"])
def test_iterate_command_when_supervised_run_live_does_refuse_without_running_and_record_it(
    supervised_repo: str, output_format: str, improved_samples_mock: CollectSamplesRecorder
):
    before = records_of_type(supervised_repo, CommandRecord, matching=False)

    result = runner.invoke(app, ["iterate", "--bench", "npm run bench", "--format", output_format])

    assert result.exit_code == 2
    assert result.stdout == ""
    stderr = " ".join(stripped_lines(result.stderr, keep_blank=False))
    assert SUPERVISED_MESSAGE in stderr
    assert SUPERVISED_HINT in stderr
    assert improved_samples_mock.call_count == 0
    assert not Path(supervised_repo, "hook-ran").exists()
    assert records_of_type(supervised_repo, CommandRecord, matching=False) == before
    assert not Path(progress_path(supervised_repo)).exists()
    commands = records_of_type(supervised_repo, CommandRecord, matching=True)
    assert len(commands) == 1
    cmd = last_command_record(supervised_repo)
    assert cmd.name == "iterate"
    assert cmd.exit_code == 2
    assert cmd.reason == "supervised-use-tool"
    assert cmd.origin == "cli"
    assert cmd.seq is None


def test_iterate_command_when_tool_hosted_under_live_budget_does_run_the_before_hook(
    supervised_repo: str, monkeypatch: pytest.MonkeyPatch
):
    set_origin(monkeypatch, "tool")

    runner.invoke(app, ["iterate", "--bench", "npm run bench"])

    assert Path(supervised_repo, "hook-ran").exists()


def _unsettled(repo: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """An open session whose last iteration was never kept or discarded."""
    write_session_log(repo, iterate_session_header(repo), (iteration_record(seq=1),))
    write_bench_config(repo)
    install_budget(repo, monkeypatch)


def _stop_condition_met(repo: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """A settled session that already reached its configured iteration cap."""
    write_session_log(
        repo, iterate_session_header(repo), (iteration_record(seq=1), committed_keep(1))
    )
    write_bench_config(repo, stop={"max_iterations": 1})
    install_budget(repo, monkeypatch)


def _budget_exceeded(repo: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """A settled session whose last iteration outlasts the 5 minutes the budget has left."""
    write_session_log(
        repo,
        iterate_session_header(repo),
        (iteration_record(seq=1, duration_ms=840_000), committed_keep(1)),
    )
    write_bench_config(repo)
    install_tight_budget(repo, monkeypatch)


#: Every readiness setup, the ``(exit_code, reason, seq)`` the tool-hosted run
#: records for it, and a phrase of the refusal it prints.
READINESS = [
    pytest.param(
        _unsettled, (2, "unsettled", 1), "Iteration 1 has not been settled", id="unsettled"
    ),
    pytest.param(
        _stop_condition_met,
        (1, "stop-condition", 1),
        "Stop condition met: max iterations (1 of 1)",
        id="stop-condition",
    ),
    pytest.param(
        _budget_exceeded,
        (1, "budget-exceeded", 1),
        "the cap would cut this one off",
        id="budget-exceeded",
    ),
]


@pytest.mark.parametrize(("setup", "expected", "phrase"), READINESS)
def test_iterate_command_when_tool_hosted_in_unready_state_does_refuse_with_its_own_reason(
    *,
    repo: str,
    monkeypatch: pytest.MonkeyPatch,
    setup: Callable[[str, pytest.MonkeyPatch], None],
    expected: tuple[int, str, int],
    phrase: str,
    improved_samples_mock: CollectSamplesRecorder,
):
    setup(repo, monkeypatch)
    set_origin(monkeypatch, "tool")

    result = runner.invoke(app, ["iterate"])

    cmd = last_command_record(repo)
    assert (result.exit_code, cmd.reason, cmd.seq) == expected
    assert phrase in " ".join(stripped_lines(result.stderr, keep_blank=False))
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
    assert SUPERVISED_MESSAGE in " ".join(stripped_lines(result.stderr, keep_blank=False))
    assert last_command_record(repo).reason == "supervised-use-tool"
    assert improved_samples_mock.call_count == 0
