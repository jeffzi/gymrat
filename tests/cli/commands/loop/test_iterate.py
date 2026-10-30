"""Iterate command tests: basic execution, progress renderer wiring, format flags, and budget.

Budget tests verify that a live budget appends a time-left line to text output
and inserts a ``budget`` key in JSON output, including on stop-condition exits.
"""

import itertools
import json
import shlex
import signal
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path
from typing import Any
from unittest.mock import create_autospec

import pytest
from rich.console import Console
from syrupy.assertion import SnapshotAssertion

from gymrat.cli.app import app
from gymrat.cli.console import stderr_console
from gymrat.cli.iterate.progress import IterateRenderer
from gymrat.progress_events import PassFinished, PassStarted, ProgressEvent
from gymrat.sampling import SamplingOptions
from gymrat.session.paths import progress_path
from gymrat.session.progress_file import SidecarWriter
from tests._ansi import (
    strip_ansi,
    stripped_lines,
    warning_lines,
)
from tests._clock import install_monotonic_clock
from tests._platform import needs_posix_signals
from tests._process_helpers import wait_for_pid_file_blocking
from tests._rich import Clock, frame_text, sealed_console
from tests.cli._command_stubs import (
    force_render_mode,
)
from tests.cli._origin import set_origin
from tests.cli._runner import (
    runner,
)
from tests.cli._session import (
    open_capped_session_under_budget,
    write_bench_config,
    write_capped_session,
)
from tests.cli._signalled_cli import (
    SETTLE_TIMEOUT_S,
    pid_recording_script,
    spawned_gymrat,
    stop_by_signal,
)
from tests.loop._settle import start_with
from tests.loop.iterate._fixtures import (
    CollectSamplesRecorder,
    install_improved_samples,
    run_before_each_call,
    write_iterate_session,
)
from tests.session._budget import install_budget_with_ten_minutes_left
from tests.session.records._fixtures import (
    last_command_record,
)

# ---------------------------------------------------------------------------
# the iterate command
# ---------------------------------------------------------------------------


def test_iterate_command_when_run_does_trace_the_measured_iteration_seq(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    write_iterate_session(repo)
    install_improved_samples(monkeypatch, repo)

    result = runner.invoke(app, ["iterate", "--bench", "npm run bench", "--samples", "10"])

    assert result.exit_code == 0
    assert last_command_record(repo).seq == 1


def test_iterate_command_when_run_ends_does_remove_the_progress_sidecar(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    write_iterate_session(repo)
    install_improved_samples(monkeypatch, repo)
    sidecar = Path(progress_path(repo))
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_text("{}", encoding="utf-8")

    result = runner.invoke(app, ["iterate", "--bench", "npm run bench"])

    assert result.exit_code == 0
    assert not sidecar.exists()


# ---------------------------------------------------------------------------
# the iterate command — progress renderer wiring
# ---------------------------------------------------------------------------


#: Rows of the terminal the live-mode tests render the dashboard into.
_LIVE_SCREEN_HEIGHT = 40


def _three_second_pass(clock: Clock[float]) -> Callable[[SamplingOptions], Awaitable[None]]:
    """A sampling hook that reports one pass taking three seconds on ``clock``."""

    async def run_pass(options: SamplingOptions) -> None:
        assert options.on_progress is not None
        pass_fields = {"round": 1, "total_rounds": 1, "target_count": 2, "label": "experiment"}
        options.on_progress(PassStarted(**pass_fields, at_ms=clock()))
        clock.tick(3000.0)
        options.on_progress(PassFinished(**pass_fields, at_ms=clock()))

    return run_pass


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


def test_iterate_command_when_live_does_render_the_session_dashboard(
    repo: str, monkeypatch: pytest.MonkeyPatch, snapshot: SnapshotAssertion
):
    write_iterate_session(repo)
    _install_live_console(monkeypatch)
    renderers = _capture_renderers(monkeypatch)
    # The bench pass takes three seconds on the clock the renderer reads its timings through.
    clock = install_monotonic_clock(monkeypatch)
    run_before_each_call(
        monkeypatch, install_improved_samples(monkeypatch, repo), _three_second_pass(clock)
    )

    result = runner.invoke(app, ["iterate", "--bench", "npm run bench"])

    (renderer,) = renderers
    assert result.exit_code == 0
    assert frame_text(renderer.frame()) == snapshot


async def _raise_bench_exploded(_options: SamplingOptions) -> None:
    """Fail the sampling call the way a crashing bench would."""
    msg = "bench exploded"
    raise RuntimeError(msg)


def test_iterate_command_when_error_does_stop_the_live_display(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    write_iterate_session(repo)
    _install_live_console(monkeypatch)
    renderers = _capture_renderers(monkeypatch)
    run_before_each_call(
        monkeypatch, install_improved_samples(monkeypatch, repo), _raise_bench_exploded
    )

    result = runner.invoke(app, ["iterate", "--bench", "npm run bench"])

    (renderer,) = renderers
    assert (result.exit_code, renderer.live) == (2, None)


@needs_posix_signals
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
    """A stand-in progress sidecar writer raising the next ``OSError`` per event, cycling."""

    def __init__(self, messages: Sequence[str]) -> None:
        self._errors = (OSError(message) for message in itertools.cycle(messages))

    def __call__(self, _event: ProgressEvent) -> None:
        raise next(self._errors)


def _wire_failing_subscriber(
    repo: str, monkeypatch: pytest.MonkeyPatch, messages: Sequence[str]
) -> None:
    """Wire ``iterate`` with a sidecar raising ``messages`` for each event, in order."""
    write_iterate_session(repo)
    install_improved_samples(monkeypatch, repo)
    monkeypatch.setattr(
        "gymrat.cli.commands.loop.SidecarWriter",
        create_autospec(SidecarWriter, return_value=_FailingSidecar(messages)),
    )


_THREE_DISK_FULL = ("disk full", "disk full", "disk full")


def test_iterate_command_when_debug_and_subscriber_raises_does_follow_the_warning_with_traceback(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    _wire_failing_subscriber(repo, monkeypatch, _THREE_DISK_FULL)

    result = runner.invoke(app, ["iterate", "--bench", "npm run bench", "--debug"])

    lines = strip_ansi(result.stderr).splitlines()
    warning_at = lines.index("warning: disk full")
    assert lines[warning_at + 1] == "Traceback (most recent call last):"


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
    _wire_failing_subscriber(repo, monkeypatch, messages)

    result = runner.invoke(app, ["iterate", "--bench", "npm run bench"])

    assert warning_lines(result.stderr) == warnings
    assert "Traceback" not in result.stderr
    assert result.exit_code == 0


# ---------------------------------------------------------------------------
# iterate --format json
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("improved_samples_mock")
def test_iterate_command_when_format_json_and_stop_condition_does_emit_stop_document_with_budget(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    write_capped_session(repo)
    install_budget_with_ten_minutes_left(repo, monkeypatch)
    set_origin(monkeypatch, "tool")

    result = runner.invoke(app, ["iterate", "--bench", "npm run bench", "--format", "json"])

    assert result.exit_code == 1
    assert json.loads(result.stdout)["budget"] == {"cap_minutes": 30, "remaining_seconds": 600}


# ---------------------------------------------------------------------------
# the iterate command — run from the tool under a live budget
# ---------------------------------------------------------------------------


@pytest.fixture
def improved_samples_mock(repo: str, monkeypatch: pytest.MonkeyPatch) -> CollectSamplesRecorder:
    """The stubbed bench, answering every sampling call with an improved run."""
    return install_improved_samples(monkeypatch, repo)


@pytest.mark.usefixtures("improved_samples_mock")
def test_iterate_command_when_tool_hosted_and_stop_condition_met_does_refuse_with_the_budget_trailer(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    open_capped_session_under_budget(repo, monkeypatch)
    set_origin(monkeypatch, "tool")

    result = runner.invoke(app, ["iterate"])

    cmd = last_command_record(repo)
    assert (result.exit_code, cmd.reason, cmd.seq) == (1, "stop-condition", 1)
    assert "left of 30m" in " ".join(stripped_lines(result.stderr, keep_blank=False))
