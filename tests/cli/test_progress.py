"""Tests for the rich-based progress renderer (live + plain modes).

Tests inject a deterministic ``Clock`` from ``tests._rich`` and capture
output through ``sealed_console``.  Frame content is pinned with syrupy
golden snapshots; plain-mode milestones use exact-line equality; live wiring
assertions check ``Live`` attributes directly.

The pure reducer behind the bar (``advance``, ``plain_line``) is driven
terminal-free: ``advance`` takes ``now`` from the event's own ``at_ms``, so a
state transition is fully determined by ``(state, event)``.
"""

from __future__ import annotations

import itertools
import sys
from dataclasses import dataclass
from io import StringIO
from typing import TYPE_CHECKING, Any, Literal, Protocol, cast
from unittest.mock import patch

import pytest

from gymrat.cli.live_display import LIVE_REFRESH_PER_SECOND, ErasableLive
from gymrat.cli.progress import ProgressReporter, ProgressState, advance, plain_line
from gymrat.progress_events import (
    HookStarted,
    PrepareFinished,
    PrepareStarted,
)
from gymrat.signals import install_termination_cleanup
from tests._rich import (
    KEPT_LINE,
    TERMINATION_SIGNAL,
    Clock,
    console_output,
    frame_text,
    screen_lines,
    sealed_console,
    track,
)
from tests.cli._progress_helpers import iterate_renderer, report_full_pass
from tests.cli._progress_helpers import ms_from_clock as _ms
from tests.cli._progress_helpers import pass_finished as _pass_finished
from tests.cli._progress_helpers import pass_started as _pass_started
from tests.cli.supervise._fixtures import (
    LIVE_CLASS_PATH,
    dashboard_console_patch,
    launch_event,
    make_reporter,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from rich.console import Console
    from syrupy.assertion import SnapshotAssertion

    from gymrat.cli.supervise.progress import SuperviseReporter

    RendererFactory = Callable[[Literal["live", "plain"], Console], "LiveRenderer"]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class LiveRenderer(Protocol):
    """The surface every CLI progress renderer shares, as the signal tests drive it."""

    @property
    def live(self) -> ErasableLive | None:
        """The live display it mounted, or ``None`` in plain mode."""
        ...

    def stop(self) -> None:
        """Stop the renderer."""
        ...


def build_progress_reporter(mode: Literal["live", "plain"], console: Console) -> LiveRenderer:
    """Build the measure/compare progress reporter on ``console``, showing its first event.

    Args:
        mode: ``"live"`` for a rich live display, ``"plain"`` for milestone lines.
        console: The console to render to.

    Returns:
        A single-target reporter with a hand-advanced clock.
    """
    reporter = track(
        ProgressReporter(
            mode=mode,
            console=console,
            target_count=1,
            sample_count=3,
            clock=Clock(0.0),
            command="measure",
        )
    )
    reporter.report(PrepareStarted(label="bench", at_ms=0))
    return reporter


def build_iterate_renderer(mode: Literal["live", "plain"], console: Console) -> LiveRenderer:
    """Build the iterate progress renderer on ``console``, showing its first event.

    Args:
        mode: ``"live"`` for a rich live checklist, ``"plain"`` for milestone lines.
        console: The console to render to.

    Returns:
        A renderer for iteration 1 with a hand-advanced clock.
    """
    _console, _clock, renderer = iterate_renderer(mode, console=console)
    renderer.report(PrepareStarted(label="bench", at_ms=0))
    return renderer


@dataclass(frozen=True, slots=True)
class _ShownSupervise:
    """The supervise reporter paired with the live display it mounted."""

    reporter: SuperviseReporter
    live: ErasableLive | None

    def stop(self) -> None:
        self.reporter.stop()


def build_supervise_reporter(mode: Literal["live", "plain"], console: Console) -> LiveRenderer:
    """Build the supervise dashboard reporter on ``console``, showing the agent's launch.

    Args:
        mode: ``"live"`` for the rich dashboard, ``"plain"`` for status lines.
        console: The console the dashboard paints on.

    Returns:
        The reporter with the display it mounted, or none in plain mode.
    """
    lives: list[ErasableLive] = []

    def build_live(**kwargs: Any) -> ErasableLive:
        live = ErasableLive(**kwargs)
        lives.append(live)
        return live

    with (
        dashboard_console_patch(console),
        patch(LIVE_CLASS_PATH, autospec=True, side_effect=build_live),
    ):
        kit = make_reporter(mode=mode, plain_write=lambda _line: None)
    kit.reporter.observer(launch_event(1000))
    return _ShownSupervise(kit.reporter, lives[0] if lives else None)


def _reporter(
    mode: Literal["live", "plain"],
    *,
    width: int = 80,
    height: int = 24,
    target_count: int = 1,
    sample_count: int = 3,
    command: str | None = None,
    target_labels: list[str] | None = None,
) -> tuple[Console, Clock[float], ProgressReporter]:
    """Wire a progress reporter to a sealed console and a hand-advanced clock."""
    clock = Clock(0.0)
    console = sealed_console(width=width, height=height, get_time=clock)
    reporter = ProgressReporter(
        mode=mode,
        console=console,
        target_count=target_count,
        sample_count=sample_count,
        clock=clock,
        command=command,
        target_labels=target_labels,
    )
    return console, clock, track(reporter)


def _summary_line(console: Console) -> str:
    """The last visible line of the rendered screen, or '' if nothing was printed."""
    visible = screen_lines(console_output(console))
    return visible[-1] if visible else ""


def _run_two_passes(reporter: ProgressReporter, clock: Clock[float]) -> None:
    """Drive prepare plus two full 2-sample passes to completion.

    The shared "measure done" setup behind the summary-line tests: prepare,
    then rounds 1 and 2 of 2, each started and finished.
    """
    reporter.report(PrepareStarted(label="bench", at_ms=0))
    clock.tick(1)
    reporter.report(PrepareFinished(label="bench", at_ms=_ms(clock)))
    for round_num in (1, 2):
        clock.tick(1)
        report_full_pass(reporter, clock, round_num, 2, duration_s=10)


@pytest.fixture
def state() -> ProgressState:
    """A fresh single-target state whose sample count is unknown, with nothing running yet."""
    return ProgressState.start(target_count=1, sample_count=None)


# ---------------------------------------------------------------------------
# ProgressState.start
# ---------------------------------------------------------------------------


def test_start_when_sample_count_known_does_set_the_total_with_nothing_visible():
    result = ProgressState.start(target_count=2, sample_count=3)

    assert (
        result.total,
        result.prepare_visible,
        result.pass_visible,
        result.current_target,
        result.run_start_ms,
        result.run_end_ms,
    ) == (6, False, False, "", None, None)


# ---------------------------------------------------------------------------
# advance -- prepare row
# ---------------------------------------------------------------------------


def test_advance_when_prepare_started_does_show_prepare_row_with_label(state: ProgressState):
    result = advance(state, PrepareStarted(label="bench", at_ms=7_000))

    assert (
        result.prepare_visible,
        result.current_target,
        result.prepare_start_ms,
        result.run_start_ms,
        result.run_end_ms,
    ) == (True, "bench", 7_000, 7_000, 7_000)


def test_advance_when_prepare_finished_does_hide_prepare_row(state: ProgressState):
    started = advance(state, PrepareStarted(label="bench", at_ms=7_000))

    finished = advance(started, PrepareFinished(label="bench", at_ms=12_000))

    assert finished.prepare_visible is False


# ---------------------------------------------------------------------------
# advance -- pass row and total
# ---------------------------------------------------------------------------


def test_advance_when_pass_started_does_show_pass_row_with_running_target(state: ProgressState):
    result = advance(state, _pass_started(1, 3, label="candidate", at_ms=2_000))

    assert (result.pass_visible, result.current_target, result.pass_start_ms) == (
        True,
        "candidate",
        2_000,
    )


def test_advance_when_pass_started_and_total_unknown_does_set_total_from_event(
    state: ProgressState,
):
    result = advance(state, _pass_started(1, 5, target_count=2, at_ms=2_000))

    assert (result.total, result.eta.total) == (10, 10)


def test_advance_when_pass_started_and_total_known_does_keep_total(state: ProgressState):
    first = advance(state, _pass_started(1, 5, target_count=2, at_ms=2_000))

    result = advance(first, _pass_started(2, 7, target_count=2, at_ms=14_000))

    assert (result.total, result.eta.total) == (10, 10)


# ---------------------------------------------------------------------------
# advance -- ETA
# ---------------------------------------------------------------------------


def test_advance_when_pass_finished_does_advance_eta_by_pass_duration(state: ProgressState):
    running = advance(state, _pass_started(1, 3, at_ms=2_000))

    result = advance(running, _pass_finished(1, 3, at_ms=12_000))

    assert (result.eta.completed, result.eta.total_time_ms, result.eta.total) == (1, 10_000, 3)


# ---------------------------------------------------------------------------
# advance -- purity
# ---------------------------------------------------------------------------


def test_advance_when_unrelated_event_does_return_state_unchanged(state: ProgressState):
    result = advance(state, HookStarted(stage="before", at_ms=7_000))

    assert result == state


# ---------------------------------------------------------------------------
# plain_line
# ---------------------------------------------------------------------------


def test_plain_line_when_multi_target_pass_finished_does_name_the_target(state: ProgressState):
    before = advance(state, _pass_started(2, 5, target_count=2, label="candidate", at_ms=2_000))
    event = _pass_finished(2, 5, target_count=2, label="candidate", at_ms=62_000)

    result = plain_line(before, advance(before, event), event)

    assert result == "pass 2/5 · candidate (1m 0s)"


@pytest.mark.parametrize(
    "event",
    [
        pytest.param(PrepareStarted(label="bench", at_ms=7_000), id="prepare-started"),
        pytest.param(_pass_started(1, 3, at_ms=7_000), id="pass-started"),
        pytest.param(HookStarted(stage="before", at_ms=7_000), id="hook-started"),
    ],
)
def test_plain_line_when_event_is_not_a_milestone_does_return_none(
    state: ProgressState, event: PrepareStarted | HookStarted
):
    assert plain_line(state, advance(state, event), event) is None


# ---------------------------------------------------------------------------
# Frame golden snapshots via frame_text(reporter.frame())
# ---------------------------------------------------------------------------


def test_frame_when_prepare_running_does_show_spinner_and_label(
    snapshot: SnapshotAssertion,
):
    _console, _clock, reporter = _reporter("live")
    reporter.report(PrepareStarted(label="bench", at_ms=0))

    result = frame_text(reporter.frame())

    assert result == snapshot


def test_frame_when_prepare_done_and_first_pass_running_does_show_pending_eta(
    snapshot: SnapshotAssertion,
):
    _console, clock, reporter = _reporter("live")
    reporter.report(PrepareStarted(label="bench", at_ms=0))
    clock.tick(2)
    reporter.report(PrepareFinished(label="bench", at_ms=_ms(clock)))
    clock.tick(1)
    reporter.report(_pass_started(1, 3, at_ms=_ms(clock)))

    result = frame_text(reporter.frame())

    assert result == snapshot


def test_frame_when_mid_run_with_computed_eta_does_show_clock_total(
    snapshot: SnapshotAssertion,
):
    _console, clock, reporter = _reporter("live")
    reporter.report(PrepareFinished(label="bench", at_ms=0))
    clock.tick(1)
    report_full_pass(reporter, clock, 1, 3, duration_s=10)
    clock.tick(1)
    reporter.report(_pass_started(2, 3, at_ms=_ms(clock)))
    clock.tick(41)

    result = frame_text(reporter.frame())

    assert result == snapshot


def test_frame_when_multi_target_compare_does_name_running_target(
    snapshot: SnapshotAssertion,
):
    _console, clock, reporter = _reporter("live", target_count=2, sample_count=5)
    reporter.report(PrepareFinished(label="main", at_ms=0))
    clock.tick(1)
    reporter.report(_pass_started(1, 5, target_count=2, label="candidate", at_ms=_ms(clock)))

    result = frame_text(reporter.frame())

    assert result == snapshot


def test_frame_when_compact_layout_on_short_console_does_show_single_row(
    snapshot: SnapshotAssertion,
):
    _console, clock, reporter = _reporter("live", height=10)
    reporter.report(PrepareFinished(label="bench", at_ms=0))
    clock.tick(1)
    reporter.report(_pass_started(1, 3, at_ms=_ms(clock)))

    result = frame_text(reporter.frame())

    assert result == snapshot


# ---------------------------------------------------------------------------
# Header line -- command name, target labels, sample count
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("command", "target_labels"),
    [
        pytest.param("measure", ["ecstatic-ts"], id="measure"),
        pytest.param("compare", ["main", "candidate"], id="compare"),
    ],
)
def test_frame_when_command_given_does_show_header_with_command_and_labels(
    command: str, target_labels: list[str], snapshot: SnapshotAssertion
):
    _console, _clock, reporter = _reporter(
        "live",
        command=command,
        target_labels=target_labels,
        target_count=len(target_labels),
        sample_count=5,
    )
    reporter.report(PrepareStarted(label=target_labels[0], at_ms=0))

    result = frame_text(reporter.frame())

    assert result == snapshot


# ---------------------------------------------------------------------------
# Plain mode -- exact timestamped milestone lines
# ---------------------------------------------------------------------------


def test_report_when_plain_mode_prepare_finished_does_print_exact_timestamped_line():
    console, clock, reporter = _reporter("plain")
    clock.tick(7)
    reporter.report(PrepareStarted(label="bench", at_ms=_ms(clock)))
    clock.tick(5)

    reporter.report(PrepareFinished(label="bench", at_ms=_ms(clock)))

    assert console_output(console) == "[00:00:05] prepared bench (5s)\n"


# ---------------------------------------------------------------------------
# Live wiring -- Live attributes and refresh path
# ---------------------------------------------------------------------------


def test_progress_reporter_when_built_in_live_mode_does_mount_a_transient_auto_refreshing_live():
    # redirect_stderr=False leaves sys.stderr as the real stream: this display
    # never swaps in rich's FileProxy, so there is nothing for the erase to
    # restore.
    real_stderr = sys.stderr

    _console, _clock, reporter = _reporter("live", command="measure", target_labels=["bench"])

    live = reporter.live
    assert live is not None
    assert (live.transient, live.auto_refresh, live.refresh_per_second) == (
        True,
        True,
        LIVE_REFRESH_PER_SECOND,
    )
    assert sys.stderr is real_stderr
    assert frame_text(live.get_renderable()) == frame_text(reporter.frame())


def test_warn_when_live_mode_does_print_through_the_console_while_the_frame_is_up(
    monkeypatch: pytest.MonkeyPatch,
):
    console, _clock, reporter = _reporter("live")
    reporter.report(PrepareStarted(label="bench", at_ms=0))
    live = cast("ErasableLive", reporter.live)
    printed: list[tuple[str, bool]] = []
    real_print = console.print

    def recording_print(*objects: object, **kwargs: Any) -> None:
        printed.extend((text, live.is_started) for text in objects if isinstance(text, str))
        real_print(*objects, **kwargs)

    monkeypatch.setattr(console, "print", recording_print)

    reporter.warn("warning: lat:100:p99 disk full")

    assert printed == [("warning: lat:100:p99 disk full", True)]


def test_warn_when_plain_mode_does_print_the_message_verbatim_on_its_own_line():
    console, _clock, reporter = _reporter("plain")
    before = console_output(console)

    reporter.warn("warning: cannot write [/tmp/lat:100:p99.json]")

    assert console_output(console) == before + "warning: cannot write [/tmp/lat:100:p99.json]\n"


def test_stop_when_live_does_clear_live():
    _console, _clock, reporter = _reporter("live")
    reporter.report(PrepareStarted(label="bench", at_ms=0))

    reporter.stop()

    assert reporter.live is None


# ---------------------------------------------------------------------------
# Summary line -- exact via injected clock
# ---------------------------------------------------------------------------


def test_stop_when_measure_done_does_print_summary():
    console, clock, reporter = _reporter("live", sample_count=2)
    _run_two_passes(reporter, clock)

    reporter.stop()

    assert _summary_line(console) == "measured in 23s"


def test_stop_when_plain_mode_does_not_print_summary():
    console, clock, reporter = _reporter("plain", sample_count=2)
    _run_two_passes(reporter, clock)

    reporter.stop()

    assert console_output(console) == (
        "[00:00:01] prepared bench (1s)\n"
        "[00:00:12] pass 1/2 · bench (10s)\n"
        "[00:00:23] pass 2/2 · bench (10s)\n"
    )


def test_stop_when_compare_done_does_print_summary():
    console, clock, reporter = _reporter("live", target_count=2, sample_count=2)
    reporter.report(PrepareStarted(label="main", at_ms=0))
    clock.tick(1)
    reporter.report(PrepareFinished(label="main", at_ms=_ms(clock)))
    for rnd, label in itertools.product((1, 2), ("main", "candidate")):
        clock.tick(1)
        report_full_pass(reporter, clock, rnd, 2, duration_s=10, target_count=2, label=label)

    reporter.stop()

    assert _summary_line(console) == "compared in 45s"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_report_when_console_width_zero_does_render_as_plain(
    monkeypatch: pytest.MonkeyPatch,
):
    # rich prints nothing on a zero-width console, so the milestone is read
    # from what the reporter handed the console rather than from its buffer.
    console, _clock, reporter = _reporter("live", width=0)
    printed: list[object] = []
    real_print = console.print

    def recording_print(*objects: object, **kwargs: Any) -> None:
        printed.extend(objects)
        real_print(*objects, **kwargs)

    monkeypatch.setattr(console, "print", recording_print)
    reporter.report(PrepareStarted(label="bench", at_ms=0))

    reporter.report(PrepareFinished(label="bench", at_ms=1000))

    assert (reporter.live, printed) == (None, ["[00:00:01] prepared bench (1s)"])


def test_report_when_plain_label_looks_like_markup_does_print_it_verbatim():
    console, _clock, reporter = _reporter("plain")
    reporter.report(PrepareStarted(label="[bold]bench[/bold]", at_ms=0))

    reporter.report(PrepareFinished(label="[bold]bench[/bold]", at_ms=1000))

    assert console_output(console) == "[00:00:01] prepared [bold]bench[/bold] (1s)\n"


# ---------------------------------------------------------------------------
# Termination signal -- every CLI progress renderer erases its live display alike
# ---------------------------------------------------------------------------


# The renderers built on LiveDisplayMixin, whose stop() does nothing once a
# signal has erased the display.
_MIXIN_RENDERERS = [
    pytest.param(build_progress_reporter, id="progress-reporter"),
    pytest.param(build_iterate_renderer, id="iterate-renderer"),
]


@pytest.fixture(
    params=[
        *_MIXIN_RENDERERS,
        pytest.param(build_supervise_reporter, id="supervise-reporter"),
    ]
)
def build_renderer(request: pytest.FixtureRequest) -> Iterator[RendererFactory]:
    """Build each CLI progress renderer in turn; stops whatever was built, even after a failure."""
    built: list[LiveRenderer] = []

    def build(mode: Literal["live", "plain"], console: Console) -> LiveRenderer:
        renderer = request.param(mode, console)
        built.append(renderer)
        return renderer

    yield build
    for renderer in built:
        # A termination signal marks the renderer stopped (as os._exit would
        # follow in production), so stop() is then a no-op; stop the Live
        # display directly so no refresh thread or console registration leaks.
        if renderer.live is not None:
            renderer.live.stop()
        renderer.stop()


def test_signal_when_live_up_does_erase_only_the_frame(
    build_renderer: RendererFactory,
    monkeypatch: pytest.MonkeyPatch,
    raise_signal: Callable[[int], int],
):
    console = sealed_console()
    console.print(KEPT_LINE)
    monkeypatch.setattr(sys, "stderr", console.file)
    build_renderer("live", console)

    raise_signal(TERMINATION_SIGNAL)

    assert screen_lines(console_output(console)) == [KEPT_LINE]


@pytest.mark.parametrize("build_renderer", _MIXIN_RENDERERS, indirect=True)
def test_stop_when_signal_already_erased_the_display_does_write_nothing(
    build_renderer: RendererFactory,
):
    console = sealed_console()
    renderer = build_renderer("live", console)
    cast("ErasableLive", renderer.live).erase_for_exit()
    before = console_output(console)

    renderer.stop()

    assert console_output(console) == before


@pytest.mark.parametrize(
    ("mode", "stopped"),
    [
        pytest.param("live", True, id="live-renderer-stopped"),
        pytest.param("plain", False, id="plain-renderer-running"),
    ],
)
def test_signal_when_no_live_display_is_up_does_write_nothing(
    mode: Literal["live", "plain"],
    stopped: bool,
    build_renderer: RendererFactory,
    monkeypatch: pytest.MonkeyPatch,
    raise_signal: Callable[[int], int],
):
    renderer = build_renderer(mode, sealed_console())
    if stopped:
        renderer.stop()
    install_termination_cleanup(lambda: None)
    buf = StringIO()
    monkeypatch.setattr(sys, "stderr", buf)

    raise_signal(TERMINATION_SIGNAL)

    assert buf.getvalue() == ""
