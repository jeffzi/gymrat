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
from io import StringIO
from typing import TYPE_CHECKING, Any, Literal, Protocol, cast

import pytest

from gymrat.cli.live_display import LIVE_REFRESH_PER_SECOND
from gymrat.cli.progress import ProgressReporter, ProgressState, advance, plain_line
from gymrat.progress_events import (
    HookStarted,
    PrepareFinished,
    PrepareStarted,
)
from gymrat.signals import install_termination_cleanup
from tests._process_helpers import fake_install
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
from tests.cli._progress_helpers import iterate_renderer
from tests.cli._progress_helpers import ms_from_clock as _ms
from tests.cli._progress_helpers import pass_finished as _pass_finished
from tests.cli._progress_helpers import pass_started as _pass_started

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from rich.console import Console
    from syrupy.assertion import SnapshotAssertion

    from gymrat.cli.live_display import ErasableLive
    from gymrat.progress_events import ProgressEvent

    RendererFactory = Callable[[Literal["live", "plain"], Console], "LiveRenderer"]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class LiveRenderer(Protocol):
    """The surface both CLI progress renderers share, as the signal tests drive it."""

    @property
    def live(self) -> ErasableLive | None:
        """The active live display, or ``None`` outside live mode or after ``stop()``."""
        ...

    def report(self, event: ProgressEvent) -> None:
        """Fold ``event`` into the display."""
        ...

    def stop(self) -> None:
        """Stop the renderer."""
        ...


def build_progress_reporter(mode: Literal["live", "plain"], console: Console) -> LiveRenderer:
    """Build the measure/compare progress reporter on ``console``.

    Args:
        mode: ``"live"`` for a rich live display, ``"plain"`` for milestone lines.
        console: The console to render to.

    Returns:
        A single-target reporter with a hand-advanced clock.
    """
    return track(
        ProgressReporter(
            mode=mode,
            console=console,
            target_count=1,
            sample_count=3,
            clock=Clock(0.0),
            command="measure",
        )
    )


def build_iterate_renderer(mode: Literal["live", "plain"], console: Console) -> LiveRenderer:
    """Build the iterate progress renderer on ``console``.

    Args:
        mode: ``"live"`` for a rich live checklist, ``"plain"`` for milestone lines.
        console: The console to render to.

    Returns:
        A renderer for iteration 1 with a hand-advanced clock.
    """
    _console, _clock, renderer = iterate_renderer(mode, console=console)
    return renderer


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
        reporter.report(_pass_started(round_num, 2, at_ms=_ms(clock)))
        clock.tick(10)
        reporter.report(_pass_finished(round_num, 2, at_ms=_ms(clock)))


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
    reporter.report(_pass_started(1, 3, at_ms=_ms(clock)))
    clock.tick(10)
    reporter.report(_pass_finished(1, 3, at_ms=_ms(clock)))
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


@pytest.mark.parametrize(
    "run_start_s",
    [
        pytest.param(0, id="run-starts-at-zero"),
        pytest.param(7, id="run-starts-later"),
    ],
)
def test_plain_renderer_when_prepare_finished_does_print_exact_timestamped_line(
    run_start_s: int,
):
    console, clock, reporter = _reporter("plain")
    clock.tick(run_start_s)
    reporter.report(PrepareStarted(label="bench", at_ms=_ms(clock)))
    clock.tick(5)
    reporter.report(PrepareFinished(label="bench", at_ms=_ms(clock)))

    output = console_output(console)
    lines = [ln for ln in output.splitlines() if ln.strip()]

    assert lines[-1] == "[00:00:05] prepared bench (5s)"


def test_plain_renderer_when_pass_finished_does_print_exact_timestamped_line():
    console, clock, reporter = _reporter("plain", sample_count=3)
    reporter.report(PrepareFinished(label="bench", at_ms=0))
    reporter.report(_pass_started(1, 3, at_ms=0))
    clock.tick(20)
    reporter.report(_pass_finished(1, 3, at_ms=_ms(clock)))

    output = console_output(console)
    lines = [ln for ln in output.splitlines() if ln.strip()]

    assert lines[-1] == "[00:00:20] pass 1/3 · bench (20s)"


def test_plain_renderer_when_any_event_does_not_emit_ansi_codes():
    console, clock, reporter = _reporter("plain")
    reporter.report(PrepareStarted(label="bench", at_ms=0))
    clock.tick(1)
    reporter.report(PrepareFinished(label="bench", at_ms=_ms(clock)))
    clock.tick(1)
    reporter.report(_pass_started(1, 3, at_ms=_ms(clock)))
    clock.tick(10)
    reporter.report(_pass_finished(1, 3, at_ms=_ms(clock)))
    reporter.stop()

    output = console_output(console)

    assert "\x1b[" not in output


# ---------------------------------------------------------------------------
# Live wiring -- Live attributes and refresh path
# ---------------------------------------------------------------------------


def test_live_wiring_when_created_does_mount_a_transient_auto_refreshing_live_on_the_frame():
    # redirect_stderr=False leaves sys.stderr as the real stream: this display
    # never swaps in rich's FileProxy, so there is nothing for the erase to
    # restore.
    real_stderr = sys.stderr
    _console, _clock, reporter = _reporter("live", command="measure", target_labels=["bench"])
    reporter.report(PrepareStarted(label="bench", at_ms=0))

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


@pytest.mark.parametrize(
    ("mode", "registrations"),
    [
        pytest.param("live", 1, id="live-registers-once"),
        pytest.param("plain", 0, id="plain-registers-none"),
    ],
)
def test_reporter_when_created_does_register_termination_cleanup_only_in_live_mode(
    mode: Literal["live", "plain"], registrations: int, monkeypatch: pytest.MonkeyPatch
):
    registered: list[Callable[[], None]] = []
    monkeypatch.setattr(
        "gymrat.cli.live_display.install_termination_cleanup",
        fake_install(registered),
    )

    _console, _clock, reporter = _reporter(mode)
    reporter.stop()

    assert len(registered) == registrations


def test_stop_when_called_twice_does_clear_live():
    _console, _clock, reporter = _reporter("live")
    reporter.report(PrepareStarted(label="bench", at_ms=0))

    reporter.stop()
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

    assert "measured in" not in console_output(console)


def test_stop_when_compare_done_does_print_summary():
    console, clock, reporter = _reporter("live", target_count=2, sample_count=2)
    reporter.report(PrepareStarted(label="main", at_ms=0))
    clock.tick(1)
    reporter.report(PrepareFinished(label="main", at_ms=_ms(clock)))
    for rnd, label in itertools.product((1, 2), ("main", "candidate")):
        clock.tick(1)
        reporter.report(_pass_started(rnd, 2, target_count=2, label=label, at_ms=_ms(clock)))
        clock.tick(10)
        reporter.report(_pass_finished(rnd, 2, target_count=2, label=label, at_ms=_ms(clock)))

    reporter.stop()

    assert _summary_line(console) == "compared in 45s"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_live_renderer_when_console_width_zero_does_render_as_plain():
    console, _clock, reporter = _reporter("live", width=0)

    reporter.report(PrepareStarted(label="bench", at_ms=0))
    reporter.report(PrepareFinished(label="bench", at_ms=1000))

    assert reporter.live is None
    assert "\x1b[" not in console_output(console)


def test_plain_renderer_when_label_looks_like_markup_does_print_it_verbatim():
    console, _clock, reporter = _reporter("plain")

    reporter.report(PrepareStarted(label="[bold]bench[/bold]", at_ms=0))
    reporter.report(PrepareFinished(label="[bold]bench[/bold]", at_ms=1000))
    reporter.stop()

    assert "[00:00:01] prepared [bold]bench[/bold] (1s)" in console_output(console)


def test_reporter_when_non_relevant_event_does_silently_ignore():
    console, _clock, reporter = _reporter("plain")

    reporter.report(HookStarted(stage="before", at_ms=0))

    output = console_output(console)
    assert output == ""


# ---------------------------------------------------------------------------
# Termination signal -- every CLI progress renderer erases its live display alike
# ---------------------------------------------------------------------------


@pytest.fixture(
    params=[
        pytest.param(build_progress_reporter, id="progress-reporter"),
        pytest.param(build_iterate_renderer, id="iterate-renderer"),
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
    renderer = build_renderer("live", console)
    renderer.report(PrepareStarted(label="bench", at_ms=0))
    monkeypatch.setattr(sys, "stderr", console.file)

    raise_signal(TERMINATION_SIGNAL)

    assert screen_lines(console_output(console)) == [KEPT_LINE]


def test_stop_when_signal_already_erased_the_display_does_write_nothing(
    build_renderer: RendererFactory,
):
    console = sealed_console()
    renderer = build_renderer("live", console)
    renderer.report(PrepareStarted(label="bench", at_ms=0))
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
    renderer.report(PrepareStarted(label="bench", at_ms=0))
    if stopped:
        renderer.stop()
    install_termination_cleanup(lambda: None)
    buf = StringIO()
    monkeypatch.setattr(sys, "stderr", buf)

    raise_signal(TERMINATION_SIGNAL)

    assert buf.getvalue() == ""
