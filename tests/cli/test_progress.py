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
from typing import TYPE_CHECKING, Literal

import pytest

from gymrat.cli.progress import ProgressReporter, ProgressState, RunShape, advance, plain_line
from gymrat.progress_events import (
    HookStarted,
    PrepareFinished,
    PrepareStarted,
)
from tests._rich import (
    Clock,
    console_output,
    frame_text,
    screen_lines,
    sealed_console,
    track,
)
from tests.cli._progress_helpers import ms_from_clock as _ms
from tests.cli._progress_helpers import pass_finished as _pass_finished
from tests.cli._progress_helpers import pass_started as _pass_started
from tests.cli._progress_helpers import report_full_pass

if TYPE_CHECKING:
    from rich.console import Console
    from syrupy.assertion import SnapshotAssertion

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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
        shape=RunShape(
            target_count=target_count,
            sample_count=sample_count,
            command=command,
            target_labels=tuple(target_labels or ()),
        ),
        clock=clock,
    )
    return console, clock, track(reporter)


def _summary_line(console: Console) -> str:
    """The last visible line of the rendered screen, or '' if nothing was printed."""
    visible = screen_lines(console_output(console))
    return visible[-1] if visible else ""


def _run_two_passes(reporter: ProgressReporter, clock: Clock[float]) -> None:
    """Drive prepare plus two full 2-sample passes to completion.

    The shared "measure done" setup behind the summary-line tests.
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


def test_start_when_sample_count_known_does_set_the_total_per_target():
    result = ProgressState.start(target_count=2, sample_count=3)

    assert result.total == 6


# ---------------------------------------------------------------------------
# advance -- prepare row
# ---------------------------------------------------------------------------


def test_advance_when_prepare_started_does_show_prepare_row_with_label(state: ProgressState):
    result = advance(state, PrepareStarted(label="bench", at_ms=7_000))

    assert (result.prepare_visible, result.current_target, result.prepare_start_ms) == (
        True,
        "bench",
        7_000,
    )


def test_advance_when_first_event_does_anchor_the_run_clock(state: ProgressState):
    result = advance(state, PrepareStarted(label="bench", at_ms=7_000))

    assert (result.run_start_ms, result.run_end_ms) == (7_000, 7_000)


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

    assert result.total == 10


def test_advance_when_pass_started_and_total_known_does_keep_total(state: ProgressState):
    first = advance(state, _pass_started(1, 5, target_count=2, at_ms=2_000))

    result = advance(first, _pass_started(2, 7, target_count=2, at_ms=14_000))

    assert result.total == 10


# ---------------------------------------------------------------------------
# advance -- ETA
# ---------------------------------------------------------------------------


def test_advance_when_pass_finished_does_advance_eta_by_pass_duration(state: ProgressState):
    running = advance(state, _pass_started(1, 3, at_ms=2_000))

    result = advance(running, _pass_finished(1, 3, at_ms=12_000))

    assert (result.eta.completed, result.eta.total_time_ms) == (1, 10_000)


# ---------------------------------------------------------------------------
# advance -- purity
# ---------------------------------------------------------------------------


def test_advance_when_unrelated_event_does_return_state_unchanged(state: ProgressState):
    result = advance(state, HookStarted(stage="before", at_ms=7_000))

    assert result == state


# ---------------------------------------------------------------------------
# plain_line
# ---------------------------------------------------------------------------


def test_plain_line_when_event_is_not_a_milestone_does_return_none(state: ProgressState):
    event = _pass_started(1, 3, at_ms=7_000)

    assert plain_line(state, advance(state, event), event) is None


# ---------------------------------------------------------------------------
# Frame golden snapshots via frame_text(reporter.frame())
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("manual_refresh")
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


@pytest.mark.usefixtures("manual_refresh")
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


@pytest.mark.usefixtures("manual_refresh")
def test_frame_when_multi_target_compare_does_name_running_target(
    snapshot: SnapshotAssertion,
):
    _console, clock, reporter = _reporter("live", target_count=2, sample_count=5)
    reporter.report(PrepareFinished(label="main", at_ms=0))
    clock.tick(1)
    reporter.report(_pass_started(1, 5, target_count=2, label="candidate", at_ms=_ms(clock)))

    result = frame_text(reporter.frame())

    assert result == snapshot


@pytest.mark.usefixtures("manual_refresh")
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
@pytest.mark.usefixtures("manual_refresh")
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
# Summary line -- exact via injected clock
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("manual_refresh")
def test_stop_when_measure_done_does_print_summary():
    console, clock, reporter = _reporter("live", sample_count=2)
    _run_two_passes(reporter, clock)

    reporter.stop()

    assert _summary_line(console) == "measured in 23s"


def test_stop_when_plain_mode_does_print_only_the_milestone_lines():
    console, clock, reporter = _reporter("plain", sample_count=2)
    _run_two_passes(reporter, clock)

    reporter.stop()

    assert console_output(console) == (
        "[00:00:01] prepared bench (1s)\n"
        "[00:00:12] pass 1/2 · bench (10s)\n"
        "[00:00:23] pass 2/2 · bench (10s)\n"
    )


@pytest.mark.usefixtures("manual_refresh")
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
