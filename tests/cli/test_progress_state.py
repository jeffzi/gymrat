"""Behavioral tests for the pure progress reducer behind the measure/compare bar.

Everything here is terminal-free: no ``rich`` import, no ``Console``, no clock.
``advance`` takes ``now`` from the event's own ``at_ms``, so a state transition
is fully determined by ``(state, event)``.
"""

from __future__ import annotations

import pytest

from gymrat.cli.progress_state import ProgressState, advance, plain_line
from gymrat.progress_events import (
    HookStarted,
    PrepareFinished,
    PrepareStarted,
)
from tests.cli._progress_helpers import (
    pass_finished as _pass_finished,
)
from tests.cli._progress_helpers import (
    pass_started as _pass_started,
)


@pytest.fixture
def state() -> ProgressState:
    """A fresh single-target, three-sample state with nothing running yet."""
    return ProgressState(target_count=1, sample_count=3)


# ---------------------------------------------------------------------------
# ProgressState value semantics
# ---------------------------------------------------------------------------


def test_progress_state_when_freshly_built_does_start_with_nothing_visible(state: ProgressState):
    assert (
        state.total,
        state.prepare_visible,
        state.pass_visible,
        state.current_target,
        state.run_start_ms,
        state.run_end_ms,
    ) == (0, False, False, "", None, None)


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


def test_advance_when_prepare_finished_does_hide_prepare_row_and_move_the_run_window(
    state: ProgressState,
):
    started = advance(state, PrepareStarted(label="bench", at_ms=7_000))

    finished = advance(started, PrepareFinished(label="bench", at_ms=12_000))

    assert finished.prepare_visible is False
    assert (finished.run_start_ms, finished.run_end_ms) == (7_000, 12_000)


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

    result = advance(first, _pass_started(2, 5, target_count=2, at_ms=14_000))

    assert (result.total, result.eta.total) == (10, 10)


# ---------------------------------------------------------------------------
# advance -- ETA
# ---------------------------------------------------------------------------


def test_advance_when_pass_finished_does_advance_eta_by_pass_duration(state: ProgressState):
    running = advance(state, _pass_started(1, 3, at_ms=2_000))

    result = advance(running, _pass_finished(1, 3, at_ms=12_000))

    assert (result.eta.completed, result.eta.total_time_ms, result.eta.total) == (1, 10_000, 3)


# ---------------------------------------------------------------------------
# advance -- run window
# ---------------------------------------------------------------------------


def test_advance_when_first_event_does_set_run_start_and_run_end_to_event_time(
    state: ProgressState,
):
    result = advance(state, PrepareStarted(label="bench", at_ms=7_000))

    assert (result.run_start_ms, result.run_end_ms) == (7_000, 7_000)


# ---------------------------------------------------------------------------
# advance -- purity
# ---------------------------------------------------------------------------


def test_advance_when_unrelated_event_does_return_state_unchanged(state: ProgressState):
    result = advance(state, HookStarted(stage="before", at_ms=7_000))

    assert result == state


def test_advance_when_called_twice_with_same_inputs_does_return_equal_states(
    state: ProgressState,
):
    event = _pass_finished(1, 3, at_ms=12_000)
    running = advance(state, _pass_started(1, 3, at_ms=2_000))

    assert advance(running, event) == advance(running, event)


def test_advance_when_applied_does_not_mutate_input_state(state: ProgressState):
    advance(state, PrepareStarted(label="bench", at_ms=7_000))

    assert (state.prepare_visible, state.current_target, state.run_start_ms) == (False, "", None)


# ---------------------------------------------------------------------------
# plain_line
# ---------------------------------------------------------------------------


def test_plain_line_when_prepare_finished_does_return_prepared_line(state: ProgressState):
    before = advance(state, PrepareStarted(label="bench", at_ms=7_000))
    event = PrepareFinished(label="bench", at_ms=12_000)

    result = plain_line(before, advance(before, event), event)

    assert result == "prepared bench (5s)"


def test_plain_line_when_pass_finished_does_return_pass_line(state: ProgressState):
    before = advance(state, _pass_started(1, 3, at_ms=2_000))
    event = _pass_finished(1, 3, at_ms=22_000)

    result = plain_line(before, advance(before, event), event)

    assert result == "pass 1/3 · bench (20s)"


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
