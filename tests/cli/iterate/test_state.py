"""Tests for the pure iterate checklist reducer.

The reducer owns every state transition the live checklist renders, so these
tests drive it directly: no terminal, no ``Console``, no ``rich`` objects.
Timestamps come from each event's own ``at_ms``, which keeps the transitions
deterministic without a clock.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from gymrat.cli.iterate.state import (
    IterateState,
    advance,
    initial_state,
    plain_line,
)
from gymrat.progress_events import (
    ConfirmFinished,
    ConfirmSkipped,
    ConfirmStarted,
    HookFinished,
    HookStarted,
    IterationRecorded,
    JudgeFinished,
    JudgeStarted,
    PrepareFinished,
    PrepareStarted,
)
from tests._imports import loaded_under, modules_imported_by
from tests.cli._progress_helpers import iteration_facts
from tests.cli._progress_helpers import pass_finished as _pass_finished
from tests.cli._progress_helpers import pass_started as _pass_started

if TYPE_CHECKING:
    from collections.abc import Sequence

    from gymrat.progress_events import ProgressEvent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _state(
    *,
    has_before_hook: bool = False,
    has_after_hook: bool = False,
    checks_cmd: str | None = None,
) -> IterateState:
    return initial_state(
        iteration_facts(
            sample_count=1,
            has_before_hook=has_before_hook,
            has_after_hook=has_after_hook,
            checks_cmd=checks_cmd,
        )
    )


def _apply(state: IterateState, *events: ProgressEvent) -> IterateState:
    for event in events:
        state = advance(state, event)
    return state


# A single round against two targets: the phase total is two passes.
_FIRST_PASS_STARTED = _pass_started(1, 1, target_count=2, label="baseline", at_ms=0)
_FIRST_PASS_FINISHED = _pass_finished(1, 1, target_count=2, label="baseline", at_ms=10000)
_LAST_PASS_STARTED = _pass_started(1, 1, target_count=2, label="experiment", at_ms=10000)
_LAST_PASS_FINISHED = _pass_finished(1, 1, target_count=2, label="experiment", at_ms=20000)


# ---------------------------------------------------------------------------
# Reducer shape: pure, no rich
# ---------------------------------------------------------------------------


def test_state_module_when_imported_in_a_fresh_interpreter_does_not_load_rich():
    loaded = modules_imported_by("gymrat.cli.iterate.state")

    assert loaded_under(loaded, "rich") == []


# ---------------------------------------------------------------------------
# Initial rows
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "has_after_hook",
    [
        pytest.param(True, id="after-hook"),
        pytest.param(False, id="no-after-hook"),
    ],
)
def test_initial_state_when_after_hook_toggled_does_hint_it_only_when_present(
    has_after_hook: bool,
):
    result = _state(has_after_hook=has_after_hook)

    assert bool(result.nodes.record.hint) is has_after_hook


def test_initial_state_when_no_before_hook_does_leave_the_hook_row_out():
    result = _state(has_before_hook=False)

    assert result.nodes.before_hook not in result.visible_nodes


# ---------------------------------------------------------------------------
# Row transitions
# ---------------------------------------------------------------------------


def test_advance_when_before_hook_starts_does_show_a_running_hook_row():
    result = advance(_state(has_before_hook=True), HookStarted(stage="before", at_ms=1000))

    assert result.nodes.before_hook in result.visible_nodes
    assert result.nodes.before_hook.status == "running"


def test_advance_when_before_hook_finished_does_mark_hook_row_done_with_its_elapsed_time():
    state = advance(_state(has_before_hook=True), HookStarted(stage="before", at_ms=1000))

    result = advance(state, HookFinished(stage="before", at_ms=3000))

    assert result.nodes.before_hook.status == "done"
    assert result.nodes.before_hook.elapsed_ms == 2000


def test_advance_when_after_hook_starts_does_leave_the_state_unchanged():
    state = _apply(
        _state(has_before_hook=True, has_after_hook=True),
        HookStarted(stage="before", at_ms=0),
        HookFinished(stage="before", at_ms=2000),
    )

    result = advance(state, HookStarted(stage="after", at_ms=5000))

    assert result == state


def test_advance_when_final_pass_finishes_does_complete_passes_row():
    state = _apply(_state(), _FIRST_PASS_STARTED, _FIRST_PASS_FINISHED, _LAST_PASS_STARTED)

    result = advance(state, _LAST_PASS_FINISHED)

    assert result.nodes.passes.status == "done"
    assert result.nodes.passes.detail == "2 passes"


def test_advance_when_two_worktrees_prepared_does_sum_their_elapsed_times():
    state = _apply(
        _state(),
        PrepareStarted(label="baseline", at_ms=0),
        PrepareFinished(label="baseline", at_ms=3000),
        PrepareStarted(label="experiment", at_ms=3000),
    )

    result = advance(state, PrepareFinished(label="experiment", at_ms=5000))

    assert result.nodes.prepare.elapsed_ms == 5000


def test_advance_when_prepare_finished_does_clear_the_target():
    state = advance(_state(), PrepareStarted(label="baseline", at_ms=0))

    result = advance(state, PrepareFinished(label="baseline", at_ms=3000))

    assert (state.nodes.prepare.target, result.nodes.prepare.target) == ("baseline", "")


def test_advance_when_judge_finished_after_started_does_time_the_judge_row():
    state = advance(_state(), JudgeStarted(at_ms=1000))

    result = advance(state, JudgeFinished(primary_delta_pct=-1.0, regressed=(), at_ms=5000))

    assert result.nodes.judge.elapsed_ms == 4000


def test_advance_when_confirm_pass_finished_does_count_it_in_the_confirm_phase():
    state = _apply(
        _state(),
        ConfirmStarted(filtered_metrics=None, at_ms=5000),
        _pass_started(1, 1, target_count=2, label="baseline", at_ms=5000, phase="confirm"),
    )

    result = advance(
        state, _pass_finished(1, 1, target_count=2, label="baseline", at_ms=10000, phase="confirm")
    )

    assert result.confirm_phase.eta.completed == 1


def test_advance_when_confirm_finished_does_time_the_confirm_row():
    state = advance(_state(), ConfirmStarted(filtered_metrics=None, at_ms=5000))

    result = advance(state, ConfirmFinished(reproduced=True, at_ms=20000))

    assert result.nodes.confirm.elapsed_ms == 15000


def test_advance_when_confirm_finished_does_settle_the_judge_alert():
    state = advance(_state(), ConfirmStarted(filtered_metrics=None, at_ms=5000))

    result = advance(state, ConfirmFinished(reproduced=True, at_ms=20000))

    assert (state.nodes.judge.alert, result.nodes.judge.alert) == (True, False)


@pytest.mark.parametrize(
    ("setup", "event", "shown"),
    [
        pytest.param(
            (),
            JudgeFinished(primary_delta_pct=-2.0, regressed=(), at_ms=6000),
            False,
            id="nothing-regressed-drops-it",
        ),
        pytest.param(
            (JudgeFinished(primary_delta_pct=-2.0, regressed=("latency",), at_ms=6000),),
            ConfirmSkipped(at_ms=6000),
            True,
            id="skipped-after-a-regression-keeps-it",
        ),
    ],
)
def test_advance_when_confirm_skipped_does_show_its_row_only_after_a_regression(
    setup: Sequence[ProgressEvent],
    event: ProgressEvent,
    shown: bool,
):
    state = _apply(_state(), *setup)

    result = advance(state, event)

    assert (result.nodes.confirm in result.visible_nodes) is shown


@pytest.mark.parametrize(
    ("filtered_metrics", "expected"),
    [
        pytest.param(None, "full suite", id="unfiltered"),
        pytest.param(("latency", "alloc"), "2 metrics", id="filtered"),
    ],
)
def test_advance_when_confirm_started_does_set_confirm_note(
    filtered_metrics: tuple[str, ...] | None, expected: str
):
    result = advance(_state(), ConfirmStarted(filtered_metrics=filtered_metrics, at_ms=5000))

    assert result.nodes.confirm.note == expected


# ---------------------------------------------------------------------------
# Plain-mode milestone lines
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("setup", "event", "expected"),
    [
        pytest.param(
            (PrepareStarted(label="baseline", at_ms=0),),
            PrepareFinished(label="baseline", at_ms=5000),
            "prepared baseline (5s)",
            id="prepare-finished",
        ),
        pytest.param(
            (_FIRST_PASS_STARTED, _FIRST_PASS_FINISHED, _LAST_PASS_STARTED),
            _LAST_PASS_FINISHED,
            "passes done (20s)",
            id="last-pass-finished",
        ),
        pytest.param(
            (),
            JudgeFinished(primary_delta_pct=-2.0, regressed=(), at_ms=6000),
            "judge -2.0% on geomean · no gating regression",
            id="judge-finished",
        ),
        pytest.param(
            (),
            JudgeFinished(primary_delta_pct=None, regressed=(), at_ms=6000),
            "judge — · no gating regression",
            id="judge-finished-missing-delta",
        ),
        pytest.param(
            (),
            JudgeFinished(primary_delta_pct=-6.8, regressed=("latency",), at_ms=6000),
            "judge -6.8% on geomean · 1 regressed: latency",
            id="judge-finished-one-regressed",
        ),
        pytest.param(
            (),
            JudgeFinished(
                primary_delta_pct=-6.8,
                regressed=("latency", "alloc", "throughput", "parse"),
                at_ms=6000,
            ),
            "judge -6.8% on geomean · 4 regressed: latency, alloc, throughput, …",
            id="judge-finished-more-regressed-than-cap",
        ),
        pytest.param(
            (ConfirmStarted(filtered_metrics=None, at_ms=5000),),
            ConfirmFinished(reproduced=True, at_ms=15000),
            "confirm 0/2 · regressions reproduced",
            id="confirm-finished",
        ),
        pytest.param(
            (ConfirmStarted(filtered_metrics=None, at_ms=5000),),
            ConfirmFinished(reproduced=False, at_ms=15000),
            "confirm 0/2 · regressions not reproduced",
            id="confirm-finished-not-reproduced",
        ),
        pytest.param(
            (),
            IterationRecorded(seq=2, outcome="improved", at_ms=15000),
            "recorded improved suggested",
            id="recorded",
        ),
    ],
)
def test_plain_line_when_milestone_event_does_return_line(
    setup: Sequence[ProgressEvent], event: ProgressEvent, expected: str
):
    before = _apply(_state(), *setup)

    result = plain_line(before, advance(before, event), event)

    assert result == expected


@pytest.mark.parametrize(
    ("setup", "event"),
    [
        pytest.param((_FIRST_PASS_STARTED,), _FIRST_PASS_FINISHED, id="non-final-pass-finished"),
        pytest.param(
            (
                _FIRST_PASS_STARTED,
                _FIRST_PASS_FINISHED,
                _LAST_PASS_STARTED,
                _LAST_PASS_FINISHED,
                _pass_started(1, 1, target_count=2, label="baseline", at_ms=20000, phase="confirm"),
            ),
            _pass_finished(1, 1, target_count=2, label="baseline", at_ms=30000, phase="confirm"),
            id="confirm-pass-finished",
        ),
    ],
)
def test_plain_line_when_non_milestone_event_does_return_none(
    setup: Sequence[ProgressEvent], event: ProgressEvent
):
    before = _apply(_state(), *setup)

    result = plain_line(before, advance(before, event), event)

    assert result is None


def test_plain_line_when_recorded_with_checks_cmd_does_name_the_checks_run_at_keep():
    before = _state(checks_cmd="npm test")
    event = IterationRecorded(seq=2, outcome="unsettled", at_ms=15000)

    result = plain_line(before, advance(before, event), event)

    assert result is not None
    assert "npm test" in result
