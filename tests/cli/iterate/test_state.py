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
    JudgeDetail,
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
from gymrat.utils import SamplingEta
from tests._imports import loaded_under, modules_imported_by
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
    sample_count: int = 1,
    metric_count: int = 3,
    checks_cmd: str | None = None,
    has_before_hook: bool = False,
    has_after_hook: bool = False,
) -> IterateState:
    return initial_state(
        sample_count=sample_count,
        metric_count=metric_count,
        primary_metric="geomean",
        checks_cmd=checks_cmd,
        has_before_hook=has_before_hook,
        has_after_hook=has_after_hook,
    )


def _apply(state: IterateState, *events: ProgressEvent) -> IterateState:
    for event in events:
        state = advance(state, event)
    return state


_INITIAL = _state()

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
# Row transitions
# ---------------------------------------------------------------------------


def test_advance_when_before_hook_finished_does_mark_hook_row_done():
    state = advance(_state(has_before_hook=True), HookStarted(stage="before", at_ms=0))

    result = advance(state, HookFinished(stage="before", at_ms=2000))

    assert result.nodes.before_hook.status == "done"


def test_advance_when_pass_finishes_does_advance_pass_phase_eta():
    state = advance(_state(), _FIRST_PASS_STARTED)

    result = advance(state, _FIRST_PASS_FINISHED)

    assert result.pass_phase.eta == SamplingEta(completed=1, total_time_ms=10000, total=2)


def test_advance_when_final_pass_finishes_does_complete_passes_row():
    state = _apply(_state(), _FIRST_PASS_STARTED, _FIRST_PASS_FINISHED, _LAST_PASS_STARTED)

    result = advance(state, _LAST_PASS_FINISHED)

    assert result.nodes.passes.status == "done"
    assert result.nodes.passes.detail == "2 passes"


def test_advance_when_judge_finished_does_store_judge_detail_as_data():
    result = advance(
        _state(),
        JudgeFinished(primary_delta_pct=-3.2, regressed=("latency", "throughput"), at_ms=6000),
    )

    assert result.nodes.judge.detail == JudgeDetail(
        primary_metric="geomean",
        primary_delta_pct=-3.2,
        regressed_names=("latency", "throughput"),
    )


def test_advance_when_confirm_started_does_set_confirm_note():
    result = advance(_state(), ConfirmStarted(filtered_metrics=("latency", "alloc"), at_ms=5000))

    assert result.nodes.confirm.note == "2 metrics"


# ---------------------------------------------------------------------------
# Plain-mode milestone lines
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("setup", "event", "expected"),
    [
        pytest.param(
            (PrepareStarted(label="baseline", at_ms=0),),
            PrepareFinished(label="baseline", at_ms=5000),
            "prepare baseline done (5s)",
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
            (ConfirmStarted(filtered_metrics=None, at_ms=5000),),
            ConfirmFinished(reproduced=True, at_ms=15000),
            "confirm 0/2 · regressions reproduced",
            id="confirm-finished",
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
        pytest.param((), HookStarted(stage="before", at_ms=0), id="hook-started"),
        pytest.param((), HookFinished(stage="before", at_ms=0), id="hook-finished"),
        pytest.param((), PrepareStarted(label="baseline", at_ms=0), id="prepare-started"),
        pytest.param((), _FIRST_PASS_STARTED, id="pass-started"),
        pytest.param((_FIRST_PASS_STARTED,), _FIRST_PASS_FINISHED, id="non-final-pass-finished"),
        pytest.param((), JudgeStarted(at_ms=0), id="judge-started"),
        pytest.param((), ConfirmStarted(filtered_metrics=None, at_ms=5000), id="confirm-started"),
        pytest.param((), ConfirmSkipped(at_ms=5000), id="confirm-skipped"),
    ],
)
def test_plain_line_when_non_milestone_event_does_return_none(
    setup: Sequence[ProgressEvent], event: ProgressEvent
):
    before = _apply(_state(), *setup)

    result = plain_line(before, advance(before, event), event)

    assert result is None
