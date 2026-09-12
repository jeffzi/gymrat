"""Tests for the pure iterate checklist reducer.

The reducer owns every state transition the live checklist renders, so these
tests drive it directly: no terminal, no ``Console``, no ``rich`` objects.
Timestamps come from each event's own ``at_ms``, which keeps the transitions
deterministic without a clock.
"""

from __future__ import annotations

import ast
import inspect
from dataclasses import FrozenInstanceError
from typing import TYPE_CHECKING

import pytest

from gymrat.cli.iterate import state as state_module
from gymrat.cli.iterate.state import (
    IterateState,
    JudgeDetail,
    advance,
    initial_state,
    plain_line,
)
from gymrat.eta import SamplingEta
from gymrat.progress_events import (
    ConfirmFinished,
    ConfirmStarted,
    HookFinished,
    HookStarted,
    IterationRecorded,
    JudgeFinished,
    JudgeStarted,
    PrepareFinished,
    PrepareStarted,
)
from tests.cli._progress_helpers import (
    pass_finished as _pass_finished,
)
from tests.cli._progress_helpers import (
    pass_started as _pass_started,
)

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


def _imported_roots(source: str) -> set[str]:
    roots: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            roots.add(node.module.split(".")[0])
    return roots


_INITIAL = _state()

# A single round against two targets: the phase total is two passes.
_FIRST_PASS_STARTED = _pass_started(1, 1, target_count=2, label="baseline", at_ms=0)
_FIRST_PASS_FINISHED = _pass_finished(1, 1, target_count=2, label="baseline", at_ms=10000)
_LAST_PASS_STARTED = _pass_started(1, 1, target_count=2, label="experiment", at_ms=10000)
_LAST_PASS_FINISHED = _pass_finished(1, 1, target_count=2, label="experiment", at_ms=20000)


# ---------------------------------------------------------------------------
# Reducer shape: frozen records, no rich
# ---------------------------------------------------------------------------


def test_state_module_when_parsed_does_import_nothing_from_rich():
    roots = _imported_roots(inspect.getsource(state_module))

    assert "rich" not in roots


@pytest.mark.parametrize(
    ("record", "attribute", "value"),
    [
        pytest.param(_INITIAL, "total", 4, id="iterate-state"),
        pytest.param(_INITIAL.nodes.prepare, "status", "done", id="node-state"),
        pytest.param(_INITIAL.pass_phase, "start_ms", 5.0, id="phase-counters"),
    ],
)
def test_state_records_when_attribute_assigned_does_raise_frozen_instance_error(
    record: object, attribute: str, value: object
):
    with pytest.raises(FrozenInstanceError):
        setattr(record, attribute, value)


def test_advance_when_applied_twice_to_same_input_does_return_equal_states():
    event = PrepareFinished(label="baseline", at_ms=5000)
    state = advance(_state(), PrepareStarted(label="baseline", at_ms=0))

    assert advance(state, event) == advance(state, event)


def test_advance_when_applied_does_leave_input_state_unchanged():
    state = _state()

    advance(state, PrepareStarted(label="baseline", at_ms=1000))

    assert state == _state()


# ---------------------------------------------------------------------------
# Row transitions
# ---------------------------------------------------------------------------


def test_advance_when_before_hook_started_does_mark_hook_row_running():
    result = advance(_state(has_before_hook=True), HookStarted(stage="before", at_ms=0))

    assert result.nodes.before_hook.status == "running"


def test_advance_when_before_hook_finished_does_mark_hook_row_done():
    state = advance(_state(has_before_hook=True), HookStarted(stage="before", at_ms=0))

    result = advance(state, HookFinished(stage="before", at_ms=2000))

    assert result.nodes.before_hook.status == "done"


def test_advance_when_prepare_started_does_mark_prepare_row_running():
    result = advance(_state(), PrepareStarted(label="baseline", at_ms=1000))

    assert result.nodes.prepare.status == "running"


def test_advance_when_second_prepare_finishes_does_accumulate_elapsed():
    state = _apply(
        _state(),
        PrepareStarted(label="baseline", at_ms=0),
        PrepareFinished(label="baseline", at_ms=3000),
        PrepareStarted(label="candidate", at_ms=3000),
    )

    result = advance(state, PrepareFinished(label="candidate", at_ms=5000))

    assert result.nodes.prepare.status == "done"
    assert result.nodes.prepare.elapsed_ms == 5000


def test_advance_when_pass_finishes_does_advance_pass_phase_eta():
    state = advance(_state(), _FIRST_PASS_STARTED)

    result = advance(state, _FIRST_PASS_FINISHED)

    assert result.pass_phase.eta == SamplingEta(
        completed=1, finish_count=1, total_time_ms=10000, total=2
    )


def test_advance_when_final_pass_finishes_does_complete_passes_row():
    state = _apply(_state(), _FIRST_PASS_STARTED, _FIRST_PASS_FINISHED, _LAST_PASS_STARTED)

    result = advance(state, _LAST_PASS_FINISHED)

    assert result.nodes.passes.status == "done"
    assert result.nodes.passes.detail == "2 passes"


def test_advance_when_judge_finished_does_store_judge_detail_as_data():
    result = advance(
        _state(),
        JudgeFinished(
            primary_delta_pct=-3.2, regressed=("latency", "throughput"), metric_count=3, at_ms=6000
        ),
    )

    assert result.nodes.judge.detail == JudgeDetail(
        primary_metric="geomean",
        primary_delta_pct=-3.2,
        regressed_names=("latency", "throughput"),
    )


@pytest.mark.parametrize(
    ("regressed", "expected_status"),
    [
        pytest.param((), "skipped", id="no-regression"),
        pytest.param(("latency",), "pending", id="regression"),
    ],
)
def test_advance_when_judge_finished_does_skip_confirm_only_without_regression(
    regressed: tuple[str, ...], expected_status: str
):
    result = advance(
        _state(),
        JudgeFinished(primary_delta_pct=-2.0, regressed=regressed, metric_count=3, at_ms=6000),
    )

    assert result.nodes.confirm.status == expected_status


@pytest.mark.parametrize(
    ("filtered_metrics", "expected_note"),
    [
        pytest.param(None, "full suite", id="unfiltered"),
        pytest.param(("latency",), "1 metric", id="one-metric"),
        pytest.param(("latency", "alloc"), "2 metrics", id="two-metrics"),
    ],
)
def test_advance_when_confirm_started_does_set_confirm_note(
    filtered_metrics: tuple[str, ...] | None, expected_note: str
):
    result = advance(_state(), ConfirmStarted(filtered_metrics=filtered_metrics, at_ms=5000))

    assert result.nodes.confirm.note == expected_note


@pytest.mark.parametrize(
    ("checks_cmd", "expected_detail"),
    [
        pytest.param(None, "improved suggested", id="no-checks-cmd"),
        pytest.param(
            "npm run check && npm test",
            "improved suggested — checks (npm run check && npm test) run at gymrat keep",
            id="with-checks-cmd",
        ),
    ],
)
def test_advance_when_iteration_recorded_does_set_record_detail(
    checks_cmd: str | None, expected_detail: str
):
    result = advance(
        _state(checks_cmd=checks_cmd), IterationRecorded(seq=2, outcome="improved", at_ms=15000)
    )

    assert result.nodes.record.detail == expected_detail


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
            JudgeFinished(primary_delta_pct=-2.0, regressed=(), metric_count=3, at_ms=6000),
            "judge -2.0% · 3 improve/noise",
            id="judge-finished",
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
    ],
)
def test_plain_line_when_non_milestone_event_does_return_none(
    setup: Sequence[ProgressEvent], event: ProgressEvent
):
    before = _apply(_state(), *setup)

    result = plain_line(before, advance(before, event), event)

    assert result is None
