"""Tests for record-to-attribute mapping (pure dict, no OTel imports)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from gymrat.session.records import (
    CommandRecord,
    IterationPrimary,
    SessionLogRecord,
    record_event,
)
from gymrat.telemetry.command_span import command_attributes
from gymrat.telemetry.provider import run_attributes
from tests.session.records._fixtures import (
    SESSION_ID,
    baseline_record,
    blocked_keep,
    command_record,
    committed_keep,
    discard_record,
    finalize_record,
    hook_record,
    iteration_record,
    stop_record,
)
from tests.supervisor._fixtures import make_launch

if TYPE_CHECKING:
    from gymrat.supervisor.events import LaunchEvent

# ---------------------------------------------------------------------------
# command_attributes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("record", "expected"),
    [
        pytest.param(
            command_record(
                name="measure",
                exit_code=0,
                duration_ms=1234,
                reason="budget-exceeded",
                seq=3,
                args={
                    "ref": "main",
                    "samples": 5,
                    "verbose": True,
                    "threshold": 0.05,
                    "paths": ["/a", "/b"],
                    "config": {"key": "val"},
                    "empty": None,
                },
            ),
            {
                "gymrat.session.id": SESSION_ID,
                "gymrat.command.name": "measure",
                "gymrat.command.exit_code": 0,
                "gymrat.command.duration_ms": 1234,
                "gymrat.command.reason": "budget-exceeded",
                "gymrat.iteration.seq": 3,
                "gymrat.command.args.ref": "main",
                "gymrat.command.args.samples": 5,
                "gymrat.command.args.verbose": True,
                "gymrat.command.args.threshold": 0.05,
            },
            id="every-field-scalar-args-only",
        ),
        pytest.param(
            command_record(name="measure", reason=None, seq=None, args={}),
            {
                "gymrat.session.id": SESSION_ID,
                "gymrat.command.name": "measure",
                "gymrat.command.exit_code": 1,
                "gymrat.command.duration_ms": 1840,
            },
            id="no-reason-seq-or-args",
        ),
    ],
)
def test_command_attributes_when_called_does_map_fields_and_scalar_args_only(
    record: CommandRecord, expected: dict[str, object]
):
    result = command_attributes(record, session_id=SESSION_ID)

    assert result == expected


# ---------------------------------------------------------------------------
# run_attributes
# ---------------------------------------------------------------------------


_FIXED_RUN_ATTRIBUTES = {
    "gymrat.session.id": SESSION_ID,
    "gymrat.run.head_sha": "abc123def",
    "gymrat.run.max_minutes": 60.0,
    "gen_ai.provider.name": "anthropic",
}


@pytest.mark.parametrize(
    ("launch", "expected"),
    [
        pytest.param(
            make_launch(session_id=SESSION_ID, max_minutes=60.0),
            _FIXED_RUN_ATTRIBUTES,
            id="options-unset-are-left-off",
        ),
        pytest.param(
            make_launch(
                session_id=SESSION_ID,
                max_minutes=60.0,
                max_usd=5.0,
                effort="high",
                model="claude-sonnet-4-20250514",
            ),
            {
                **_FIXED_RUN_ATTRIBUTES,
                "gymrat.run.max_usd": 5.0,
                "gymrat.run.effort": "high",
                "gen_ai.request.model": "claude-sonnet-4-20250514",
            },
            id="options-set-are-recorded",
        ),
    ],
)
def test_run_attributes_when_called_does_map_the_launch_leaving_unset_options_off(
    launch: LaunchEvent, expected: dict[str, object]
):
    result = run_attributes(launch)

    assert result == expected


# ---------------------------------------------------------------------------
# record_event
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("record", "expected"),
    [
        pytest.param(
            iteration_record(
                seq=3, outcome="improved", primary=IterationPrimary(kind="geomean", delta_pct=-7.2)
            ),
            (
                "gymrat.iteration",
                {
                    "gymrat.iteration.seq": 3,
                    "gymrat.iteration.outcome": "improved",
                    "gymrat.iteration.delta_pct": -7.2,
                },
            ),
            id="iteration",
        ),
        pytest.param(
            iteration_record(primary=IterationPrimary(kind="geomean", delta_pct=None)),
            (
                "gymrat.iteration",
                {"gymrat.iteration.seq": 1, "gymrat.iteration.outcome": "improved"},
            ),
            id="iteration-without-delta",
        ),
        pytest.param(
            hook_record(stage="before", exit_code=0, seq=5),
            (
                "gymrat.hook",
                {
                    "gymrat.iteration.seq": 5,
                    "gymrat.hook.stage": "before",
                    "gymrat.hook.exit_code": 0,
                    "gymrat.hook.duration_ms": 120,
                    "gymrat.hook.stdout_bytes": 80,
                    "gymrat.hook.timed_out": False,
                },
            ),
            id="hook",
        ),
        pytest.param(
            committed_keep(1),
            (
                "gymrat.keep",
                {
                    "gymrat.iteration.seq": 1,
                    "gymrat.keep.status": "committed",
                    "gymrat.keep.commit": "b" * 40,
                    "gymrat.keep.message": "cache the regex",
                },
            ),
            id="committed-keep",
        ),
        pytest.param(
            blocked_keep(1),
            (
                "gymrat.keep",
                {
                    "gymrat.iteration.seq": 1,
                    "gymrat.keep.status": "blocked",
                    "gymrat.keep.reason": "checks-failed",
                },
            ),
            id="blocked-keep",
        ),
        pytest.param(
            baseline_record(label="initial"),
            ("gymrat.baseline", {"gymrat.baseline.label": "initial"}),
            id="baseline-has-no-seq",
        ),
        pytest.param(
            stop_record(),
            ("gymrat.stop", {"gymrat.stop.message": "user requested stop"}),
            id="stop-has-no-seq",
        ),
        pytest.param(
            discard_record(2), ("gymrat.discard", {"gymrat.iteration.seq": 2}), id="discard"
        ),
        pytest.param(
            finalize_record(branch="gymrat/test-final"),
            (
                "gymrat.finalize",
                {
                    "gymrat.finalize.branch": "gymrat/test-final",
                    "gymrat.finalize.commit": "c" * 40,
                    "gymrat.finalize.message": "squash 1 kept iteration",
                },
            ),
            id="finalize-has-no-seq",
        ),
    ],
)
def test_record_event_when_called_does_return_the_event_name_and_its_scalar_attributes(
    record: SessionLogRecord, expected: tuple[str, dict[str, object]]
):
    result = record_event(record)

    assert result == expected
