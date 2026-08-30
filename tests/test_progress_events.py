"""Behavioral tests for the progress-event types and default clock."""

import dataclasses
import time

import pytest

from gymrat.progress_events import (
    ConfirmFinished,
    ConfirmStarted,
    HookFinished,
    HookStarted,
    IterationRecorded,
    JudgeFinished,
    JudgeStarted,
    PassFinished,
    PassStarted,
    PrepareFinished,
    PrepareStarted,
    ProgressCallback,
    ProgressEvent,
    default_clock,
)
from gymrat.session.schema import HookStage

# ---------------------------------------------------------------------------
# default_clock
# ---------------------------------------------------------------------------


def test_default_clock_when_called_does_return_perf_counter_in_milliseconds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(time, "perf_counter", lambda: 1.5)

    result = default_clock()

    assert result == 1500.0


def _one_of_each_event(at_ms: float) -> list[ProgressEvent]:
    """One instance of every ``ProgressEvent`` subtype, stamped with ``at_ms``.

    The other fields are arbitrary — only ``at_ms`` matters to the tests that
    share this factory; each event type's own fields are covered separately
    below under "field shapes".
    """
    return [
        PrepareStarted(label="A", at_ms=at_ms),
        PrepareFinished(label="A", at_ms=at_ms),
        PassStarted(
            round=1, total_rounds=3, target_index=0, target_count=2, label="A", at_ms=at_ms
        ),
        PassFinished(
            round=1, total_rounds=3, target_index=0, target_count=2, label="A", at_ms=at_ms
        ),
        HookStarted(stage="before", at_ms=at_ms),
        HookFinished(stage="after", at_ms=at_ms),
        JudgeStarted(at_ms=at_ms),
        JudgeFinished(primary_delta_pct=1.5, regressed=(), at_ms=at_ms),
        ConfirmStarted(filtered_metrics=None, at_ms=at_ms),
        ConfirmFinished(reproduced=True, at_ms=at_ms),
        IterationRecorded(seq=1, outcome="improved", at_ms=at_ms),
    ]


# ---------------------------------------------------------------------------
# frozen + slots
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("event", _one_of_each_event(at_ms=0), ids=lambda e: type(e).__name__)
def test_event_when_field_assigned_does_raise_frozen(event: ProgressEvent) -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        event.at_ms = 999.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# at_ms carried by every event
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("event", _one_of_each_event(at_ms=42.0), ids=lambda e: type(e).__name__)
def test_event_when_constructed_does_carry_at_ms(event: ProgressEvent) -> None:
    assert event.at_ms == 42.0


# ---------------------------------------------------------------------------
# field shapes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("event_class", [PrepareStarted, PrepareFinished], ids=lambda c: c.__name__)
def test_prepare_event_when_constructed_does_carry_label(
    event_class: type[PrepareStarted | PrepareFinished],
) -> None:
    event = event_class(label="build", at_ms=0)

    assert event.label == "build"


@pytest.mark.parametrize("event_class", [PassStarted, PassFinished], ids=lambda c: c.__name__)
def test_pass_event_when_constructed_does_carry_all_fields(
    event_class: type[PassStarted | PassFinished],
) -> None:
    event = event_class(
        round=2, total_rounds=5, target_index=1, target_count=3, label="bench", at_ms=100
    )

    assert event.round == 2
    assert event.total_rounds == 5
    assert event.target_index == 1
    assert event.target_count == 3
    assert event.label == "bench"
    assert event.phase == "measure"


def test_pass_started_when_phase_set_to_confirm_does_carry_confirm() -> None:
    event = PassStarted(
        round=1, total_rounds=2, target_index=0, target_count=1, label="x", phase="confirm", at_ms=0
    )

    assert event.phase == "confirm"


@pytest.mark.parametrize(
    ("event_class", "stage"),
    [
        pytest.param(HookStarted, "before", id="HookStarted"),
        pytest.param(HookFinished, "after", id="HookFinished"),
    ],
)
def test_hook_event_when_constructed_does_carry_stage(
    event_class: type[HookStarted | HookFinished], stage: HookStage
) -> None:
    event = event_class(stage=stage, at_ms=0)

    assert event.stage == stage


def test_judge_finished_when_constructed_does_carry_delta_and_regressed() -> None:
    event = JudgeFinished(primary_delta_pct=2.5, regressed=("x", "y"), at_ms=0)

    assert event.primary_delta_pct == 2.5
    assert event.regressed == ("x", "y")


def test_judge_finished_when_delta_none_does_carry_none() -> None:
    event = JudgeFinished(primary_delta_pct=None, regressed=(), at_ms=0)

    assert event.primary_delta_pct is None


def test_confirm_started_when_filtered_none_does_carry_none() -> None:
    event = ConfirmStarted(filtered_metrics=None, at_ms=0)

    assert event.filtered_metrics is None


def test_confirm_started_when_filtered_given_does_carry_tuple() -> None:
    event = ConfirmStarted(filtered_metrics=("x", "y"), at_ms=0)

    assert event.filtered_metrics == ("x", "y")


def test_confirm_finished_when_constructed_does_carry_reproduced() -> None:
    event = ConfirmFinished(reproduced=True, at_ms=0)

    assert event.reproduced is True


def test_iteration_recorded_when_constructed_does_carry_seq_and_outcome() -> None:
    event = IterationRecorded(seq=3, outcome="improved", at_ms=0)

    assert event.seq == 3
    assert event.outcome == "improved"


# ---------------------------------------------------------------------------
# type aliases
# ---------------------------------------------------------------------------


def test_progress_callback_type_alias_accepts_callable() -> None:
    calls: list[ProgressEvent] = []
    cb: ProgressCallback = calls.append

    cb(PrepareStarted(label="A", at_ms=0))

    assert len(calls) == 1
