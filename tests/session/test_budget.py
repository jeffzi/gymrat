"""Behavioral tests for the session budget file (read / clear / remaining / arm / headroom).

``read_budget`` returns the budget only when the file parses, the deadline has
not passed, and the supervise lock for that root is held; otherwise it returns
``None``.  ``remaining_ms`` reports milliseconds left against a supplied
current time, clamped at zero.  ``arm_budget`` writes a budget due its minutes
from now and returns a release that removes it, and ``iterate_headroom`` gives
up when any of its inputs is missing.
"""

from collections.abc import Sequence
from pathlib import Path

import pytest

from gymrat.session.budget import (
    Budget,
    DurationEstimate,
    arm_budget,
    estimate_iterate_duration,
    iterate_headroom,
    read_budget,
    write_budget,
)
from gymrat.session.paths import budget_path
from gymrat.session.records import SessionLogRecord
from tests._clock import freeze_now_ms
from tests._process_helpers import track_cleanups
from tests.session._budget import LIVE_BUDGET
from tests.session.records._fixtures import baseline_record, iteration_record


def _budget_file(root: str) -> Path:
    return Path(budget_path(root))


def _make_budget(*, max_minutes: float = 30, deadline_ms: float = 1_800_000.0) -> Budget:
    return Budget(max_minutes=max_minutes, deadline_ms=deadline_ms)


# ---------------------------------------------------------------------------
# remaining_ms
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("deadline_ms", "now_ms", "expected"),
    [
        pytest.param(10_000.0, 4_000.0, 6_000.0, id="time-left"),
        pytest.param(10_000.0, 15_000.0, 0.0, id="past-deadline-clamps-to-zero"),
    ],
)
def test_remaining_ms_when_called_does_return_clamped_difference(
    deadline_ms: float, now_ms: float, expected: float
):
    budget = _make_budget(deadline_ms=deadline_ms)

    result = budget.remaining_ms(now_ms)

    assert result == expected


# ---------------------------------------------------------------------------
# read_budget
# ---------------------------------------------------------------------------


_WRITTEN = _make_budget(deadline_ms=5000.0)


@pytest.mark.parametrize(
    ("supervise_lock", "now_ms", "expected"),
    [
        pytest.param(True, 1000.0, _WRITTEN, id="lock-held-and-deadline-ahead"),
        pytest.param(True, 6000.0, None, id="deadline-passed"),
        pytest.param(False, 1000.0, None, id="supervise-lock-not-held"),
    ],
    indirect=["supervise_lock"],
)
@pytest.mark.usefixtures("supervise_lock")
def test_read_budget_when_file_written_does_return_it_only_while_held_and_ahead(
    root: str, now_ms: float, expected: Budget | None
):
    write_budget(root, _WRITTEN)

    result = read_budget(root, now_ms=now_ms)

    assert result == expected


@pytest.mark.usefixtures("supervise_lock")
def test_read_budget_when_file_absent_does_return_none(root: str):
    result = read_budget(root, now_ms=0.0)

    assert result is None


# ---------------------------------------------------------------------------
# arm_budget
# ---------------------------------------------------------------------------


def test_arm_budget_when_armed_does_write_a_budget_due_its_minutes_from_now(
    root: str, monkeypatch: pytest.MonkeyPatch
):
    freeze_now_ms(monkeypatch, 1_000)

    budget, _release = arm_budget(root, 30, warn=lambda _message: None)

    assert budget == Budget(max_minutes=30, deadline_ms=1_801_000.0)
    assert Budget.model_validate_json(_budget_file(root).read_text()) == budget


def test_arm_budget_when_released_twice_does_remove_the_file_and_uninstall_the_hook_once(
    root: str, monkeypatch: pytest.MonkeyPatch
):
    registry = track_cleanups(monkeypatch, "gymrat.session.budget")
    _budget, release = arm_budget(root, 30, warn=lambda _message: None)

    release()
    release()

    assert registry.uninstall_counts() == [1]
    assert not _budget_file(root).exists()


def test_arm_budget_when_removal_refused_does_warn_through_the_given_sink(
    root: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    track_cleanups(monkeypatch, "gymrat.session.budget")
    warned: list[str] = []
    _budget, release = arm_budget(root, 30, warn=warned.append)
    _budget_file(root).unlink()
    _budget_file(root).mkdir()

    release()

    assert len(warned) == 1
    assert str(_budget_file(root)) in warned[0]
    assert capsys.readouterr().err == ""


# ---------------------------------------------------------------------------
# estimate_iterate_duration
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("records", "expected"),
    [
        pytest.param([baseline_record(), iteration_record()], None, id="no-durations"),
        pytest.param(
            [baseline_record(duration_ms=420_000), iteration_record()],
            DurationEstimate(duration_ms=840_000, source="baseline", source_duration_ms=420_000),
            id="only-baseline-has-duration-doubles-it",
        ),
        pytest.param(
            [iteration_record(duration_ms=900_000), baseline_record(duration_ms=420_000)],
            DurationEstimate(duration_ms=900_000, source="iteration", source_duration_ms=900_000),
            id="both-have-durations-prefers-iteration",
        ),
        pytest.param(
            [
                baseline_record(),
                iteration_record(duration_ms=600_000, seq=1),
                iteration_record(duration_ms=840_000, seq=2),
            ],
            DurationEstimate(duration_ms=840_000, source="iteration", source_duration_ms=840_000),
            id="multiple-iterations-uses-newest",
        ),
        pytest.param(
            [
                baseline_record(),
                iteration_record(duration_ms=600_000, seq=1),
                iteration_record(seq=2),
            ],
            DurationEstimate(duration_ms=600_000, source="iteration", source_duration_ms=600_000),
            id="newest-iteration-lacks-duration-uses-earlier",
        ),
        pytest.param(
            [baseline_record(duration_ms=420_000), baseline_record()],
            DurationEstimate(duration_ms=840_000, source="baseline", source_duration_ms=420_000),
            id="newest-baseline-lacks-duration-uses-earlier",
        ),
    ],
)
def test_estimate_iterate_duration_when_records_vary_does_prefer_newest_iteration_duration_over_baseline(
    records: list[SessionLogRecord], expected: DurationEstimate | None
):
    result = estimate_iterate_duration(records)

    assert result == expected


# ---------------------------------------------------------------------------
# iterate_headroom
# ---------------------------------------------------------------------------


_TIMED_RECORDS: list[SessionLogRecord] = [baseline_record(duration_ms=420_000)]


@pytest.mark.parametrize(
    ("budget_written", "records"),
    [
        pytest.param(False, _TIMED_RECORDS, id="no-budget-live"),
        pytest.param(True, None, id="records-unreadable"),
        pytest.param(True, [baseline_record(), iteration_record()], id="no-record-timed"),
    ],
)
@pytest.mark.usefixtures("supervise_lock")
def test_iterate_headroom_when_an_input_is_missing_does_return_none(
    root: str, budget_written: bool, records: Sequence[SessionLogRecord] | None
):
    if budget_written:
        write_budget(root, LIVE_BUDGET)

    result = iterate_headroom(root, lambda: records)

    assert result is None
