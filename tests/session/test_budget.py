"""Behavioral tests for the session budget file (write / read / clear / remaining).

A budget is a frozen pydantic model with ``max_minutes`` and ``deadline_ms``, and
the file carries exactly those two keys.  ``write_budget`` writes atomically via
temp-file-and-replace so a concurrent reader never sees a partial file.
``read_budget`` returns the budget only when the file parses, the deadline has
not passed, and the supervise lock for that root is held; otherwise it returns
``None``.  ``remaining_ms`` reports milliseconds left against a supplied
current time, clamped at zero.  ``clear_budget`` removes the file and succeeds
when the file is already gone.
"""

import json
from pathlib import Path

import pytest

from gymrat.session.budget import (
    Budget,
    DurationEstimate,
    clear_budget,
    estimate_iterate_duration,
    read_budget,
    write_budget,
)
from gymrat.session.paths import budget_path
from gymrat.session.records import SessionLogRecord
from tests.session.records._fixtures import baseline_record, iteration_record

_FAR_FUTURE_DEADLINE_MS = 999_999_999.0


def _budget_file(root: str) -> Path:
    return Path(budget_path(root))


def _make_budget(*, max_minutes: float = 30, deadline_ms: float = 1_800_000.0) -> Budget:
    """Build a Budget with sensible defaults, overridable per-field."""
    return Budget(max_minutes=max_minutes, deadline_ms=deadline_ms)


def _budget_json(**overrides: object) -> str:
    """Serialize a budget JSON payload with sensible defaults, overridable per-field."""
    defaults: dict[str, object] = {
        "max_minutes": 30,
        "deadline_ms": _FAR_FUTURE_DEADLINE_MS,
    }
    defaults.update(overrides)
    return json.dumps(defaults)


# ---------------------------------------------------------------------------
# remaining_ms
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("deadline_ms", "now_ms", "expected"),
    [
        pytest.param(10_000.0, 4_000.0, 6_000.0, id="time-left"),
        pytest.param(10_000.0, 10_000.0, 0.0, id="exactly-at-deadline"),
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
# write_budget
# ---------------------------------------------------------------------------


def test_write_budget_when_called_does_write_compact_json_bytes(root: str):
    write_budget(root, _make_budget())

    assert _budget_file(root).read_bytes() == b'{"max_minutes":30.0,"deadline_ms":1800000.0}'


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


@pytest.mark.parametrize(
    "contents",
    [
        pytest.param(_budget_json(deadline_ms=True).encode(), id="deadline-bool"),
        pytest.param(_budget_json(extra=1).encode(), id="unexpected-field"),
    ],
)
@pytest.mark.usefixtures("supervise_lock")
def test_read_budget_when_file_not_a_budget_does_return_none(root: str, contents: bytes):
    _budget_file(root).write_bytes(contents)

    result = read_budget(root, now_ms=0.0)

    assert result is None


# ---------------------------------------------------------------------------
# clear_budget
# ---------------------------------------------------------------------------


def test_clear_budget_when_file_exists_does_remove_it(root: str):
    write_budget(root, _make_budget())

    clear_budget(root)

    assert not _budget_file(root).exists()


def test_clear_budget_when_file_absent_does_not_raise(root: str):
    clear_budget(root)


# ---------------------------------------------------------------------------
# estimate_iterate_duration
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("records", "expected"),
    [
        pytest.param([], None, id="no-records"),
        pytest.param([baseline_record(), iteration_record()], None, id="no-durations"),
        pytest.param(
            [baseline_record(), iteration_record(duration_ms=840_000)],
            DurationEstimate(duration_ms=840_000, source="iteration", source_duration_ms=840_000),
            id="iteration-has-duration",
        ),
        pytest.param(
            [baseline_record(duration_ms=420_000), iteration_record()],
            DurationEstimate(duration_ms=840_000, source="baseline", source_duration_ms=420_000),
            id="only-baseline-has-duration-doubles-it",
        ),
        pytest.param(
            [baseline_record(duration_ms=420_000), iteration_record(duration_ms=900_000)],
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
            [
                baseline_record(duration_ms=420_000),
                iteration_record(duration_ms=840_000),
                baseline_record(),
            ],
            DurationEstimate(duration_ms=840_000, source="iteration", source_duration_ms=840_000),
            id="a-keep-appended-baseline-times-nothing-and-is-skipped",
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
