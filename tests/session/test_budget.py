"""Behavioral tests for the session budget file (write / read / clear / remaining).

A budget is a frozen dataclass with ``version``, ``started_at_ms``,
``max_minutes``, and ``deadline_ms``.  ``write_budget`` writes atomically via
temp-file-and-replace so a concurrent reader never sees a partial file.
``read_budget`` returns the budget only when the file parses, the deadline has
not passed, and the supervise lock for that root is held; otherwise it returns
``None``.  ``remaining_ms`` reports milliseconds left against a supplied
current time, clamped at zero.  ``clear_budget`` removes the file and succeeds
when the file is already gone.
"""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from gymrat.session import BaselineRecord, IterationRecord, SessionLogRecord
from gymrat.session.budget import (
    Budget,
    clear_budget,
    estimate_iterate_duration,
    read_budget,
    write_budget,
)
from gymrat.session.paths import budget_path, session_dir
from tests.session.records._fixtures import iteration_record


@pytest.fixture
def root(tmp_path: Path) -> str:
    """A fake repo root with the .gymrat session directory pre-created."""
    session = tmp_path / ".gymrat"
    session.mkdir()
    return str(tmp_path)


def _budget_file(root: str) -> Path:
    return Path(budget_path(root))


def _read_json(root: str) -> dict[str, object]:
    """Read and parse the raw budget JSON under *root*."""
    return json.loads(_budget_file(root).read_text(encoding="utf-8"))


def _make_budget(**overrides: object) -> Budget:
    """Build a Budget with sensible defaults, overridable per-field."""
    defaults: dict[str, object] = {
        "started_at_ms": 1000.0,
        "max_minutes": 30,
        "deadline_ms": 1_800_000.0,
    }
    defaults.update(overrides)
    return Budget(**defaults)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# budget_path
# ---------------------------------------------------------------------------


def test_budget_path_when_given_root_does_place_file_under_session_dir(root: str):
    result = budget_path(root)

    expected = str(Path(session_dir(root)) / "budget.json")
    assert result == expected


# ---------------------------------------------------------------------------
# Budget dataclass
# ---------------------------------------------------------------------------


def test_budget_when_constructed_does_be_frozen():
    budget = _make_budget()

    with pytest.raises(AttributeError):
        budget.max_minutes = 99  # type: ignore[misc]


def test_budget_when_constructed_does_have_version_1():
    budget = _make_budget()

    assert budget.version == 1


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


def test_write_budget_when_called_does_create_readable_json_file(root: str):
    budget = _make_budget()

    write_budget(root, budget)

    raw = _read_json(root)
    assert raw["version"] == 1
    assert raw["started_at_ms"] == 1000.0
    assert raw["max_minutes"] == 30
    assert raw["deadline_ms"] == 1_800_000.0


def test_write_budget_when_called_twice_does_overwrite_previous(root: str):
    write_budget(root, _make_budget(max_minutes=10))
    write_budget(root, _make_budget(max_minutes=20))

    assert _read_json(root)["max_minutes"] == 20


# ---------------------------------------------------------------------------
# read_budget
# ---------------------------------------------------------------------------


def test_read_budget_when_file_exists_and_lock_held_and_deadline_ahead_does_return_budget(
    root: str,
):
    original = _make_budget(deadline_ms=999_999_999.0)
    write_budget(root, original)

    with patch("gymrat.session.budget.is_held", autospec=True, return_value=True):
        result = read_budget(root, now_ms=1000.0)

    assert result == original


def test_read_budget_when_file_absent_does_return_none(root: str):
    with patch("gymrat.session.budget.is_held", autospec=True, return_value=True):
        result = read_budget(root, now_ms=0.0)

    assert result is None


def test_read_budget_when_file_contains_invalid_json_does_return_none(root: str):
    _budget_file(root).write_text("not valid json{{{", encoding="utf-8")

    with patch("gymrat.session.budget.is_held", autospec=True, return_value=True):
        result = read_budget(root, now_ms=0.0)

    assert result is None


def test_read_budget_when_file_contains_wrong_schema_does_return_none(root: str):
    _budget_file(root).write_text(json.dumps({"unexpected_field": 42}), encoding="utf-8")

    with patch("gymrat.session.budget.is_held", autospec=True, return_value=True):
        result = read_budget(root, now_ms=0.0)

    assert result is None


def test_read_budget_when_version_unrecognized_does_return_none(root: str):
    write_budget(root, _make_budget())
    raw = _read_json(root)
    raw["version"] = 999
    _budget_file(root).write_text(json.dumps(raw), encoding="utf-8")

    with patch("gymrat.session.budget.is_held", autospec=True, return_value=True):
        result = read_budget(root, now_ms=0.0)

    assert result is None


def test_read_budget_when_deadline_passed_does_return_none(root: str):
    write_budget(root, _make_budget(deadline_ms=5000.0))

    with patch("gymrat.session.budget.is_held", autospec=True, return_value=True):
        result = read_budget(root, now_ms=6000.0)

    assert result is None


def test_read_budget_when_supervise_lock_not_held_does_return_none(root: str):
    write_budget(root, _make_budget(deadline_ms=999_999_999.0))

    with patch("gymrat.session.budget.is_held", autospec=True, return_value=False):
        result = read_budget(root, now_ms=1000.0)

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


def _baseline(duration_ms: float | None = None) -> BaselineRecord:
    """A baseline record with only the fields ``estimate_iterate_duration`` inspects."""
    return BaselineRecord(
        type="baseline",
        at="2026-08-08T14:15:30.000Z",
        label="main",
        samples=({"total_ms": 15200},),
        duration_ms=duration_ms,
    )


def _iteration(duration_ms: float | None = None, *, seq: int = 1) -> IterationRecord:
    """An iteration record with only the fields ``estimate_iterate_duration`` inspects."""
    return iteration_record(seq=seq, duration_ms=duration_ms)


def test_estimate_iterate_duration_when_no_records_does_return_none():
    result = estimate_iterate_duration([])

    assert result is None


def test_estimate_iterate_duration_when_no_durations_does_return_none():
    records: list[SessionLogRecord] = [_baseline(), _iteration()]

    result = estimate_iterate_duration(records)

    assert result is None


def test_estimate_iterate_duration_when_iteration_has_duration_does_return_it():
    records: list[SessionLogRecord] = [_baseline(), _iteration(duration_ms=840_000)]

    result = estimate_iterate_duration(records)

    assert result is not None
    assert result.duration_ms == 840_000
    assert result.source == "iteration"
    assert result.source_duration_ms == 840_000


def test_estimate_iterate_duration_when_only_baseline_has_duration_does_return_double():
    records: list[SessionLogRecord] = [_baseline(duration_ms=420_000), _iteration()]

    result = estimate_iterate_duration(records)

    assert result is not None
    assert result.duration_ms == 840_000
    assert result.source == "baseline"
    assert result.source_duration_ms == 420_000


def test_estimate_iterate_duration_when_both_have_durations_does_prefer_iteration():
    records: list[SessionLogRecord] = [
        _baseline(duration_ms=420_000),
        _iteration(duration_ms=900_000),
    ]

    result = estimate_iterate_duration(records)

    assert result is not None
    assert result.duration_ms == 900_000
    assert result.source == "iteration"


def test_estimate_iterate_duration_when_multiple_iterations_does_use_newest():
    records: list[SessionLogRecord] = [
        _baseline(),
        _iteration(duration_ms=600_000, seq=1),
        _iteration(duration_ms=840_000, seq=2),
    ]

    result = estimate_iterate_duration(records)

    assert result is not None
    assert result.duration_ms == 840_000
    assert result.source_duration_ms == 840_000


def test_estimate_iterate_duration_when_newest_iteration_lacks_duration_does_use_earlier():
    records: list[SessionLogRecord] = [
        _baseline(),
        _iteration(duration_ms=600_000, seq=1),
        _iteration(seq=2),
    ]

    result = estimate_iterate_duration(records)

    assert result is not None
    assert result.duration_ms == 600_000
    assert result.source == "iteration"
