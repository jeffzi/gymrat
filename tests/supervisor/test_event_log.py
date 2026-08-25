"""Behavioral tests for the supervisor event-log writer.

``create_event_log_writer`` returns a ``SessionObserver`` that appends one
serialized JSON line per event to a log file, creating the parent directory
tree lazily on the first write. Serialization is delegated to ``to_json_line``
(compact camelCase); these tests pin the file-writing side effects, the lazy
directory creation, the failure surface, and ``SessionObserver`` compatibility.
"""

from pathlib import Path

import pytest

from gymrat_py.errors import GymratError
from gymrat_py.supervisor import (
    CapEvent,
    UsageUpdateEvent,
    combine_observers,
    create_event_log_writer,
)
from tests.supervisor._fixtures import read_log_lines

# ---------------------------------------------------------------------------
# create_event_log_writer
# ---------------------------------------------------------------------------


def test_create_event_log_writer_when_observing_events_does_append_one_line_each(
    tmp_path: Path,
):
    log_path = tmp_path / "events.jsonl"
    writer = create_event_log_writer(log_path)
    event1 = UsageUpdateEvent(timestamp=1000, cost_usd=0.01)
    event2 = UsageUpdateEvent(timestamp=2000, cost_usd=0.02)

    writer(event1)
    writer(event2)

    assert read_log_lines(log_path) == [
        {"type": "usage_update", "timestamp": 1000, "costUsd": 0.01},
        {"type": "usage_update", "timestamp": 2000, "costUsd": 0.02},
    ]


def test_create_event_log_writer_when_writing_does_terminate_each_line_with_newline(
    tmp_path: Path,
):
    log_path = tmp_path / "events.jsonl"
    writer = create_event_log_writer(log_path)

    writer(UsageUpdateEvent(timestamp=1000, cost_usd=0.01))

    assert log_path.read_text(encoding="utf-8").endswith("\n")


def test_create_event_log_writer_when_parent_missing_does_create_tree_on_first_write(
    tmp_path: Path,
):
    log_path = tmp_path / "nested" / "deep" / "events.jsonl"
    writer = create_event_log_writer(log_path)

    writer(UsageUpdateEvent(timestamp=1000, cost_usd=0.01))

    assert read_log_lines(log_path) == [
        {"type": "usage_update", "timestamp": 1000, "costUsd": 0.01},
    ]


def test_create_event_log_writer_when_write_fails_does_raise_gymrat_error_naming_path(
    tmp_path: Path,
):
    log_path = tmp_path / "a-directory"
    log_path.mkdir()
    writer = create_event_log_writer(log_path)

    with pytest.raises(GymratError, match=str(log_path)):
        writer(UsageUpdateEvent(timestamp=1000, cost_usd=0.01))


def test_create_event_log_writer_when_cap_event_written_does_round_trip(
    tmp_path: Path,
):
    log_path = tmp_path / "events.jsonl"
    writer = create_event_log_writer(log_path)

    writer(CapEvent(cap="wall-clock", timestamp=5000))

    assert read_log_lines(log_path) == [
        {"type": "cap", "timestamp": 5000, "cap": "wall-clock"},
    ]


def test_create_event_log_writer_when_wrapped_in_combine_observers_does_write_one_line(
    tmp_path: Path,
):
    log_path = tmp_path / "events.jsonl"
    combined = combine_observers(create_event_log_writer(log_path))

    combined(UsageUpdateEvent(timestamp=1000, cost_usd=0.01))

    assert read_log_lines(log_path) == [
        {"type": "usage_update", "timestamp": 1000, "costUsd": 0.01},
    ]
