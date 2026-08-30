"""Behavioral tests for the supervisor event-log writer.

``create_event_log_writer`` returns a ``SessionObserver`` that appends one
serialized JSON line per event to a log file, creating the parent directory
tree lazily on the first write. Serialization is delegated to ``to_json_line``
(compact camelCase); these tests pin the file-writing side effects, the lazy
directory creation, the failure surface, and ``SessionObserver`` compatibility.
"""

import re
from pathlib import Path

import pytest

from gymrat.errors import GymratError
from gymrat.supervisor import (
    CapEvent,
    UsageUpdateEvent,
    combine_observers,
    create_event_log_writer,
)
from gymrat.supervisor.event_log import probe_event_log_path
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

    with pytest.raises(GymratError, match=re.escape(str(log_path))):
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


# ---------------------------------------------------------------------------
# event log directory re-creation
# ---------------------------------------------------------------------------


def test_create_event_log_writer_when_parent_removed_after_first_write_does_recreate_on_next(
    tmp_path: Path,
):
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    log_path = log_dir / "events.jsonl"
    writer = create_event_log_writer(log_path)

    writer(UsageUpdateEvent(timestamp=1000, cost_usd=0.01))
    import shutil

    shutil.rmtree(log_dir)
    writer(UsageUpdateEvent(timestamp=2000, cost_usd=0.02))

    assert read_log_lines(log_path) == [
        {"type": "usage_update", "timestamp": 2000, "costUsd": 0.02},
    ]


# ---------------------------------------------------------------------------
# probe_event_log_path — up-front write check
# ---------------------------------------------------------------------------


def test_probe_event_log_path_when_parent_is_a_file_does_raise_gymrat_error_naming_path(
    tmp_path: Path,
):
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("I am a file", encoding="utf-8")
    log_path = blocker / "events.jsonl"

    with pytest.raises(GymratError, match=re.escape(str(log_path))):
        probe_event_log_path(log_path)


def test_probe_event_log_path_when_path_writable_does_not_raise(tmp_path: Path):
    log_path = tmp_path / "events.jsonl"

    probe_event_log_path(log_path)
