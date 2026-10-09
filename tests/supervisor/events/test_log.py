"""Behavioral tests for the supervisor event-log writer.

``create_event_log_writer`` returns a ``SessionObserver`` that appends one
serialized JSON line per event to a log file, creating the parent directory
tree lazily on the first write. Serialization is delegated to ``to_json_line``
(compact snake_case); these tests pin the file-writing side effects, the lazy
directory creation, and the failure surface.
"""

import json
import re
import shutil
from collections.abc import Callable
from pathlib import Path

import pytest

from gymrat.errors import GymratError
from gymrat.supervisor.events import (
    TextDeltaEvent,
    UsageUpdateEvent,
    create_event_log_writer,
    probe_event_log_path,
    to_json_line,
)
from tests.supervisor._fixtures import read_log_lines

# ---------------------------------------------------------------------------
# create_event_log_writer
# ---------------------------------------------------------------------------


def test_create_event_log_writer_when_observing_events_does_append_one_utf8_lf_line_each(
    tmp_path: Path,
):
    log_path = tmp_path / "events.jsonl"
    writer = create_event_log_writer(log_path)
    events = [
        UsageUpdateEvent(at=1_000_000_000_000, cost_usd=0.01),
        TextDeltaEvent(at=2_000_000_000_000, chunk="café"),
    ]
    expected = "".join(to_json_line(event) + "\n" for event in events)

    for event in events:
        writer(event)

    assert log_path.read_bytes() == expected.encode("utf-8")


def test_create_event_log_writer_when_created_does_not_create_the_parent_before_a_write(
    tmp_path: Path,
):
    log_path = tmp_path / "nested" / "events.jsonl"

    create_event_log_writer(log_path)

    assert not log_path.parent.exists()


def test_create_event_log_writer_when_parent_missing_does_create_tree_on_first_write(
    tmp_path: Path,
):
    log_path = tmp_path / "nested" / "deep" / "events.jsonl"
    writer = create_event_log_writer(log_path)
    event = UsageUpdateEvent(at=1_000_000_000_000, cost_usd=0.01)

    writer(event)

    assert read_log_lines(log_path) == [json.loads(to_json_line(event))]


def test_create_event_log_writer_when_write_fails_does_raise_gymrat_error_naming_path(
    tmp_path: Path,
):
    log_path = tmp_path / "a-directory"
    log_path.mkdir()
    writer = create_event_log_writer(log_path)

    with pytest.raises(GymratError, match=re.escape(str(log_path))):
        writer(UsageUpdateEvent(at=1_000_000_000_000, cost_usd=0.01))


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
    writer(UsageUpdateEvent(at=1_000_000_000_000, cost_usd=0.01))
    shutil.rmtree(log_dir)
    event = UsageUpdateEvent(at=2_000_000_000_000, cost_usd=0.02)

    writer(event)

    assert read_log_lines(log_path) == [json.loads(to_json_line(event))]


# ---------------------------------------------------------------------------
# probe_event_log_path — up-front write check
# ---------------------------------------------------------------------------


def _log_under_a_file(tmp_path: Path) -> Path:
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("I am a file", encoding="utf-8")
    return blocker / "events.jsonl"


def _log_at_a_directory(tmp_path: Path) -> Path:
    log_path = tmp_path / "a-directory"
    log_path.mkdir()
    return log_path


@pytest.mark.parametrize(
    "build_log_path",
    [
        pytest.param(_log_under_a_file, id="parent-is-a-file"),
        pytest.param(_log_at_a_directory, id="path-is-a-directory"),
    ],
)
def test_probe_event_log_path_when_path_not_writable_does_raise_gymrat_error_naming_path(
    tmp_path: Path, build_log_path: Callable[[Path], Path]
):
    log_path = build_log_path(tmp_path)

    with pytest.raises(GymratError, match=re.escape(str(log_path))):
        probe_event_log_path(log_path)


def test_probe_event_log_path_when_parent_missing_does_create_it(tmp_path: Path):
    log_path = tmp_path / "nested" / "events.jsonl"

    probe_event_log_path(log_path)

    assert log_path.parent.is_dir()
