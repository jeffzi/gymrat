"""Behavioral tests for the supervisor event vocabulary and helpers.

Events are frozen pydantic models whose base ``_EventModel`` carries a per-event
``type`` discriminator and ``at: int`` (nanoseconds since epoch), with snake_case
wire keys via ``validate_by_name`` / ``serialize_by_alias``. ``to_json_line``
writes one compact JSON line with snake_case keys, omitting an unset optional
field, keeping non-ASCII text raw except the Unicode line breaks (escaped, or
every non-ASCII character when the event holds a lone surrogate), writing a NaN
or infinite float nested in a free-form payload as ``null`` on either path,
refusing a non-finite value in a typed float field at construction, and
writing a value pydantic cannot serialize as its ``str()``; ``event_from_wire``
inverts it and returns ``None`` for anything it cannot reconstruct;
``combine_observers`` sends every observer failure to its warn sink.

The event-log writer, ``create_event_log_writer``, appends one ``to_json_line``
line per event to a log file; its tests pin the file-writing side effects, the
lazy directory creation (including re-creation after the directory is removed),
and the failure surface, alongside the up-front write check
``probe_event_log_path``.
"""

import json
import re
import shutil
from collections.abc import Callable
from pathlib import Path

import pytest
from pydantic import BaseModel

from gymrat.errors import GymratError
from gymrat.supervisor.events import (
    FollowUpEvent,
    SessionEvent,
    SessionObserver,
    TextDeltaEvent,
    ToolStartEvent,
    UsageUpdateEvent,
    combine_observers,
    create_event_log_writer,
    event_from_wire,
    probe_event_log_path,
    to_json_line,
)
from gymrat.utils import loads_strict_json
from tests.supervisor._fixtures import (
    make_launch,
    make_turn_end,
    raising_observer,
    read_log_lines,
)

# ---------------------------------------------------------------------------
# Event vocabulary — at: int (nanoseconds)
# ---------------------------------------------------------------------------

_USAGE_UPDATE = UsageUpdateEvent(at=6_000_000_000, cost_usd=0.01)
_FOLLOW_UP = FollowUpEvent(at=13_000_000_000, action="replied", reason="user asked", text="hello")
_FOLLOW_UP_NO_OPTIONALS = FollowUpEvent(at=20_000_000_000, action="waiting")

#: Text holding a lone surrogate, which forces the ASCII-escaping fallback
#: serializer, next to a raw non-ASCII character it must escape.
_SURROGATE_TEXT = "café \ud800"


class _Limit(BaseModel):
    """A model nested in a free-form payload: JSON mode keeps its float field's NaN/inf."""

    max: float


# ---------------------------------------------------------------------------
# to_json_line — snake_case keys, unset optional fields left off
# ---------------------------------------------------------------------------

JSON_CASES = [
    pytest.param(
        _FOLLOW_UP,
        {
            "type": "follow_up",
            "at": 13_000_000_000,
            "action": "replied",
            "reason": "user asked",
            "text": "hello",
        },
        id="follow_up-with-optionals",
    ),
    pytest.param(
        _FOLLOW_UP_NO_OPTIONALS,
        {"type": "follow_up", "at": 20_000_000_000, "action": "waiting"},
        id="follow_up-no-optionals",
    ),
]


@pytest.mark.parametrize(("event", "expected"), JSON_CASES)
def test_to_json_line_when_given_event_does_write_its_wire_object(
    event: SessionEvent, expected: dict[str, object]
):
    parsed = json.loads(to_json_line(event))

    assert parsed == expected


def test_to_json_line_when_text_holds_a_line_separator_does_escape_it():
    event = TextDeltaEvent(at=5_000_000_000, chunk="a\u2028b")

    line = to_json_line(event)

    assert line == '{"type":"text_delta","at":5000000000,"chunk":"a\\u2028b"}'


def test_to_json_line_when_event_holds_lone_surrogate_does_escape_all_non_ascii():
    event = make_turn_end(at=12_000_000_000, text=_SURROGATE_TEXT, cost_usd=1e-07)

    line = to_json_line(event)

    assert line == (
        '{"type":"turn_end","at":12000000000,"text":"caf\\u00e9 \\ud800",'
        '"cost_usd":1e-07,"origin":"agent","budget_exhausted":false}'
    )


def test_to_json_line_when_lone_surrogate_and_nested_non_finite_float_does_write_null():
    event = ToolStartEvent(
        at=2_000_000_000,
        tool_use_id="t1",
        tool_name="Probe",
        input={"weights": [float("nan"), 1.5, float("-inf")], "limit": _Limit(max=float("inf"))},
        input_summary=_SURROGATE_TEXT,
    )

    decoded = loads_strict_json(to_json_line(event))

    assert decoded == {
        "type": "tool_start",
        "at": 2_000_000_000,
        "tool_use_id": "t1",
        "tool_name": "Probe",
        "input": {"weights": [None, 1.5, None], "limit": {"max": None}},
        "input_summary": _SURROGATE_TEXT,
    }


# ---------------------------------------------------------------------------
# event_from_wire — snake_case wire
# ---------------------------------------------------------------------------

ROUND_TRIP_EVENTS = [
    pytest.param(make_launch(), id="launch"),
    pytest.param(_USAGE_UPDATE, id="usage_update"),
]


@pytest.mark.parametrize("event", ROUND_TRIP_EVENTS)
def test_event_from_wire_when_given_serialized_event_does_reconstruct_it(event: SessionEvent):
    reconstructed = event_from_wire(json.loads(to_json_line(event)))

    assert reconstructed == event


@pytest.mark.parametrize(
    "obj",
    [
        pytest.param([1, 2, 3], id="non-dict-list"),
        pytest.param({"at": 1}, id="missing-type"),
        pytest.param(
            {"type": "usage_update", "at": 6_000_000_000, "cost_usd": 0.01, "schema": 1},
            id="schema-on-non-launch-event",
        ),
    ],
)
def test_event_from_wire_when_input_unrecognized_does_return_none(obj: object):
    reconstructed = event_from_wire(obj)

    assert reconstructed is None


# ---------------------------------------------------------------------------
# combine_observers
# ---------------------------------------------------------------------------


_OBSERVER_FAILURE = "observer failure"


def test_combine_observers_when_an_observer_raises_does_send_the_failure_to_the_sink():
    messages: list[str] = []
    combined = combine_observers(raising_observer(_OBSERVER_FAILURE), warn=messages.append)

    combined(UsageUpdateEvent(at=1_000_000_000, cost_usd=0.01))

    assert [_OBSERVER_FAILURE in message for message in messages] == [True]


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


def _removed_after_a_write(tmp_path: Path) -> tuple[SessionObserver, Path]:
    """A writer whose parent directory is removed after its first write."""
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    log_path = log_dir / "events.jsonl"
    writer = create_event_log_writer(log_path)
    writer(UsageUpdateEvent(at=1_000_000_000_000, cost_usd=0.01))
    shutil.rmtree(log_dir)
    return writer, log_path


def test_create_event_log_writer_when_parent_missing_does_create_it_on_write(tmp_path: Path):
    writer, log_path = _removed_after_a_write(tmp_path)
    event = UsageUpdateEvent(at=2_000_000_000_000, cost_usd=0.02)

    writer(event)

    assert read_log_lines(log_path) == [json.loads(to_json_line(event))]


def test_create_event_log_writer_when_write_fails_does_raise_gymrat_error_naming_path_from_os_error(
    tmp_path: Path,
):
    log_path = tmp_path / "a-directory"
    log_path.mkdir()
    writer = create_event_log_writer(log_path)

    with pytest.raises(GymratError, match=re.escape(str(log_path))) as exc_info:
        writer(UsageUpdateEvent(at=1_000_000_000_000, cost_usd=0.01))

    assert isinstance(exc_info.value.__cause__, OSError)


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
def test_probe_event_log_path_when_not_writable_does_raise_gymrat_error_naming_path_from_os_error(
    tmp_path: Path, build_log_path: Callable[[Path], Path]
):
    log_path = build_log_path(tmp_path)

    with pytest.raises(GymratError, match=re.escape(str(log_path))) as exc_info:
        probe_event_log_path(log_path)

    assert isinstance(exc_info.value.__cause__, OSError)


def test_probe_event_log_path_when_parent_missing_does_create_it(tmp_path: Path):
    log_path = tmp_path / "nested" / "events.jsonl"

    probe_event_log_path(log_path)

    assert log_path.parent.is_dir()
