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
``combine_observers`` sends every observer failure to its warn sink, which
defaults to stderr.

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
from functools import partial
from pathlib import Path

import pytest
from pydantic import BaseModel, ValidationError

from gymrat.errors import GymratError
from gymrat.session.records import decode_log_line
from gymrat.supervisor.events import (
    CapEvent,
    CompactionEvent,
    DirtyInfo,
    FollowUpEvent,
    ModelPhaseEvent,
    SessionEvent,
    TextDeltaEvent,
    ThinkingUpdateEvent,
    ToolEndEvent,
    ToolStartEvent,
    TurnEndEvent,
    UsageUpdateEvent,
    combine_observers,
    create_event_log_writer,
    event_from_wire,
    probe_event_log_path,
    to_json_line,
)
from tests.session.records._fixtures import (
    SUPERVISED_SESSION_ID,
)
from tests.supervisor._fixtures import (
    NotJsonEncodable,
    make_launch,
    make_turn_end,
    raising_observer,
    read_log_lines,
)

# ---------------------------------------------------------------------------
# Event vocabulary — at: int (nanoseconds)
# ---------------------------------------------------------------------------

# One instance per event type, shared between the JSON-serialization and
# wire-round-trip tests below so each event's fields are declared exactly once.
_THINKING_UPDATE = ThinkingUpdateEvent(at=1_000_000_000, estimated_tokens=100, delta=10)
_TOOL_START = ToolStartEvent(
    at=2_000_000_000, tool_use_id="t1", tool_name="Read", input={"path": "/x"}, input_summary="/x"
)
_TOOL_END = ToolEndEvent(
    at=4_000_000_000,
    tool_use_id="t1",
    tool_name="Read",
    duration_ms=750,
    result="ok",
    result_summary="ok",
)
_TEXT_DELTA = TextDeltaEvent(at=5_000_000_000, chunk="hello")
_USAGE_UPDATE = UsageUpdateEvent(at=6_000_000_000, cost_usd=0.01)
_CAP = CapEvent(at=7_000_000_000, cap="wall-clock", action="interrupting")
_MODEL_PHASE_THINKING = ModelPhaseEvent(at=8_000_000_000, phase="thinking")
_MODEL_PHASE_TOOL_INPUT = ModelPhaseEvent(
    at=10_000_000_000, phase="tool_input", tool_name="Read", parent_tool_use_id="p1"
)
_TURN_END = TurnEndEvent(
    at=12_000_000_000, text="done", cost_usd=0.05, origin="agent", budget_exhausted=False
)
_FOLLOW_UP = FollowUpEvent(at=13_000_000_000, action="replied", reason="user asked", text="hello")
_COMPACTION = CompactionEvent(at=14_000_000_000)
_LAUNCH_WITH_OPTIONALS = make_launch(
    max_usd=1.5, model="opus", effort="high", dirty=DirtyInfo(file_count=4)
)

#: Every event sample with a unique parametrize id. The two ``model_phase``
#: variants carry phase-specific ids so pytest's strict parametrize-id
#: uniqueness check passes.
EVENT_SAMPLES: list[tuple[SessionEvent, str]] = [
    (_THINKING_UPDATE, "thinking_update"),
    (_TOOL_START, "tool_start"),
    (_TOOL_END, "tool_end"),
    (_TEXT_DELTA, "text_delta"),
    (_USAGE_UPDATE, "usage_update"),
    (_CAP, "cap"),
    (_MODEL_PHASE_THINKING, "model_phase-thinking"),
    (_MODEL_PHASE_TOOL_INPUT, "model_phase-tool_input"),
    (make_launch(), "launch"),
    (_TURN_END, "turn_end"),
    (_FOLLOW_UP, "follow_up"),
    (_COMPACTION, "compaction"),
]

# Sub-agent events: the samples above with a ``parent_tool_use_id`` set.
_THINKING_UPDATE_WITH_PARENT = ThinkingUpdateEvent(
    at=1_000_000_000, estimated_tokens=100, delta=10, parent_tool_use_id="p1"
)
_TOOL_START_WITH_PARENT = ToolStartEvent(
    at=2_000_000_000,
    tool_use_id="t1",
    tool_name="Read",
    input={"path": "/x"},
    input_summary="/x",
    parent_tool_use_id="p2",
)
_TOOL_END_WITH_PARENT = ToolEndEvent(
    at=4_000_000_000,
    tool_use_id="t1",
    tool_name="Read",
    duration_ms=750,
    result="ok",
    result_summary="ok",
    parent_tool_use_id="p3",
)
_TEXT_DELTA_WITH_PARENT = TextDeltaEvent(at=5_000_000_000, chunk="hello", parent_tool_use_id="p4")

_TOOL_START_INPUT_NONE = ToolStartEvent(
    at=2_000_000_000, tool_use_id="t1", tool_name="Read", input=None, input_summary="none"
)
_FOLLOW_UP_NO_OPTIONALS = FollowUpEvent(at=20_000_000_000, action="waiting")

#: Every non-finite float, each with its parametrize id.
_NON_FINITE_FLOATS = [
    (float("nan"), "nan"),
    (float("inf"), "positive-infinity"),
    (float("-inf"), "negative-infinity"),
]

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
        _THINKING_UPDATE,
        {
            "type": "thinking_update",
            "at": 1_000_000_000,
            "estimated_tokens": 100,
            "delta": 10,
        },
        id="thinking_update",
    ),
    pytest.param(
        _TOOL_START,
        {
            "type": "tool_start",
            "at": 2_000_000_000,
            "tool_use_id": "t1",
            "tool_name": "Read",
            "input": {"path": "/x"},
            "input_summary": "/x",
        },
        id="tool_start",
    ),
    pytest.param(
        _TOOL_START_INPUT_NONE,
        {
            "type": "tool_start",
            "at": 2_000_000_000,
            "tool_use_id": "t1",
            "tool_name": "Read",
            "input": None,
            "input_summary": "none",
        },
        id="tool_start-input-none-keeps-key",
    ),
    pytest.param(
        _TOOL_END,
        {
            "type": "tool_end",
            "at": 4_000_000_000,
            "tool_use_id": "t1",
            "tool_name": "Read",
            "duration_ms": 750,
            "result": "ok",
            "result_summary": "ok",
        },
        id="tool_end",
    ),
    pytest.param(
        _TEXT_DELTA,
        {"type": "text_delta", "at": 5_000_000_000, "chunk": "hello"},
        id="text_delta",
    ),
    pytest.param(
        _USAGE_UPDATE,
        {"type": "usage_update", "at": 6_000_000_000, "cost_usd": 0.01},
        id="usage_update",
    ),
    pytest.param(
        _CAP,
        {"type": "cap", "at": 7_000_000_000, "cap": "wall-clock", "action": "interrupting"},
        id="cap",
    ),
    pytest.param(
        _MODEL_PHASE_THINKING,
        {
            "type": "model_phase",
            "at": 8_000_000_000,
            "phase": "thinking",
        },
        id="model_phase-thinking",
    ),
    pytest.param(
        _MODEL_PHASE_TOOL_INPUT,
        {
            "type": "model_phase",
            "at": 10_000_000_000,
            "phase": "tool_input",
            "tool_name": "Read",
            "parent_tool_use_id": "p1",
        },
        id="model_phase-tool_input",
    ),
    pytest.param(
        make_launch(),
        {
            "type": "launch",
            "at": 1_000_000_000_000,
            "schema": 1,
            "session_id": SUPERVISED_SESSION_ID,
            "head_sha": "abc123def",
            "dirty": False,
            "max_minutes": 5,
            "runbook_path": "/path/to/runbook.md",
            "kickoff_summary": "test kickoff",
        },
        id="launch-no-optionals",
    ),
    pytest.param(
        _LAUNCH_WITH_OPTIONALS,
        {
            "type": "launch",
            "at": 1_000_000_000_000,
            "schema": 1,
            "session_id": SUPERVISED_SESSION_ID,
            "head_sha": "abc123def",
            "dirty": {"file_count": 4},
            "max_minutes": 5,
            "max_usd": 1.5,
            "model": "opus",
            "effort": "high",
            "runbook_path": "/path/to/runbook.md",
            "kickoff_summary": "test kickoff",
        },
        id="launch-with-optionals",
    ),
    pytest.param(
        _TURN_END,
        {
            "type": "turn_end",
            "at": 12_000_000_000,
            "text": "done",
            "cost_usd": 0.05,
            "origin": "agent",
            "budget_exhausted": False,
        },
        id="turn_end",
    ),
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
    pytest.param(
        _COMPACTION,
        {
            "type": "compaction",
            "at": 14_000_000_000,
        },
        id="compaction",
    ),
    pytest.param(
        _THINKING_UPDATE_WITH_PARENT,
        {
            "type": "thinking_update",
            "at": 1_000_000_000,
            "estimated_tokens": 100,
            "delta": 10,
            "parent_tool_use_id": "p1",
        },
        id="thinking_update-with-parent",
    ),
    pytest.param(
        _TOOL_START_WITH_PARENT,
        {
            "type": "tool_start",
            "at": 2_000_000_000,
            "tool_use_id": "t1",
            "tool_name": "Read",
            "input": {"path": "/x"},
            "input_summary": "/x",
            "parent_tool_use_id": "p2",
        },
        id="tool_start-with-parent",
    ),
    pytest.param(
        _TOOL_END_WITH_PARENT,
        {
            "type": "tool_end",
            "at": 4_000_000_000,
            "tool_use_id": "t1",
            "tool_name": "Read",
            "duration_ms": 750,
            "result": "ok",
            "result_summary": "ok",
            "parent_tool_use_id": "p3",
        },
        id="tool_end-with-parent",
    ),
    pytest.param(
        _TEXT_DELTA_WITH_PARENT,
        {"type": "text_delta", "at": 5_000_000_000, "chunk": "hello", "parent_tool_use_id": "p4"},
        id="text_delta-with-parent",
    ),
    *(
        pytest.param(
            ToolStartEvent(
                at=1_000_000_000,
                tool_use_id="t1",
                tool_name="Probe",
                input={"limit": value},
                input_summary="probe",
            ),
            {
                "type": "tool_start",
                "at": 1_000_000_000,
                "tool_use_id": "t1",
                "tool_name": "Probe",
                "input": {"limit": None},
                "input_summary": "probe",
            },
            id=f"nested-{case_id}-as-null",
        )
        for value, case_id in _NON_FINITE_FLOATS
    ),
    pytest.param(
        ToolStartEvent(
            at=1_000_000_000,
            tool_use_id="t1",
            tool_name="Test",
            input={"key": NotJsonEncodable()},
            input_summary="test",
        ),
        {
            "type": "tool_start",
            "at": 1_000_000_000,
            "tool_use_id": "t1",
            "tool_name": "Test",
            "input": {"key": "not-json-encodable"},
            "input_summary": "test",
        },
        id="not-json-encodable-stringified",
    ),
]


@pytest.mark.parametrize(("event", "expected"), JSON_CASES)
def test_to_json_line_when_given_event_does_write_its_wire_object(
    event: SessionEvent, expected: dict[str, object]
):
    parsed = json.loads(to_json_line(event))

    assert parsed == expected


@pytest.mark.parametrize(
    ("event", "expected_line"),
    [
        pytest.param(
            TextDeltaEvent(at=5_000_000_000, chunk="café \U0001f3af"),
            '{"type":"text_delta","at":5000000000,"chunk":"café \U0001f3af"}',
            id="non-ascii-written-raw",
        ),
        pytest.param(
            UsageUpdateEvent(at=6_000_000_000, cost_usd=1e-7),
            '{"type":"usage_update","at":6000000000,"cost_usd":1e-7}',
            id="float-exponent-without-leading-zero",
        ),
        pytest.param(
            TextDeltaEvent(at=5_000_000_000, chunk="a\x85b"),
            '{"type":"text_delta","at":5000000000,"chunk":"a\\u0085b"}',
            id="next-line-escaped",
        ),
        pytest.param(
            TextDeltaEvent(at=5_000_000_000, chunk="a\u2028b"),
            '{"type":"text_delta","at":5000000000,"chunk":"a\\u2028b"}',
            id="line-separator-escaped",
        ),
        pytest.param(
            TextDeltaEvent(at=5_000_000_000, chunk="a\u2029b"),
            '{"type":"text_delta","at":5000000000,"chunk":"a\\u2029b"}',
            id="paragraph-separator-escaped",
        ),
    ],
)
def test_to_json_line_when_text_is_utf8_encodable_does_write_exact_compact_line(
    event: SessionEvent, expected_line: str
):
    line = to_json_line(event)

    assert line == expected_line


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

    decoded = decode_log_line(to_json_line(event))

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

ROUND_TRIP_EVENTS = [pytest.param(event, id=id_) for event, id_ in EVENT_SAMPLES] + [
    pytest.param(_LAUNCH_WITH_OPTIONALS, id="launch-with-optionals"),
    pytest.param(
        TextDeltaEvent(at=5_000_000_000, chunk="a\x85b\u2028c\u2029d"),
        id="text_delta-unicode-line-breaks",
    ),
    pytest.param(
        TextDeltaEvent(at=5_000_000_000, chunk=_SURROGATE_TEXT),
        id="text_delta-lone-surrogate",
    ),
    pytest.param(_TOOL_START_INPUT_NONE, id="tool_start-input-none"),
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
        pytest.param({"type": "mystery", "at": 1}, id="unknown-type"),
        pytest.param({"type": "usage_update", "at": 6}, id="missing-required-field"),
        pytest.param(
            {"type": "model_phase", "at": 1_000_000_000, "phase": "unknown"},
            id="model-phase-unknown",
        ),
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


def test_combine_observers_when_an_observer_raises_twice_does_send_each_failure_to_the_sink():
    messages: list[str] = []
    combined = combine_observers(raising_observer(_OBSERVER_FAILURE), warn=messages.append)
    event = UsageUpdateEvent(at=1_000_000_000, cost_usd=0.01)

    combined(event)
    combined(event)

    assert [_OBSERVER_FAILURE in message for message in messages] == [True, True]


def test_combine_observers_when_no_sink_given_does_write_the_failure_to_stderr(
    capsys: pytest.CaptureFixture[str],
):
    combined = combine_observers(raising_observer(_OBSERVER_FAILURE))
    event = UsageUpdateEvent(at=1_000_000_000, cost_usd=0.01)

    combined(event)

    assert _OBSERVER_FAILURE in capsys.readouterr().err


# ---------------------------------------------------------------------------
# non-finite floats refused at construction
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value", [pytest.param(value, id=case_id) for value, case_id in _NON_FINITE_FLOATS]
)
@pytest.mark.parametrize(
    ("build", "field"),
    [
        pytest.param(
            partial(UsageUpdateEvent, at=1_000_000_000), "cost_usd", id="usage-update-cost"
        ),
        pytest.param(
            partial(make_turn_end, at=1_000_000_000, text="done"), "cost_usd", id="turn-end-cost"
        ),
        pytest.param(make_launch, "max_minutes", id="launch-max-minutes"),
        pytest.param(make_launch, "max_usd", id="launch-max-usd"),
    ],
)
def test_event_when_float_field_is_non_finite_does_raise(
    build: Callable[..., object], field: str, value: float
):
    with pytest.raises(ValidationError, match="finite number"):
        build(**{field: value})


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
