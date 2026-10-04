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
accepts only the snake_case wire; ``combine_observers`` is pinned on ordering,
identity, and error propagation.
"""

import json
import typing
from collections.abc import Callable

import pytest
from pydantic import ValidationError

from gymrat.session.records import decode_log_line
from gymrat.supervisor.events import (
    CapEvent,
    CompactionEvent,
    DirtyInfo,
    FollowUpEvent,
    LaunchEvent,
    ModelPhaseEvent,
    SessionEvent,
    TextDeltaEvent,
    ThinkingUpdateEvent,
    ToolEndEvent,
    ToolStartEvent,
    TurnEndEvent,
    UsageUpdateEvent,
    combine_observers,
    event_from_wire,
    to_json_line,
)
from tests.supervisor._fixtures import (
    NotJsonEncodable,
    collecting_observer,
    make_launch,
    make_prompt,
)

# ---------------------------------------------------------------------------
# Event vocabulary — at: int (nanoseconds), no timestamp field
# ---------------------------------------------------------------------------

# One instance per event type, shared between the type-literal, JSON-serialization,
# and wire-round-trip tests below so each event's fields are declared exactly once.
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
_MODEL_PHASE_RESPONDING = ModelPhaseEvent(at=9_000_000_000, phase="responding")
_MODEL_PHASE_TOOL_INPUT = ModelPhaseEvent(
    at=10_000_000_000, phase="tool_input", tool_name="Read", parent_tool_use_id="p1"
)
_MODEL_PHASE_TURN_END = ModelPhaseEvent(at=11_000_000_000, phase="turn_end")
_TURN_END = TurnEndEvent(
    at=12_000_000_000, text="done", cost_usd=0.05, origin="agent", budget_exhausted=False
)
_FOLLOW_UP = FollowUpEvent(at=13_000_000_000, action="replied", reason="user asked", text="hello")
_COMPACTION = CompactionEvent(at=14_000_000_000)

#: Every event sample with its type literal and a unique parametrize id.
#: The four ``model_phase`` variants carry phase-specific ids so pytest's
#: strict parametrize-id uniqueness check passes.
EVENT_SAMPLES: list[tuple[object, str, str]] = [
    (_THINKING_UPDATE, "thinking_update", "thinking_update"),
    (_TOOL_START, "tool_start", "tool_start"),
    (_TOOL_END, "tool_end", "tool_end"),
    (_TEXT_DELTA, "text_delta", "text_delta"),
    (_USAGE_UPDATE, "usage_update", "usage_update"),
    (_CAP, "cap", "cap"),
    (_MODEL_PHASE_THINKING, "model_phase", "model_phase-thinking"),
    (_MODEL_PHASE_RESPONDING, "model_phase", "model_phase-responding"),
    (_MODEL_PHASE_TOOL_INPUT, "model_phase", "model_phase-tool_input"),
    (_MODEL_PHASE_TURN_END, "model_phase", "model_phase-turn_end"),
    (make_launch(), "launch", "launch"),
    (_TURN_END, "turn_end", "turn_end"),
    (_FOLLOW_UP, "follow_up", "follow_up"),
    (_COMPACTION, "compaction", "compaction"),
]


@pytest.mark.parametrize(
    ("event", "expected_type"),
    [pytest.param(event, type_, id=id_) for event, type_, id_ in EVENT_SAMPLES],
)
def test_event_when_constructed_does_expose_its_type_literal(event: object, expected_type: str):
    assert event.type == expected_type  # type: ignore[attr-defined]


def test_session_event_union_when_enumerated_does_expose_exactly_eleven_type_literals():
    event_classes = typing.get_args(SessionEvent)
    types = {cls.model_fields["type"].default for cls in event_classes}

    assert types == {
        "thinking_update",
        "tool_start",
        "tool_end",
        "text_delta",
        "usage_update",
        "cap",
        "model_phase",
        "launch",
        "turn_end",
        "follow_up",
        "compaction",
    }


def test_event_when_field_reassigned_does_raise():
    event = UsageUpdateEvent(at=6_000_000_000, cost_usd=0.01)

    with pytest.raises(ValidationError, match="frozen"):
        event.cost_usd = 0.02  # type: ignore[misc]


def test_dirty_info_when_constructed_does_carry_file_count():
    dirty = DirtyInfo(file_count=3)

    assert dirty.file_count == 3


# ---------------------------------------------------------------------------
# EventEnvelope — at: int (nanoseconds), no timestamp
# ---------------------------------------------------------------------------


def test_event_when_constructed_does_carry_at_in_nanoseconds():
    event = UsageUpdateEvent(at=1_234_567_890_000_000_000, cost_usd=0.01)

    assert event.at == 1_234_567_890_000_000_000


def test_event_when_constructed_does_not_have_timestamp_attribute():
    event = UsageUpdateEvent(at=1_000_000_000, cost_usd=0.01)

    assert not hasattr(event, "timestamp")


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
        _COMPACTION,
        {
            "type": "compaction",
            "at": 14_000_000_000,
        },
        id="compaction",
    ),
]


@pytest.mark.parametrize(("event", "expected"), JSON_CASES)
def test_to_json_line_when_serializing_does_use_snake_case_keys(
    event: object, expected: dict[str, object]
):
    parsed = json.loads(to_json_line(event))  # type: ignore[arg-type]

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
        pytest.param(
            TextDeltaEvent(at=5_000_000_000, chunk="café \ud800"),
            '{"type":"text_delta","at":5000000000,"chunk":"caf\\u00e9 \\ud800"}',
            id="lone-surrogate-escapes-all-non-ascii",
        ),
    ],
)
def test_to_json_line_when_serializing_does_write_exact_compact_line(
    event: SessionEvent, expected_line: str
):
    assert to_json_line(event) == expected_line


def test_to_json_line_when_none_field_does_omit_it():
    event = ThinkingUpdateEvent(at=1_000_000_000, estimated_tokens=100, delta=10)

    parsed = json.loads(to_json_line(event))

    assert "parent_tool_use_id" not in parsed


def test_to_json_line_when_tool_start_input_is_none_does_keep_input_key():
    event = ToolStartEvent(
        at=2_000_000_000,
        tool_use_id="t1",
        tool_name="Read",
        input=None,
        input_summary="none",
    )

    parsed = json.loads(to_json_line(event))

    assert "input" in parsed
    assert parsed["input"] is None


#: The wire form of a no-optionals ``LaunchEvent``, shared between the
#: ``to_json_line`` omission test below and the schema/session_id wire
#: round-trip test further down.
_LAUNCH_WIRE_NO_OPTIONALS: dict[str, object] = {
    "type": "launch",
    "at": 1_000_000_000_000,
    "schema": 1,
    "session_id": "20260813-125044-34ec",
    "head_sha": "abc123def",
    "dirty": False,
    "max_minutes": 5,
    "runbook_path": "/path/to/runbook.md",
    "kickoff_summary": "test kickoff",
}


def test_to_json_line_when_launch_has_no_optionals_does_omit_max_usd_model_and_effort():
    event = make_launch(max_usd=None, model=None, effort=None)

    parsed = json.loads(to_json_line(event))

    assert parsed == _LAUNCH_WIRE_NO_OPTIONALS


def test_to_json_line_when_launch_has_optionals_does_emit_them():
    event = make_launch(max_usd=1.5, model="opus", effort="high", dirty=DirtyInfo(file_count=4))

    parsed = json.loads(to_json_line(event))

    assert parsed["max_usd"] == 1.5
    assert parsed["model"] == "opus"
    assert parsed["effort"] == "high"
    assert parsed["dirty"] == {"file_count": 4}


# ---------------------------------------------------------------------------
# LaunchEvent — schema and session_id
# ---------------------------------------------------------------------------


def test_launch_event_when_constructed_does_carry_schema_one():
    event = make_launch()

    assert event.schema_version == 1


def test_launch_event_when_constructed_does_carry_session_id():
    event = make_launch(session_id="my-session-id")

    assert event.session_id == "my-session-id"


def test_to_json_line_when_launch_does_emit_schema_and_session_id():
    event = make_launch()

    parsed = json.loads(to_json_line(event))

    assert parsed["schema"] == 1
    assert parsed["session_id"] == "20260813-125044-34ec"


# ---------------------------------------------------------------------------
# FollowUpEvent — unset optional fields omitted
# ---------------------------------------------------------------------------


def test_to_json_line_when_follow_up_has_no_optionals_does_omit_reason_and_text():
    event = FollowUpEvent(at=20_000_000_000, action="waiting")

    parsed = json.loads(to_json_line(event))

    assert parsed == {
        "type": "follow_up",
        "at": 20_000_000_000,
        "action": "waiting",
    }


def test_to_json_line_when_follow_up_has_optionals_does_emit_them():
    event = FollowUpEvent(at=20_000_000_000, action="ended", reason="budget", text="bye")

    parsed = json.loads(to_json_line(event))

    assert parsed["reason"] == "budget"
    assert parsed["text"] == "bye"


# ---------------------------------------------------------------------------
# SessionPrompt — max_budget_usd
# ---------------------------------------------------------------------------


def test_session_prompt_when_max_budget_usd_given_does_carry_it():
    prompt = make_prompt(max_budget_usd=2.5)

    assert prompt.max_budget_usd == 2.5


def test_session_prompt_when_max_budget_usd_omitted_does_default_to_none():
    prompt = make_prompt()

    assert prompt.max_budget_usd is None


# ---------------------------------------------------------------------------
# event_from_wire — snake_case wire
# ---------------------------------------------------------------------------

ROUND_TRIP_EVENTS = [pytest.param(event, id=id_) for event, _, id_ in EVENT_SAMPLES] + [
    pytest.param(
        make_launch(max_usd=1.5, model="opus", dirty=DirtyInfo(file_count=4)),
        id="launch-with-optionals",
    ),
    pytest.param(
        FollowUpEvent(at=20_000_000_000, action="waiting"),
        id="follow_up-no-optionals",
    ),
    pytest.param(
        TextDeltaEvent(at=5_000_000_000, chunk="a\x85b\u2028c\u2029d"),
        id="text_delta-unicode-line-breaks",
    ),
    pytest.param(
        TextDeltaEvent(at=5_000_000_000, chunk="café \ud800"),
        id="text_delta-lone-surrogate",
    ),
]


@pytest.mark.parametrize("event", ROUND_TRIP_EVENTS)
def test_event_from_wire_when_given_serialized_event_does_reconstruct_it(event: object):
    assert event_from_wire(json.loads(to_json_line(event))) == event  # type: ignore[arg-type]


def test_event_from_wire_when_tool_start_input_is_none_does_round_trip():
    event = ToolStartEvent(
        at=2_000_000_000,
        tool_use_id="t1",
        tool_name="Read",
        input=None,
        input_summary="none",
    )

    assert event_from_wire(json.loads(to_json_line(event))) == event


@pytest.mark.parametrize("settled", [True, False])
def test_event_from_wire_when_usage_update_line_has_settled_does_parse_with_same_cost(
    settled: bool,
):
    obj = {"type": "usage_update", "at": 6_000_000_000, "cost_usd": 0.01, "settled": settled}

    event = event_from_wire(obj)

    assert event == UsageUpdateEvent(at=6_000_000_000, cost_usd=0.01)


@pytest.mark.parametrize(
    "obj",
    [
        pytest.param([1, 2, 3], id="non-dict-list"),
        pytest.param("nope", id="non-dict-str"),
        pytest.param({"at": 1}, id="missing-type"),
        pytest.param({"type": "mystery", "at": 1}, id="unknown-type"),
        pytest.param({"type": "usage_update", "at": 6}, id="missing-required-field"),
        pytest.param(
            {"type": "cap", "at": 7_000_000_000, "cap": "wall-clock"}, id="cap-missing-action"
        ),
    ],
)
def test_event_from_wire_when_input_unrecognized_does_return_none(obj: object):
    assert event_from_wire(obj) is None


def test_event_from_wire_when_camel_case_keys_does_return_none():
    obj = {"type": "usage_update", "timestamp": 6, "costUsd": 0.01}

    assert event_from_wire(obj) is None


@pytest.mark.parametrize(
    "obj",
    [
        pytest.param(
            {"type": "model_phase", "at": 1_000_000_000, "phase": "unknown"},
            id="unknown-phase",
        ),
        pytest.param(
            {"type": "model_phase", "phase": "thinking"},
            id="missing-at",
        ),
    ],
)
def test_event_from_wire_when_model_phase_invalid_does_return_none(obj: object):
    assert event_from_wire(obj) is None


# ---------------------------------------------------------------------------
# launch event — schema key required
# ---------------------------------------------------------------------------


def test_event_from_wire_when_launch_lacks_schema_does_return_none():
    wire = json.loads(to_json_line(make_launch()))
    del wire["schema"]

    assert event_from_wire(wire) is None


# ---------------------------------------------------------------------------
# schema key rejected on non-launch events
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "event",
    [
        pytest.param(_USAGE_UPDATE, id="usage_update"),
        pytest.param(_COMPACTION, id="compaction"),
        pytest.param(_TURN_END, id="turn_end"),
    ],
)
def test_event_from_wire_when_schema_on_non_launch_event_does_return_none(event: object):
    wire = {**json.loads(to_json_line(event)), "schema": 1}  # type: ignore[arg-type]

    assert event_from_wire(wire) is None


# ---------------------------------------------------------------------------
# parent_tool_use_id on existing events
# ---------------------------------------------------------------------------


# Each entry is (event without parent_tool_use_id, event with it set, expected id).
# The no-parent events reuse the shared samples above so each event's fields are
# still declared exactly once.
_PARENT_ID_CASES = [
    (
        _THINKING_UPDATE,
        ThinkingUpdateEvent(
            at=1_000_000_000, estimated_tokens=100, delta=10, parent_tool_use_id="p1"
        ),
        "p1",
        "thinking_update",
    ),
    (
        _TOOL_START,
        ToolStartEvent(
            at=2_000_000_000,
            tool_use_id="t1",
            tool_name="Read",
            input={"path": "/x"},
            input_summary="/x",
            parent_tool_use_id="p2",
        ),
        "p2",
        "tool_start",
    ),
    (
        _TOOL_END,
        ToolEndEvent(
            at=4_000_000_000,
            tool_use_id="t1",
            tool_name="Read",
            duration_ms=750,
            result="ok",
            result_summary="ok",
            parent_tool_use_id="p3",
        ),
        "p3",
        "tool_end",
    ),
    (
        _TEXT_DELTA,
        TextDeltaEvent(at=5_000_000_000, chunk="hello", parent_tool_use_id="p4"),
        "p4",
        "text_delta",
    ),
]

_WITHOUT_PARENT_ID_PARAMS = [
    pytest.param(no_parent, id=case_id) for no_parent, _, _, case_id in _PARENT_ID_CASES
]
_WITH_PARENT_ID_PARAMS = [
    pytest.param(with_parent, expected_id, id=case_id)
    for _, with_parent, expected_id, case_id in _PARENT_ID_CASES
]


@pytest.mark.parametrize("event", _WITHOUT_PARENT_ID_PARAMS)
def test_event_when_constructed_without_parent_tool_use_id_does_default_to_none(event: object):
    assert event.parent_tool_use_id is None  # type: ignore[attr-defined]


@pytest.mark.parametrize(("event", "expected_id"), _WITH_PARENT_ID_PARAMS)
def test_event_when_constructed_with_parent_tool_use_id_does_carry_it(
    event: object, expected_id: str
):
    assert event.parent_tool_use_id == expected_id  # type: ignore[attr-defined]


@pytest.mark.parametrize(("event", "expected_id"), _WITH_PARENT_ID_PARAMS)
def test_to_json_line_when_parent_tool_use_id_set_does_render_snake_case(
    event: object, expected_id: str
):
    parsed = json.loads(to_json_line(event))  # type: ignore[arg-type]

    assert parsed["parent_tool_use_id"] == expected_id


@pytest.mark.parametrize(
    "event",
    [pytest.param(with_parent, id=case_id) for _, with_parent, _, case_id in _PARENT_ID_CASES],
)
def test_event_from_wire_when_parent_tool_use_id_set_does_round_trip(event: object):
    assert event_from_wire(json.loads(to_json_line(event))) == event  # type: ignore[arg-type]


def test_event_from_wire_when_text_delta_lacks_parent_tool_use_id_does_default_to_none():
    wire = {"type": "text_delta", "at": 5_000_000_000, "chunk": "hello"}

    event = event_from_wire(wire)

    assert isinstance(event, TextDeltaEvent)
    assert event.parent_tool_use_id is None


# ---------------------------------------------------------------------------
# combine_observers
# ---------------------------------------------------------------------------


def test_combine_observers_when_invoked_does_call_each_with_the_identical_event():
    first = collecting_observer()
    second = collecting_observer()
    combined = combine_observers(first.observer, second.observer)
    event = UsageUpdateEvent(at=1_000_000_000, cost_usd=0.01)

    combined(event)

    assert first.events[0] is event
    assert second.events[0] is event


def test_combine_observers_when_given_no_observers_does_not_raise():
    combined = combine_observers()
    event = UsageUpdateEvent(at=1_000_000_000, cost_usd=0.01)

    combined(event)


def test_combine_observers_when_an_observer_raises_does_warn_and_call_remaining():
    boom_message = "observer failure"

    def boom(_: object) -> None:
        raise RuntimeError(boom_message)

    later = collecting_observer()
    combined = combine_observers(boom, later.observer)
    event = UsageUpdateEvent(at=1_000_000_000, cost_usd=0.01)

    with pytest.warns(RuntimeWarning, match=boom_message) as caught:
        combined(event)

    assert later.events == [event]
    assert [warning.filename for warning in caught] == [__file__]


# ---------------------------------------------------------------------------
# non-finite float serialization
# ---------------------------------------------------------------------------


_NON_FINITE_FLOATS = [
    pytest.param(float("nan"), id="nan"),
    pytest.param(float("inf"), id="positive-infinity"),
    pytest.param(float("-inf"), id="negative-infinity"),
]


@pytest.mark.parametrize("value", _NON_FINITE_FLOATS)
def test_to_json_line_when_nested_float_is_non_finite_does_serialize_as_null(value: float):
    event = ToolStartEvent(
        at=1_000_000_000,
        tool_use_id="t1",
        tool_name="Probe",
        input={"limit": value},
        input_summary="probe",
    )

    line = to_json_line(event)

    parsed = json.loads(line)
    assert parsed["input"]["limit"] is None


#: A turn-end text holding a lone surrogate, which forces the ASCII-escaping
#: fallback serializer, next to a raw non-ASCII character it must escape.
_SURROGATE_TEXT = "café \ud800"


@pytest.mark.parametrize(
    ("cost", "expected_cost"),
    [
        pytest.param(0.05, "0.05", id="finite-unchanged"),
        pytest.param(1e-07, "1e-07", id="finite-exponent-unchanged"),
    ],
)
def test_to_json_line_when_lone_surrogate_and_float_does_write_escaped_line_with_json_float(
    cost: float, expected_cost: str
):
    event = TurnEndEvent(
        at=12_000_000_000,
        text=_SURROGATE_TEXT,
        cost_usd=cost,
        origin="agent",
        budget_exhausted=False,
    )

    line = to_json_line(event)

    assert line == (
        '{"type":"turn_end","at":12000000000,"text":"caf\\u00e9 \\ud800",'
        f'"cost_usd":{expected_cost},"origin":"agent","budget_exhausted":false}}'
    )


def test_to_json_line_when_lone_surrogate_and_nested_non_finite_float_does_write_null():
    event = ToolStartEvent(
        at=2_000_000_000,
        tool_use_id="t1",
        tool_name="Probe",
        input={"weights": [float("nan"), 1.5, float("-inf")], "limit": {"max": float("inf")}},
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
# non-finite floats refused at construction
# ---------------------------------------------------------------------------


def _usage_update_with_cost(value: float) -> UsageUpdateEvent:
    return UsageUpdateEvent(at=1_000_000_000, cost_usd=value)


def _turn_end_with_cost(value: float) -> TurnEndEvent:
    return TurnEndEvent(
        at=1_000_000_000,
        text="done",
        cost_usd=value,
        origin="agent",
        budget_exhausted=False,
    )


def _launch_with_max_minutes(value: float) -> LaunchEvent:
    return make_launch(max_minutes=value)


def _launch_with_max_usd(value: float) -> LaunchEvent:
    return make_launch(max_usd=value)


@pytest.mark.parametrize("value", _NON_FINITE_FLOATS)
@pytest.mark.parametrize(
    "build",
    [
        pytest.param(_usage_update_with_cost, id="usage-update-cost"),
        pytest.param(_turn_end_with_cost, id="turn-end-cost"),
        pytest.param(_launch_with_max_minutes, id="launch-max-minutes"),
        pytest.param(_launch_with_max_usd, id="launch-max-usd"),
    ],
)
def test_event_when_float_field_is_non_finite_does_raise(
    build: Callable[[float], object], value: float
):
    with pytest.raises(ValidationError, match="finite number"):
        build(value)


# ---------------------------------------------------------------------------
# non-JSON-encodable field fallback
# ---------------------------------------------------------------------------


def test_to_json_line_when_field_not_json_encodable_does_stringify_instead_of_raising():
    event = ToolStartEvent(
        at=1_000_000_000,
        tool_use_id="t1",
        tool_name="Test",
        input={"key": NotJsonEncodable()},
        input_summary="test",
    )

    line = to_json_line(event)

    parsed = json.loads(line)
    assert parsed["input"]["key"] == "not-json-encodable"
