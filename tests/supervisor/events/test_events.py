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
``combine_observers`` is pinned on ordering, identity, and error propagation.
"""

import json
from collections.abc import Callable
from functools import partial

import pytest
from pydantic import ValidationError

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
    event_from_wire,
    to_json_line,
)
from tests.session.records._fixtures import (
    SUPERVISED_SESSION_ID,
)
from tests.supervisor._fixtures import (
    NotJsonEncodable,
    collecting_observer,
    make_launch,
    make_turn_end,
)

# ---------------------------------------------------------------------------
# Event vocabulary — at: int (nanoseconds)
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
_MODEL_PHASE_TOOL_INPUT = ModelPhaseEvent(
    at=10_000_000_000, phase="tool_input", tool_name="Read", parent_tool_use_id="p1"
)
_TURN_END = TurnEndEvent(
    at=12_000_000_000, text="done", cost_usd=0.05, origin="agent", budget_exhausted=False
)
_FOLLOW_UP = FollowUpEvent(at=13_000_000_000, action="replied", reason="user asked", text="hello")
_COMPACTION = CompactionEvent(at=14_000_000_000)

#: Every event sample with its type literal and a unique parametrize id.
#: The two ``model_phase`` variants carry phase-specific ids so pytest's
#: strict parametrize-id uniqueness check passes.
EVENT_SAMPLES: list[tuple[SessionEvent, str, str]] = [
    (_THINKING_UPDATE, "thinking_update", "thinking_update"),
    (_TOOL_START, "tool_start", "tool_start"),
    (_TOOL_END, "tool_end", "tool_end"),
    (_TEXT_DELTA, "text_delta", "text_delta"),
    (_USAGE_UPDATE, "usage_update", "usage_update"),
    (_CAP, "cap", "cap"),
    (_MODEL_PHASE_THINKING, "model_phase", "model_phase-thinking"),
    (_MODEL_PHASE_TOOL_INPUT, "model_phase", "model_phase-tool_input"),
    (make_launch(), "launch", "launch"),
    (_TURN_END, "turn_end", "turn_end"),
    (_FOLLOW_UP, "follow_up", "follow_up"),
    (_COMPACTION, "compaction", "compaction"),
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
        make_launch(max_usd=1.5, model="opus", effort="high", dirty=DirtyInfo(file_count=4)),
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
        for value, case_id in [
            (float("nan"), "nan"),
            (float("inf"), "positive-infinity"),
            (float("-inf"), "negative-infinity"),
        ]
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
def test_to_json_line_when_serializing_does_use_snake_case_keys(
    event: SessionEvent, expected: dict[str, object]
):
    parsed = json.loads(to_json_line(event))

    assert parsed == expected


#: A turn-end text holding a lone surrogate, which forces the ASCII-escaping
#: fallback serializer, next to a raw non-ASCII character it must escape.
_SURROGATE_TEXT = "café \ud800"


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
        pytest.param(
            make_turn_end(at=12_000_000_000, text=_SURROGATE_TEXT, cost_usd=0.05),
            '{"type":"turn_end","at":12000000000,"text":"caf\\u00e9 \\ud800",'
            '"cost_usd":0.05,"origin":"agent","budget_exhausted":false}',
            id="lone-surrogate-keeps-finite-float",
        ),
        pytest.param(
            make_turn_end(at=12_000_000_000, text=_SURROGATE_TEXT, cost_usd=1e-07),
            '{"type":"turn_end","at":12000000000,"text":"caf\\u00e9 \\ud800",'
            '"cost_usd":1e-07,"origin":"agent","budget_exhausted":false}',
            id="lone-surrogate-keeps-float-exponent",
        ),
    ],
)
def test_to_json_line_when_serializing_does_write_exact_compact_line(
    event: SessionEvent, expected_line: str
):
    assert to_json_line(event) == expected_line


# ---------------------------------------------------------------------------
# event_from_wire — snake_case wire
# ---------------------------------------------------------------------------

ROUND_TRIP_EVENTS = [pytest.param(event, id=id_) for event, _, id_ in EVENT_SAMPLES] + [
    pytest.param(
        make_launch(max_usd=1.5, model="opus", dirty=DirtyInfo(file_count=4)),
        id="launch-with-optionals",
    ),
    pytest.param(_FOLLOW_UP_NO_OPTIONALS, id="follow_up-no-optionals"),
    pytest.param(
        TextDeltaEvent(at=5_000_000_000, chunk="a\x85b\u2028c\u2029d"),
        id="text_delta-unicode-line-breaks",
    ),
    pytest.param(
        TextDeltaEvent(at=5_000_000_000, chunk="café \ud800"),
        id="text_delta-lone-surrogate",
    ),
    pytest.param(_TOOL_START_INPUT_NONE, id="tool_start-input-none"),
    pytest.param(_THINKING_UPDATE_WITH_PARENT, id="thinking_update-with-parent"),
    pytest.param(_TOOL_START_WITH_PARENT, id="tool_start-with-parent"),
    pytest.param(_TOOL_END_WITH_PARENT, id="tool_end-with-parent"),
    pytest.param(_TEXT_DELTA_WITH_PARENT, id="text_delta-with-parent"),
]


@pytest.mark.parametrize("event", ROUND_TRIP_EVENTS)
def test_event_from_wire_when_given_serialized_event_does_reconstruct_it(event: SessionEvent):
    assert event_from_wire(json.loads(to_json_line(event))) == event


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
        pytest.param(
            {"type": "model_phase", "at": 1_000_000_000, "phase": "unknown"},
            id="model-phase-unknown",
        ),
        pytest.param({"type": "model_phase", "phase": "thinking"}, id="model-phase-missing-at"),
        pytest.param(
            {"type": "usage_update", "at": 6_000_000_000, "cost_usd": 0.01, "schema": 1},
            id="schema-on-non-launch-event",
        ),
    ],
)
def test_event_from_wire_when_input_unrecognized_does_return_none(obj: object):
    assert event_from_wire(obj) is None


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


@pytest.mark.parametrize("value", _NON_FINITE_FLOATS)
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
