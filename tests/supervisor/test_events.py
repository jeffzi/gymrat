"""Behavioral tests for the supervisor event vocabulary and helpers.

Events are frozen pydantic models whose base ``_EventModel`` carries a per-event
``type`` discriminator and ``at: int`` (nanoseconds since epoch), with snake_case
wire keys via ``validate_by_name`` / ``serialize_by_alias``. ``to_json_line``
writes snake_case keys with ``exclude_none=True`` globally; ``event_from_wire``
accepts only the snake_case wire; ``combine_observers`` is pinned on ordering,
identity, and error propagation.
"""

import json
import typing
from pathlib import Path

import pytest
from pydantic import ValidationError

from gymrat.supervisor.events import (
    SUMMARY_MAX_CHARS,
    CapEvent,
    CompactionEvent,
    DirtyInfo,
    FollowUpEvent,
    ModelPhaseEvent,
    SessionEvent,
    TextDeltaEvent,
    ThinkingUpdateEvent,
    ToolEndEvent,
    ToolProgressEvent,
    ToolStartEvent,
    TurnEndEvent,
    UsageUpdateEvent,
    combine_observers,
    event_from_wire,
    summarize,
    summarize_input,
    to_json_line,
)
from tests.supervisor._fixtures import collecting_observer, make_launch, make_prompt

# ---------------------------------------------------------------------------
# Event vocabulary — at: int (nanoseconds), no timestamp field
# ---------------------------------------------------------------------------

# One instance per event type, shared between the type-literal, JSON-serialization,
# and wire-round-trip tests below so each event's fields are declared exactly once.
_THINKING_UPDATE = ThinkingUpdateEvent(at=1_000_000_000, estimated_tokens=100, delta=10)
_TOOL_START = ToolStartEvent(
    at=2_000_000_000, tool_use_id="t1", tool_name="Read", input={"path": "/x"}, input_summary="/x"
)
_TOOL_PROGRESS = ToolProgressEvent(at=3_000_000_000, tool_use_id="t1", elapsed_ms=500)
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
    (_TOOL_PROGRESS, "tool_progress", "tool_progress"),
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


def test_session_event_union_when_enumerated_does_expose_exactly_twelve_type_literals():
    event_classes = typing.get_args(SessionEvent)
    types = {cls.model_fields["type"].default for cls in event_classes}

    assert types == {
        "thinking_update",
        "tool_start",
        "tool_progress",
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
# to_json_line — snake_case keys, exclude_none=True
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
        _TOOL_PROGRESS,
        {"type": "tool_progress", "at": 3_000_000_000, "tool_use_id": "t1", "elapsed_ms": 500},
        id="tool_progress",
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
        {"type": "usage_update", "at": 6_000_000_000, "cost_usd": 0.01, "settled": False},
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


def test_to_json_line_when_usage_update_unsettled_does_still_write_settled_false():
    event = UsageUpdateEvent(at=6_000_000_000, cost_usd=0.01, settled=False)

    parsed = json.loads(to_json_line(event))

    assert parsed["settled"] is False


def test_to_json_line_when_usage_update_settled_does_write_settled_true():
    event = UsageUpdateEvent(at=6_000_000_000, cost_usd=0.01, settled=True)

    parsed = json.loads(to_json_line(event))

    assert parsed["settled"] is True


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
# FollowUpEvent — None-optional omission via global exclude_none
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
# SUMMARY_MAX_CHARS
# ---------------------------------------------------------------------------


def test_summarize_when_called_without_max_chars_does_truncate_to_summary_max_chars():
    overflow = 50
    text = "a" * (SUMMARY_MAX_CHARS + overflow)

    result = summarize(text)

    assert result == "a" * SUMMARY_MAX_CHARS + "…"


# ---------------------------------------------------------------------------
# summarize
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        pytest.param("short text", "short text", id="plain-fit"),
        pytest.param("hello   world", "hello world", id="internal-whitespace-collapsed"),
        pytest.param("  trimmed  ", "trimmed", id="leading-trailing-trimmed"),
        pytest.param("line1\nline2\nline3", "line1 line2 line3", id="newlines-collapsed"),
    ],
)
def test_summarize_when_within_budget_does_return_collapsed_text(text: str, expected: str):
    assert summarize(text, 100) == expected


def test_summarize_when_over_budget_does_truncate_with_bare_ellipsis():
    overflow = 250
    max_chars = 50

    result = summarize("a" * (max_chars + overflow), max_chars)

    assert result == "a" * max_chars + "…"


def test_summarize_when_multiline_over_budget_does_truncate_with_bare_ellipsis():
    result = summarize("line1\nline2\nline3\nline4", 20)

    assert result == "line1 line2 line3 li…"


@pytest.mark.parametrize(
    ("text", "max_chars", "expected"),
    [
        pytest.param(
            "\U0001f3af" * 8,
            5,
            "\U0001f3af\U0001f3af\U0001f3af\U0001f3af\U0001f3af…",
            id="all-emoji",
        ),
        pytest.param(  # cspell:disable-next-line
            "ab\U0001f3af\U0001f3afcd\U0001f3af", 3, "ab\U0001f3af…", id="mixed-width"
        ),
    ],
)
def test_summarize_when_truncating_does_split_on_code_point_boundaries(
    text: str, max_chars: int, expected: str
):
    assert summarize(text, max_chars) == expected


# ---------------------------------------------------------------------------
# summarize_input
# ---------------------------------------------------------------------------


class _NotJsonEncodable:
    """A value ``json.dumps`` cannot encode, with a deterministic string form."""

    def __str__(self) -> str:
        return "not-json-encodable"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        pytest.param({"key": "value"}, '{"key":"value"}', id="dict-no-spaces"),
        pytest.param(None, "null", id="none-to-json-null"),
        pytest.param(_NotJsonEncodable(), "not-json-encodable", id="non-serializable-str-fallback"),
    ],
)
def test_summarize_input_when_given_value_does_summarize_its_json_form(
    value: object, expected: str
):
    assert summarize_input(value, 200) == expected


# ---------------------------------------------------------------------------
# summarize_input — tool-specific extraction
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("tool_name", "tool_input", "expected"),
    [
        pytest.param("Read", {"file_path": "/a/b.py"}, "/a/b.py", id="read"),
        pytest.param(
            "Edit",
            {"file_path": "/a/b.py", "old_string": "x", "new_string": "y"},
            "/a/b.py",
            id="edit",
        ),
        pytest.param("Write", {"file_path": "/a/b.py", "content": "..."}, "/a/b.py", id="write"),
        pytest.param(
            "NotebookEdit",
            {"notebook_path": "/a/nb.ipynb"},
            "/a/nb.ipynb",
            id="notebook-edit",
        ),
    ],
)
def test_summarize_input_when_file_tool_does_extract_path_only(
    tool_name: str, tool_input: dict[str, object], expected: str
):
    assert summarize_input(tool_input, tool_name=tool_name) == expected


def test_summarize_input_when_path_under_root_does_render_relative():
    result = summarize_input(
        {"file_path": "/project/src/main.py"},
        tool_name="Read",
        supervised_root="/project",
    )

    assert result == "src/main.py"


def test_summarize_input_when_path_under_home_does_render_tilde_prefixed():
    home = str(Path.home())

    result = summarize_input(
        {"file_path": f"{home}/Documents/notes.md"},
        tool_name="Read",
        supervised_root="/other/project",
    )

    assert result == "~/Documents/notes.md"


def test_summarize_input_when_path_under_root_and_home_does_prefer_root_relative():
    home = str(Path.home())
    root = f"{home}/project"

    result = summarize_input(
        {"file_path": f"{root}/src/main.py"},
        tool_name="Read",
        supervised_root=root,
    )

    assert result == "src/main.py"


def test_summarize_input_when_path_outside_root_and_home_does_render_verbatim():
    result = summarize_input(
        {"file_path": "/etc/config.ini"},
        tool_name="Read",
        supervised_root="/project",
    )

    assert result == "/etc/config.ini"


def test_summarize_input_when_bash_does_extract_command():
    result = summarize_input({"command": "echo hello"}, tool_name="Bash")

    assert result == "echo hello"


@pytest.mark.parametrize(
    ("tool_name", "tool_input"),
    [
        pytest.param(
            "Agent",
            {
                "subagent_type": "Explore",
                "description": "Explore ECS source architecture",
                "prompt": "long prompt body that should never appear",
            },
            id="agent-subagent-type",
        ),
        pytest.param(
            "Task",
            {
                "type": "Explore",
                "description": "Explore ECS source architecture",
                "prompt": "long prompt body that should never appear",
            },
            id="task-type",
        ),
    ],
)
def test_summarize_input_when_agent_or_task_does_extract_type_and_description(
    tool_name: str, tool_input: dict[str, object]
):
    result = summarize_input(tool_input, tool_name=tool_name)

    assert result == "Explore: Explore ECS source architecture"


def test_summarize_input_when_agent_has_no_subagent_type_does_show_description_only():
    result = summarize_input(
        {"description": "Explore ECS source architecture", "prompt": "..."},
        tool_name="Agent",
    )

    assert result == "Explore ECS source architecture"


@pytest.mark.parametrize(
    ("tool_name", "tool_input", "expected"),
    [
        pytest.param("Skill", {"skill": "gymrat"}, "gymrat", id="skill-name-only"),
        pytest.param(
            "Skill",
            {"skill": "gymrat", "args": "some args"},
            "gymrat some args",
            id="skill-name-with-args",
        ),
    ],
)
def test_summarize_input_when_skill_tool_does_extract_skill_and_args(
    tool_name: str, tool_input: dict[str, object], expected: str
):
    assert summarize_input(tool_input, tool_name=tool_name) == expected


@pytest.mark.parametrize(
    ("tool_name", "tool_input"),
    [
        pytest.param("Read", {"not_file_path": "/a.py"}, id="read-missing-file-path"),
        pytest.param("Bash", {"not_command": "echo"}, id="bash-missing-command"),
        pytest.param("Agent", {"prompt": "..."}, id="agent-missing-type"),
        pytest.param("Skill", {"not_skill": "foo"}, id="skill-missing-skill"),
    ],
)
def test_summarize_input_when_expected_field_missing_does_fall_back_to_json(
    tool_name: str, tool_input: dict[str, object]
):
    result = summarize_input(tool_input, tool_name=tool_name)

    assert result == json.dumps(tool_input, separators=(",", ":"))


def test_summarize_input_when_unknown_tool_does_fall_back_to_json():
    tool_input = {"some_key": "some_value"}

    result = summarize_input(tool_input, tool_name="UnknownTool")

    assert result == '{"some_key":"some_value"}'


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

    with pytest.warns(RuntimeWarning, match=boom_message):
        combined(event)

    assert later.events == [event]


# ---------------------------------------------------------------------------
# non-finite float serialization
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "cost",
    [
        pytest.param(float("nan"), id="nan"),
        pytest.param(float("inf"), id="positive-infinity"),
        pytest.param(float("-inf"), id="negative-infinity"),
    ],
)
def test_to_json_line_when_cost_is_non_finite_does_serialize_as_null(cost: float):
    event = UsageUpdateEvent(at=1_000_000_000, cost_usd=cost)

    line = to_json_line(event)

    parsed = json.loads(line)
    assert parsed["cost_usd"] is None


# ---------------------------------------------------------------------------
# non-JSON-encodable field fallback
# ---------------------------------------------------------------------------


def test_to_json_line_when_field_not_json_encodable_does_stringify_instead_of_raising():
    event = ToolStartEvent(
        at=1_000_000_000,
        tool_use_id="t1",
        tool_name="Test",
        input={"key": _NotJsonEncodable()},
        input_summary="test",
    )

    line = to_json_line(event)

    parsed = json.loads(line)
    assert parsed["input"]["key"] == "not-json-encodable"
