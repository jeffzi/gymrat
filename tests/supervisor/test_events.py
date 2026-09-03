"""Behavioral tests for the supervisor event vocabulary and helpers.

Events are frozen pydantic models, so behavior is pinned via construction,
immutability, and JSON-serialization assertions rather than type-level checks.
``summarize`` and ``summarize_input`` are pinned against the Python JSON style
(no spaces, ``null`` for ``None``); ``combine_observers`` is pinned on
ordering, identity, and error propagation.
"""

import json
import typing
from pathlib import Path

import pytest
from pydantic import ValidationError

from gymrat.supervisor.events import (
    SUMMARY_MAX_CHARS,
    CapEvent,
    DirtyInfo,
    ModelPhaseEvent,
    SessionEvent,
    TextDeltaEvent,
    ThinkingUpdateEvent,
    ToolEndEvent,
    ToolProgressEvent,
    ToolStartEvent,
    UsageUpdateEvent,
    combine_observers,
    event_from_wire,
    summarize,
    summarize_input,
    to_json_line,
)
from tests.supervisor._fixtures import collecting_observer, make_launch

# ---------------------------------------------------------------------------
# Event vocabulary
# ---------------------------------------------------------------------------

# One instance per event type, shared between the type-literal, JSON-serialization,
# and wire-round-trip tests below so each event's fields are declared exactly once.
_THINKING_UPDATE = ThinkingUpdateEvent(timestamp=1, estimated_tokens=100, delta=10)
_TOOL_START = ToolStartEvent(
    timestamp=2, tool_use_id="t1", tool_name="Read", input={"path": "/x"}, input_summary="/x"
)
_TOOL_PROGRESS = ToolProgressEvent(timestamp=3, tool_use_id="t1", elapsed_ms=500)
_TOOL_END = ToolEndEvent(
    timestamp=4,
    tool_use_id="t1",
    tool_name="Read",
    duration_ms=750,
    result="ok",
    result_summary="ok",
)
_TEXT_DELTA = TextDeltaEvent(timestamp=5, chunk="hello")
_USAGE_UPDATE = UsageUpdateEvent(timestamp=6, cost_usd=0.01)
_CAP = CapEvent(timestamp=7, cap="wall-clock")
_MODEL_PHASE_THINKING = ModelPhaseEvent(timestamp=8, phase="thinking")
_MODEL_PHASE_RESPONDING = ModelPhaseEvent(timestamp=9, phase="responding")
_MODEL_PHASE_TOOL_INPUT = ModelPhaseEvent(
    timestamp=10, phase="tool_input", tool_name="Read", parent_tool_use_id="p1"
)
_MODEL_PHASE_TURN_END = ModelPhaseEvent(timestamp=11, phase="turn_end")

EVENT_SAMPLES: list[tuple[object, str]] = [
    (_THINKING_UPDATE, "thinking_update"),
    (_TOOL_START, "tool_start"),
    (_TOOL_PROGRESS, "tool_progress"),
    (_TOOL_END, "tool_end"),
    (_TEXT_DELTA, "text_delta"),
    (_USAGE_UPDATE, "usage_update"),
    (_CAP, "cap"),
    (_MODEL_PHASE_THINKING, "model_phase"),
    (_MODEL_PHASE_RESPONDING, "model_phase"),
    (_MODEL_PHASE_TOOL_INPUT, "model_phase"),
    (_MODEL_PHASE_TURN_END, "model_phase"),
    (make_launch(), "launch"),
]


@pytest.mark.parametrize(
    ("event", "expected_type"),
    [pytest.param(event, type_, id=type_) for event, type_ in EVENT_SAMPLES],
)
def test_event_when_constructed_does_expose_its_type_literal(event: object, expected_type: str):
    assert event.type == expected_type  # type: ignore[attr-defined]


def test_session_event_union_when_enumerated_does_expose_exactly_nine_type_literals():
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
    }


def test_event_when_field_reassigned_does_raise():
    event = UsageUpdateEvent(timestamp=6, cost_usd=0.01)

    with pytest.raises(ValidationError, match="frozen"):
        event.cost_usd = 0.02  # type: ignore[misc]


def test_dirty_info_when_constructed_does_carry_file_count():
    dirty = DirtyInfo(file_count=3)

    assert dirty.file_count == 3


# ---------------------------------------------------------------------------
# to_json_line
# ---------------------------------------------------------------------------

JSON_CASES = [
    pytest.param(
        _THINKING_UPDATE,
        {
            "type": "thinking_update",
            "timestamp": 1,
            "estimatedTokens": 100,
            "delta": 10,
            "parentToolUseId": None,
        },
        id="thinking_update",
    ),
    pytest.param(
        _TOOL_START,
        {
            "type": "tool_start",
            "timestamp": 2,
            "toolUseId": "t1",
            "toolName": "Read",
            "input": {"path": "/x"},
            "inputSummary": "/x",
            "parentToolUseId": None,
        },
        id="tool_start",
    ),
    pytest.param(
        _TOOL_PROGRESS,
        {"type": "tool_progress", "timestamp": 3, "toolUseId": "t1", "elapsedMs": 500},
        id="tool_progress",
    ),
    pytest.param(
        _TOOL_END,
        {
            "type": "tool_end",
            "timestamp": 4,
            "toolUseId": "t1",
            "toolName": "Read",
            "durationMs": 750,
            "result": "ok",
            "resultSummary": "ok",
            "parentToolUseId": None,
        },
        id="tool_end",
    ),
    pytest.param(
        _TEXT_DELTA,
        {"type": "text_delta", "timestamp": 5, "chunk": "hello", "parentToolUseId": None},
        id="text_delta",
    ),
    pytest.param(
        _USAGE_UPDATE,
        {"type": "usage_update", "timestamp": 6, "costUsd": 0.01},
        id="usage_update",
    ),
    pytest.param(
        _CAP,
        {"type": "cap", "timestamp": 7, "cap": "wall-clock"},
        id="cap",
    ),
    pytest.param(
        _MODEL_PHASE_THINKING,
        {
            "type": "model_phase",
            "timestamp": 8,
            "phase": "thinking",
            "toolName": None,
            "parentToolUseId": None,
        },
        id="model_phase-thinking",
    ),
    pytest.param(
        _MODEL_PHASE_TOOL_INPUT,
        {
            "type": "model_phase",
            "timestamp": 10,
            "phase": "tool_input",
            "toolName": "Read",
            "parentToolUseId": "p1",
        },
        id="model_phase-tool_input",
    ),
]


@pytest.mark.parametrize(("event", "expected"), JSON_CASES)
def test_to_json_line_when_serializing_does_use_camel_case_keys(
    event: object, expected: dict[str, object]
):
    parsed = json.loads(to_json_line(event))  # type: ignore[arg-type]

    assert parsed == expected


def test_to_json_line_when_launch_has_no_optionals_does_omit_max_usd_and_model():
    event = make_launch(max_usd=None, model=None)

    parsed = json.loads(to_json_line(event))

    assert parsed == {
        "type": "launch",
        "timestamp": 1000,
        "headSha": "abc123def",
        "dirty": False,
        "maxMinutes": 5,
        "runbookPath": "/path/to/runbook.md",
        "kickoffSummary": "test kickoff",
    }


def test_to_json_line_when_launch_has_optionals_does_emit_them():
    event = make_launch(max_usd=1.5, model="opus", dirty=DirtyInfo(file_count=4))

    parsed = json.loads(to_json_line(event))

    assert parsed["maxUsd"] == 1.5
    assert parsed["model"] == "opus"
    assert parsed["dirty"] == {"fileCount": 4}


# ---------------------------------------------------------------------------
# LaunchEvent.model_dump — None-optional omission
# ---------------------------------------------------------------------------

BY_ALIAS_CASES = [
    pytest.param(False, "max_usd", "model", id="python-names"),
    pytest.param(True, "maxUsd", "model", id="aliased-names"),
]


@pytest.mark.parametrize(("by_alias", "usd_key", "model_key"), BY_ALIAS_CASES)
def test_launch_event_model_dump_when_optionals_are_none_does_omit_them(
    by_alias: bool, usd_key: str, model_key: str
):
    event = make_launch(max_usd=None, model=None)

    dumped = event.model_dump(by_alias=by_alias)

    assert usd_key not in dumped
    assert model_key not in dumped


@pytest.mark.parametrize(("by_alias", "usd_key", "model_key"), BY_ALIAS_CASES)
def test_launch_event_model_dump_when_optionals_are_set_does_include_them(
    by_alias: bool, usd_key: str, model_key: str
):
    event = make_launch(max_usd=1.5, model="opus")

    dumped = event.model_dump(by_alias=by_alias)

    assert dumped[usd_key] == 1.5
    assert dumped[model_key] == "opus"


# ---------------------------------------------------------------------------
# event_from_wire
# ---------------------------------------------------------------------------

ROUND_TRIP_EVENTS = [pytest.param(event, id=type_) for event, type_ in EVENT_SAMPLES] + [
    pytest.param(
        make_launch(max_usd=1.5, model="opus", dirty=DirtyInfo(file_count=4)),
        id="launch-with-optionals",
    ),
]


@pytest.mark.parametrize("event", ROUND_TRIP_EVENTS)
def test_event_from_wire_when_given_serialized_event_does_reconstruct_it(event: object):
    assert event_from_wire(json.loads(to_json_line(event))) == event  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "obj",
    [
        pytest.param([1, 2, 3], id="non-dict-list"),
        pytest.param("nope", id="non-dict-str"),
        pytest.param({"timestamp": 1}, id="missing-type"),
        pytest.param({"type": "mystery", "timestamp": 1}, id="unknown-type"),
        pytest.param({"type": "usage_update", "timestamp": 6}, id="missing-required-field"),
    ],
)
def test_event_from_wire_when_input_unrecognized_does_return_none(obj: object):
    assert event_from_wire(obj) is None


@pytest.mark.parametrize(
    "obj",
    [
        pytest.param(
            {"type": "model_phase", "timestamp": 1, "phase": "unknown"},
            id="unknown-phase",
        ),
        pytest.param(
            {"type": "model_phase", "phase": "thinking"},
            id="missing-timestamp",
        ),
    ],
)
def test_event_from_wire_when_model_phase_invalid_does_return_none(obj: object):
    assert event_from_wire(obj) is None


# ---------------------------------------------------------------------------
# parent_tool_use_id on existing events
# ---------------------------------------------------------------------------


# Each entry is (event without parent_tool_use_id, event with it set, expected id).
# The no-parent events reuse the shared samples above so each event's fields are
# still declared exactly once.
_PARENT_ID_CASES = [
    (
        _THINKING_UPDATE,
        ThinkingUpdateEvent(timestamp=1, estimated_tokens=100, delta=10, parent_tool_use_id="p1"),
        "p1",
        "thinking_update",
    ),
    (
        _TOOL_START,
        ToolStartEvent(
            timestamp=2,
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
            timestamp=4,
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
        TextDeltaEvent(timestamp=5, chunk="hello", parent_tool_use_id="p4"),
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
def test_to_json_line_when_parent_tool_use_id_set_does_render_camel_case(
    event: object, expected_id: str
):
    parsed = json.loads(to_json_line(event))  # type: ignore[arg-type]

    assert parsed["parentToolUseId"] == expected_id


@pytest.mark.parametrize(
    "event",
    [pytest.param(with_parent, id=case_id) for _, with_parent, _, case_id in _PARENT_ID_CASES],
)
def test_event_from_wire_when_parent_tool_use_id_set_does_round_trip(event: object):
    assert event_from_wire(json.loads(to_json_line(event))) == event  # type: ignore[arg-type]


def test_event_from_wire_when_text_delta_lacks_parent_tool_use_id_does_default_to_none():
    """Old log files written before the field was added lack the key entirely."""
    wire = {"type": "text_delta", "timestamp": 5, "chunk": "hello"}

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
        pytest.param("🎯" * 8, 5, "🎯🎯🎯🎯🎯…", id="all-emoji"),
        pytest.param("ab🎯🎯cd🎯", 3, "ab🎯…", id="mixed-width"),
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
    event = UsageUpdateEvent(timestamp=1, cost_usd=0.01)

    combined(event)

    assert first.events[0] is event
    assert second.events[0] is event


def test_combine_observers_when_given_no_observers_does_not_raise():
    combined = combine_observers()
    event = UsageUpdateEvent(timestamp=1, cost_usd=0.01)

    combined(event)


def test_combine_observers_when_an_observer_raises_does_warn_and_call_remaining():
    boom_message = "observer failure"

    def boom(_: object) -> None:
        raise RuntimeError(boom_message)

    later = collecting_observer()
    combined = combine_observers(boom, later.observer)
    event = UsageUpdateEvent(timestamp=1, cost_usd=0.01)

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
    event = UsageUpdateEvent(timestamp=1, cost_usd=cost)

    line = to_json_line(event)

    parsed = json.loads(line)
    assert parsed["costUsd"] is None


# ---------------------------------------------------------------------------
# non-JSON-encodable field fallback
# ---------------------------------------------------------------------------


def test_to_json_line_when_field_not_json_encodable_does_stringify_instead_of_raising():
    """A non-JSON-encodable value in an event field must not drop the log line.

    ``to_json_line`` should fall back to ``str()`` for values that
    ``json.dumps`` cannot encode, so the line is always written.
    """
    event = ToolStartEvent(
        timestamp=1,
        tool_use_id="t1",
        tool_name="Test",
        input={"key": _NotJsonEncodable()},
        input_summary="test",
    )

    line = to_json_line(event)

    parsed = json.loads(line)
    assert parsed["input"]["key"] == "not-json-encodable"
