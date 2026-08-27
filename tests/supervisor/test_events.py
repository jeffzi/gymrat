"""Behavioral tests for the supervisor event vocabulary and helpers.

The upstream suite pins most of these behaviors at the type level; those checks
do not survive the port, so they are re-expressed as construction, immutability,
and JSON-serialization assertions on the dataclasses. ``summarize`` and
``summarize_input`` are pinned against the Python JSON style (no spaces, ``null``
for ``None``); ``combine_observers`` is pinned on ordering, identity, and
error propagation.
"""

import dataclasses
import json
import typing

import pytest

from gymrat_py.supervisor.events import (
    SUMMARY_MAX_CHARS,
    CapEvent,
    DirtyInfo,
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

EVENT_SAMPLES: list[tuple[object, str]] = [
    (ThinkingUpdateEvent(timestamp=1, estimated_tokens=100, delta=10), "thinking_update"),
    (
        ToolStartEvent(
            timestamp=2,
            tool_use_id="t1",
            tool_name="Read",
            input={"path": "/x"},
            input_summary="/x",
        ),
        "tool_start",
    ),
    (ToolProgressEvent(timestamp=3, tool_use_id="t1", elapsed_ms=500), "tool_progress"),
    (
        ToolEndEvent(
            timestamp=4,
            tool_use_id="t1",
            tool_name="Read",
            duration_ms=750,
            result="ok",
            result_summary="ok",
        ),
        "tool_end",
    ),
    (TextDeltaEvent(timestamp=5, chunk="hello"), "text_delta"),
    (UsageUpdateEvent(timestamp=6, cost_usd=0.01), "usage_update"),
    (CapEvent(timestamp=7, cap="wall-clock"), "cap"),
    (make_launch(), "launch"),
]


@pytest.mark.parametrize(
    ("event", "expected_type"),
    [pytest.param(event, type_, id=type_) for event, type_ in EVENT_SAMPLES],
)
def test_event_when_constructed_does_expose_its_type_literal(event: object, expected_type: str):
    assert event.type == expected_type  # type: ignore[attr-defined]


def test_session_event_union_when_enumerated_does_expose_exactly_eight_type_literals():
    event_classes = typing.get_args(SessionEvent)
    types = {
        next(field.default for field in dataclasses.fields(cls) if field.name == "type")
        for cls in event_classes
    }

    assert types == {
        "thinking_update",
        "tool_start",
        "tool_progress",
        "tool_end",
        "text_delta",
        "usage_update",
        "cap",
        "launch",
    }


def test_event_when_field_reassigned_does_raise_frozen_instance_error():
    event = UsageUpdateEvent(timestamp=6, cost_usd=0.01)

    with pytest.raises(dataclasses.FrozenInstanceError):
        event.cost_usd = 0.02  # type: ignore[misc]


def test_dirty_info_when_constructed_does_carry_file_count():
    dirty = DirtyInfo(file_count=3)

    assert dirty.file_count == 3


# ---------------------------------------------------------------------------
# to_json_line
# ---------------------------------------------------------------------------

JSON_CASES = [
    pytest.param(
        ThinkingUpdateEvent(timestamp=1, estimated_tokens=100, delta=10),
        {"type": "thinking_update", "timestamp": 1, "estimatedTokens": 100, "delta": 10},
        id="thinking_update",
    ),
    pytest.param(
        ToolStartEvent(
            timestamp=2,
            tool_use_id="t1",
            tool_name="Read",
            input={"path": "/x"},
            input_summary="/x",
        ),
        {
            "type": "tool_start",
            "timestamp": 2,
            "toolUseId": "t1",
            "toolName": "Read",
            "input": {"path": "/x"},
            "inputSummary": "/x",
        },
        id="tool_start",
    ),
    pytest.param(
        ToolProgressEvent(timestamp=3, tool_use_id="t1", elapsed_ms=500),
        {"type": "tool_progress", "timestamp": 3, "toolUseId": "t1", "elapsedMs": 500},
        id="tool_progress",
    ),
    pytest.param(
        ToolEndEvent(
            timestamp=4,
            tool_use_id="t1",
            tool_name="Read",
            duration_ms=750,
            result="ok",
            result_summary="ok",
        ),
        {
            "type": "tool_end",
            "timestamp": 4,
            "toolUseId": "t1",
            "toolName": "Read",
            "durationMs": 750,
            "result": "ok",
            "resultSummary": "ok",
        },
        id="tool_end",
    ),
    pytest.param(
        TextDeltaEvent(timestamp=5, chunk="hello"),
        {"type": "text_delta", "timestamp": 5, "chunk": "hello"},
        id="text_delta",
    ),
    pytest.param(
        UsageUpdateEvent(timestamp=6, cost_usd=0.01),
        {"type": "usage_update", "timestamp": 6, "costUsd": 0.01},
        id="usage_update",
    ),
    pytest.param(
        CapEvent(timestamp=7, cap="spend-cap"),
        {"type": "cap", "timestamp": 7, "cap": "spend-cap"},
        id="cap",
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


# ---------------------------------------------------------------------------
# SUMMARY_MAX_CHARS
# ---------------------------------------------------------------------------


def test_summarize_when_called_without_max_chars_does_truncate_to_summary_max_chars():
    overflow = 50
    text = "a" * (SUMMARY_MAX_CHARS + overflow)

    result = summarize(text)

    assert result == "a" * SUMMARY_MAX_CHARS + f"… ({overflow} more chars, 1 lines)"


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


def test_summarize_when_over_budget_does_report_remaining_chars_and_line_count():
    result = summarize("a" * 300, 50)

    assert result == "a" * 50 + "… (250 more chars, 1 lines)"


def test_summarize_when_over_budget_does_count_original_lines_before_collapsing():
    result = summarize("line1\nline2\nline3\nline4", 20)

    assert "4 lines" in result


@pytest.mark.parametrize(
    ("text", "max_chars", "expected"),
    [
        pytest.param("🎯" * 8, 5, "🎯🎯🎯🎯🎯… (3 more chars, 1 lines)", id="all-emoji"),
        pytest.param("ab🎯🎯cd🎯", 3, "ab🎯… (4 more chars, 1 lines)", id="mixed-width"),
    ],
)
def test_summarize_when_truncating_does_split_on_code_point_boundaries(
    text: str, max_chars: int, expected: str
):
    assert summarize(text, max_chars) == expected


# ---------------------------------------------------------------------------
# summarize_input
# ---------------------------------------------------------------------------


class _Weird:
    """A value ``json.dumps`` cannot encode, with a deterministic string form."""

    def __str__(self) -> str:
        return "weird-value"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        pytest.param({"key": "value"}, '{"key":"value"}', id="dict-no-spaces"),
        pytest.param(None, "null", id="none-to-json-null"),
        pytest.param(_Weird(), "weird-value", id="non-serializable-str-fallback"),
    ],
)
def test_summarize_input_when_given_value_does_summarize_its_json_form(
    value: object, expected: str
):
    assert summarize_input(value, 200) == expected


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


def test_combine_observers_when_an_observer_raises_does_still_call_later_observers():
    boom_message = "observer failure"

    def boom(_: object) -> None:
        raise RuntimeError(boom_message)

    later = collecting_observer()
    combined = combine_observers(boom, later.observer)
    event = UsageUpdateEvent(timestamp=1, cost_usd=0.01)

    combined(event)

    assert later.events == [event]


def test_combine_observers_when_an_observer_raises_does_warn_about_the_failure():
    boom_message = "observer failure"

    def boom(_: object) -> None:
        raise RuntimeError(boom_message)

    later = collecting_observer()
    combined = combine_observers(boom, later.observer)
    event = UsageUpdateEvent(timestamp=1, cost_usd=0.01)

    with pytest.warns(RuntimeWarning, match=boom_message):
        combined(event)


# ---------------------------------------------------------------------------
# B34 — non-finite float serialization
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
