"""Stream-event mapping in the Claude Agent SDK driver.

These cover how partial-message stream events become session events: thinking
deltas and their throttling, model phase transitions, event types that emit
nothing, thinking scoped per parent tool call, and ``parent_tool_use_id``
carried onto the emitted events. The driver runs against an injected fake
client that yields the SDK's own dataclasses.
"""

from math import ceil

import pytest
from claude_agent_sdk import (
    StreamEvent,
)

from gymrat.supervisor.events import (
    ModelPhaseEvent,
    ThinkingUpdateEvent,
)
from tests.supervisor._fixtures import (
    events_of,
    run_with_messages,
    stream_event,
)

# ---------------------------------------------------------------------------
# stream events — thinking deltas with throttling
# ---------------------------------------------------------------------------


async def test_stream_when_thinking_delta_short_does_flush_only_on_block_stop():
    messages = [
        stream_event({"type": "content_block_start", "content_block": {"type": "thinking"}}),
        stream_event({
            "type": "content_block_delta",
            "delta": {"type": "thinking_delta", "thinking": "a" * 100},
        }),
        stream_event({"type": "content_block_stop"}),
    ]

    events = await run_with_messages(messages)

    updates = events_of(events, ThinkingUpdateEvent)
    # block_start emits (0, 0); the 100-char delta stays under the 200-char
    # throttle, so the only other update is the block_stop flush.
    assert [(u.delta, u.estimated_tokens) for u in updates] == [(0, 0), (25, 25)]


async def test_stream_when_thinking_delta_crosses_throttle_does_emit_mid_block():
    messages = [
        stream_event({"type": "content_block_start", "content_block": {"type": "thinking"}}),
        stream_event({
            "type": "content_block_delta",
            "delta": {"type": "thinking_delta", "thinking": "a" * 220},
        }),
        stream_event({
            "type": "content_block_delta",
            "delta": {"type": "thinking_delta", "thinking": "a" * 30},
        }),
        stream_event({"type": "content_block_stop"}),
    ]

    events = await run_with_messages(messages)

    updates = events_of(events, ThinkingUpdateEvent)
    # block_start (0, 0); the first delta crosses the 200-char throttle mid-block
    # (55, 55); the trailing 30 chars flush at block_stop (63, 8).
    assert [(u.delta, u.estimated_tokens) for u in updates] == [
        (0, 0),
        (ceil(220 / 4), ceil(220 / 4)),
        (ceil(250 / 4) - ceil(220 / 4), ceil(250 / 4)),
    ]


async def test_stream_when_thinking_deltas_accumulated_does_bound_update_count():
    chunk = "a" * 50
    num_chunks = 20  # 1000 chars total
    messages = [
        stream_event({"type": "content_block_start", "content_block": {"type": "thinking"}}),
    ]
    for _ in range(num_chunks):
        messages.append(
            stream_event({
                "type": "content_block_delta",
                "delta": {"type": "thinking_delta", "thinking": chunk},
            })
        )
    messages.append(stream_event({"type": "content_block_stop"}))

    events = await run_with_messages(messages)

    updates = events_of(events, ThinkingUpdateEvent)
    max_allowed = ceil(1000 / 200) + 2
    assert len(updates) <= max_allowed
    assert updates[-1].estimated_tokens == ceil(1000 / 4)


async def test_stream_when_single_thinking_block_does_report_delta_equal_to_estimated_tokens():
    messages = [
        stream_event({"type": "content_block_start", "content_block": {"type": "thinking"}}),
        stream_event({
            "type": "content_block_delta",
            "delta": {"type": "thinking_delta", "thinking": "abcd"},
        }),
        stream_event({"type": "content_block_stop"}),
    ]

    events = await run_with_messages(messages)

    updates = events_of(events, ThinkingUpdateEvent)
    assert len(updates) == 2
    assert updates[0].delta == 0
    assert updates[0].estimated_tokens == 0
    assert updates[-1].delta == 1
    assert updates[-1].estimated_tokens == 1


async def test_stream_when_multiple_thinking_blocks_does_accumulate_estimated_tokens():
    messages = [
        stream_event({"type": "content_block_start", "content_block": {"type": "thinking"}}),
        stream_event({
            "type": "content_block_delta",
            "delta": {"type": "thinking_delta", "thinking": "abcd"},
        }),
        stream_event({"type": "content_block_stop"}),
        stream_event({"type": "content_block_start", "content_block": {"type": "thinking"}}),
        stream_event({
            "type": "content_block_delta",
            "delta": {"type": "thinking_delta", "thinking": "abcdefgh"},
        }),
        stream_event({"type": "content_block_stop"}),
    ]

    events = await run_with_messages(messages)

    updates = events_of(events, ThinkingUpdateEvent)
    first_block = [u for u in updates if u.estimated_tokens <= 1]
    second_block_final = updates[-1]
    assert first_block[-1].estimated_tokens == 1
    assert second_block_final.estimated_tokens == 3


async def test_stream_when_thinking_delta_has_parent_does_carry_parent_tool_use_id():
    messages = [
        stream_event(
            {"type": "content_block_start", "content_block": {"type": "thinking"}},
            parent_tool_use_id="tu_42",
        ),
        stream_event({"type": "content_block_stop"}, parent_tool_use_id="tu_42"),
    ]

    events = await run_with_messages(messages)

    updates = events_of(events, ThinkingUpdateEvent)
    assert updates[0].parent_tool_use_id == "tu_42"


# ---------------------------------------------------------------------------
# stream events — phase transitions
# ---------------------------------------------------------------------------


_THINKING_BLOCK_MESSAGES = [
    stream_event(
        {"type": "content_block_start", "content_block": {"type": "thinking"}},
        parent_tool_use_id="tu_x",
    ),
    stream_event({"type": "content_block_stop"}, parent_tool_use_id="tu_x"),
]
_TEXT_BLOCK_MESSAGES = [
    stream_event(
        {"type": "content_block_start", "content_block": {"type": "text"}},
        parent_tool_use_id="tu_x",
    ),
    stream_event({"type": "content_block_stop"}, parent_tool_use_id="tu_x"),
]


@pytest.mark.parametrize(
    ("messages", "expected_phase"),
    [
        pytest.param(_THINKING_BLOCK_MESSAGES, "thinking", id="thinking-block-start"),
        pytest.param(_TEXT_BLOCK_MESSAGES, "responding", id="text-block-start"),
        pytest.param(
            [stream_event({"type": "message_stop"}, parent_tool_use_id="tu_x")],
            "turn_end",
            id="message-stop",
        ),
    ],
)
async def test_stream_when_phase_event_received_does_emit_model_phase(
    messages: list[StreamEvent], expected_phase: str
):
    events = await run_with_messages(messages)

    phases = events_of(events, ModelPhaseEvent)
    assert any((p.phase, p.parent_tool_use_id) == (expected_phase, "tu_x") for p in phases)


async def test_stream_when_tool_use_block_start_does_emit_model_phase_tool_input():
    messages = [
        stream_event({
            "type": "content_block_start",
            "content_block": {"type": "tool_use", "name": "Read"},
        }),
        stream_event({"type": "content_block_stop"}),
    ]

    events = await run_with_messages(messages)

    phases = events_of(events, ModelPhaseEvent)
    tool_phases = [p for p in phases if p.phase == "tool_input"]
    assert len(tool_phases) == 1
    assert tool_phases[0].tool_name == "Read"


# ---------------------------------------------------------------------------
# stream events — silent event types
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "event_type",
    [
        pytest.param("text_delta", id="text-delta"),
        pytest.param("input_json_delta", id="input-json-delta"),
        pytest.param("signature_delta", id="signature-delta"),
        pytest.param("message_start", id="message-start"),
        pytest.param("message_delta", id="message-delta"),
        pytest.param("totally_unknown_type", id="unrecognized"),
    ],
)
async def test_stream_when_silent_event_type_does_emit_nothing(event_type: str):
    messages = [stream_event({"type": event_type})]

    events = await run_with_messages(messages)

    assert events == []


# ---------------------------------------------------------------------------
# stream events — per-parent thinking scoping
# ---------------------------------------------------------------------------


async def test_stream_when_subagent_thinking_does_not_inflate_top_level_total():
    messages = [
        stream_event(
            {"type": "content_block_start", "content_block": {"type": "thinking"}},
            parent_tool_use_id=None,
        ),
        stream_event(
            {
                "type": "content_block_delta",
                "delta": {"type": "thinking_delta", "thinking": "aaaa"},
            },
            parent_tool_use_id=None,
        ),
        stream_event({"type": "content_block_stop"}, parent_tool_use_id=None),
        stream_event(
            {"type": "content_block_start", "content_block": {"type": "thinking"}},
            parent_tool_use_id="tu_sub",
        ),
        stream_event(
            {
                "type": "content_block_delta",
                "delta": {"type": "thinking_delta", "thinking": "b" * 400},
            },
            parent_tool_use_id="tu_sub",
        ),
        stream_event({"type": "content_block_stop"}, parent_tool_use_id="tu_sub"),
    ]

    events = await run_with_messages(messages)

    updates = events_of(events, ThinkingUpdateEvent)
    top = [u for u in updates if u.parent_tool_use_id is None]
    sub = [u for u in updates if u.parent_tool_use_id == "tu_sub"]
    assert top[-1].estimated_tokens == ceil(4 / 4)
    assert sub[-1].estimated_tokens == ceil(400 / 4)
