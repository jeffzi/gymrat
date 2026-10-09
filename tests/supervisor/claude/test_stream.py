"""Stream-event mapping in the Claude Agent SDK driver.

These cover how partial-message stream events become session events: thinking
deltas and their throttling, model phase transitions, event types that emit
nothing, thinking scoped per parent tool call, and ``parent_tool_use_id``
carried onto the emitted events. The driver runs against an injected fake
client that yields the SDK's own dataclasses.
"""

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
)

_SESSION_ID = "sdk-session"


def stream_event(event: dict[str, object], *, parent_tool_use_id: str | None = None) -> StreamEvent:
    """Build an SDK ``StreamEvent`` wrapping one raw API stream event."""
    return StreamEvent(
        uuid="event-uuid",
        session_id=_SESSION_ID,
        event=event,
        parent_tool_use_id=parent_tool_use_id,
    )


# ---------------------------------------------------------------------------
# stream events — thinking deltas with throttling
# ---------------------------------------------------------------------------


def _thinking_start(parent: str | None = None) -> StreamEvent:
    return stream_event(
        {"type": "content_block_start", "content_block": {"type": "thinking"}},
        parent_tool_use_id=parent,
    )


def _thinking_delta(text: str, parent: str | None = None) -> StreamEvent:
    return stream_event(
        {"type": "content_block_delta", "delta": {"type": "thinking_delta", "thinking": text}},
        parent_tool_use_id=parent,
    )


def _block_stop(parent: str | None = None) -> StreamEvent:
    return stream_event({"type": "content_block_stop"}, parent_tool_use_id=parent)


# Each block start reports the running estimate with no delta. Deltas under the
# 200-char throttle flush only at block stop; a block's estimate builds on the
# earlier blocks' at four chars per token.
@pytest.mark.parametrize(
    ("messages", "expected"),
    [
        pytest.param(
            [
                _thinking_start(),
                _thinking_delta("a" * 100),
                _block_stop(),
                _thinking_start(),
                _thinking_delta("abcdefgh"),
                _block_stop(),
            ],
            [(0, 0), (25, 25), (0, 25), (2, 27)],
            id="under-throttle-flush-at-stop",
        ),
        pytest.param(
            [
                _thinking_start(),
                _thinking_delta("a" * 220),
                _thinking_delta("a" * 30),
                _block_stop(),
            ],
            [(0, 0), (55, 55), (8, 63)],
            id="single-delta-crosses-throttle",
        ),
        pytest.param(
            [_thinking_start(), *[_thinking_delta("a" * 50)] * 20, _block_stop()],
            [(0, 0), (50, 50), (50, 100), (50, 150), (50, 200), (50, 250)],
            id="small-deltas-accumulate-past-throttle",
        ),
    ],
)
async def test_start_when_thinking_deltas_stream_does_emit_throttled_estimates(
    messages: list[StreamEvent], expected: list[tuple[int, int]]
):
    events = await run_with_messages(messages)

    updates = events_of(events, ThinkingUpdateEvent)
    assert [(u.delta, u.estimated_tokens) for u in updates] == expected


# ---------------------------------------------------------------------------
# stream events — phase transitions
# ---------------------------------------------------------------------------


_THINKING_BLOCK_MESSAGES = [_thinking_start("tu_x"), _block_stop("tu_x")]
_TEXT_BLOCK_MESSAGES = [
    stream_event(
        {"type": "content_block_start", "content_block": {"type": "text"}},
        parent_tool_use_id="tu_x",
    ),
    _block_stop("tu_x"),
]
_TOOL_USE_BLOCK_MESSAGES = [
    stream_event(
        {"type": "content_block_start", "content_block": {"type": "tool_use", "name": "Read"}},
        parent_tool_use_id="tu_x",
    ),
    _block_stop("tu_x"),
]


@pytest.mark.parametrize(
    ("messages", "expected_phase", "expected_tool_name"),
    [
        pytest.param(_THINKING_BLOCK_MESSAGES, "thinking", None, id="thinking-block-start"),
        pytest.param(_TEXT_BLOCK_MESSAGES, "responding", None, id="text-block-start"),
        pytest.param(_TOOL_USE_BLOCK_MESSAGES, "tool_input", "Read", id="tool-use-block-start"),
        pytest.param(
            [stream_event({"type": "message_stop"}, parent_tool_use_id="tu_x")],
            "turn_end",
            None,
            id="message-stop",
        ),
    ],
)
async def test_start_when_phase_event_received_does_emit_model_phase(
    messages: list[StreamEvent], expected_phase: str, expected_tool_name: str | None
):
    events = await run_with_messages(messages)

    phases = events_of(events, ModelPhaseEvent)
    assert [(p.phase, p.tool_name, p.parent_tool_use_id) for p in phases] == [
        (expected_phase, expected_tool_name, "tu_x")
    ]


# ---------------------------------------------------------------------------
# stream events — silent event types
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "event",
    [
        pytest.param(
            {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "hi"}},
            id="text-delta",
        ),
        pytest.param(
            {
                "type": "content_block_delta",
                "delta": {"type": "input_json_delta", "partial_json": '{"a"'},
            },
            id="input-json-delta",
        ),
        pytest.param(
            {
                "type": "content_block_delta",
                "delta": {"type": "signature_delta", "signature": "sig"},
            },
            id="signature-delta",
        ),
        pytest.param({"type": "message_start"}, id="message-start"),
        pytest.param({"type": "message_delta"}, id="message-delta"),
        pytest.param({"type": "totally_unknown_type"}, id="unrecognized"),
    ],
)
async def test_start_when_silent_event_type_does_emit_nothing(event: dict[str, object]):
    messages = [stream_event(event)]

    events = await run_with_messages(messages)

    assert events == []


# ---------------------------------------------------------------------------
# stream events — per-parent thinking scoping
# ---------------------------------------------------------------------------


async def test_start_when_subagent_thinking_does_not_inflate_top_level_total():
    messages = [
        _thinking_start(),
        _thinking_delta("aaaa"),
        _block_stop(),
        _thinking_start("tu_sub"),
        _thinking_delta("b" * 400, "tu_sub"),
        _block_stop("tu_sub"),
        _thinking_start(),
    ]

    events = await run_with_messages(messages)

    updates = events_of(events, ThinkingUpdateEvent)
    top = [(u.delta, u.estimated_tokens) for u in updates if u.parent_tool_use_id is None]
    sub = [(u.delta, u.estimated_tokens) for u in updates if u.parent_tool_use_id == "tu_sub"]
    assert top == [(0, 0), (1, 1), (0, 1)]
    assert sub == [(0, 0), (100, 100)]
