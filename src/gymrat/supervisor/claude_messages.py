"""Message-to-event mapping for the Claude Agent SDK driver.

Extracted from :mod:`gymrat.supervisor.claude` so the session lifecycle and
the message vocabulary live in separate modules. This module imports nothing
from the SDK — duck-typed attribute access handles both real and fake messages.
"""

import json
from collections.abc import Mapping
from math import ceil
from typing import Literal

from gymrat.session.clock import now_ms
from gymrat.supervisor.driver import SessionOutcome
from gymrat.supervisor.events import (
    ModelPhaseEvent,
    SessionObserver,
    TextDeltaEvent,
    ThinkingUpdateEvent,
    ToolEndEvent,
    ToolStartEvent,
    summarize,
    summarize_input,
)

#: Rough chars-per-token ratio used to estimate thinking-block token counts,
#: since the SDK reports thinking as text, not a token count.
_CHARS_PER_TOKEN_ESTIMATE = 4
#: A ThinkingUpdateEvent is emitted only after this many characters accumulate
#: since the last emit, keeping the event rate bounded during long thinking blocks.
_THINKING_EMIT_CHARS = 200
_ModelPhase = Literal["thinking", "responding", "tool_input", "turn_end"]


class _ThinkingStream:
    """Per-parent running state for streamed thinking deltas."""

    __slots__ = ("chars_since_emit", "estimated_tokens", "text_len")

    def __init__(self) -> None:
        self.estimated_tokens: int = 0
        self.chars_since_emit: int = 0
        self.text_len: int = 0


def _stringify_result(content: object) -> str:
    if isinstance(content, str):
        return content
    try:
        return json.dumps(content)
    except (TypeError, ValueError):
        return str(content)


def detect_origin(message: object) -> Literal["agent", "injected"]:
    """Classify a result message's origin for the turn-end event.

    ``"agent"`` when the origin is absent, ``None``, or a mapping/object
    whose ``kind`` is ``"human"`` (a human-initiated turn is still the
    agent's response). Any other ``kind`` — e.g. ``"task-notification"`` —
    means the turn was injected into the conversation.
    """
    origin = getattr(message, "origin", None)
    if origin is None:
        return "agent"
    kind = origin.get("kind") if isinstance(origin, Mapping) else getattr(origin, "kind", None)
    if kind is None or kind == "human":
        return "agent"
    return "injected"


def result_outcome_from(message: object, subtype: str, cost_usd: float) -> SessionOutcome:
    """Classify a settled error result message."""
    result_text = getattr(message, "result", None)
    return SessionOutcome(
        reason="error",
        cost_usd=cost_usd,
        message=result_text if isinstance(result_text, str) else subtype,
    )


def read_cost(message: object) -> float | None:
    """Extract cumulative cost from a message, or ``None`` when absent or zero."""
    cost = getattr(message, "total_cost_usd", None)
    if isinstance(cost, (int, float)) and not isinstance(cost, bool) and cost > 0:
        return float(cost)
    return None


class MessageMapper:
    """Maps SDK messages to session events, owning the per-turn mapping state.

    Emits session events directly through the observer. Turn-text tracking
    and tool-call state live here; cost commitment and outcome settlement
    stay in the session.
    """

    def __init__(self, observer: SessionObserver, supervised_root: str) -> None:
        self._observer = observer
        self._supervised_root = supervised_root
        self._thinking_streams: dict[str | None, _ThinkingStream] = {}
        self._tool_starts: dict[str, int] = {}
        self._tool_names: dict[str, str] = {}
        self._last_top_level_text: str = ""

    @property
    def last_top_level_text(self) -> str:
        """The most recent top-level text block from the agent."""
        return self._last_top_level_text

    def reset_turn_text(self) -> None:
        """Clear accumulated turn text after a turn boundary."""
        self._last_top_level_text = ""

    def map_stream_event(self, event: dict[str, object], parent: str | None) -> None:
        """Dispatch a raw SDK stream event to the appropriate handler."""
        event_type = event.get("type")
        if event_type == "content_block_start":
            content_block = event.get("content_block")
            if isinstance(content_block, dict):
                self._handle_block_start(content_block, parent)
        elif event_type == "content_block_delta":
            delta_obj = event.get("delta")
            if isinstance(delta_obj, dict) and delta_obj.get("type") == "thinking_delta":
                text = delta_obj.get("thinking", "")
                if isinstance(text, str):
                    self._handle_thinking_delta(text, parent)
        elif event_type == "content_block_stop":
            self._flush_thinking_remainder(parent)
        elif event_type == "message_stop":
            self._emit_phase("turn_end", parent)

    def map_blocks(self, content: list[object], parent: str | None) -> None:
        """Map settled content blocks (text, tool-use, tool-result) to events."""
        for block in content:
            self._map_block(block, parent)

    def _emit_phase(
        self, phase: _ModelPhase, parent: str | None, tool_name: str | None = None
    ) -> None:
        self._observer(
            ModelPhaseEvent(
                timestamp=now_ms(), phase=phase, tool_name=tool_name, parent_tool_use_id=parent
            )
        )

    def _handle_block_start(self, content_block: dict[str, object], parent: str | None) -> None:
        block_type = content_block.get("type")
        if block_type == "thinking":
            self._emit_phase("thinking", parent)
            stream = self._thinking_streams.setdefault(parent, _ThinkingStream())
            self._observer(
                ThinkingUpdateEvent(
                    timestamp=now_ms(),
                    estimated_tokens=stream.estimated_tokens,
                    delta=0,
                    parent_tool_use_id=parent,
                )
            )
        elif block_type == "text":
            self._emit_phase("responding", parent)
        elif block_type == "tool_use":
            name = content_block.get("name")
            self._emit_phase(
                "tool_input", parent, tool_name=name if isinstance(name, str) else None
            )

    def _emit_thinking_update(self, stream: _ThinkingStream, parent: str | None) -> None:
        new_estimate = ceil(stream.text_len / _CHARS_PER_TOKEN_ESTIMATE)
        delta = new_estimate - stream.estimated_tokens
        stream.estimated_tokens = new_estimate
        stream.chars_since_emit = 0
        self._observer(
            ThinkingUpdateEvent(
                timestamp=now_ms(),
                estimated_tokens=stream.estimated_tokens,
                delta=delta,
                parent_tool_use_id=parent,
            )
        )

    def _handle_thinking_delta(self, text: str, parent: str | None) -> None:
        stream = self._thinking_streams.setdefault(parent, _ThinkingStream())
        stream.text_len += len(text)
        stream.chars_since_emit += len(text)
        if stream.chars_since_emit >= _THINKING_EMIT_CHARS:
            self._emit_thinking_update(stream, parent)

    def _flush_thinking_remainder(self, parent: str | None) -> None:
        stream = self._thinking_streams.get(parent)
        if stream is not None and stream.chars_since_emit != 0:
            self._emit_thinking_update(stream, parent)

    def _map_block(self, block: object, parent: str | None = None) -> None:
        text = getattr(block, "text", None)
        if isinstance(text, str):
            if parent is None:
                self._last_top_level_text = text
            self._observer(
                TextDeltaEvent(timestamp=now_ms(), chunk=text, parent_tool_use_id=parent)
            )
            return

        if hasattr(block, "thinking"):
            return

        block_id = getattr(block, "id", None)
        name = getattr(block, "name", None)
        if isinstance(block_id, str) and isinstance(name, str):
            self._tool_starts[block_id] = now_ms()
            self._tool_names[block_id] = name
            tool_input = getattr(block, "input", None)
            self._observer(
                ToolStartEvent(
                    timestamp=now_ms(),
                    tool_use_id=block_id,
                    tool_name=name,
                    input=tool_input,
                    input_summary=summarize_input(
                        tool_input,
                        tool_name=name,
                        supervised_root=self._supervised_root,
                    ),
                    parent_tool_use_id=parent,
                )
            )
            return

        tool_use_id = getattr(block, "tool_use_id", None)
        if isinstance(tool_use_id, str):
            start = self._tool_starts.get(tool_use_id)
            duration_ms = now_ms() - start if start is not None else 0
            result = _stringify_result(getattr(block, "content", None))
            self._observer(
                ToolEndEvent(
                    timestamp=now_ms(),
                    tool_use_id=tool_use_id,
                    tool_name=self._tool_names.get(tool_use_id, "unknown"),
                    duration_ms=duration_ms,
                    result=result,
                    result_summary=summarize(result),
                    parent_tool_use_id=parent,
                )
            )
