"""Message-to-event mapping for the Claude Agent SDK driver.

This module owns the message vocabulary only; session-lifecycle concerns
(connect, stream, interrupt, send, end) belong in
:mod:`gymrat.supervisor.claude`. Content blocks are matched structurally on the
SDK's own dataclasses; the SDK import is deferred to call time so importing this
module never loads the SDK.
"""

from __future__ import annotations

import json
from math import ceil
from typing import TYPE_CHECKING, Literal

from gymrat import clock
from gymrat.clock import now_ns
from gymrat.supervisor.driver import SessionOutcome, usable_cost
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

if TYPE_CHECKING:
    from claude_agent_sdk import ContentBlock, ResultMessage

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


def detect_origin(message: ResultMessage) -> Literal["agent", "injected"]:
    """Classify a result message's origin for the turn-end event.

    A human-initiated turn is still the agent's response.

    Args:
        message: The result message to classify.

    Returns:
        ``"agent"`` when the origin is absent, lacks a ``kind``, or has
        ``kind == "human"``; ``"injected"`` for any other ``kind`` (e.g.
        ``"task-notification"``).
    """
    origin = message.origin
    if origin is None or origin.get("kind", "human") == "human":
        return "agent"
    return "injected"


def result_outcome_from(message: ResultMessage, cost_usd: float) -> SessionOutcome:
    """Classify a settled error result message.

    Args:
        message: The error result message.
        cost_usd: Cumulative cost to record on the outcome.

    Returns:
        A ``SessionOutcome`` with ``reason="error"`` whose message is the
        result text, or the subtype when the result has no text.
    """
    return SessionOutcome(
        reason="error",
        cost_usd=cost_usd,
        message=message.result if message.result is not None else message.subtype,
    )


def read_cost(message: ResultMessage) -> float | None:
    """Extract cumulative cost from a result message.

    Args:
        message: The result message to read.

    Returns:
        The cost as a float when it passes
        :func:`~gymrat.supervisor.driver.usable_cost`, ``None`` otherwise.
    """
    return usable_cost(message.total_cost_usd)


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
        self._tool_starts: dict[str, float] = {}
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
        """Dispatch a raw SDK stream event to the appropriate handler.

        Args:
            event: A raw SDK stream event dict keyed by ``"type"``.
            parent: The tool-use ID of the enclosing tool call, or ``None``
                for top-level content.
        """
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

    def map_blocks(self, content: list[ContentBlock], parent: str | None) -> None:
        """Map settled content blocks (text, tool-use, tool-result) to events.

        Client and server tool blocks map alike. Thinking blocks emit nothing:
        their tokens were already counted from the streamed thinking deltas.

        Args:
            content: The settled content blocks from a message response.
            parent: The tool-use ID of the enclosing tool call, or ``None``
                for top-level content.
        """
        from claude_agent_sdk import (  # noqa: PLC0415 -- deferred to avoid import-time SDK load
            ServerToolResultBlock,
            ServerToolUseBlock,
            TextBlock,
            ToolResultBlock,
            ToolUseBlock,
        )

        for block in content:
            match block:
                case TextBlock(text=text):
                    self._emit_text(text, parent)
                case (
                    ToolUseBlock(id=tool_use_id, name=name, input=tool_input)
                    | ServerToolUseBlock(id=tool_use_id, name=name, input=tool_input)
                ):
                    self._start_tool(tool_use_id, name, tool_input, parent)
                case (
                    ToolResultBlock(tool_use_id=tool_use_id, content=result)
                    | ServerToolResultBlock(tool_use_id=tool_use_id, content=result)
                ):
                    self._end_tool(tool_use_id, result, parent)

    def _emit_phase(
        self, phase: _ModelPhase, parent: str | None, tool_name: str | None = None
    ) -> None:
        self._observer(
            ModelPhaseEvent(
                at=now_ns(), phase=phase, tool_name=tool_name, parent_tool_use_id=parent
            )
        )

    def _handle_block_start(self, content_block: dict[str, object], parent: str | None) -> None:
        block_type = content_block.get("type")
        if block_type == "thinking":
            self._emit_phase("thinking", parent)
            stream = self._thinking_streams.setdefault(parent, _ThinkingStream())
            self._observer(
                ThinkingUpdateEvent(
                    at=now_ns(),
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
                at=now_ns(),
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

    def _emit_text(self, text: str, parent: str | None) -> None:
        if parent is None:
            self._last_top_level_text = text
        self._observer(TextDeltaEvent(at=now_ns(), chunk=text, parent_tool_use_id=parent))

    def _start_tool(
        self, tool_use_id: str, name: str, tool_input: dict[str, object], parent: str | None
    ) -> None:
        self._tool_starts[tool_use_id] = clock.monotonic_ms()
        self._tool_names[tool_use_id] = name
        self._observer(
            ToolStartEvent(
                at=now_ns(),
                tool_use_id=tool_use_id,
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

    def _end_tool(self, tool_use_id: str, content: object, parent: str | None) -> None:
        start = self._tool_starts.get(tool_use_id)
        duration_ms = int(clock.monotonic_ms() - start) if start is not None else 0
        result = _stringify_result(content)
        self._observer(
            ToolEndEvent(
                at=now_ns(),
                tool_use_id=tool_use_id,
                tool_name=self._tool_names.get(tool_use_id, "unknown"),
                duration_ms=duration_ms,
                result=result,
                result_summary=summarize(result),
                parent_tool_use_id=parent,
            )
        )
