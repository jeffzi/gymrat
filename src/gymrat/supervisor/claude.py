"""The Claude Agent SDK driver: a streaming session backed by the real SDK.

``create_claude_driver`` returns a :class:`~gymrat.supervisor.driver.Driver`
that drives one agent session per :meth:`~gymrat.supervisor.driver.Driver.start`
through the SDK's streaming client. The client is obtained from an injectable
``client_factory`` so the whole unit suite runs against a fake; the real SDK is
imported lazily inside the run task (never at import or construction) so that
merely importing this module — the import-latency guard depends on it — does not
require ``claude-agent-sdk`` to be installed.

Each SDK message is mapped to a session event by attribute, not by class name,
so any duck-typed object shaped like the SDK's message and block dataclasses
maps correctly and anything malformed is skipped in silence.
"""

import asyncio
import contextlib
import json
import warnings
from collections.abc import AsyncIterator, Callable, Mapping
from math import ceil
from typing import Literal, Protocol

from gymrat.session.clock import now_ms
from gymrat.supervisor.driver import (
    Driver,
    DriverSession,
    SessionObserver,
    SessionOutcome,
    SessionPrompt,
)
from gymrat.supervisor.events import (
    ModelPhaseEvent,
    TextDeltaEvent,
    ThinkingUpdateEvent,
    ToolEndEvent,
    ToolStartEvent,
    UsageUpdateEvent,
    summarize,
    summarize_input,
)


class ClaudeClient(Protocol):
    """The streaming client surface the driver depends on.

    Structural, not nominal: the injected fake and the real
    ``ClaudeSDKClient`` both satisfy it without a shared base class.
    """

    async def connect(self) -> None:
        """Establish the session with the agent backend."""
        ...

    async def query(self, prompt: str) -> None:
        """Send the kickoff prompt that starts the agent's work."""
        ...

    def receive_messages(self) -> AsyncIterator[object]:
        """Stream SDK messages until the session ends."""
        ...

    async def interrupt(self) -> None:
        """Ask the agent to stop without tearing down the connection."""
        ...

    async def disconnect(self) -> None:
        """Tear down the session and release its resources."""
        ...


ClientFactory = Callable[[Mapping[str, object]], ClaudeClient]
"""Builds a :class:`ClaudeClient` from an SDK-native options mapping."""


def _load_default_factory() -> ClientFactory:  # pragma: no cover - needs the package + live CLI
    # Imported lazily, not at module top: keeps claude-agent-sdk an optional
    # dependency and off the import-latency path the guard test protects.
    import claude_agent_sdk  # noqa: PLC0415

    def factory(options: Mapping[str, object]) -> ClaudeClient:
        # options is a validated dict from _build_options; checker can't verify
        # the **spread into ClaudeAgentOptions's typed kwargs.
        return claude_agent_sdk.ClaudeSDKClient(claude_agent_sdk.ClaudeAgentOptions(**options))  # pyrefly: ignore[bad-argument-type]

    return factory


def _build_options(prompt: SessionPrompt) -> dict[str, object]:
    """Assemble the SDK options mapping; the kickoff is sent via ``query`` instead."""
    options: dict[str, object] = {
        "cwd": prompt.cwd,
        "permission_mode": "bypassPermissions",
        "include_partial_messages": True,
    }
    if prompt.system_prompt_append is not None:
        options["system_prompt"] = {
            "type": "preset",
            "preset": "claude_code",
            "append": prompt.system_prompt_append,
        }
    if prompt.model is not None:
        options["model"] = prompt.model
    return options


#: Rough chars-per-token ratio used to estimate thinking-block token counts,
#: since the SDK reports thinking as text, not a token count.
_CHARS_PER_TOKEN_ESTIMATE = 4

#: A ThinkingUpdateEvent is emitted only after this many characters accumulate
#: since the last emit, keeping the event rate bounded during long thinking blocks.
_THINKING_EMIT_CHARS = 200

#: Mirrors :class:`~gymrat.supervisor.events.ModelPhaseEvent`'s ``phase`` literal.
_ModelPhase = Literal["thinking", "responding", "tool_input", "turn_end"]


class _ThinkingStream:
    """Per-parent running state for streamed thinking deltas."""

    __slots__ = ("chars_since_emit", "estimated_tokens", "text_len")

    def __init__(self) -> None:
        self.estimated_tokens: int = 0
        self.chars_since_emit: int = 0
        self.text_len: int = 0


async def _disconnect_quietly(client: ClaudeClient) -> None:
    """Disconnect ``client``, warning instead of raising.

    The caller's outcome is already settled by the time this runs; a
    disconnect failure must not replace it.
    """
    try:
        await client.disconnect()
    except Exception as err:  # noqa: BLE001 - must not replace the already-settled outcome
        warnings.warn(f"claude client disconnect failed: {err!s}", RuntimeWarning, stacklevel=2)


def _stringify_result(content: object) -> str:
    if isinstance(content, str):
        return content
    try:
        return json.dumps(content)
    except (TypeError, ValueError):
        return str(content)


class _ClaudeSession:
    """Runs one SDK session as a task; ``outcome`` settles when the stream ends.

    ``interrupt`` and an external ``abort`` both drive the session to an
    ``interrupted`` outcome. The first stop wins: whichever fires first captures
    the running cost, and later stops leave that cost untouched. ``interrupt``
    is soft — it calls ``client.interrupt()`` without tearing down the
    connection — while ``abort`` disconnects the client to unblock the stream.
    """

    def __init__(
        self,
        client_factory: ClientFactory | None,
        prompt: SessionPrompt,
        observer: SessionObserver,
        abort: asyncio.Event | None,
    ) -> None:
        self._client_factory = client_factory
        self._prompt = prompt
        self._observer = observer
        self._abort = abort
        self._client: ClaudeClient | None = None
        self._abort_task: asyncio.Task[None] | None = None
        self._cost_usd = 0.0
        self._thinking_streams: dict[str | None, _ThinkingStream] = {}
        self._tool_starts: dict[str, int] = {}
        self._tool_names: dict[str, str] = {}
        self._stopped: SessionOutcome | None = None
        self._result_outcome: SessionOutcome | None = None
        self._task: asyncio.Task[SessionOutcome] = asyncio.create_task(self._run())

    @property
    def outcome(self) -> asyncio.Task[SessionOutcome]:
        return self._task

    def _claim_interrupted(self) -> bool:
        """Mark the session interrupted if unset; report whether this call did the marking."""
        if self._stopped is not None:
            return False
        self._stopped = SessionOutcome(reason="interrupted", cost_usd=self._cost_usd)
        return True

    async def interrupt(self) -> None:
        if self._claim_interrupted() and self._client is not None:
            await self._client.interrupt()

    async def _watch_abort(self, abort: asyncio.Event, client: ClaudeClient) -> None:
        await abort.wait()
        self._claim_interrupted()
        # Disconnect so the streaming loop unblocks; the first stop already
        # captured the cost, so a later abort leaves it untouched.
        await _disconnect_quietly(client)

    def _settled_or(self, default: SessionOutcome) -> SessionOutcome:
        return self._stopped if self._stopped is not None else default

    async def _resolve_factory(self) -> ClientFactory | SessionOutcome:
        if self._client_factory is not None:
            return self._client_factory
        try:
            return _load_default_factory()
        except ModuleNotFoundError as err:
            detail = str(err)
            return SessionOutcome(
                reason="error",
                cost_usd=0.0,
                message=f"The claude-agent-sdk package is not installed: {detail}",
            )

    async def _run(self) -> SessionOutcome:
        if self._abort is not None and self._abort.is_set():
            return SessionOutcome(reason="interrupted", cost_usd=0.0)

        factory = await self._resolve_factory()
        if isinstance(factory, SessionOutcome):
            return factory

        try:
            return await self._stream(factory)
        except Exception as err:  # noqa: BLE001 - any construction or stream failure becomes an error outcome, never a raise
            return self._settled_or(
                SessionOutcome(reason="error", cost_usd=self._cost_usd, message=str(err))
            )
        finally:
            await self._teardown()

    async def _stream(self, factory: ClientFactory) -> SessionOutcome:
        client = factory(_build_options(self._prompt))
        self._client = client
        if self._abort is not None:
            self._abort_task = asyncio.create_task(self._watch_abort(self._abort, client))
        await client.connect()
        if self._stopped is not None:
            return self._stopped
        await client.query(self._prompt.kickoff)
        async for message in client.receive_messages():
            if self._stopped is not None:
                break
            self._map_message(message)
            if self._result_outcome is not None:
                break
            # Yield so an observer-scheduled interrupt or a fired abort is
            # applied before the next message is drawn from the stream.
            await asyncio.sleep(0)
            if self._stopped is not None:
                break
        if self._stopped is not None:
            return self._stopped
        if self._result_outcome is not None:
            return self._result_outcome
        return SessionOutcome(
            reason="error",
            cost_usd=self._cost_usd,
            message="Agent stream ended without a result message",
        )

    async def _teardown(self) -> None:
        if self._abort_task is not None:
            # Cancel then await under suppression so the abort watcher's
            # CancelledError is retrieved here, never surfaced by the loop as a
            # forgotten-task diagnostic (matches the stdio driver's teardown).
            self._abort_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._abort_task
        if self._client is not None:
            await _disconnect_quietly(self._client)

    def _map_message(self, message: object) -> None:
        """Map one SDK message to session events."""
        # Stream events carry an ``event`` dict and no ``content``.
        event = getattr(message, "event", None)
        if isinstance(event, dict) and not hasattr(message, "content"):
            parent = getattr(message, "parent_tool_use_id", None)
            self._map_stream_event(event, parent)
            return

        # A result message carries both ``subtype`` and ``num_turns``; a
        # system message has ``subtype`` alone and is silently passed through.
        # Checked ahead of the usage emit below: a result settles the session
        # on its own, so its cost must never also be routed through the
        # observer, where a spend-cap callback would treat it as a live
        # threshold crossing the session has already outrun.
        subtype = getattr(message, "subtype", None)
        num_turns = getattr(message, "num_turns", None)
        if isinstance(subtype, str) and num_turns is not None:
            cost = getattr(message, "total_cost_usd", None)
            if isinstance(cost, (int, float)) and not isinstance(cost, bool) and cost > 0:
                self._cost_usd = float(cost)
            self._result_outcome = self._result_outcome_from(message, subtype)
            return

        cost = getattr(message, "total_cost_usd", None)
        if isinstance(cost, (int, float)) and not isinstance(cost, bool) and cost > 0:
            # Commit the cost before notifying observers: a spend-cap
            # callback reads self._cost_usd synchronously and must see the
            # value that just crossed the threshold.
            self._cost_usd = float(cost)
            self._observer(UsageUpdateEvent(timestamp=now_ms(), cost_usd=self._cost_usd))

        parent = getattr(message, "parent_tool_use_id", None)
        content = getattr(message, "content", None)
        if isinstance(content, list):
            for block in content:
                self._map_block(block, parent)

    def _result_outcome_from(self, message: object, subtype: str) -> SessionOutcome:
        """Classify a settled result message as completed or errored."""
        is_error = getattr(message, "is_error", False)
        if is_error:
            result_text = getattr(message, "result", None)
            return SessionOutcome(
                reason="error",
                cost_usd=self._cost_usd,
                message=result_text if isinstance(result_text, str) else subtype,
            )
        return SessionOutcome(reason="completed", cost_usd=self._cost_usd)

    def _emit_phase(
        self, phase: _ModelPhase, parent: str | None, tool_name: str | None = None
    ) -> None:
        self._observer(
            ModelPhaseEvent(
                timestamp=now_ms(), phase=phase, tool_name=tool_name, parent_tool_use_id=parent
            )
        )

    def _map_stream_event(self, event: dict[str, object], parent: str | None) -> None:
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
            self._observer(
                TextDeltaEvent(timestamp=now_ms(), chunk=text, parent_tool_use_id=parent)
            )
            return

        # Complete ThinkingBlocks are skipped — streamed deltas already counted them.
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
                        supervised_root=self._prompt.cwd,
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


class _ClaudeDriver:
    def __init__(self, client_factory: ClientFactory | None) -> None:
        self._client_factory = client_factory

    def start(
        self,
        prompt: SessionPrompt,
        observer: SessionObserver,
        abort: asyncio.Event | None = None,
    ) -> DriverSession:
        return _ClaudeSession(self._client_factory, prompt, observer, abort)


def create_claude_driver(client_factory: ClientFactory | None = None) -> Driver:
    """Build a :class:`Driver` backed by the Claude Agent SDK.

    Args:
        client_factory: Builds the streaming client from an SDK options
            mapping. When ``None``, the real SDK is imported lazily inside the
            session's run task and the default factory constructs a
            ``ClaudeSDKClient``.

    Returns:
        A driver whose ``start`` launches one SDK session per call.
    """
    return _ClaudeDriver(client_factory)
