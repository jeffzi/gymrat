"""The Claude Agent SDK driver: a streaming session backed by the real SDK.

``create_claude_driver`` returns a :class:`~gymrat.supervisor.driver.Driver`
that drives one agent session per :meth:`~gymrat.supervisor.driver.Driver.start`
through the SDK's streaming client. The client is obtained from an injectable
``client_factory`` so the whole unit suite runs against a fake. The real SDK,
``claude-agent-sdk``, is a required dependency; it is imported lazily inside
the run task (never at import or construction) only so that merely importing
this module does not pay the SDK's import cost — the import-latency guard
depends on it.

The module owns both halves of the driver: the session lifecycle (connect,
stream, interrupt, send, end) and, in :class:`MessageMapper`, the mapping from
SDK messages to session events. Content blocks are matched structurally on the
SDK's own dataclasses.
"""

import asyncio
import contextlib
import json
import warnings
from collections.abc import AsyncIterator, Callable, Mapping
from dataclasses import dataclass
from math import ceil, isfinite
from typing import TYPE_CHECKING, Literal, Protocol

from gymrat import clock
from gymrat.agent_env import TRACEPARENT_ENV
from gymrat.clock import now_ns
from gymrat.supervisor.driver import (
    Driver,
    DriverSession,
    SessionOutcome,
    SessionPrompt,
)
from gymrat.supervisor.events import (
    CompactionEvent,
    ModelPhase,
    ModelPhaseEvent,
    SessionObserver,
    TextDeltaEvent,
    ThinkingUpdateEvent,
    ToolEndEvent,
    ToolStartEvent,
    TurnEndEvent,
    UsageUpdateEvent,
    summarize,
    summarize_input,
)
from gymrat.supervisor.hooks import HooksFactory
from gymrat.supervisor.tools import ToolsFactory

if TYPE_CHECKING:
    from claude_agent_sdk import ContentBlock, ResultMessage

#: Rough chars-per-token ratio used to estimate thinking-block token counts,
#: since the SDK reports thinking as text, not a token count.
_CHARS_PER_TOKEN_ESTIMATE = 4
#: A ThinkingUpdateEvent is emitted only after this many characters accumulate
#: since the last emit, keeping the event rate bounded during long thinking blocks.
_THINKING_EMIT_CHARS = 200


def usable_cost(value: object) -> float | None:
    """Accept a reported session cost only when it is a finite, positive number.

    A missing, non-numeric, NaN, infinite, zero, or negative cost is unusable:
    the caller keeps the cost it already has.

    Args:
        value: The cost as reported, of any type.

    Returns:
        The cost as a float, or ``None`` when it is unusable. Booleans are
        unusable even though ``bool`` subclasses ``int``.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    cost = float(value)
    if isfinite(cost) and cost > 0:
        return cost
    return None


@dataclass(slots=True)
class _ThinkingStream:
    """Per-parent running state for streamed thinking deltas."""

    estimated_tokens: int = 0
    chars_since_emit: int = 0
    text_len: int = 0


def _stringify_result(content: object) -> str:
    if isinstance(content, str):
        return content
    try:
        return json.dumps(content)
    except (TypeError, ValueError):
        return str(content)


def detect_origin(message: "ResultMessage") -> Literal["agent", "injected"]:
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


class MessageMapper:
    """Maps SDK messages to session events, owning the per-turn mapping state.

    Emits session events directly through the observer. Turn-text tracking
    and tool-call state live here; cost commitment and outcome settlement
    stay in the session.

    Args:
        observer: Receives every event the mapper emits.
        supervised_root: The supervised repository root, for summarizing tool
            inputs relative to it.

    Attributes:
        last_top_level_text: The most recent top-level text block from the
            agent. The session clears it at each turn boundary.
    """

    def __init__(self, observer: SessionObserver, supervised_root: str) -> None:
        self._observer = observer
        self._supervised_root = supervised_root
        self._thinking_streams: dict[str | None, _ThinkingStream] = {}
        self._tool_starts: dict[str, float] = {}
        self._tool_names: dict[str, str] = {}
        self.last_top_level_text: str = ""

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

    def map_blocks(self, content: "list[ContentBlock]", parent: str | None) -> None:
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
        self, phase: ModelPhase, parent: str | None, tool_name: str | None = None
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
            self.last_top_level_text = text
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


class ClaudeClient(Protocol):
    """The streaming client surface the driver depends on.

    Structural, not nominal: the injected fake and the real
    ``ClaudeSDKClient`` both satisfy it without a shared base class.
    """

    async def connect(self) -> None:
        """Establish the session with the agent backend."""
        ...

    async def query(self, prompt: str) -> None:
        """Send the kickoff prompt that starts the agent's work.

        Args:
            prompt: The initial prompt text to send.
        """
        ...

    def receive_messages(self) -> AsyncIterator[object]:
        """Stream SDK messages until the session ends.

        Returns:
            An async iterator of raw SDK message objects.
        """
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
    # claude-agent-sdk is a required dependency. It is imported lazily, not at
    # module top, only to keep it off the import-latency path the guard test
    # protects.
    import claude_agent_sdk  # noqa: PLC0415

    def factory(options: Mapping[str, object]) -> ClaudeClient:
        # options is a validated dict from _build_options; checker can't verify
        # the **spread into ClaudeAgentOptions's typed kwargs.
        opts = claude_agent_sdk.ClaudeAgentOptions(**options)  # pyrefly: ignore[bad-argument-type]
        return claude_agent_sdk.ClaudeSDKClient(opts)

    return factory


def _traceparent_env(traceparent: str | None) -> dict[str, str]:
    """The :data:`TRACEPARENT_ENV` entry for *traceparent*, or empty when unset."""
    return {} if traceparent is None else {TRACEPARENT_ENV: traceparent}


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
    if prompt.effort is not None:
        options["effort"] = prompt.effort
    env: dict[str, str] = _traceparent_env(prompt.traceparent)
    if prompt.command_timeout_ms is not None:
        # Raising the shell timeout ceiling to the wall-clock cap requires setting
        # both the default and max tool-use timeout variables. Automatically moving
        # a command to the background is disabled (empty string) because it would
        # otherwise detach a long `gymrat` command into the background, where the
        # agent can no longer observe its output or exit status.
        timeout_ms = str(prompt.command_timeout_ms)
        env["CLAUDE_CODE_DEFAULT_TOOL_USE_TIMEOUT_MS"] = timeout_ms
        env["CLAUDE_CODE_MAX_TOOL_USE_TIMEOUT_MS"] = timeout_ms
        env["CLAUDE_CODE_AUTO_BACKGROUND_TIMEOUT_MS"] = ""
        env["MCP_TOOL_TIMEOUT"] = timeout_ms
    if env:
        options["env"] = env
    if prompt.max_budget_usd is not None:
        options["max_budget_usd"] = prompt.max_budget_usd
    return options


async def _disconnect_quietly(client: ClaudeClient) -> None:
    """Disconnect ``client``, warning instead of raising.

    The caller's outcome is already settled by the time this runs; a
    disconnect failure must not replace it.

    Args:
        client: The client to disconnect.
    """
    try:
        await client.disconnect()
    except Exception as err:  # noqa: BLE001 - must not replace the already-settled outcome
        warnings.warn(f"claude client disconnect failed: {err!s}", RuntimeWarning, stacklevel=2)


@dataclass(frozen=True, slots=True)
class _OptionFactories:
    """Per-session builders whose results are merged into the SDK options."""

    tools: ToolsFactory | None
    hooks: HooksFactory | None


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
        factories: _OptionFactories,
    ) -> None:
        self._client_factory = client_factory
        self._prompt = prompt
        self._observer = observer
        # No event supplied is the same as one that is never set.
        self._abort = abort if abort is not None else asyncio.Event()
        self._factories = factories
        self._client: ClaudeClient | None = None
        self._abort_task: asyncio.Task[None] | None = None
        self._cost_usd = 0.0
        self._mapper = MessageMapper(observer, prompt.cwd)
        self._turn_end_count: int = 0
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

    async def send(self, text: str) -> None:
        if self._stopped is not None or self._result_outcome is not None:
            return
        if self._client is None:
            return
        try:
            await self._client.query(text)
        except Exception as err:  # noqa: BLE001 - query failure settles the session as error
            self._stopped = SessionOutcome(
                reason="error", cost_usd=self._cost_usd, message=str(err)
            )
            await _disconnect_quietly(self._client)

    async def end(self) -> None:
        if self._stopped is not None or self._result_outcome is not None:
            return
        self._stopped = SessionOutcome(reason="completed", cost_usd=self._cost_usd)
        self._commit_cost(self._cost_usd, settled=True)
        if self._client is not None:
            await _disconnect_quietly(self._client)

    async def _watch_abort(self, client: ClaudeClient) -> None:
        await self._abort.wait()
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
            return SessionOutcome(
                reason="error",
                cost_usd=0.0,
                message=f"The claude-agent-sdk package is not installed: {err!s}",
            )

    async def _run(self) -> SessionOutcome:
        if self._abort.is_set():
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
        options = _build_options(self._prompt)
        tools = self._factories.tools
        if tools is not None:
            env = _traceparent_env(self._prompt.traceparent)
            options["mcp_servers"] = {"gymrat": tools(self._abort, env)}
        hooks = self._factories.hooks
        if hooks is not None:
            options["hooks"] = hooks()
        client = factory(options)
        self._client = client
        self._abort_task = asyncio.create_task(self._watch_abort(client))
        await client.connect()
        if self._stopped is not None:
            return self._stopped
        await client.query(self._prompt.kickoff)
        async for message in client.receive_messages():
            if self._stopped is not None:
                break
            turns_before = self._turn_end_count
            self._map_message(message)
            if self._result_outcome is not None:
                break
            # After a turn-end result the recv iterator's own yield between
            # messages provides the interrupt-check window; an extra sleep
            # here would double the per-message yield count needlessly.
            if self._turn_end_count > turns_before:
                continue
            # Yield so an observer-scheduled interrupt or a fired abort is
            # applied before the next message is drawn from the stream.
            await asyncio.sleep(0)
            if self._stopped is not None:
                break
        return self._settled_or(
            self._result_outcome
            or (
                SessionOutcome(reason="completed", cost_usd=self._cost_usd)
                if self._turn_end_count > 0
                else SessionOutcome(
                    reason="error",
                    cost_usd=self._cost_usd,
                    message="Agent stream ended without a result message",
                )
            )
        )

    async def _teardown(self) -> None:
        if self._abort_task is not None:
            # Cancel then await under suppression so the abort watcher's
            # CancelledError is retrieved here, never surfaced by the loop as a
            # forgotten-task diagnostic.
            self._abort_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._abort_task
        if self._client is not None:
            await _disconnect_quietly(self._client)

    def _map_message(self, message: object) -> None:
        """Dispatch one SDK message by class; messages of other classes emit nothing."""
        from claude_agent_sdk import (  # noqa: PLC0415 -- deferred to avoid import-time SDK load
            AssistantMessage,
            ResultMessage,
            StreamEvent,
            SystemMessage,
            UserMessage,
        )

        match message:
            case StreamEvent(event=event, parent_tool_use_id=parent):
                self._mapper.map_stream_event(event, parent)
            case SystemMessage(subtype="compact_boundary"):
                self._observer(CompactionEvent(at=now_ns()))
            case ResultMessage():
                self._map_result(message)
            case AssistantMessage(content=blocks, parent_tool_use_id=parent):
                self._mapper.map_blocks(blocks, parent)
            case UserMessage(content=list() as blocks, parent_tool_use_id=parent):
                self._mapper.map_blocks(blocks, parent)

    def _map_result(self, message: "ResultMessage") -> None:
        """Settle on an error result, or close the turn on any other result.

        A budget-exhausted result is an error the session survives: it closes
        the turn with ``budget_exhausted`` set instead of settling.
        """
        budget_exhausted = message.is_error and message.subtype == "error_max_budget_usd"
        settles = message.is_error and not budget_exhausted
        cost = usable_cost(message.total_cost_usd)
        if cost is not None:
            self._commit_cost(cost, settled=settles)
        if settles:
            self._result_outcome = SessionOutcome(
                reason="error",
                cost_usd=self._cost_usd,
                message=message.result if message.result is not None else message.subtype,
            )
            return

        self._observer(
            TurnEndEvent(
                at=now_ns(),
                text=self._mapper.last_top_level_text,
                cost_usd=self._cost_usd,
                origin=detect_origin(message),
                budget_exhausted=budget_exhausted,
            )
        )
        self._turn_end_count += 1
        self._mapper.last_top_level_text = ""

    def _commit_cost(self, cost: float, *, settled: bool = False) -> None:
        """Set the running cost and notify observers.

        Commits before notifying: a spend-cap callback reads ``self._cost_usd``
        synchronously and must see the value that just crossed the threshold.

        Args:
            cost: The running cost in USD.
            settled: Whether a result message settled the session on its own, so a
                spend-cap observer does not mistake it for a live cap crossing.
        """
        self._cost_usd = cost
        self._observer(UsageUpdateEvent(at=now_ns(), cost_usd=self._cost_usd, settled=settled))


class _ClaudeDriver:
    def __init__(self, client_factory: ClientFactory | None, factories: _OptionFactories) -> None:
        self._client_factory = client_factory
        self._factories = factories

    def start(
        self,
        prompt: SessionPrompt,
        observer: SessionObserver,
        abort: asyncio.Event | None = None,
    ) -> DriverSession:
        return _ClaudeSession(self._client_factory, prompt, observer, abort, self._factories)


def create_claude_driver(
    client_factory: ClientFactory | None = None,
    tools: ToolsFactory | None = None,
    hooks: HooksFactory | None = None,
) -> Driver:
    """Build a :class:`Driver` backed by the Claude Agent SDK.

    Args:
        client_factory: Builds the streaming client from an SDK options
            mapping. When ``None``, the real SDK is imported lazily inside the
            session's run task and the default factory constructs a
            ``ClaudeSDKClient``.
        tools: Builds the MCP server config for gymrat tools. Called once per
            session start with the session's abort event and an env mapping.
            When ``None``, no ``mcp_servers`` key is added to the options.
        hooks: Builds the SDK hooks mapping. Called once per session start;
            its result goes under ``hooks`` in the options. When ``None``, no
            ``hooks`` key is added to the options.

    Returns:
        A driver whose ``start`` launches one SDK session per call.
    """
    return _ClaudeDriver(client_factory, _OptionFactories(tools=tools, hooks=hooks))
