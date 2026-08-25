"""The Claude Agent SDK driver: a streaming session backed by the real SDK.

``create_claude_driver`` returns a :class:`~gymrat_py.supervisor.driver.Driver`
that drives one agent session per :meth:`~gymrat_py.supervisor.driver.Driver.start`
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
import json
import warnings
from collections.abc import AsyncIterator, Callable, Mapping
from math import ceil
from typing import Any, Protocol, cast

from gymrat_py.errors import message_of
from gymrat_py.session.clock import now_ms
from gymrat_py.supervisor.driver import (
    Driver,
    DriverSession,
    SessionObserver,
    SessionOutcome,
    SessionPrompt,
)
from gymrat_py.supervisor.events import (
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

    # The SDK constructors expose many typed keyword-only fields; the driver
    # forwards a validated subset as a mapping, so treat the module as untyped
    # at this one boundary rather than thread every optional field through.
    sdk = cast("Any", claude_agent_sdk)

    def factory(options: Mapping[str, object]) -> ClaudeClient:
        return sdk.ClaudeSDKClient(sdk.ClaudeAgentOptions(**options))

    return factory


def _build_options(prompt: SessionPrompt) -> dict[str, object]:
    """Assemble the SDK options mapping; the kickoff is sent via ``query`` instead."""
    options: dict[str, object] = {"cwd": prompt.cwd, "permission_mode": "bypassPermissions"}
    if prompt.system_prompt_append is not None:
        options["system_prompt"] = {
            "type": "preset",
            "preset": "claude_code",
            "append": prompt.system_prompt_append,
        }
    if prompt.model is not None:
        options["model"] = prompt.model
    return options


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
        self._estimated_tokens = 0
        self._tool_starts: dict[str, int] = {}
        self._tool_names: dict[str, str] = {}
        self._stopped: SessionOutcome | None = None
        self._task: asyncio.Task[SessionOutcome] = asyncio.ensure_future(self._run())

    @property
    def outcome(self) -> asyncio.Task[SessionOutcome]:
        return self._task

    async def interrupt(self) -> None:
        if self._stopped is None:
            self._stopped = SessionOutcome(reason="interrupted", cost_usd=self._cost_usd)
            if self._client is not None:
                await self._client.interrupt()

    async def _watch_abort(self, abort: asyncio.Event, client: ClaudeClient) -> None:
        await abort.wait()
        if self._stopped is None:
            self._stopped = SessionOutcome(reason="interrupted", cost_usd=self._cost_usd)
        # Disconnect so the streaming loop unblocks; the first stop already
        # captured the cost, so a later abort leaves it untouched.
        await client.disconnect()

    def _settled_or(self, default: SessionOutcome) -> SessionOutcome:
        return self._stopped if self._stopped is not None else default

    async def _resolve_factory(self) -> ClientFactory | SessionOutcome:
        if self._client_factory is not None:
            return self._client_factory
        try:
            return _load_default_factory()
        except ModuleNotFoundError as err:
            detail = message_of(err)
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
                SessionOutcome(reason="error", cost_usd=self._cost_usd, message=message_of(err))
            )
        finally:
            await self._teardown()

    async def _stream(self, factory: ClientFactory) -> SessionOutcome:
        client = factory(_build_options(self._prompt))
        self._client = client
        if self._abort is not None:
            self._abort_task = asyncio.ensure_future(self._watch_abort(self._abort, client))
        await client.connect()
        await client.query(self._prompt.kickoff)
        async for message in client.receive_messages():
            if self._stopped is not None:
                break
            self._map_message(message)
            # Yield so an observer-scheduled interrupt or a fired abort is
            # applied before the next message is drawn from the stream.
            await asyncio.sleep(0)
            if self._stopped is not None:
                break
        return self._settled_or(SessionOutcome(reason="completed", cost_usd=self._cost_usd))

    async def _teardown(self) -> None:
        if self._abort_task is not None:
            self._abort_task.cancel()
        if self._client is not None:
            try:
                await self._client.disconnect()
            except Exception as err:  # noqa: BLE001 - teardown must not mask the already-settled outcome
                warnings.warn(
                    f"claude client disconnect failed: {message_of(err)}",
                    RuntimeWarning,
                    stacklevel=2,
                )

    def _map_message(self, message: object) -> None:
        cost = getattr(message, "total_cost_usd", None)
        if isinstance(cost, (int, float)) and not isinstance(cost, bool) and cost > 0:
            # Commit the cost before the observer fires so a callback reading it
            # (e.g. to interrupt at a threshold) sees the just-crossed value.
            self._cost_usd = float(cost)
            self._observer(UsageUpdateEvent(timestamp=now_ms(), cost_usd=self._cost_usd))
            return

        content = getattr(message, "content", None)
        if isinstance(content, list):
            for block in content:
                self._map_block(block)

    def _map_block(self, block: object) -> None:
        text = getattr(block, "text", None)
        if isinstance(text, str):
            self._observer(TextDeltaEvent(timestamp=now_ms(), chunk=text))
            return

        thinking = getattr(block, "thinking", None)
        if isinstance(thinking, str):
            delta = ceil(len(thinking) / 4)
            self._estimated_tokens += delta
            self._observer(
                ThinkingUpdateEvent(
                    timestamp=now_ms(),
                    estimated_tokens=self._estimated_tokens,
                    delta=delta,
                )
            )
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
                    input_summary=summarize_input(tool_input),
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
