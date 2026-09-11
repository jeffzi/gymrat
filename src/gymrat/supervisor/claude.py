"""The Claude Agent SDK driver: a streaming session backed by the real SDK.

``create_claude_driver`` returns a :class:`~gymrat.supervisor.driver.Driver`
that drives one agent session per :meth:`~gymrat.supervisor.driver.Driver.start`
through the SDK's streaming client. The client is obtained from an injectable
``client_factory`` so the whole unit suite runs against a fake; the real SDK is
imported lazily inside the run task (never at import or construction) so that
merely importing this module — the import-latency guard depends on it — does not
require ``claude-agent-sdk`` to be installed.

Message-to-event mapping lives in :mod:`gymrat.supervisor.claude_messages`;
this module owns the session lifecycle: connect, stream, interrupt, send, end.
"""

import asyncio
import contextlib
import warnings
from collections.abc import AsyncIterator, Callable, Mapping
from typing import Protocol

from gymrat.session.clock import now_ns
from gymrat.supervisor.claude_messages import (
    MessageMapper,
    detect_origin,
    read_cost,
    result_outcome_from,
)
from gymrat.supervisor.driver import (
    Driver,
    DriverSession,
    SessionObserver,
    SessionOutcome,
    SessionPrompt,
)
from gymrat.supervisor.events import CompactionEvent, TurnEndEvent, UsageUpdateEvent


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
    # Imported lazily, not at module top: keeps claude-agent-sdk an optional
    # dependency and off the import-latency path the guard test protects.
    import claude_agent_sdk  # noqa: PLC0415

    def factory(options: Mapping[str, object]) -> ClaudeClient:
        # options is a validated dict from _build_options; checker can't verify
        # the **spread into ClaudeAgentOptions's typed kwargs.
        opts = claude_agent_sdk.ClaudeAgentOptions(**options)  # pyrefly: ignore[bad-argument-type]
        return claude_agent_sdk.ClaudeSDKClient(opts)

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
    if prompt.effort is not None:
        options["effort"] = prompt.effort
    env: dict[str, str] = {}
    if prompt.traceparent is not None:
        env["GYMRAT_TRACEPARENT"] = prompt.traceparent
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
    if env:
        options["env"] = env
    if prompt.max_budget_usd is not None:
        options["max_budget_usd"] = prompt.max_budget_usd
    return options


async def _disconnect_quietly(client: ClaudeClient) -> None:
    """Disconnect ``client``, warning instead of raising.

    The caller's outcome is already settled by the time this runs; a
    disconnect failure must not replace it.
    """
    try:
        await client.disconnect()
    except Exception as err:  # noqa: BLE001 - must not replace the already-settled outcome
        warnings.warn(f"claude client disconnect failed: {err!s}", RuntimeWarning, stacklevel=2)


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
        self._mapper = MessageMapper(observer, prompt.cwd)
        self._had_turn_end: bool = False
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
                if self._had_turn_end
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
            # forgotten-task diagnostic (matches the stdio driver's teardown).
            self._abort_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._abort_task
        if self._client is not None:
            await _disconnect_quietly(self._client)

    def _map_message(self, message: object) -> None:
        """Dispatch by message shape, in order: stream event, result, content blocks."""
        event = getattr(message, "event", None)
        if isinstance(event, dict) and not hasattr(message, "content"):
            parent = getattr(message, "parent_tool_use_id", None)
            self._mapper.map_stream_event(event, parent)
            return

        subtype = getattr(message, "subtype", None)
        num_turns = getattr(message, "num_turns", None)

        # System messages carry ``subtype`` but no ``num_turns``.
        if isinstance(subtype, str) and num_turns is None:
            if subtype == "compact_boundary":
                self._observer(CompactionEvent(at=now_ns()))
            return

        if isinstance(subtype, str) and num_turns is not None:
            is_error = getattr(message, "is_error", False)

            if is_error and subtype != "error_max_budget_usd":
                cost = read_cost(message)
                if cost is not None:
                    self._commit_cost(cost, settled=True)
                self._result_outcome = result_outcome_from(message, subtype, self._cost_usd)
                return

            cost = read_cost(message)
            if cost is not None:
                self._commit_cost(cost)
            self._observer(
                TurnEndEvent(
                    at=now_ns(),
                    text=self._mapper.last_top_level_text,
                    cost_usd=self._cost_usd,
                    origin=detect_origin(message),
                    budget_exhausted=bool(is_error and subtype == "error_max_budget_usd"),
                )
            )
            self._had_turn_end = True
            self._turn_end_count += 1
            self._mapper.reset_turn_text()
            return

        cost = read_cost(message)
        if cost is not None:
            self._commit_cost(cost)

        parent = getattr(message, "parent_tool_use_id", None)
        content = getattr(message, "content", None)
        if isinstance(content, list):
            self._mapper.map_blocks(content, parent)

    def _commit_cost(self, cost: float, *, settled: bool = False) -> None:
        """Set the running cost and notify observers.

        Commits before notifying: a spend-cap callback reads ``self._cost_usd``
        synchronously and must see the value that just crossed the threshold.
        ``settled`` marks a result message settling the session on its own, so
        a spend-cap observer does not mistake it for a live cap crossing.
        """
        self._cost_usd = cost
        self._observer(UsageUpdateEvent(at=now_ns(), cost_usd=self._cost_usd, settled=settled))


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
