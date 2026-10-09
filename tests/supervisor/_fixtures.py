"""Shared builders and probes for the supervisor event tests.

These helpers are reused across the supervisor suites, so they live in one
module rather than being duplicated per test file. ``collecting_observer`` hands back an appending observer paired with
the list it fills; ``make_launch`` builds a fully-populated ``LaunchEvent`` from
overridable defaults; ``read_log_lines`` parses a JSONL log into dicts;
``NotJsonEncodable`` is a value ``json.dumps`` cannot encode.
``seed_session_log``, ``append_step``, ``driver_calls``,
``events_log_path``, and ``_supervise`` share the turn-loop test boilerplate;
``LockSwitch`` is a repository lock a test holds and releases between driver
steps; ``SupervisorClock`` is the supervisor's wall clock, moved by hand; ``_WrapDriver``
captures the abort event, observer, and session a supervised driver hands out;
``FollowUpWatch`` lets a test await the follow-ups a supervised run emits;
``WAIT_FINISHED_LINE`` is the line a reply closes on after a lock wait;
``_SENTINEL_SERVER`` and ``_SENTINEL_HOOKS`` are what the stubbed tools and
hooks factories hand back; ``HooksFactoryProbe`` counts its calls and
``ToolsFactoryProbe`` records the context of each.
``result_message``, ``system_message``, ``assistant``, ``tool_results``, and
``stream_event`` build the real ``claude-agent-sdk`` message dataclasses the
Claude driver consumes. ``start_claude_session``,
``run_interrupting_on_first_usage_update``, ``settled_outcome``,
``start_past_turns``, and ``run_outcome`` start a Claude session over a fake
client and await it, every wait bounded by ``SESSION_TIMEOUT_S``;
``end_and_settle`` ends a running session and awaits its outcome.
``wait_for_event_or_task`` waits on an event a background task should set,
failing instead of hanging when the task settles first.
"""

import asyncio
import contextlib
import json
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, NamedTuple, override
from unittest.mock import create_autospec

import pytest
from claude_agent_sdk import (
    AssistantMessage,
    ContentBlock,
    HookMatcher,
    MessageOrigin,
    ResultMessage,
    StreamEvent,
    SystemMessage,
    ToolResultBlock,
    UserMessage,
)
from claude_agent_sdk.types import HookEvent

from gymrat.clock import now_ms, now_ns
from gymrat.config import BenchlessConfig, Effort
from gymrat.session.paths import lockfile_path
from gymrat.session.records import SessionLogRecord
from gymrat.supervisor.claude import create_claude_driver
from gymrat.supervisor.driver import Driver, DriverSession, SessionOutcome, SessionPrompt
from gymrat.supervisor.events import (
    DirtyInfo,
    FollowUpEvent,
    LaunchEvent,
    SessionEvent,
    SessionObserver,
    TurnEndEvent,
    UsageUpdateEvent,
)
from gymrat.supervisor.hooks import HooksFactory
from gymrat.supervisor.supervise import SupervisedSession, SupervisionResult, supervise
from gymrat.supervisor.tools import ToolsFactory
from tests._config import benchless_config
from tests.session.records._fixtures import (
    SUPERVISED_SESSION_ID,
    append_records,
    session_record,
)
from tests.supervisor._mock_driver import ActionStep, EmitStep, _MockSession

_SESSION_ID = "sdk-session"
_MODEL = "claude-test"

#: How long a test waits on a session's outcome before failing instead of hanging.
SESSION_TIMEOUT_S = 30.0

#: How long a test waits for a follow-up before failing instead of hanging.
FOLLOW_UP_TIMEOUT_S = 5

#: The line a reply closes on when the agent's last command was still running.
WAIT_FINISHED_LINE = (
    "The command you left running has finished; its record, if any, is in the session log."
)


class NotJsonEncodable:
    """A value ``json.dumps`` cannot encode, with a deterministic string form."""

    @override
    def __str__(self) -> str:
        return "not-json-encodable"


class ObserverProbe(NamedTuple):
    """An observer paired with the list it appends every received event to."""

    events: list[SessionEvent]
    observer: SessionObserver


def collecting_observer() -> ObserverProbe:
    """Return an observer that records each event it receives, and its list."""
    events: list[SessionEvent] = []
    return ObserverProbe(events, events.append)


def make_launch(
    *,
    at: int = 1_000_000_000_000,
    head_sha: str = "abc123def",
    dirty: Literal[False] | DirtyInfo = False,
    max_minutes: float = 5,
    max_usd: float | None = None,
    model: str | None = None,
    effort: Effort | None = None,
    runbook_path: str = "/path/to/runbook.md",
    kickoff_summary: str = "test kickoff",
    session_id: str = SUPERVISED_SESSION_ID,
) -> LaunchEvent:
    """Build a ``LaunchEvent`` from shared defaults, overridden per keyword."""
    return LaunchEvent(
        at=at,
        schema_version=1,
        head_sha=head_sha,
        dirty=dirty,
        max_minutes=max_minutes,
        max_usd=max_usd,
        model=model,
        effort=effort,
        runbook_path=runbook_path,
        kickoff_summary=kickoff_summary,
        session_id=session_id,
    )


def read_log_lines(log_path: str | Path) -> list[dict[str, object]]:
    """Parse a JSONL log file into a list of decoded JSON objects."""
    text = Path(log_path).read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def make_prompt(
    *,
    kickoff: str = "do the thing",
    cwd: str = "/tmp/test",
    system_prompt_append: str = "follow the runbook",
    model: str | None = None,
    effort: Effort | None = None,
    command_timeout_ms: int = 60_000,
    max_budget_usd: float | None = None,
    traceparent: str | None = None,
) -> SessionPrompt:
    """Build a ``SessionPrompt`` from shared defaults, overridden per keyword."""
    return SessionPrompt(
        kickoff=kickoff,
        cwd=cwd,
        system_prompt_append=system_prompt_append,
        model=model,
        effort=effort,
        command_timeout_ms=command_timeout_ms,
        max_budget_usd=max_budget_usd,
        traceparent=traceparent,
    )


def noop_observer() -> SessionObserver:
    """Return an observer that discards every event it receives."""

    def _observer(_event: SessionEvent) -> None:
        return None

    return _observer


# ---------------------------------------------------------------------------
# driver double: interrupt emits TurnEndEvent
# ---------------------------------------------------------------------------


class DelegatingSession:
    """Let subclasses override one method while inheriting the rest."""

    def __init__(self, inner: DriverSession) -> None:
        self._inner = inner

    @property
    def outcome(self) -> Awaitable[SessionOutcome]:
        """Forward the inner session's outcome."""
        return self._inner.outcome

    async def interrupt(self) -> None:
        await self._inner.interrupt()

    async def send(self, text: str) -> None:
        await self._inner.send(text)

    async def end(self) -> None:
        await self._inner.end()


class _InterruptEmitsEndSession(DelegatingSession):
    """A session whose ``interrupt`` also emits a ``TurnEndEvent`` to the observer.

    Models a driver that, on ``interrupt()``, pushes one more agent
    ``TurnEndEvent`` before the session settles — exercising the cap guard in
    ``_handle_turn_end``.
    """

    def __init__(self, inner: DriverSession, observer: SessionObserver) -> None:
        super().__init__(inner)
        self._observer = observer

    @override
    async def interrupt(self) -> None:
        await self._inner.interrupt()
        self._observer(make_turn_end(cost_usd=0.0))


class SlowEndSession(DelegatingSession):
    """A session whose ``end`` settles only after a delay."""

    def __init__(self, inner: DriverSession, delay_ms: int) -> None:
        super().__init__(inner)
        self._delay_ms = delay_ms

    @override
    async def end(self) -> None:
        await asyncio.sleep(self._delay_ms / 1000)
        await self._inner.end()


class InterruptEmitsEndDriver:
    """A driver wrapper whose sessions emit a ``TurnEndEvent`` on ``interrupt``.

    Wraps a driver (typically from ``create_mock_driver``) and intercepts the
    observer from ``start``.  Each session's ``interrupt`` delegates to the
    inner session and then fires a ``TurnEndEvent(origin="agent")`` into the
    observer, simulating a driver that delivers a final turn boundary on
    interrupt.
    """

    def __init__(self, inner: Driver) -> None:
        self._inner = inner

    def start(
        self,
        prompt: SessionPrompt,
        observer: SessionObserver,
        abort: asyncio.Event,
    ) -> DriverSession:
        """Start a session that emits a ``TurnEndEvent`` on ``interrupt``."""
        inner_session = self._inner.start(prompt, observer, abort)
        return _InterruptEmitsEndSession(inner_session, observer)


# ---------------------------------------------------------------------------
# SDK message builders
# ---------------------------------------------------------------------------


def result_message(
    *,
    subtype: str = "success",
    is_error: bool = False,
    num_turns: int = 1,
    total_cost_usd: float | None = None,
    result: str | None = None,
    origin: MessageOrigin | None = None,
) -> ResultMessage:
    """Build an SDK ``ResultMessage`` with fixed bookkeeping fields."""
    return ResultMessage(
        subtype=subtype,
        duration_ms=0,
        duration_api_ms=0,
        is_error=is_error,
        num_turns=num_turns,
        session_id=_SESSION_ID,
        total_cost_usd=total_cost_usd,
        result=result,
        origin=origin,
    )


def system_message(
    *, subtype: str = "init", data: dict[str, object] | None = None
) -> SystemMessage:
    """Build an SDK ``SystemMessage`` carrying ``data`` (empty when omitted)."""
    return SystemMessage(subtype=subtype, data=data if data is not None else {})


def assistant(*blocks: ContentBlock, parent_tool_use_id: str | None = None) -> AssistantMessage:
    """Build an SDK ``AssistantMessage`` carrying the given content blocks."""
    return AssistantMessage(
        content=list(blocks), model=_MODEL, parent_tool_use_id=parent_tool_use_id
    )


def tool_results(*blocks: ToolResultBlock, parent_tool_use_id: str | None = None) -> UserMessage:
    """Build an SDK ``UserMessage`` carrying tool results, as the CLI delivers them."""
    return UserMessage(content=list(blocks), parent_tool_use_id=parent_tool_use_id)


def stream_event(event: dict[str, object], *, parent_tool_use_id: str | None = None) -> StreamEvent:
    """Build an SDK ``StreamEvent`` wrapping one raw API stream event."""
    return StreamEvent(
        uuid="event-uuid",
        session_id=_SESSION_ID,
        event=event,
        parent_tool_use_id=parent_tool_use_id,
    )


# ---------------------------------------------------------------------------
# fake streaming client
# ---------------------------------------------------------------------------


class FakeClient:
    """A stand-in for the SDK streaming client that replays scripted messages.

    ``receive_messages`` yields each supplied message after an
    ``asyncio.sleep(0)`` handshake so an observer-scheduled interrupt or an
    abort lands deterministically between messages.  After the scripted
    messages, the stream blocks until ``disconnect`` releases it, mirroring
    the real SDK whose ``receive_messages`` iterator never terminates.
    """

    def __init__(
        self,
        messages: Sequence[object],
        *,
        throw: Exception | None = None,
    ) -> None:
        self.messages = messages
        self.throw = throw
        self.options: dict[str, object] | None = None
        self.query_prompts: list[str] = []
        self.interrupt_called = False
        self.disconnect_count = 0
        self._released = asyncio.Event()

    async def connect(self) -> None:
        return None

    async def query(self, prompt: str) -> None:
        self.query_prompts.append(prompt)

    async def receive_messages(self) -> AsyncIterator[object]:
        for message in self.messages:
            await asyncio.sleep(0)
            yield message
        if self.throw is not None:
            raise self.throw
        await self._released.wait()

    async def interrupt(self) -> None:
        self.interrupt_called = True

    async def disconnect(self) -> None:
        self.disconnect_count += 1
        self._released.set()


class FiniteClient(FakeClient):
    """A client whose stream ends naturally instead of blocking after the script."""

    def __init__(self, messages: Sequence[object], *, throw: Exception | None = None) -> None:
        super().__init__(messages, throw=throw)
        # Released from the start, so the stream ends once the script is spent.
        self._released.set()


class FactoryProbe:
    """A client factory that records its call count and the options it saw."""

    def __init__(self, client: FakeClient) -> None:
        self._client = client
        self.calls = 0

    def __call__(self, options: Mapping[str, object]) -> FakeClient:
        self.calls += 1
        self._client.options = dict(options)
        return self._client


def make_turn_end(**overrides: Any) -> TurnEndEvent:
    """Build an agent ``TurnEndEvent`` from shared defaults, overridden per keyword.

    Args:
        **overrides: ``TurnEndEvent`` fields to set in place of the defaults.

    Returns:
        An agent turn end with no text, stamped now, costing one cent, with
        budget left, except where ``overrides`` says otherwise.
    """
    defaults: dict[str, Any] = {
        "at": now_ns(),
        "text": "",
        "cost_usd": 0.01,
        "origin": "agent",
        "budget_exhausted": False,
    }
    return TurnEndEvent(**(defaults | overrides))


def make_context(
    *,
    root: str,
    log_path: str = "/tmp/events.jsonl",
    lock_path: str | None = None,
    config: BenchlessConfig | None = None,
    deadline_ms: float | None = None,
    max_minutes: float = 10,
    max_usd: float | None = None,
) -> SupervisedSession:
    """Build a ``SupervisedSession`` from shared defaults, overridden per keyword.

    Args:
        root: The repository the session runs in.
        log_path: Where the supervisor event log is written.
        lock_path: The repository lock the session probes, or None for the
            lock gymrat keeps for ``root``.
        config: The settled config, or None for the benchless defaults.
        deadline_ms: The wall-clock deadline, or None for ``max_minutes`` from
            now, matching the computation ``_run_session`` performs.
        max_minutes: The wall-clock cap in minutes.
        max_usd: The spend cap in dollars, or None for no cap.

    Returns:
        The supervised session.
    """
    if deadline_ms is None:
        deadline_ms = now_ms() + max_minutes * 60_000
    return SupervisedSession(
        root=root,
        log_path=log_path,
        lock_path=lock_path if lock_path is not None else lockfile_path(root),
        config=config if config is not None else benchless_config(),
        deadline_ms=deadline_ms,
        max_minutes=max_minutes,
        max_usd=max_usd,
    )


# ---------------------------------------------------------------------------
# turn-loop test helpers
# ---------------------------------------------------------------------------


def follow_ups_with_action(events: list[SessionEvent], action: str) -> list[FollowUpEvent]:
    """Return every ``FollowUpEvent`` in ``events`` whose ``action`` matches."""
    return [e for e in events_of(events, FollowUpEvent) if e.action == action]


def seed_session_log(root: str) -> None:
    """Write a minimal session header so ``read_records`` / ``fold_session`` work."""
    append_records(root, session_record())


def append_step(root: str, *records: SessionLogRecord, delay_ms: int | None = None) -> ActionStep:
    """Build a driver step that appends ``records`` to the session log under ``root``.

    Args:
        root: The repository whose session log receives the records.
        *records: The records to append, in order.
        delay_ms: How long the step waits before appending, or None for no wait.

    Returns:
        The step, ready to place in a mock driver script.
    """

    async def append() -> None:
        append_records(root, *records)

    return ActionStep(action=append, delay_ms=delay_ms)


def sent_texts(session: _MockSession) -> list[str | None]:
    """Return the text of every ``send`` call the mock session recorded."""
    return [text for call_type, text in session.calls if call_type == "send"]


def driver_calls(session: _MockSession) -> list[str]:
    """Return every ``send`` and ``end`` call the mock session recorded, in order."""
    return [call for call, _ in session.calls if call in {"send", "end"}]


@dataclass(slots=True)
class LockSwitch:
    """A repository lock a test holds and releases between driver steps."""

    held: bool

    def is_held(self) -> bool:
        """Report whether the lock is held."""
        return self.held

    async def hold(self) -> None:
        """Take the lock."""
        self.held = True

    async def release(self) -> None:
        """Free the lock."""
        self.held = False

    def release_on_waiting(self, observer: SessionObserver) -> SessionObserver:
        """Wrap ``observer`` so the lock frees once the supervisor says it is waiting on it.

        Args:
            observer: Receives every event before the lock is checked.

        Returns:
            An observer that forwards each event, then frees the lock on a
            ``waiting`` follow-up.
        """

        def forward(event: SessionEvent) -> None:
            observer(event)
            if isinstance(event, FollowUpEvent) and event.action == "waiting":
                self.held = False

        return forward


def emit_turn_end(
    *, cost_usd: float = 0.01, origin: Literal["agent", "injected"] = "agent"
) -> EmitStep:
    """Build an ``EmitStep`` for a ``TurnEndEvent`` with the fields every caller shares."""
    return EmitStep(emit=make_turn_end(cost_usd=cost_usd, origin=origin))


#: The MCP server config a stubbed tools or hooks factory hands back, so a test
#: can find it again in the client options.
_SENTINEL_SERVER: dict[str, str] = {"type": "stdio", "command": "fake"}

#: The hook mapping ``HooksFactoryProbe`` hands back, so a test can find it again
#: in the client options.
_SENTINEL_HOOKS: dict[HookEvent, list[HookMatcher]] = {
    "PreToolUse": [HookMatcher(matcher="Bash", hooks=[])],
}


class HooksFactoryProbe:
    """A stub hooks factory that counts its calls and returns ``_SENTINEL_HOOKS``."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self) -> dict[HookEvent, list[HookMatcher]]:
        self.calls += 1
        return _SENTINEL_HOOKS


class ToolsFactoryProbe:
    """A stub tools factory that records each call's context and returns ``_SENTINEL_SERVER``."""

    def __init__(self) -> None:
        self.calls: list[tuple[asyncio.Event, Mapping[str, str]]] = []

    def __call__(self, abort: asyncio.Event, env: Mapping[str, str]) -> object:
        self.calls.append((abort, env))
        return _SENTINEL_SERVER


def _same_session(session: DriverSession, _abort: asyncio.Event) -> DriverSession:
    return session


class _WrapDriver:
    """Capture what the supervisor hands a driver, and wrap the session it starts.

    ``make_session`` receives the inner driver's session and the abort event,
    and returns the session the supervisor sees; by default the inner one.
    """

    def __init__(
        self,
        inner: Driver,
        make_session: Callable[[DriverSession, asyncio.Event], DriverSession] = _same_session,
    ) -> None:
        self._inner = inner
        self._make_session = make_session
        self.captured_abort: asyncio.Event | None = None
        self.captured_observer: SessionObserver | None = None
        self.session: DriverSession | None = None

    def start(
        self,
        prompt: SessionPrompt,
        observer: SessionObserver,
        abort: asyncio.Event,
    ) -> DriverSession:
        self.captured_abort = abort
        self.captured_observer = observer
        self.session = self._make_session(self._inner.start(prompt, observer, abort), abort)
        return self.session


class FollowUpWatch:
    """An observer that records every event and lets a test await follow-ups."""

    def __init__(self) -> None:
        self.events: list[SessionEvent] = []
        self._follow_up = asyncio.Event()

    def __call__(self, event: SessionEvent) -> None:
        self.events.append(event)
        if isinstance(event, FollowUpEvent):
            self._follow_up.set()

    def actions(self) -> list[str]:
        """The action of every follow-up received so far, in order."""
        return [e.action for e in events_of(self.events, FollowUpEvent)]

    async def until(self, predicate: Callable[[list[str]], bool]) -> None:
        """Wait until ``predicate`` holds for the follow-up actions received so far.

        Args:
            predicate: Tests the follow-up actions received so far.

        Raises:
            TimeoutError: When ``predicate`` still fails after
                ``FOLLOW_UP_TIMEOUT_S``.
        """
        async with asyncio.timeout(FOLLOW_UP_TIMEOUT_S):
            while not predicate(self.actions()):
                self._follow_up.clear()
                await self._follow_up.wait()


class SupervisorClock:
    """The wall clock the supervisor reads, standing still until a test moves it.

    Moving the clock only once a prerequisite is observed orders a wall-clock
    cap after that prerequisite, instead of racing a real-time deadline.

    Args:
        monkeypatch: Patches the supervisor's wall clock.
        start_ms: Where the clock stands until a test moves it.
        deadline_ms: The wall-clock deadline a test hands the supervisor, or
            None for one minute after ``start_ms``.
    """

    def __init__(
        self,
        monkeypatch: pytest.MonkeyPatch,
        start_ms: int,
        *,
        deadline_ms: int | None = None,
    ) -> None:
        self.now_ms = start_ms
        self.deadline_ms = start_ms + 60_000 if deadline_ms is None else deadline_ms
        monkeypatch.setattr(
            "gymrat.supervisor.supervise.now_ms",
            create_autospec(now_ms, side_effect=lambda: self.now_ms),
        )

    def jump_to(self, to_ms: int) -> None:
        """Move the clock to ``to_ms``."""
        self.now_ms = to_ms

    def jump_on(
        self,
        event_type: type[SessionEvent],
        observer: SessionObserver | None = None,
    ) -> SessionObserver:
        """Build an observer that moves the clock to the deadline when an ``event_type`` arrives.

        Args:
            event_type: The event that passes the deadline.
            observer: Receives every event first, or None to forward nothing.

        Returns:
            The forwarding observer, ready to hand to ``_supervise``.
        """

        def observe(event: SessionEvent) -> None:
            if observer is not None:
                observer(event)
            if isinstance(event, event_type):
                self.jump_to(self.deadline_ms)

        return observe

    def jump_step(self, to_ms: int, *, after: asyncio.Event | None = None) -> ActionStep:
        """Build a driver step that moves the clock to ``to_ms``.

        Args:
            to_ms: Where the clock lands.
            after: An event the step waits on before moving the clock, or None
                to move it at once.

        Returns:
            The step, ready to place in a mock driver script.

        Raises:
            TimeoutError: When ``after`` is still unset after
                ``FOLLOW_UP_TIMEOUT_S``.
        """

        async def jump() -> None:
            if after is not None:
                async with asyncio.timeout(FOLLOW_UP_TIMEOUT_S):
                    await after.wait()
            self.jump_to(to_ms)

        return ActionStep(action=jump)


def events_log_path(root: str) -> Path:
    """The supervisor event log ``_supervise`` writes for the repository at ``root``."""
    return Path(root).parent / "events.jsonl"


def lock_file_path(root: str) -> Path:
    """The repository lock file ``_supervise`` probes for the repository at ``root``."""
    return Path(root).parent / "lockfile"


async def _supervise(
    root: str,
    driver: Driver,
    *,
    config: BenchlessConfig | None = None,
    max_minutes: float = 10,
    max_usd: float | None = None,
    deadline_ms: float | None = None,
    launch: LaunchEvent | None = None,
    observer: SessionObserver | None = None,
    is_lock_held: Callable[[], bool] | None = lambda: False,
    grace_ms: int = 30_000,
    settle_window_ms: int = 0,
    wall_clock_poll_ms: int = 1,
) -> SupervisionResult:
    """Supervise ``driver`` over the repository at ``root`` with fast polls and no settle window.

    The event log and the repository lock file sit beside ``root``; the
    session's caps come from the keywords, and the launch event carries the
    same wall-clock and spend caps unless ``launch`` says otherwise.

    Args:
        root: The repository the session runs in.
        driver: The agent driver to supervise.
        config: The settled config, or None for the benchless defaults.
        max_minutes: The wall-clock cap in minutes.
        max_usd: The spend cap in dollars, or None for no cap.
        deadline_ms: The wall-clock deadline, or None for ``max_minutes`` from now.
        launch: The launch event to emit, or None for one carrying ``max_minutes``
            and ``max_usd``.
        observer: Receives every session event.
        is_lock_held: Answers whether the repository lock is held; None probes
            the real lock file instead.
        grace_ms: How long a stop request waits before forcing cancellation.
        settle_window_ms: Idle time after a turn ends before the log is read.
        wall_clock_poll_ms: How often the caps are checked.

    Returns:
        How the supervised session ended.
    """
    return await supervise(
        driver,
        make_prompt(cwd=root),
        context=make_context(
            root=root,
            log_path=str(events_log_path(root)),
            lock_path=str(lock_file_path(root)),
            config=config,
            deadline_ms=deadline_ms,
            max_minutes=max_minutes,
            max_usd=max_usd,
        ),
        launch=make_launch(max_minutes=max_minutes, max_usd=max_usd) if launch is None else launch,
        observer=observer,
        grace_ms=grace_ms,
        wall_clock_poll_ms=wall_clock_poll_ms,
        settle_window_ms=settle_window_ms,
        lock_poll_ms=1,
        is_lock_held=is_lock_held,
    )


async def settled_outcome(session: DriverSession) -> SessionOutcome:
    """Await ``session``'s outcome, failing after ``SESSION_TIMEOUT_S`` instead of hanging."""
    return await asyncio.wait_for(session.outcome, SESSION_TIMEOUT_S)


async def run_session(
    driver: Driver,
    observer: SessionObserver,
    prompt: SessionPrompt | None = None,
    abort: asyncio.Event | None = None,
) -> SessionOutcome:
    """Start a session on ``driver`` and await its settled outcome.

    Args:
        driver: The driver that starts the session.
        observer: Receives every session event.
        prompt: The prompt to start with, or None for ``make_prompt()``.
        abort: The session's abort event, or None for a fresh, never-set one.

    Returns:
        The settled outcome.
    """
    session = driver.start(prompt or make_prompt(), observer, abort or asyncio.Event())
    return await settled_outcome(session)


def start_claude_session(
    client: FakeClient,
    observer: SessionObserver | None = None,
    *,
    prompt: SessionPrompt | None = None,
    abort: asyncio.Event | None = None,
    hooks: HooksFactory | None = None,
    tools: ToolsFactory | None = None,
) -> DriverSession:
    """Start a Claude driver session that streams from ``client``.

    Args:
        client: The fake client the session streams from.
        observer: Receives every session event; None discards them.
        prompt: The prompt to start with; ``make_prompt()`` when omitted.
        abort: The session's abort event; a fresh, never-set one when omitted.
        hooks: The hooks factory the driver passes on, if any.
        tools: The tools factory the driver passes on, if any.

    Returns:
        The running session.
    """
    driver = create_claude_driver(client_factory=FactoryProbe(client), hooks=hooks, tools=tools)
    return driver.start(
        prompt or make_prompt(), observer or noop_observer(), abort or asyncio.Event()
    )


async def end_and_settle(session: DriverSession) -> SessionOutcome:
    """End ``session`` and await its settled outcome.

    Args:
        session: The running session to end.

    Returns:
        The settled outcome.
    """
    await session.end()
    return await settled_outcome(session)


async def run_interrupting_on_first_usage_update(client: FakeClient) -> SessionOutcome:
    """Run a Claude session over ``client`` that interrupts itself on its first usage update.

    Usage updates come from result messages, which leave the session idle
    between turns, so the soft stop lands when the stream delivers its next
    message. The interrupt is awaited once the outcome settles, so an
    exception it raises fails the test instead of being lost.

    Args:
        client: The fake client the session streams from; it must carry a
            message after the first result.

    Returns:
        The settled outcome.
    """
    sessions: list[DriverSession] = []
    interrupts: list[asyncio.Task[None]] = []

    def interrupting(event: SessionEvent) -> None:
        if isinstance(event, UsageUpdateEvent) and not interrupts:
            interrupts.append(asyncio.ensure_future(sessions[0].interrupt()))

    sessions.append(start_claude_session(client, interrupting))
    outcome = await settled_outcome(sessions[0])
    await asyncio.gather(*interrupts)
    return outcome


async def run_outcome(
    client: FakeClient,
    observer: SessionObserver | None = None,
    *,
    prompt: SessionPrompt | None = None,
    abort: asyncio.Event | None = None,
    hooks: HooksFactory | None = None,
    tools: ToolsFactory | None = None,
) -> SessionOutcome:
    """Drive ``client`` through a Claude session and return its settled outcome.

    Args:
        client: The fake client the session streams from.
        observer: Receives every session event; None discards them.
        prompt: The prompt to start with; ``make_prompt()`` when omitted.
        abort: The session's abort event; a fresh, never-set one when omitted.
        hooks: The hooks factory the driver passes on, if any.
        tools: The tools factory the driver passes on, if any.

    Returns:
        The settled outcome.
    """
    session = start_claude_session(
        client, observer, prompt=prompt, abort=abort, hooks=hooks, tools=tools
    )
    return await settled_outcome(session)


async def start_past_turns(
    client: FakeClient, turns: int, prompt: SessionPrompt | None = None
) -> tuple[DriverSession, list[SessionEvent]]:
    """Start a Claude session over ``client`` and wait until it has closed ``turns`` turns.

    Waiting on the turn ends themselves, not on a count of event-loop yields,
    keeps a later ``end()`` or ``send()`` from landing before the stream has
    drawn every result.

    Args:
        client: The fake client the session streams from.
        turns: How many ``TurnEndEvent`` the session must emit before returning.
        prompt: The prompt to start with; ``make_prompt()`` when omitted.

    Returns:
        The running session and the list its observer keeps appending events to.
    """
    events: list[SessionEvent] = []
    turns_closed = asyncio.Event()

    def observer(event: SessionEvent) -> None:
        events.append(event)
        if len(events_of(events, TurnEndEvent)) == turns:
            turns_closed.set()

    session = start_claude_session(client, observer, prompt=prompt)
    await wait_for_event_or_task(turns_closed, session.outcome)
    return session, events


async def run_with_messages(messages: Sequence[object]) -> list[SessionEvent]:
    """Drive a session over scripted ``messages`` and return the events it emitted.

    The stream terminates naturally after the scripted messages so mapping
    tests exercise message-to-event logic without a result message or an
    explicit ``end()`` call.  Tests that need a specific outcome or the
    turn-end protocol use ``run_session`` directly.

    Args:
        messages: The scripted SDK messages the fake client replays.

    Returns:
        The events the observer received, in emission order.
    """
    driver = create_claude_driver(client_factory=FactoryProbe(FiniteClient(list(messages))))
    probe = collecting_observer()
    await run_session(driver, probe.observer)
    return probe.events


def events_of[T: SessionEvent](events: Sequence[SessionEvent], event_type: type[T]) -> list[T]:
    """The events of ``event_type`` in ``events``, in emission order."""
    return [e for e in events if isinstance(e, event_type)]


async def wait_for_event_or_task(event: asyncio.Event, awaitable: Awaitable[object]) -> None:
    """Wait until ``event`` is set, failing fast when ``awaitable`` settles first.

    A task that settles before setting the event never will, so an unbounded
    wait would hang the suite. Its exception is re-raised; a task that returned
    instead fails the test naming what it returned, such as an error outcome.

    Args:
        event: The event the test waits on.
        awaitable: The task, future, or coroutine expected to set ``event``. A
            task or future is never cancelled or consumed, so the test can still
            await it afterwards.
    """
    task = asyncio.ensure_future(awaitable)
    waiter = asyncio.ensure_future(event.wait())
    await asyncio.wait({waiter, task}, return_when=asyncio.FIRST_COMPLETED)
    if waiter.done():
        return
    waiter.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await waiter
    pytest.fail(f"the task settled before the event was set: {task.result()!r}")
