"""Shared builders, doubles and runners for the supervisor test suites.

The supervisor suites — the orchestrator, the Claude driver, the event log and
the command line that wraps them — build the same events, prompts and contexts,
script the same mock and fake drivers, and run the supervisor the same way.
Those pieces live here once instead of being copied into each test file.
"""

import asyncio
import contextlib
import json
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from operator import methodcaller
from pathlib import Path
from typing import Any, Literal, NamedTuple, override
from unittest.mock import create_autospec

import pytest
from claude_agent_sdk import (
    AssistantMessage,
    ContentBlock,
    MessageOrigin,
    ResultMessage,
)

from gymrat.clock import now_ms, now_ns
from gymrat.config import BenchlessConfig, Effort
from gymrat.session.budget import Budget
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
    TextDeltaEvent,
    ToolEndEvent,
    ToolStartEvent,
    TurnEndEvent,
    UsageUpdateEvent,
)
from gymrat.supervisor.hooks import HooksFactory
from gymrat.supervisor.supervise import SupervisedSession, SupervisionResult, supervise
from gymrat.supervisor.tools import ToolsFactory
from gymrat.utils import NS_PER_MS
from tests._config import benchless_config
from tests.session.records._fixtures import (
    SUPERVISED_SESSION_ID,
    append_records,
)

_SESSION_ID = "sdk-session"
_MODEL = "claude-test"

#: How long a test waits on a session's outcome before failing instead of hanging.
SESSION_TIMEOUT_S = 30.0

#: How long a test waits for a follow-up before failing instead of hanging.
FOLLOW_UP_TIMEOUT_S = 5

#: A step delay long enough that only an end or a cap, never the script, cuts it short.
_BLOCKED_MS = 60_000

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


def raising_observer(message: str, *, on: type[SessionEvent] | None = None) -> SessionObserver:
    """Return an observer that raises ``RuntimeError(message)`` on the events it is handed.

    Args:
        message: The error message, for the test to match the warning or report against.
        on: The one event type the observer fails on; ``None`` fails on every event.

    Returns:
        The failing observer.
    """

    def _observer(event: SessionEvent) -> None:
        if on is None or isinstance(event, on):
            raise RuntimeError(message)

    return _observer


# ---------------------------------------------------------------------------
# scripted mock driver
# ---------------------------------------------------------------------------

# ``create_mock_driver`` builds a ``Driver`` whose ``start`` runs a script of
# steps in order on the running event loop, without a real agent backend. A
# step emits an event, awaits an async action, reports a cost, or simulates a
# turn boundary. Each step's optional ``delay_ms`` races a timer against the
# abort — the driver's own ``interrupt`` or the external abort event — so a
# delayed step yields the moment the session is interrupted or aborted.


@dataclass(frozen=True, slots=True, kw_only=True)
class EmitStep:
    """Delivers ``emit`` to the observer, optionally after ``delay_ms``."""

    emit: SessionEvent
    delay_ms: int | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class ActionStep:
    """Awaits ``action``, optionally after ``delay_ms``."""

    action: Callable[[], Awaitable[None]]
    delay_ms: int | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class CostStep:
    """Sets the running cost to ``cost_usd`` and emits a usage update."""

    cost_usd: float
    delay_ms: int | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class TurnEndStep:
    """Simulates a turn boundary.

    Emits a ``TurnEndEvent`` and blocks until the supervisor calls ``send``
    or ``end``, or the session is interrupted.
    """

    text: str = ""
    cost_usd: float | None = None
    origin: Literal["agent", "injected"] = "agent"
    budget_exhausted: bool = False
    delay_ms: int | None = None


MockStep = EmitStep | ActionStep | CostStep | TurnEndStep
"""A single step in a mock driver script."""


class _MockSession:
    """Runs a mock script as a task; ``outcome`` settles when the script returns."""

    def __init__(
        self,
        steps: Sequence[MockStep],
        observer: SessionObserver,
        external_abort: asyncio.Event | None,
    ) -> None:
        self._observer = observer
        self._external = external_abort
        self._abort = asyncio.Event()
        self._cost_usd = 0.0
        self._turn_gate = asyncio.Event()
        self._end_requested = False
        self._settled = False
        self.calls: list[tuple[str, str | None]] = []
        self._script: asyncio.Task[SessionOutcome] = asyncio.ensure_future(self._run(steps))

    @property
    def outcome(self) -> Awaitable[SessionOutcome]:
        return self._script

    async def interrupt(self) -> None:
        if not self._settled:
            self.calls.append(("interrupt", None))
        self._abort.set()
        self._turn_gate.set()

    async def send(self, text: str) -> None:
        if self._settled:
            return
        self.calls.append(("send", text))
        self._turn_gate.set()

    async def end(self) -> None:
        if self._settled:
            return
        self.calls.append(("end", None))
        self._end_requested = True
        self._turn_gate.set()

    def _aborted(self) -> bool:
        return self._abort.is_set() or (self._external is not None and self._external.is_set())

    def _ended(self) -> bool:
        return self._end_requested or self._aborted()

    def _interrupted(self) -> SessionOutcome:
        return SessionOutcome(reason="interrupted", cost_usd=self._cost_usd)

    async def _delay(self, ms: int) -> None:
        """Wait up to ``ms`` milliseconds, returning early when the abort fires."""
        if self._aborted():
            return
        waiters = [asyncio.ensure_future(self._abort.wait())]
        if self._external is not None:
            waiters.append(asyncio.ensure_future(self._external.wait()))
        try:
            await asyncio.wait(waiters, timeout=ms / 1000, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for waiter in waiters:
                waiter.cancel()

    async def _execute(self, step: MockStep) -> None:
        match step:
            case EmitStep():
                if not self._aborted():
                    self._observer(step.emit)
            case ActionStep():
                await step.action()
            case CostStep():
                if self._aborted():
                    return
                self._cost_usd = step.cost_usd
                self._observer(UsageUpdateEvent(at=now_ns(), cost_usd=step.cost_usd))
            case TurnEndStep():
                if self._aborted():
                    return
                if step.cost_usd is not None:
                    self._cost_usd = step.cost_usd
                self._observer(
                    TurnEndEvent(
                        at=now_ns(),
                        text=step.text,
                        cost_usd=self._cost_usd,
                        origin=step.origin,
                        budget_exhausted=step.budget_exhausted,
                    )
                )
                self._turn_gate.clear()
                await self._turn_gate.wait()

    async def _run(self, steps: Sequence[MockStep]) -> SessionOutcome:
        for step in steps:
            # Yield between steps so an interrupt scheduled by the prior step's
            # observer is applied before the successor runs. The supervisor fires
            # ``interrupt`` as a task, so its abort lands on the next loop turn.
            await asyncio.sleep(0)

            if self._ended():
                break

            if step.delay_ms is not None and step.delay_ms > 0:
                await self._delay(step.delay_ms)

            if self._ended():
                break

            try:
                await self._execute(step)
            except Exception as error:  # noqa: BLE001 - the mock's contract turns any action failure into an error outcome
                self._settled = True
                return SessionOutcome(reason="error", cost_usd=self._cost_usd, message=str(error))

            if self._aborted():
                self._settled = True
                return self._interrupted()

        self._settled = True
        if self._aborted():
            return self._interrupted()
        return SessionOutcome(reason="completed", cost_usd=self._cost_usd)


class _MockDriver:
    def __init__(self, steps: Sequence[MockStep]) -> None:
        self._steps = steps
        self.sessions: list[_MockSession] = []

    def start(
        self,
        prompt: SessionPrompt,
        observer: SessionObserver,
        abort: asyncio.Event | None = None,
    ) -> DriverSession:
        session = _MockSession(self._steps, observer, abort)
        self.sessions.append(session)
        return session


def create_mock_driver(steps: Sequence[MockStep]) -> _MockDriver:
    """Return a driver that runs ``steps`` in order on each ``start``.

    Args:
        steps: The scripted steps every started session plays in order.

    Returns:
        A driver satisfying the :class:`Driver` protocol, whose ``.sessions``
        list holds each started session for assertions on ``send`` and ``end``
        calls.
    """
    return _MockDriver(tuple(steps))


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


def assistant(*blocks: ContentBlock, parent_tool_use_id: str | None = None) -> AssistantMessage:
    """Build an SDK ``AssistantMessage`` carrying the given content blocks."""
    return AssistantMessage(
        content=list(blocks), model=_MODEL, parent_tool_use_id=parent_tool_use_id
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
    the real SDK whose ``receive_messages`` iterator never terminates; with
    ``finite``, the stream is released from the start and ends once the script
    is spent.  With ``fail_follow_up``, every ``query`` after the kickoff raises
    ``RuntimeError("connection lost")``.
    """

    def __init__(
        self,
        messages: Sequence[object],
        *,
        throw: Exception | None = None,
        fail_follow_up: bool = False,
        finite: bool = False,
    ) -> None:
        self.messages = messages
        self.throw = throw
        self.fail_follow_up = fail_follow_up
        self.options: dict[str, object] | None = None
        self.query_prompts: list[str] = []
        self.interrupt_called = False
        self.disconnect_count = 0
        self._released = asyncio.Event()
        if finite:
            self._released.set()

    async def connect(self) -> None:
        return None

    async def query(self, prompt: str) -> None:
        if self.fail_follow_up and self.query_prompts:
            message = "connection lost"
            raise RuntimeError(message)
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
        super().__init__(messages, throw=throw, finite=True)


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


#: Default timestamp of a tool start, in milliseconds; a tool end's default
#: duration is measured from it, so the two stay in sync.
TOOL_START_MS = 2000


def tool_start_event(
    tool_name: str,
    tool_use_id: str,
    at_ms: int = TOOL_START_MS,
    *,
    input_summary: str = "...",
    parent_tool_use_id: str | None = None,
) -> ToolStartEvent:
    """A ``ToolStartEvent`` for *tool_name* stamped at *at_ms* milliseconds."""
    return ToolStartEvent(
        at=at_ms * NS_PER_MS,
        tool_use_id=tool_use_id,
        tool_name=tool_name,
        input={},
        input_summary=input_summary,
        parent_tool_use_id=parent_tool_use_id,
    )


def tool_end_event(
    tool_name: str,
    tool_use_id: str,
    at_ms: int = 3000,
    *,
    result: str = "ok",
    started_at_ms: int = TOOL_START_MS,
    parent_tool_use_id: str | None = None,
) -> ToolEndEvent:
    """A ``ToolEndEvent`` whose duration is measured from *started_at_ms*."""
    return ToolEndEvent(
        at=at_ms * NS_PER_MS,
        tool_use_id=tool_use_id,
        tool_name=tool_name,
        duration_ms=at_ms - started_at_ms,
        result=result,
        result_summary="ok",
        parent_tool_use_id=parent_tool_use_id,
    )


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
        budget=Budget(max_minutes=max_minutes, deadline_ms=deadline_ms),
        max_usd=max_usd,
    )


# ---------------------------------------------------------------------------
# turn-loop test helpers
# ---------------------------------------------------------------------------


def follow_ups_with_action(events: list[SessionEvent], action: str) -> list[FollowUpEvent]:
    """Return every ``FollowUpEvent`` in ``events`` whose ``action`` matches."""
    return [e for e in events_of(events, FollowUpEvent) if e.action == action]


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


def blocked_step() -> EmitStep:
    """Build a script tail that only an end or a cap can cut short."""
    return EmitStep(emit=TextDeltaEvent(at=now_ns(), chunk="late"), delay_ms=_BLOCKED_MS)


def _same_session(session: DriverSession, _abort: asyncio.Event) -> DriverSession:
    return session


class WrapDriver:
    """Capture what the supervisor hands a driver, and wrap the session it starts.

    ``make_session`` receives the inner driver's session and the abort event,
    and returns the session the supervisor sees; by default the inner one.
    ``abort`` and ``observer`` fail the test when the driver never started.
    """

    def __init__(
        self,
        inner: Driver,
        make_session: Callable[[DriverSession, asyncio.Event], DriverSession] = _same_session,
    ) -> None:
        self._inner = inner
        self._make_session = make_session
        self._abort: asyncio.Event | None = None
        self._observer: SessionObserver | None = None
        self.session: DriverSession | None = None

    @property
    def abort(self) -> asyncio.Event:
        """The abort event the supervisor handed the driver."""
        if self._abort is None:
            pytest.fail("the driver never started, so no abort event was captured")
        return self._abort

    @property
    def observer(self) -> SessionObserver:
        """The observer the supervisor handed the driver."""
        if self._observer is None:
            pytest.fail("the driver never started, so no observer was captured")
        return self._observer

    def start(
        self,
        prompt: SessionPrompt,
        observer: SessionObserver,
        abort: asyncio.Event,
    ) -> DriverSession:
        self._abort = abort
        self._observer = observer
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
            The forwarding observer, ready to hand to ``run_supervised``.
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
    """The supervisor event log ``run_supervised`` writes for the repository at ``root``."""
    return Path(root).parent / "events.jsonl"


def event_log_markers(root: str) -> list[str]:
    """Label each event ``run_supervised`` logged for ``root``, in log order.

    A ``tool_end`` line becomes ``tool_end:<id>``, a ``follow_up`` line becomes
    ``follow_up:<action>:<reason>``, and every other line becomes its type.

    Args:
        root: The repository root ``run_supervised`` ran against.

    Returns:
        One marker per logged event.
    """
    markers: list[str] = []
    for line in read_log_lines(events_log_path(root)):
        match line["type"]:
            case "tool_end":
                markers.append(f"tool_end:{line['tool_use_id']}")
            case "follow_up":
                markers.append(f"follow_up:{line['action']}:{line.get('reason')}")
            case other:
                markers.append(str(other))
    return markers


def lock_file_path(root: str) -> Path:
    """The repository lock file ``run_supervised`` probes for the repository at ``root``."""
    return Path(root).parent / "lockfile"


async def run_supervised(
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


def abort_on_first_usage_update(abort: asyncio.Event) -> SessionObserver:
    """Build an observer that sets ``abort`` on the first usage update it receives.

    Args:
        abort: The session's abort event.

    Returns:
        The observer to start the session with.
    """

    def observer(event: SessionEvent) -> None:
        if isinstance(event, UsageUpdateEvent):
            abort.set()

    return observer


async def run_acting_on_first(
    client: FakeClient,
    event_type: type[SessionEvent],
    action: Callable[[DriverSession], Awaitable[None]],
    *,
    abort: asyncio.Event | None = None,
) -> SessionOutcome:
    """Run a Claude session over ``client`` that acts on itself at its first ``event_type`` event.

    ``action`` is called inside the observer, so anything it does before
    returning happens before the stream moves on; what it returns is scheduled
    as a task. That task is awaited once the outcome settles, so an exception it
    raises fails the test instead of being lost.

    Args:
        client: The fake client the session streams from.
        event_type: The event whose first occurrence triggers ``action``.
        action: Receives the running session and returns the stop to await.
        abort: The session's abort event; a fresh, never-set one when omitted.

    Returns:
        The settled outcome.
    """
    sessions: list[DriverSession] = []
    actions: list[asyncio.Future[None]] = []

    def acting(event: SessionEvent) -> None:
        if isinstance(event, event_type) and not actions:
            actions.append(asyncio.ensure_future(action(sessions[0])))

    sessions.append(start_claude_session(client, acting, abort=abort))
    outcome = await settled_outcome(sessions[0])
    await asyncio.gather(*actions)
    return outcome


async def run_interrupting_on_first_usage_update(client: FakeClient) -> SessionOutcome:
    """Run a Claude session over ``client`` that interrupts itself on its first usage update.

    Usage updates come from result messages, which leave the session idle
    between turns, so the soft stop lands when the stream delivers its next
    message.

    Args:
        client: The fake client the session streams from; it must carry a
            message after the first result.

    Returns:
        The settled outcome.
    """
    return await run_acting_on_first(client, UsageUpdateEvent, methodcaller("interrupt"))


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
