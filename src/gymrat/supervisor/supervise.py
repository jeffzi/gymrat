"""The supervisor orchestrator: run a driver session under time and spend caps.

:func:`supervise` starts a driver session, tees every event to a JSONL log and
an optional observer, and enforces a wall-clock cap plus an optional spend cap.
On each ``TurnEndEvent`` the supervisor enters an idle state, waits for a settle
window to elapse, reads and folds the session log, probes the repository lock,
and delegates to :func:`~gymrat.supervisor.turns.classify` for the next action.
The returned :class:`SupervisionResult` reports the session outcome and how it
ended.
"""

import asyncio
import time
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from gymrat.errors import GymratError
from gymrat.session.clock import now_ms, now_ns
from gymrat.session.lock import is_held
from gymrat.session.paths import session_jsonl_path
from gymrat.session.store import fold_session, read_records
from gymrat.supervisor.context import SupervisedSession
from gymrat.supervisor.driver import Driver, DriverSession, SessionOutcome, SessionPrompt
from gymrat.supervisor.event_log import create_event_log_writer
from gymrat.supervisor.events import (
    CapEvent,
    FollowUpEvent,
    LaunchEvent,
    SessionEvent,
    SessionObserver,
    TurnEndEvent,
    UsageUpdateEvent,
    combine_observers,
)
from gymrat.supervisor.turns import (
    Decision,
    End,
    GuardState,
    Reply,
    WaitForLock,
    classify,
    outcome_record_count,
)
from gymrat.warn import warn_to_stderr

WALL_CLOCK_POLL_MS = 1000
"""Default interval (in milliseconds) for polling wall-clock time against the
deadline. Tests override this to avoid real waits."""

SETTLE_WINDOW_MS = 800
"""Default settle window (in milliseconds) after a turn end before reading the
session log and classifying."""

LOCK_POLL_MS = 5000
"""Default interval (in milliseconds) for polling the repository lock while
waiting for another process to release it."""

CapType = Literal["wall-clock", "spend-cap"]
EndedBy = Literal["session", "wall-clock", "spend-cap", "guard"]

_IN_FLIGHT_EXCLUSION = frozenset({
    "usage_update",
    "cap",
    "launch",
    "follow_up",
    "turn_end",
    "compaction",
})
"""Event types that do NOT cancel a pending settle window or lock poll."""


@dataclass(frozen=True, slots=True, kw_only=True)
class SupervisionResult:
    """How a supervised session ended.

    - ``outcome``: how the session settled (completed, interrupted, or error).
    - ``ended_by``: whether the session ended on its own or was stopped by a cap.
    - ``end_reason``: for ``guard``, the guard reason; for a cap, the cap name;
      for ``session``, ``None`` unless a log-read error set it.
    - ``duration_ms``: wall-clock duration from start to settlement.
    - ``cost_usd``: the final cost reported by the session.
    """

    outcome: SessionOutcome
    ended_by: EndedBy
    duration_ms: int
    cost_usd: float
    end_reason: str | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class _SuperviseConfig:
    """The full input surface of one :func:`supervise` call."""

    driver: Driver
    prompt: SessionPrompt
    context: SupervisedSession
    log_path: str | Path
    launch: LaunchEvent
    max_usd: float | None
    observer: SessionObserver | None
    grace_ms: int
    deadline_ms: float
    wall_clock_poll_ms: int
    settle_window_ms: int
    lock_poll_ms: int
    is_lock_held: Callable[[], bool]


def _fire_and_report_interrupt(session: DriverSession) -> asyncio.Task[None] | None:
    """Interrupt the session, isolating any failure so grace setup continues.

    ``interrupt`` may throw synchronously or its coroutine may reject; either way
    the fallback recovery still runs, so the failure is warned, never raised.
    Returns the interrupt task so the caller can cancel it on teardown.

    Args:
        session: The driver session to interrupt.

    Returns:
        The interrupt task, or ``None`` when the interrupt could not be started.
    """
    try:
        pending = session.interrupt()
    except Exception as error:  # noqa: BLE001 - interrupt failure must not abort grace setup
        warn_to_stderr(f"session interrupt failed: {error!s}")
        return None

    task = asyncio.create_task(pending)

    def _report(finished: asyncio.Task[None]) -> None:
        if finished.cancelled():
            return
        error = finished.exception()
        if error is not None:
            warn_to_stderr(f"session interrupt failed: {error!s}")

    task.add_done_callback(_report)
    return task


def _warn_unhandled(finished: asyncio.Task[None]) -> None:
    """Done-callback that surfaces exceptions from fire-and-forget tasks."""
    if finished.cancelled():
        return
    error = finished.exception()
    if error is not None:
        warn_to_stderr(f"background task failed: {error!s}")


class _Supervision:
    """Runs one supervised session, holding the mutable cap/timer state."""

    def __init__(self, config: _SuperviseConfig) -> None:
        self._config = config
        self._abort_event = asyncio.Event()
        log_writer = create_event_log_writer(config.log_path)

        observers: list[SessionObserver] = [self._event_router, log_writer]
        if config.observer is not None:
            observers.append(config.observer)
        self._combined = combine_observers(*observers)

        self._ended_by: EndedBy = "session"
        self._end_reason: str | None = None
        self._cap_fired = False
        self._wall_task: asyncio.Task[None] | None = None
        self._grace_task: asyncio.Task[None] | None = None
        self._interrupt_task: asyncio.Task[None] | None = None
        self._session: DriverSession | None = None
        self._pending_cap: CapType | None = None

        self._guards = GuardState(
            initial_record_count=self._initial_record_count(),
        )
        self._reply_outstanding = False
        self._tasks: dict[Literal["settle", "lock_poll"], asyncio.Task[None]] = {}
        self._in_flight = False
        self._last_cost_usd = 0.0
        self._background_tasks: set[asyncio.Task[None]] = set()

    def _spawn(self, target: Coroutine[object, object, None]) -> None:
        """Create a background task and prevent it from being garbage-collected."""
        task = asyncio.create_task(target)
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
        task.add_done_callback(_warn_unhandled)

    def _initial_record_count(self) -> int:
        try:
            records = read_records(
                session_jsonl_path(self._config.context.root),
            )
        except GymratError:
            return 0
        return outcome_record_count(records)

    def _event_router(self, event: SessionEvent) -> None:
        """Route events for in-flight detection and cost tracking."""
        if isinstance(event, UsageUpdateEvent):
            self._last_cost_usd = event.cost_usd
            return

        if isinstance(event, TurnEndEvent):
            self._handle_turn_end(event)
            return

        if event.type not in _IN_FLIGHT_EXCLUSION:
            self._enter_in_flight()

    def _enter_in_flight(self) -> None:
        self._in_flight = True
        self._cancel("settle")
        self._cancel("lock_poll")

    def _cancel(self, slot: Literal["settle", "lock_poll"]) -> None:
        task = self._tasks.pop(slot, None)
        if task is not None:
            task.cancel()

    def _schedule(
        self,
        slot: Literal["settle", "lock_poll"],
        routine: Coroutine[object, object, None],
    ) -> None:
        task = asyncio.create_task(routine)

        def _clear_on_done(finished: asyncio.Task[None]) -> None:
            if self._tasks.get(slot) is finished:
                del self._tasks[slot]

        task.add_done_callback(_clear_on_done)
        task.add_done_callback(_warn_unhandled)
        self._tasks[slot] = task

    def _handle_turn_end(self, event: TurnEndEvent) -> None:
        if self._reply_outstanding:
            if event.origin == "agent":
                self._reply_outstanding = False
                self._schedule_settle(event)
            return

        self._in_flight = False
        self._schedule_settle(event)

    def _schedule_settle(self, event: TurnEndEvent) -> None:
        self._cancel("settle")
        self._schedule("settle", self._run_settle(event))

    async def _run_settle(self, turn: TurnEndEvent, *, after_wait: bool = False) -> None:
        if self._config.settle_window_ms > 0:
            await asyncio.sleep(self._config.settle_window_ms / 1000)

        try:
            records = read_records(
                session_jsonl_path(self._config.context.root),
            )
            state = fold_session(records)
        except GymratError as error:
            self._handle_log_error(str(error))
            return

        lock_held = self._config.is_lock_held()

        decision = classify(
            config=self._config.context.config,
            state=state,
            records=records,
            guards=self._guards,
            lock_held=lock_held,
            turn=turn,
            max_usd=self._config.max_usd,
            deadline_ms=self._config.deadline_ms,
            max_minutes=self._config.context.max_minutes,
            now_ms=now_ms(),
            after_wait=after_wait,
        )

        self._execute_decision(decision, turn)

    def _end_session(
        self,
        reason: str,
        *,
        ended_by: EndedBy,
        end_reason: str | None = None,
    ) -> None:
        if self._cap_fired:
            return
        self._combined(
            FollowUpEvent(at=now_ns(), action="ended", reason=reason),
        )
        self._cap_fired = True
        self._ended_by = ended_by
        self._end_reason = end_reason
        if self._session is not None:
            self._spawn(self._session.end())

    def _handle_log_error(self, message: str) -> None:
        self._end_session(message, ended_by="session", end_reason=message)

    def _execute_decision(self, decision: Decision, turn: TurnEndEvent) -> None:
        match decision:
            case End(reason="finished"):
                self._end_session("finished", ended_by="session")

            case End(reason="spend-cap"):
                self._combined(CapEvent(at=now_ns(), cap="spend-cap"))
                self._end_session("spend-cap", ended_by="spend-cap", end_reason="spend-cap")

            case End(reason=reason):
                self._end_session(reason, ended_by="guard", end_reason=reason)

            case Reply(text=text):
                self._combined(
                    FollowUpEvent(at=now_ns(), action="replied", text=text),
                )
                self._reply_outstanding = True
                if self._session is not None:
                    self._spawn(self._session.send(text))

            case WaitForLock():
                self._combined(
                    FollowUpEvent(at=now_ns(), action="waiting"),
                )
                self._schedule("lock_poll", self._run_lock_poll(turn))

    async def _run_lock_poll(self, turn: TurnEndEvent) -> None:
        poll_s = self._config.lock_poll_ms / 1000
        while self._config.is_lock_held():  # noqa: ASYNC110 - lock poll uses real file probes
            await asyncio.sleep(poll_s)
        await self._run_settle(turn, after_wait=True)

    def _is_idle(self) -> bool:
        return bool(self._tasks)

    def _trigger_cap(self, cap: CapType) -> None:
        if self._cap_fired:
            return
        if self._session is None:
            self._pending_cap = cap
            return
        self._cap_fired = True
        self._ended_by = cap
        self._end_reason = cap
        if self._wall_task is not None:
            self._wall_task.cancel()

        was_idle = self._is_idle()
        self._cancel("settle")
        self._cancel("lock_poll")

        self._combined(CapEvent(at=now_ns(), cap=cap))

        if was_idle:
            self._spawn(self._session.end())
        else:
            self._interrupt_task = _fire_and_report_interrupt(self._session)
            self._grace_task = asyncio.create_task(self._run_grace())

    async def _run_grace(self) -> None:
        await asyncio.sleep(self._config.grace_ms / 1000)
        self._abort_event.set()

    async def _run_wall_clock(self) -> None:
        deadline = self._config.deadline_ms
        poll_s = self._config.wall_clock_poll_ms / 1000
        while now_ms() < deadline:  # noqa: ASYNC110 - wall-clock poll survives machine sleep
            await asyncio.sleep(poll_s)
        self._trigger_cap("wall-clock")

    async def run(self) -> SupervisionResult:
        self._combined(self._config.launch)

        start_time = time.perf_counter()
        self._session = self._config.driver.start(
            self._config.prompt,
            self._combined,
            self._abort_event,
        )
        if self._pending_cap is not None:
            self._trigger_cap(self._pending_cap)

        if not self._cap_fired:
            self._wall_task = asyncio.create_task(self._run_wall_clock())

        try:
            outcome = await self._session.outcome
            duration_ms = int((time.perf_counter() - start_time) * 1000)

            if self._end_reason is not None and self._ended_by == "session":
                return SupervisionResult(
                    outcome=SessionOutcome(
                        reason="error",
                        cost_usd=self._last_cost_usd,
                        message=self._end_reason,
                    ),
                    ended_by="session",
                    end_reason=self._end_reason,
                    duration_ms=duration_ms,
                    cost_usd=self._last_cost_usd,
                )

            return SupervisionResult(
                outcome=outcome,
                ended_by=self._ended_by,
                end_reason=self._end_reason,
                duration_ms=duration_ms,
                cost_usd=outcome.cost_usd,
            )
        finally:
            self._cancel("settle")
            self._cancel("lock_poll")
            if self._wall_task is not None:
                self._wall_task.cancel()
            if self._grace_task is not None:
                self._grace_task.cancel()
            if self._interrupt_task is not None:
                self._interrupt_task.cancel()


async def supervise(  # noqa: PLR0913 - one parameter per supervision knob
    driver: Driver,
    prompt: SessionPrompt,
    *,
    context: SupervisedSession,
    launch: LaunchEvent,
    observer: SessionObserver | None = None,
    grace_ms: int = 30_000,
    wall_clock_poll_ms: int = WALL_CLOCK_POLL_MS,
    settle_window_ms: int = SETTLE_WINDOW_MS,
    lock_poll_ms: int = LOCK_POLL_MS,
    is_lock_held: Callable[[], bool] | None = None,
) -> SupervisionResult:
    """Run a supervised agent session with wall-clock and spend caps.

    Starts the driver session, tees every event to a JSONL log and an optional
    observer, enforces the time and cost limits, and returns the session outcome
    with metadata about how the session ended.

    On each ``TurnEndEvent`` the supervisor enters an idle state, waits for the
    settle window to elapse, reads and folds the session log, probes the
    repository lock, and delegates to ``classify`` for the next action.

    ``is_lock_held`` defaults to probing ``context.lock_path`` via filelock's
    ``is_held``. Tests inject a callable to avoid filesystem contention.

    Args:
        driver: The agent driver that starts and sends messages to the session.
        prompt: The initial prompt and any system instructions.
        context: Session metadata — paths, caps, and the lock path.
        launch: The launch event emitted when the session starts.
        observer: Optional callback receiving every session event.
        grace_ms: Milliseconds to wait after requesting a stop before forcing
            cancellation.
        wall_clock_poll_ms: How often (ms) to check wall-clock and spend caps.
        settle_window_ms: Idle time (ms) after a turn ends before reading the
            session log and deciding the next action.
        lock_poll_ms: How often (ms) to probe the repository lock while the
            agent is idle.
        is_lock_held: Callable returning whether the repository lock is held.
            Defaults to probing ``context.lock_path`` on disk.

    Returns:
        The supervision result containing the session outcome and metadata.

    Raises:
        Exception: Whatever the driver session's ``outcome`` raises, propagated
            after the wall-clock and grace timers are cancelled.
    """
    if is_lock_held is None:
        lock_path = Path(context.lock_path)

        def is_lock_held() -> bool:
            return is_held(lock_path)

    config = _SuperviseConfig(
        driver=driver,
        prompt=prompt,
        context=context,
        log_path=context.log_path,
        launch=launch,
        max_usd=context.max_usd,
        observer=observer,
        grace_ms=grace_ms,
        deadline_ms=context.deadline_ms,
        wall_clock_poll_ms=wall_clock_poll_ms,
        settle_window_ms=settle_window_ms,
        lock_poll_ms=lock_poll_ms,
        is_lock_held=is_lock_held,
    )
    return await _Supervision(config).run()
