"""The supervisor orchestrator: run a driver session under time and spend caps.

:func:`supervise` starts a driver session, tees every event to a JSONL log and
an optional observer, and enforces a wall-clock cap plus an optional spend cap.
After each ``ToolEndEvent`` whose session log has grown, the supervisor scans
the log for a failed hook or a newly met stop condition and ends the run itself.
On each ``TurnEndEvent`` the supervisor enters an idle state, waits for a settle
window to elapse, reads and folds the session log, probes the repository lock,
and delegates to :func:`~gymrat.supervisor.turns.classify` for the next action.
The returned :class:`SupervisionResult` reports the session outcome and how it
ended.
"""

import asyncio
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Literal

from gymrat import clock
from gymrat.clock import now_ms, now_ns
from gymrat.config import BenchlessConfig
from gymrat.errors import GymratError
from gymrat.loop.iterate.run import stop_reason
from gymrat.session.budget import Budget
from gymrat.session.lock import is_held
from gymrat.session.paths import session_jsonl_path
from gymrat.session.records import HookRecord, SessionLogRecord
from gymrat.session.store import SessionState, fold_session, read_records
from gymrat.supervisor.driver import Driver, DriverSession, SessionOutcome, SessionPrompt
from gymrat.supervisor.events import (
    CapAction,
    CapEvent,
    CapType,
    CompactionEvent,
    FollowUpEvent,
    LaunchEvent,
    SessionEvent,
    SessionObserver,
    ToolEndEvent,
    TurnEndEvent,
    UsageUpdateEvent,
    combine_observers,
    create_event_log_writer,
)
from gymrat.supervisor.turns import (
    Decision,
    End,
    GuardState,
    Reply,
    WaitForLock,
    classify,
    outcome_record_count,
    spend_cap_reached,
)
from gymrat.utils import MS_PER_SECOND, warn_to_stderr

WALL_CLOCK_POLL_MS = 1000
"""Default interval (in milliseconds) for polling wall-clock time against the
deadline. Tests override this to avoid real waits."""

SETTLE_WINDOW_MS = 800
"""Default settle window (in milliseconds) after a turn end before reading the
session log and classifying."""

LOCK_POLL_MS = 5000
"""Default interval (in milliseconds) for polling the repository lock while
waiting for another process to release it."""

EndedBy = Literal["session", "guard", "stop-condition", "hook-failure"] | CapType
"""How a supervised run ended. See ``SupervisionResult.ended_by`` for the meaning of each member."""

_TaskSlot = Literal["settle", "lock_poll"]
"""Names of the at-most-one-per-slot background tasks the supervisor schedules."""

_IN_FLIGHT_EXCLUSION = (CapEvent, LaunchEvent, FollowUpEvent, CompactionEvent)
"""Event classes that do NOT cancel a pending settle window or lock poll.

Usage updates and turn ends never reach the check: the event router handles
both before it.
"""


@dataclass(frozen=True, slots=True)
class EndCondition:
    """A condition read off the session log that ends supervision."""

    ended_by: EndedBy
    reason: str


def _hook_failure_reason(record: HookRecord) -> str:
    failure = "timed out" if record.timed_out else f"exit {record.exit_code}"
    stderr = "?" if record.stderr_bytes is None else record.stderr_bytes
    return (
        f"{record.stage} hook failed on iteration {record.seq}: {failure} "
        f"(stdout {record.stdout_bytes} B, stderr {stderr} B)"
    )


def detect_end_condition(
    config: BenchlessConfig,
    records: list[SessionLogRecord],
    state: SessionState,
    *,
    cursor: int | None,
    check_stop: bool,
) -> EndCondition | None:
    """Find the condition in the session log that ends supervision, if any.

    A failed or timed-out hook record at or past *cursor* wins over a met stop
    condition. Hooks are scanned from the cursor rather than the tail because
    ``iterate`` appends a before-hook, the iteration, then an after-hook, so a
    failed before-hook sits two records back.

    Args:
        config: The session's Benchless configuration, read for stop conditions.
        records: The raw session log records, command records included.
        state: The session state already folded from *records*.
        cursor: The index of the first record not yet scanned for hook
            failures, or ``None`` to scan no hook records.
        check_stop: Whether a met stop condition is reported.

    Returns:
        The end condition, or ``None`` when nothing ends supervision.
    """
    if cursor is not None:
        for record in records[cursor:]:
            if isinstance(record, HookRecord) and record.failed:
                return EndCondition("hook-failure", _hook_failure_reason(record))

    if check_stop and (reason := stop_reason(config, state)) is not None:
        return EndCondition("stop-condition", reason)

    return None


def _log_size(path: str) -> int:
    """Size in bytes of the session log; a missing log is empty.

    Args:
        path: The session log path.

    Returns:
        The log size in bytes, or ``0`` when the log does not exist.

    Raises:
        GymratError: The log exists but cannot be inspected.
    """
    try:
        return Path(path).stat().st_size
    except FileNotFoundError:
        return 0
    except OSError as error:
        message = f"Cannot inspect session log {path}: {error.strerror or error}"
        raise GymratError(message) from error


class EndConditionScan:
    """Track how far the session log has been scanned and the end it found.

    Hook records are scanned only past the cursor, so a hook failure already in
    the log when the run started never ends it. Later scans leave a pending
    condition alone.

    Args:
        config: The settled config whose stop conditions the scan checks.
        log_path: The session log to scan.

    Attributes:
        pending: The first end condition found, held until the supervisor fires
            it; ``None`` while nothing is pending.
    """

    def __init__(self, config: BenchlessConfig, log_path: str) -> None:
        self._config = config
        self._path = log_path
        self._size = 0
        self._cursor: int | None = None
        self._check_stop = True
        self.pending: EndCondition | None = None

    def seed(self) -> list[SessionLogRecord] | None:
        """Read the log at launch and start the scan from its current end.

        A stop condition already met at launch disarms stop-condition detection:
        the run was forced past it. When the read fails the cursor stays unset,
        and the first clean scan sets it and arms stop-condition detection from
        the state it folds, without reporting hook failures.

        Returns:
            The launch records, or ``None`` when the log cannot be inspected, read, or folded.
        """
        try:
            size = _log_size(self._path)
            records = read_records(self._path)
            state = fold_session(records)
        except GymratError:
            return None
        self._size = size
        self._cursor = len(records)
        self._arm_stop_check(state)
        return records

    def scan_if_grown(self) -> None:
        """Read the log and detect an end condition when it grew since the last scan.

        Does nothing while an end is pending or when the log size is unchanged,
        so event delivery never re-parses an untouched log.

        Raises:
            GymratError: The grown log cannot be inspected, read, or folded.
        """
        if self.pending is not None:
            return
        size = _log_size(self._path)
        if size == self._size:
            return
        records = read_records(self._path)
        state = fold_session(records)
        self._size = size
        self.detect(records, state)

    def detect(self, records: list[SessionLogRecord], state: SessionState) -> None:
        """Record the end condition in ``records`` as pending, unless one already is.

        The first scan after a failed launch read decides whether stop-condition
        detection is armed, exactly as a clean launch read would: a stop condition
        already met in that state means the run was forced past it.

        Args:
            records: The raw session log records, command records included.
            state: The session state already folded from ``records``.
        """
        if self.pending is not None:
            return
        if self._cursor is None:
            self._arm_stop_check(state)
        self.pending = detect_end_condition(
            self._config,
            records,
            state,
            cursor=self._cursor,
            check_stop=self._check_stop,
        )
        # Every record has now been scanned for hook failures, whatever was found.
        self._cursor = len(records)

    def _arm_stop_check(self, state: SessionState) -> None:
        """Arm stop-condition detection unless ``state`` already satisfies one."""
        self._check_stop = stop_reason(self._config, state) is None


def _warn_on_task_failure(finished: asyncio.Task[None], *, context: str) -> None:
    """Warn to stderr with ``context`` when ``finished`` raised; ignore cancellation."""
    if finished.cancelled():
        return
    error = finished.exception()
    if error is not None:
        warn_to_stderr(f"{context} failed: {error!s}")


_warn_unhandled = partial(_warn_on_task_failure, context="background task")
"""Done-callback that surfaces exceptions from fire-and-forget tasks."""


def _fire_and_report_interrupt(session: DriverSession) -> asyncio.Task[None] | None:
    """Interrupt the session, isolating any failure so grace setup continues.

    ``interrupt`` may throw synchronously or its coroutine may reject; either way
    the fallback recovery still runs, so the failure is warned, never raised.

    Args:
        session: The driver session to interrupt.

    Returns:
        The interrupt task, so the caller can cancel it on teardown, or ``None``
        when the interrupt could not be started.
    """
    try:
        pending = session.interrupt()
    except Exception as error:  # noqa: BLE001 - interrupt failure must not abort grace setup
        warn_to_stderr(f"session interrupt failed: {error!s}")
        return None

    task = asyncio.create_task(pending)
    task.add_done_callback(partial(_warn_on_task_failure, context="session interrupt"))
    return task


@dataclass(frozen=True, slots=True)
class SupervisedSession:
    """Immutable snapshot of everything a supervised session needs to run.

    Built by the CLI layer and passed to :func:`supervise`.

    Attributes:
        root: The repository root the session runs in.
        log_path: Where the supervisor writes its JSONL event log.
        lock_path: The repository lock (``lockfile_path(root)``), not the
            supervise lock.
        config: The settled config whose stop conditions and guards apply.
        budget: The session's time budget; its deadline is when the wall-clock cap trips.
        max_usd: The spend cap in dollars, or ``None`` for no cap.
    """

    root: str
    log_path: str
    lock_path: str
    config: BenchlessConfig
    budget: Budget
    max_usd: float | None


@dataclass(frozen=True, slots=True, kw_only=True)
class SupervisionResult:
    """How a supervised session ended.

    Attributes:
        outcome: How the session settled (completed, interrupted, or error).
        ended_by: Whether the session ended on its own, was stopped by a cap,
            or was ended by a condition read off the session log.
        end_reason: For ``guard``, the guard reason; for a cap, the cap name;
            for ``stop-condition`` and ``hook-failure``, the condition's
            summary; for ``session``, ``None`` unless a log-read error set it.
        duration_ms: Elapsed milliseconds from start to settlement, measured
            on the monotonic clock.
    """

    outcome: SessionOutcome
    ended_by: EndedBy
    duration_ms: int
    end_reason: str | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class _SuperviseConfig:
    """The full input surface of one :func:`supervise` call."""

    driver: Driver
    prompt: SessionPrompt
    context: SupervisedSession
    launch: LaunchEvent
    observer: SessionObserver | None
    grace_ms: int
    wall_clock_poll_ms: int
    settle_window_ms: int
    lock_poll_ms: int
    is_lock_held: Callable[[], bool]


class _Supervision:
    """Runs one supervised session, holding the mutable cap/timer state."""

    def __init__(self, config: _SuperviseConfig) -> None:
        self._config = config
        self._abort_event = asyncio.Event()
        log_writer = create_event_log_writer(config.context.log_path)

        observers: list[SessionObserver] = [self._event_router, log_writer]
        if config.observer is not None:
            observers.append(config.observer)
        self._combined = combine_observers(*observers)

        self._ended_by: EndedBy = "session"
        self._end_reason: str | None = None
        self._cap_fired = False
        self._wall_task: asyncio.Task[None] | None = None
        self._grace_timer: asyncio.TimerHandle | None = None
        self._interrupt_task: asyncio.Task[None] | None = None
        self._session: DriverSession | None = None

        self._end_scan = EndConditionScan(
            config.context.config, session_jsonl_path(config.context.root)
        )
        launch_records = self._end_scan.seed()
        self._guards = GuardState(
            initial_record_count=0
            if launch_records is None
            else outcome_record_count(launch_records),
        )
        self._reply_outstanding = False
        self._tasks: dict[_TaskSlot, asyncio.Task[None]] = {}
        self._last_cost_usd = 0.0
        self._background_tasks: set[asyncio.Task[None]] = set()

    def _spawn(self, target: Coroutine[object, object, None]) -> None:
        """Create a background task and prevent it from being garbage-collected."""
        task = asyncio.create_task(target)
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
        task.add_done_callback(_warn_unhandled)

    def _event_router(self, event: SessionEvent) -> None:
        """Route events for in-flight detection, cost tracking, and end-condition scans."""
        if isinstance(event, UsageUpdateEvent):
            self._last_cost_usd = event.cost_usd
            return

        if isinstance(event, TurnEndEvent):
            self._handle_turn_end(event)
            return

        if not isinstance(event, _IN_FLIGHT_EXCLUSION):
            self._cancel_pending()

        if isinstance(event, ToolEndEvent):
            self._scan_at_tool_end()

    def _scan_at_tool_end(self) -> None:
        """Scan a grown session log for an end condition and fire it once the lock is free.

        Runs inside event delivery, so every end it causes is scheduled rather
        than emitted inline: the router is the first observer, and an inline
        emit would reach the event log ahead of the ``tool_end`` that caused it.
        """
        if self._cap_fired:
            return
        loop = asyncio.get_running_loop()
        try:
            self._end_scan.scan_if_grown()
        except GymratError as error:
            loop.call_soon(self._end_with_error, str(error))
            return
        if self._end_scan.pending is not None and not self._config.is_lock_held():
            loop.call_soon(self._fire_pending_end)

    def _fire_pending_end(self) -> None:
        pending = self._end_scan.pending
        if pending is None:
            return
        self._trigger_end(
            FollowUpEvent(at=now_ns(), action="ended", reason=pending.reason),
            ended_by=pending.ended_by,
            end_reason=pending.reason,
        )

    def _cancel_pending(self) -> None:
        self._cancel("settle")
        self._cancel("lock_poll")

    def _cancel(self, slot: _TaskSlot) -> None:
        task = self._tasks.pop(slot, None)
        if task is not None:
            task.cancel()

    def _schedule(
        self,
        slot: _TaskSlot,
        routine: Coroutine[object, object, None],
    ) -> None:
        task = asyncio.create_task(routine)

        def _clear_on_done(finished: asyncio.Task[None]) -> None:
            if self._tasks.get(slot) is finished:
                del self._tasks[slot]

        task.add_done_callback(_clear_on_done)
        task.add_done_callback(self._end_on_task_failure)
        self._tasks[slot] = task

    def _end_on_task_failure(self, finished: asyncio.Task[None]) -> None:
        # A failed settle or lock poll leaves nothing to answer the turn end,
        # so the run would otherwise idle until the wall-clock cap.
        if finished.cancelled():
            return
        error = finished.exception()
        if error is not None:
            self._end_with_error(str(error))

    def _handle_turn_end(self, event: TurnEndEvent) -> None:
        if self._cap_fired:
            return
        if self._reply_outstanding:
            if event.origin != "agent":
                return
            self._reply_outstanding = False

        self._cancel_pending()
        self._schedule("settle", self._run_settle(event))

    async def _run_settle(self, turn: TurnEndEvent, *, after_wait: bool = False) -> None:
        if self._config.settle_window_ms > 0:
            await asyncio.sleep(self._config.settle_window_ms / MS_PER_SECOND)

        try:
            records = read_records(
                session_jsonl_path(self._config.context.root),
            )
            state = fold_session(records)
        except GymratError as error:
            self._end_with_error(str(error))
            return

        self._end_scan.detect(records, state)
        lock_held = self._config.is_lock_held()

        if (pending := self._end_scan.pending) is not None:
            if spend_cap_reached(turn, self._config.context.max_usd):
                self._execute_decision(End(reason="spend-cap"), turn)
            elif lock_held:
                self._wait_for_lock(turn)
            else:
                self._end_session(
                    pending.reason, ended_by=pending.ended_by, end_reason=pending.reason
                )
            return

        decision = classify(
            state=state,
            records=records,
            guards=self._guards,
            lock_held=lock_held,
            turn=turn,
            max_usd=self._config.context.max_usd,
            budget=self._config.context.budget,
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
            self._arm_grace()

    def _arm_grace(self) -> None:
        self._grace_timer = asyncio.get_running_loop().call_later(
            self._config.grace_ms / MS_PER_SECOND, self._abort_event.set
        )

    def _end_with_error(self, message: str) -> None:
        self._end_session(message, ended_by="session", end_reason=message)

    def _execute_decision(self, decision: Decision, turn: TurnEndEvent) -> None:
        match decision:
            case End(reason="finished"):
                self._end_session("finished", ended_by="session")

            case End(reason="spend-cap"):
                self._combined(CapEvent(at=now_ns(), cap="spend-cap", action="ending"))
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
                self._wait_for_lock(turn)

    def _wait_for_lock(self, turn: TurnEndEvent) -> None:
        self._combined(
            FollowUpEvent(at=now_ns(), action="waiting"),
        )
        self._schedule("lock_poll", self._run_lock_poll(turn))

    async def _run_lock_poll(self, turn: TurnEndEvent) -> None:
        poll_s = self._config.lock_poll_ms / MS_PER_SECOND
        while self._config.is_lock_held():  # noqa: ASYNC110 - lock poll uses real file probes
            await asyncio.sleep(poll_s)
        await self._run_settle(turn, after_wait=True)

    def _is_idle(self) -> bool:
        return bool(self._tasks)

    def _trigger_cap(self, cap: CapType) -> None:
        action: CapAction = "ending" if self._is_idle() else "interrupting"
        self._trigger_end(
            CapEvent(at=now_ns(), cap=cap, action=action), ended_by=cap, end_reason=cap
        )

    def _trigger_end(
        self, event: CapEvent | FollowUpEvent, *, ended_by: EndedBy, end_reason: str
    ) -> None:
        """Stop the running session once.

        An idle session is ended; one with a turn in flight is interrupted.
        Either way grace is armed, so the abort event fires if the driver does
        not settle.

        Args:
            event: The cap or follow-up event announcing the end.
            ended_by: How the session is reported to have ended.
            end_reason: The reason reported alongside ``ended_by``.
        """
        if self._cap_fired or self._session is None:
            return
        self._cap_fired = True
        self._ended_by = ended_by
        self._end_reason = end_reason
        if self._wall_task is not None:
            self._wall_task.cancel()

        was_idle = self._is_idle()
        self._cancel_pending()
        self._combined(event)

        if was_idle:
            self._spawn(self._session.end())
        else:
            self._interrupt_task = _fire_and_report_interrupt(self._session)
        self._arm_grace()

    async def _run_wall_clock(self) -> None:
        deadline = self._config.context.budget.deadline_ms
        poll_s = self._config.wall_clock_poll_ms / MS_PER_SECOND
        while now_ms() < deadline:  # noqa: ASYNC110 - wall-clock poll survives machine sleep
            await asyncio.sleep(poll_s)
        self._trigger_cap("wall-clock")

    async def run(self) -> SupervisionResult:
        self._combined(self._config.launch)

        start_time = clock.monotonic_ms()
        self._session = self._config.driver.start(
            self._config.prompt,
            self._combined,
            self._abort_event,
        )

        self._wall_task = asyncio.create_task(self._run_wall_clock())

        try:
            outcome = await self._session.outcome
            duration_ms = int(clock.monotonic_ms() - start_time)

            if self._end_reason is not None and self._ended_by == "session":
                outcome = SessionOutcome(
                    reason="error", cost_usd=self._last_cost_usd, message=self._end_reason
                )

            return SupervisionResult(
                outcome=outcome,
                ended_by=self._ended_by,
                end_reason=self._end_reason,
                duration_ms=duration_ms,
            )
        finally:
            self._cancel_pending()
            for handle in (self._wall_task, self._grace_timer, self._interrupt_task):
                if handle is not None:
                    handle.cancel()


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

    After each ``ToolEndEvent`` whose session log has grown, the supervisor scans
    the log for a failed hook or a stop condition met during the run and ends the
    session, deferring the end while the repository lock is held.

    On each ``TurnEndEvent`` the supervisor enters an idle state, waits for the
    settle window to elapse, reads and folds the session log, probes the
    repository lock, and delegates to ``classify`` for the next action.

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
    config = _SuperviseConfig(
        driver=driver,
        prompt=prompt,
        context=context,
        launch=launch,
        observer=observer,
        grace_ms=grace_ms,
        wall_clock_poll_ms=wall_clock_poll_ms,
        settle_window_ms=settle_window_ms,
        lock_poll_ms=lock_poll_ms,
        is_lock_held=is_lock_held or partial(is_held, context.lock_path),
    )
    return await _Supervision(config).run()
