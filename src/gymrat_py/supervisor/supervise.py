"""The supervisor orchestrator: run a driver session under time and spend caps.

:func:`supervise` starts a driver session, tees every event to a JSONL log and
an optional observer, and enforces a wall-clock cap plus an optional spend cap.
When a cap fires it emits a ``cap`` event, interrupts the session, and — because
a driver may ignore or be slow to honour the interrupt — arms a grace timer that
sets the driver's abort event after ``grace_ms``. The returned
:class:`SupervisionResult` reports the session outcome and how it ended.
"""

import asyncio
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from gymrat_py.errors import message_of
from gymrat_py.session.clock import now_ms
from gymrat_py.supervisor.driver import Driver, DriverSession, SessionOutcome, SessionPrompt
from gymrat_py.supervisor.event_log import create_event_log_writer
from gymrat_py.supervisor.events import (
    CapEvent,
    LaunchEvent,
    SessionEvent,
    SessionObserver,
    combine_observers,
)
from gymrat_py.warn import warn_to_stderr

# Seconds per minute, for converting ``max_minutes`` to an ``asyncio.sleep``.
_MINUTE_S = 60.0

CapType = Literal["wall-clock", "spend-cap"]
EndedBy = Literal["session", "wall-clock", "spend-cap"]


@dataclass(frozen=True, slots=True, kw_only=True)
class SupervisionResult:
    """How a supervised session ended.

    - ``outcome``: how the session settled (completed, interrupted, or error).
    - ``ended_by``: whether the session ended on its own or was stopped by a cap.
    - ``duration_ms``: wall-clock duration from start to settlement.
    - ``cost_usd``: the final cost reported by the session.
    """

    outcome: SessionOutcome
    ended_by: EndedBy
    duration_ms: int
    cost_usd: float


@dataclass(frozen=True, slots=True, kw_only=True)
class _SuperviseConfig:
    """The full input surface of one :func:`supervise` call."""

    driver: Driver
    prompt: SessionPrompt
    max_minutes: float
    log_path: str | Path
    launch: LaunchEvent
    max_usd: float | None
    observer: SessionObserver | None
    grace_ms: int


def _fire_and_report_interrupt(session: DriverSession) -> None:
    """Interrupt the session, isolating any failure so grace setup continues.

    ``interrupt`` may throw synchronously or its coroutine may reject; either way
    the fallback recovery still runs, so the failure is warned, never raised.
    """
    try:
        pending = session.interrupt()
    except Exception as error:  # noqa: BLE001 - interrupt failure must not abort grace setup
        warn_to_stderr(f"session interrupt failed: {message_of(error)}")
        return

    task = asyncio.ensure_future(pending)

    def _report(finished: asyncio.Task[None]) -> None:
        if finished.cancelled():
            return
        error = finished.exception()
        if error is not None:
            warn_to_stderr(f"session interrupt failed: {message_of(error)}")

    task.add_done_callback(_report)


class _Supervision:
    """Runs one supervised session, holding the mutable cap/timer state."""

    def __init__(self, config: _SuperviseConfig) -> None:
        self._config = config
        self._abort_event = asyncio.Event()
        log_writer = create_event_log_writer(config.log_path)

        observers: list[SessionObserver] = [self._cost_observer, log_writer]
        if config.observer is not None:
            observers.append(config.observer)
        self._combined = combine_observers(*observers)

        self._ended_by: EndedBy = "session"
        self._cap_fired = False
        self._wall_task: asyncio.Task[None] | None = None
        self._grace_task: asyncio.Task[None] | None = None
        self._session: DriverSession | None = None
        self._pending_cap: CapType | None = None

    def _cost_observer(self, event: SessionEvent) -> None:
        max_usd = self._config.max_usd
        if max_usd is not None and event.type == "usage_update" and event.cost_usd >= max_usd:
            self._trigger_cap("spend-cap")

    def _trigger_cap(self, cap: CapType) -> None:
        if self._cap_fired:
            return
        if self._session is None:
            # A cap requested before ``start`` returns is deferred until it does.
            self._pending_cap = cap
            return
        self._cap_fired = True
        self._ended_by = cap
        if self._wall_task is not None:
            self._wall_task.cancel()
        self._combined(CapEvent(timestamp=now_ms(), cap=cap))
        _fire_and_report_interrupt(self._session)
        self._grace_task = asyncio.ensure_future(self._run_grace())

    async def _run_grace(self) -> None:
        await asyncio.sleep(self._config.grace_ms / 1000)
        self._abort_event.set()

    async def _run_wall_clock(self) -> None:
        await asyncio.sleep(self._config.max_minutes * _MINUTE_S)
        self._trigger_cap("wall-clock")

    async def run(self) -> SupervisionResult:
        self._combined(self._config.launch)

        start_time = time.perf_counter()
        self._session = self._config.driver.start(
            self._config.prompt, self._combined, self._abort_event
        )
        if self._pending_cap is not None:
            self._trigger_cap(self._pending_cap)

        # Arm the wall-clock timer only when no cap fired during start.
        if not self._cap_fired:
            self._wall_task = asyncio.ensure_future(self._run_wall_clock())

        try:
            outcome = await self._session.outcome
            duration_ms = int((time.perf_counter() - start_time) * 1000)
            return SupervisionResult(
                outcome=outcome,
                ended_by=self._ended_by,
                duration_ms=duration_ms,
                cost_usd=outcome.cost_usd,
            )
        finally:
            if self._wall_task is not None:
                self._wall_task.cancel()
            if self._grace_task is not None:
                self._grace_task.cancel()


async def supervise(  # noqa: PLR0913 - one parameter per supervision knob, mirroring the driver-seam option surface
    driver: Driver,
    prompt: SessionPrompt,
    *,
    max_minutes: float,
    log_path: str | Path,
    launch: LaunchEvent,
    max_usd: float | None = None,
    observer: SessionObserver | None = None,
    grace_ms: int = 30_000,
) -> SupervisionResult:
    """Run a supervised agent session with wall-clock and spend caps.

    Starts the driver session, tees every event to a JSONL log and an optional
    observer, enforces the time and cost limits, and returns the session outcome
    with metadata about how the session ended. A raising ``outcome`` propagates
    after the wall-clock and grace timers are cancelled.
    """
    config = _SuperviseConfig(
        driver=driver,
        prompt=prompt,
        max_minutes=max_minutes,
        log_path=log_path,
        launch=launch,
        max_usd=max_usd,
        observer=observer,
        grace_ms=grace_ms,
    )
    return await _Supervision(config).run()
