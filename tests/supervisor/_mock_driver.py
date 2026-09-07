"""A scripted mock driver for exercising supervisor orchestration.

``create_mock_driver`` builds a :class:`~gymrat.supervisor.driver.Driver`
whose ``start`` runs a caller-supplied script of steps in order on the running
event loop, without a real agent backend. A step emits an event, awaits an async
action, reports a cost, or simulates a turn boundary. Each step's optional
``delay_ms`` races a timer against the abort — the driver's own ``interrupt`` or
the external abort event — so a delayed step yields the moment the session is
interrupted or aborted.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Sequence

from gymrat.session.clock import now_ms
from gymrat.supervisor.driver import (
    DriverSession,
    SessionOutcome,
    SessionPrompt,
)
from gymrat.supervisor.events import (
    SessionEvent,
    SessionObserver,
    TurnEndEvent,
    UsageUpdateEvent,
)


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
                self._observer(UsageUpdateEvent(timestamp=now_ms(), cost_usd=step.cost_usd))
            case TurnEndStep():
                if self._aborted():
                    return
                if step.cost_usd is not None:
                    self._cost_usd = step.cost_usd
                self._observer(
                    TurnEndEvent(
                        timestamp=now_ms(),
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

    The returned ``_MockDriver`` satisfies the :class:`Driver` protocol and
    exposes a ``.sessions`` list for test assertions on ``send``/``end`` calls.
    """
    return _MockDriver(tuple(steps))
