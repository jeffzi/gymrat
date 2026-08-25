"""A scripted mock driver for exercising supervisor orchestration.

``create_mock_driver`` builds a :class:`~gymrat_py.supervisor.driver.Driver`
whose ``start`` runs a caller-supplied script of steps in order on the running
event loop, without a real agent backend. A step emits an event, awaits an async
action, or reports a cost. Each step's optional ``delay_ms`` races a timer
against the abort — the driver's own ``interrupt`` or the external abort event —
so a delayed step yields the moment the session is interrupted or aborted.
"""

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass

from gymrat_py.errors import message_of
from gymrat_py.session.clock import now_ms
from gymrat_py.supervisor.driver import (
    Driver,
    DriverSession,
    SessionOutcome,
    SessionPrompt,
)
from gymrat_py.supervisor.events import SessionEvent, SessionObserver, UsageUpdateEvent


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


MockStep = EmitStep | ActionStep | CostStep
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
        self._script: asyncio.Task[SessionOutcome] = asyncio.ensure_future(self._run(steps))

    @property
    def outcome(self) -> Awaitable[SessionOutcome]:
        return self._script

    async def interrupt(self) -> None:
        self._abort.set()

    def _aborted(self) -> bool:
        return self._abort.is_set() or (self._external is not None and self._external.is_set())

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

    async def _run(self, steps: Sequence[MockStep]) -> SessionOutcome:
        for step in steps:
            # Yield between steps so an interrupt scheduled by the prior step's
            # observer is applied before the successor runs. The supervisor fires
            # ``interrupt`` as a task, so its abort lands on the next loop turn.
            await asyncio.sleep(0)

            if self._aborted():
                return self._interrupted()

            if step.delay_ms is not None and step.delay_ms > 0:
                await self._delay(step.delay_ms)

            if self._aborted():
                return self._interrupted()

            try:
                await self._execute(step)
            except Exception as error:  # noqa: BLE001 - the mock's contract turns any action failure into an error outcome
                return SessionOutcome(
                    reason="error", cost_usd=self._cost_usd, message=message_of(error)
                )

            if self._aborted():
                return self._interrupted()

        return SessionOutcome(reason="completed", cost_usd=self._cost_usd)


class _MockDriver:
    def __init__(self, steps: Sequence[MockStep]) -> None:
        self._steps = steps

    def start(
        self,
        prompt: SessionPrompt,
        observer: SessionObserver,
        abort: asyncio.Event | None = None,
    ) -> DriverSession:
        return _MockSession(self._steps, observer, abort)


def create_mock_driver(steps: Sequence[MockStep]) -> Driver:
    """Return a :class:`Driver` that runs ``steps`` in order on each ``start``."""
    return _MockDriver(tuple(steps))
