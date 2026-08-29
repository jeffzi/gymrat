"""The driver seam: the contract the supervisor drives an agent session through.

A :class:`Driver` launches a session synchronously via :meth:`Driver.start` and
hands back a :class:`DriverSession` whose asynchronous work is exposed behind an
awaitable ``outcome``. The supervisor tees every event through the observer it
passes in, and stops a session either by setting the ``abort`` event or by
calling :meth:`DriverSession.interrupt`.
"""

import asyncio
from collections.abc import Awaitable
from dataclasses import dataclass
from typing import Literal, Protocol

from gymrat.supervisor.events import SessionObserver


@dataclass(frozen=True, slots=True, kw_only=True)
class SessionPrompt:
    """What the driver is asked to run."""

    kickoff: str
    cwd: str
    system_prompt_append: str | None = None
    model: str | None = None


SessionEndReason = Literal["completed", "interrupted", "error"]
"""Why a session ended."""


@dataclass(frozen=True, slots=True, kw_only=True)
class SessionOutcome:
    """The result of a settled session; ``message`` is set only on ``error``."""

    reason: SessionEndReason
    cost_usd: float
    message: str | None = None


class DriverSession(Protocol):
    """A live agent session returned by :meth:`Driver.start`."""

    @property
    def outcome(self) -> Awaitable[SessionOutcome]:
        """Resolves with the session's outcome once it settles."""
        ...

    async def interrupt(self) -> None:
        """Ask the session to stop; the outcome then settles as ``interrupted``."""
        ...


class Driver(Protocol):
    """Launches and controls an agent session."""

    def start(
        self,
        prompt: SessionPrompt,
        observer: SessionObserver,
        abort: asyncio.Event | None = None,
    ) -> DriverSession:
        """Start a session synchronously; async work runs behind ``outcome``."""
        ...
