"""Inputs shared by the turn-classifier tests.

Builders for the turn-end event and guard state ``classify`` takes, and a
wrapper that fills its boilerplate keyword arguments with neutral defaults.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from gymrat.supervisor.events import TurnEndEvent

if TYPE_CHECKING:
    from gymrat.config.types import BenchlessConfig
    from gymrat.session.records import SessionLogRecord
    from gymrat.session.store import SessionState
from gymrat.supervisor.turns import (
    Decision,
    GuardState,
    classify,
)
from tests.cli.supervise._fixtures import session_state
from tests.supervisor._fixtures import default_benchless_config


def turn_end(
    *,
    cost_usd: float = 0.01,
    budget_exhausted: bool = False,
    text: str = "done",
    origin: Literal["agent", "injected"] = "agent",
    at: int = 5_000_000_000_000,
) -> TurnEndEvent:
    """A turn-end event with classifier-neutral defaults."""
    return TurnEndEvent(
        at=at,
        text=text,
        cost_usd=cost_usd,
        origin=origin,
        budget_exhausted=budget_exhausted,
    )


def guard_state(
    *,
    initial_record_count: int = 0,
    replies_sent: int = 0,
    no_progress_count: int = 0,
    last_record_count: int | None = None,
) -> GuardState:
    """A guard state preset to the given counters."""
    gs = GuardState(initial_record_count=initial_record_count)
    gs.replies_sent = replies_sent
    gs.no_progress_count = no_progress_count
    if last_record_count is not None:
        gs.last_record_count = last_record_count
    return gs


def classify_with_defaults(
    *,
    config: BenchlessConfig,
    state: SessionState,
    records: list[SessionLogRecord],
    guards: GuardState,
    turn: TurnEndEvent,
    lock_held: bool = False,
    max_usd: float | None = None,
    deadline_ms: float = 999_999_999.0,
    max_minutes: float = 60,
    now_ms: float = 0.0,
    after_wait: bool = False,
) -> Decision:
    """Delegates to ``classify`` with defaults for the boilerplate keyword args."""
    return classify(
        config=config,
        state=state,
        records=records,
        guards=guards,
        turn=turn,
        lock_held=lock_held,
        max_usd=max_usd,
        deadline_ms=deadline_ms,
        max_minutes=max_minutes,
        now_ms=now_ms,
        after_wait=after_wait,
    )


def classify_discards(records: list[SessionLogRecord]) -> Decision:
    """Runs classify with the discard-streak defaults, varying only ``records``."""
    return classify_with_defaults(
        config=default_benchless_config(),
        state=session_state(),
        records=records,
        guards=guard_state(),
        turn=turn_end(),
    )
