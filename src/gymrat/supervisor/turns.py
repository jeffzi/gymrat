"""Turn classifier: decide whether to reply, wait, or end after each agent turn.

The classifier is a pure function over its arguments plus the mutable
``GuardState`` it updates. It never reads the filesystem or the driver.
Evaluation order is fixed — the first matching rule wins:

1. Session finished (finalized, stop record, or configured stop condition).
2. Spend cap (budget exhausted or cost exceeds ``max_usd``).
3. Lock held (another process holds the repository lock).
4. Guard tripped (follow-up ceiling, no-progress, consecutive discards).
5. Otherwise: reply with the runbook instruction and time-left trailer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from gymrat.eta import format_duration
from gymrat.loop.iterate.run import stop_condition
from gymrat.session.records.models import DiscardRecord, KeepRecord

if TYPE_CHECKING:
    from gymrat.config import BenchlessConfig
    from gymrat.session import SessionLogRecord
    from gymrat.session.store import SessionState
    from gymrat.supervisor.events import TurnEndEvent

FOLLOW_UP_CEILING = 100
"""Maximum follow-up replies before the classifier ends the session."""

NO_PROGRESS_LIMIT = 3
"""Consecutive stale turn ends (no new records) before ending with no-progress."""

CONSECUTIVE_DISCARD_LIMIT = 5
"""Settlement records ending in this many discards with no committed keep triggers an end."""

_RUNBOOK_INSTRUCTION = (
    "No human is present. Run gymrat status to re-read the session, "
    "then decide from the runbook and continue. When the work is done, "
    "record your report with gymrat stop -m and end the turn."
)

_AFTER_WAIT_LINE = (
    "The command you left running has finished; its record, if any, is in the session log."
)


@dataclass(slots=True)
class GuardState:
    """Mutable per-run counters the classifier reads and updates."""

    initial_record_count: int
    replies_sent: int = 0
    no_progress_count: int = 0
    last_record_count: int = field(init=False)

    def __post_init__(self) -> None:
        self.last_record_count = self.initial_record_count


class Decision:
    """Base type for classifier outcomes."""


@dataclass(frozen=True, slots=True)
class End(Decision):
    """The session should end for the given reason."""

    reason: str


@dataclass(frozen=True, slots=True)
class Reply(Decision):
    """Send a follow-up message with the given text."""

    text: str


@dataclass(frozen=True, slots=True)
class WaitForLock(Decision):
    """Another process holds the lock; wait without updating counters."""


def _consecutive_discard_count(
    records: list[SessionLogRecord],
    initial_record_count: int,
) -> int:
    """Count trailing discards among settlement records appended since launch.

    Only keep and discard records are settlements. Iteration and hook records
    between discards do not break the streak. A committed keep resets it. A
    keep that is not committed neither counts nor resets the streak.
    """
    settlements = [
        r for r in records[initial_record_count:] if isinstance(r, KeepRecord | DiscardRecord)
    ]
    count = 0
    for record in reversed(settlements):
        if isinstance(record, DiscardRecord):
            count += 1
        elif isinstance(record, KeepRecord) and record.status == "committed":
            break
    return count


def _format_reply(
    *,
    deadline_ms: float,
    max_minutes: float,
    now_ms: float,
    after_wait: bool,
) -> str:
    remaining = max(0.0, deadline_ms - now_ms)
    parts = [
        _RUNBOOK_INSTRUCTION,
        f"\n{format_duration(remaining)} left of {max_minutes:g}m",
    ]
    if after_wait:
        parts.append(f"\n{_AFTER_WAIT_LINE}")
    return "".join(parts)


def classify(  # noqa: PLR0913, PLR0911 - one parameter per classification input
    *,
    config: BenchlessConfig,
    state: SessionState,
    records: list[SessionLogRecord],
    guards: GuardState,
    lock_held: bool,
    turn: TurnEndEvent,
    max_usd: float | None,
    deadline_ms: float,
    max_minutes: float,
    now_ms: float,
    after_wait: bool = False,
) -> Decision:
    """Evaluate the turn and return the decision.

    Mutates *guards* on every non-``WaitForLock`` outcome: updates the
    record-count baseline, the no-progress counter, and the reply counter.
    """
    # Rule 1: session finished
    if (
        state.finalized is not None
        or state.ends_on_stop
        or stop_condition(config, state) is not None
    ):
        return End(reason="finished")

    # Rule 2: spend cap
    if turn.budget_exhausted or (max_usd is not None and turn.cost_usd >= max_usd):
        return End(reason="spend-cap")

    # Rule 3: lock held — freeze all counters
    if lock_held:
        return WaitForLock()

    # From here, counters are updated on every non-WaitForLock outcome.

    current_count = len(records)

    # No-progress accounting: only after at least one reply has been sent
    if guards.replies_sent > 0:
        if current_count > guards.last_record_count:
            guards.no_progress_count = 0
        else:
            guards.no_progress_count += 1

    # Move baseline on every non-WaitForLock classification
    guards.last_record_count = current_count

    # Rule 4: guards
    if guards.replies_sent >= FOLLOW_UP_CEILING:
        return End(reason="follow-up-ceiling")

    if guards.no_progress_count >= NO_PROGRESS_LIMIT:
        return End(reason="no-progress")

    trailing_discards = _consecutive_discard_count(records, guards.initial_record_count)
    if trailing_discards >= CONSECUTIVE_DISCARD_LIMIT:
        return End(reason="consecutive-discards")

    # Rule 5: reply
    guards.replies_sent += 1
    text = _format_reply(
        deadline_ms=deadline_ms,
        max_minutes=max_minutes,
        now_ms=now_ms,
        after_wait=after_wait,
    )
    return Reply(text=text)
