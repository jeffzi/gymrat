"""Shared progress-event builders for the progress renderer and progress state tests."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

from gymrat.progress_events import PassFinished, PassStarted

if TYPE_CHECKING:
    from typing import Literal

    from tests._rich import Clock


def ms_from_clock(clock: Clock[float]) -> int:
    """Return the clock's current time in milliseconds, for ``at_ms`` fields."""
    return int(clock.now * 1000)


def _pass_event[E: (PassStarted, PassFinished)](
    event_type: type[E],
    round_num: int,
    total_rounds: int,
    *,
    at_ms: int,
    target_count: int = 1,
    label: str = "bench",
    phase: Literal["measure", "confirm"] = "measure",
) -> E:
    return event_type(
        round=round_num,
        total_rounds=total_rounds,
        target_count=target_count,
        label=label,
        at_ms=at_ms,
        phase=phase,
    )


#: Build the event a bench pass emits as it starts, one target per pass unless overridden.
pass_started = partial(_pass_event, PassStarted)

#: Build the event a bench pass emits as it finishes, one target per pass unless overridden.
pass_finished = partial(_pass_event, PassFinished)
