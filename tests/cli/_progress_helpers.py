"""Shared progress-event builders for the progress renderer and progress state tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

from gymrat.progress_events import PassFinished, PassStarted

if TYPE_CHECKING:
    from typing import Literal

    from tests._rich import Clock


def ms_from_clock(clock: Clock) -> int:
    """Return the clock's current time in milliseconds, for ``at_ms`` fields."""
    return int(clock.now * 1000)


def pass_started(
    round_num: int,
    total_rounds: int,
    *,
    at_ms: int,
    target_count: int = 1,
    label: str = "bench",
    phase: Literal["measure", "confirm"] = "measure",
) -> PassStarted:
    return PassStarted(
        round=round_num,
        total_rounds=total_rounds,
        target_count=target_count,
        label=label,
        at_ms=at_ms,
        phase=phase,
    )


def pass_finished(
    round_num: int,
    total_rounds: int,
    *,
    at_ms: int,
    target_count: int = 1,
    label: str = "bench",
    phase: Literal["measure", "confirm"] = "measure",
) -> PassFinished:
    return PassFinished(
        round=round_num,
        total_rounds=total_rounds,
        target_count=target_count,
        label=label,
        at_ms=at_ms,
        phase=phase,
    )
