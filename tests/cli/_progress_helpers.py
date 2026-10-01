"""Shared progress-event builders and renderer factories for the progress renderer tests.

Used by ``test_progress`` and ``iterate/test_progress``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from gymrat.cli.iterate.progress import IterateRenderer
from gymrat.cli.progress import ProgressReporter
from gymrat.progress_events import PassFinished, PassStarted
from tests._rich import Clock

if TYPE_CHECKING:
    from typing import Literal

    from rich.console import Console

    from gymrat.cli.style import ErasableLive
    from gymrat.progress_events import ProgressEvent

__all__ = [
    "LiveRenderer",
    "build_iterate_renderer",
    "build_progress_reporter",
    "ms_from_clock",
    "pass_finished",
    "pass_started",
]


class LiveRenderer(Protocol):
    """The surface both CLI progress renderers share, as the signal tests drive it."""

    @property
    def live(self) -> ErasableLive | None:
        """The active live display, or ``None`` outside live mode or after ``stop()``."""
        ...

    def report(self, event: ProgressEvent) -> None:
        """Fold ``event`` into the display."""
        ...

    def stop(self) -> None:
        """Stop the renderer."""
        ...


def build_progress_reporter(mode: Literal["live", "plain"], console: Console) -> LiveRenderer:
    """Build the measure/compare progress reporter on ``console``.

    Args:
        mode: ``"live"`` for a rich live display, ``"plain"`` for milestone lines.
        console: The console to render to.

    Returns:
        A single-target reporter with a hand-advanced clock.
    """
    return ProgressReporter(
        mode=mode,
        console=console,
        target_count=1,
        sample_count=3,
        clock=Clock(),
        command="measure",
    )


def build_iterate_renderer(mode: Literal["live", "plain"], console: Console) -> LiveRenderer:
    """Build the iterate progress renderer on ``console``.

    Args:
        mode: ``"live"`` for a rich live checklist, ``"plain"`` for milestone lines.
        console: The console to render to.

    Returns:
        A renderer for iteration 1 with a hand-advanced clock.
    """
    return IterateRenderer(
        mode=mode,
        console=console,
        seq=1,
        session_id="test-session",
        sample_count=5,
        metric_count=3,
        primary_metric="geomean",
        clock=Clock(),
    )


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
