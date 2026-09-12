"""Pure state machine behind the measure/compare progress display.

This module owns every decision the progress bar makes -- which rows are
visible, how many passes are done, what the remaining estimate is, and which
milestone line plain mode prints. It imports nothing from ``rich`` and reads no
clock: ``now`` always comes from the event's own ``at_ms``, so a transition is
fully determined by ``(state, event)``.

:mod:`gymrat.cli.progress` is the shell around it, owning the terminal and the
``rich`` objects.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Self

from gymrat.eta import SamplingEta, format_duration
from gymrat.progress_events import (
    PassFinished,
    PassStarted,
    PrepareFinished,
    PrepareStarted,
    ProgressEvent,
)

__all__ = ["ProgressState", "advance", "plain_line"]


@dataclass(frozen=True, slots=True)
class ProgressState:
    """Everything the progress display needs to paint a frame.

    ``total`` is ``0`` while the pass count is still unknown, which happens when
    no ``--samples`` flag pinned it up front; the first ``PassStarted`` fills it
    in. ``run_start_ms`` and ``run_end_ms`` are ``None`` until the first event
    arrives.
    """

    target_count: int
    sample_count: int | None = None
    total: int = 0
    prepare_start_ms: float = 0.0
    pass_start_ms: float = 0.0
    run_start_ms: float | None = None
    run_end_ms: float | None = None
    eta: SamplingEta = field(default_factory=lambda: SamplingEta.start(0))
    prepare_visible: bool = False
    pass_visible: bool = False
    current_target: str = ""

    @classmethod
    def start(cls, *, target_count: int, sample_count: int | None) -> Self:
        """Build the state a run begins in.

        Args:
            target_count: How many targets (baseline plus candidates) the run covers.
            sample_count: Samples per target, or ``None`` when the total is only
                discovered at runtime from ``PassStarted.total_rounds``.

        Returns:
            A state with nothing visible and the total pre-sized when it is known.
        """
        total = (sample_count or 0) * target_count
        return cls(
            target_count=target_count,
            sample_count=sample_count,
            total=total,
            eta=SamplingEta.start(total),
        )


def _pass_started(state: ProgressState, event: PassStarted) -> ProgressState:
    total = state.total or event.total_rounds * event.target_count
    return replace(
        state,
        total=total,
        eta=state.eta.with_total(total),
        pass_start_ms=event.at_ms,
        pass_visible=True,
        current_target=event.label,
    )


def advance(state: ProgressState, event: ProgressEvent) -> ProgressState:
    """Fold ``event`` into ``state``.

    Args:
        state: The state the run is in before the event.
        event: The event to apply; anything outside the four prepare/pass
            milestones is not a display change.

    Returns:
        The state the display should paint next, or ``state`` itself when the
        event carries nothing the display shows.
    """
    match event:
        case PrepareStarted():
            updated = replace(
                state,
                prepare_start_ms=event.at_ms,
                prepare_visible=True,
                current_target=event.label,
            )
        case PrepareFinished():
            # The prepare row has nothing left to say once sampling starts, so it
            # leaves the display rather than lingering as a completed row.
            updated = replace(state, prepare_visible=False)
        case PassStarted():
            updated = _pass_started(state, event)
        case PassFinished():
            updated = replace(state, eta=state.eta.advanced(event.at_ms - state.pass_start_ms))
        case _:
            return state

    return replace(
        updated,
        run_start_ms=event.at_ms if state.run_start_ms is None else state.run_start_ms,
        run_end_ms=event.at_ms,
    )


def plain_line(before: ProgressState, after: ProgressState, event: ProgressEvent) -> str | None:
    """Render the milestone line plain mode prints for ``event``, without its timestamp.

    Args:
        before: The state before ``event`` was applied.
        after: The state ``advance`` returned for ``event``.
        event: The event being reported.

    Returns:
        The line to print, or ``None`` when the event is not a milestone plain
        mode announces.
    """
    match event:
        case PrepareFinished():
            elapsed = format_duration(event.at_ms - before.prepare_start_ms)
            return f"prepared {event.label} ({elapsed})"
        case PassFinished():
            # Taking the duration from the ETA delta rather than recomputing it
            # keeps the printed number and the bar's estimate from ever disagreeing.
            elapsed = format_duration(after.eta.total_time_ms - before.eta.total_time_ms)
            return f"pass {event.round}/{event.total_rounds} · {event.label} ({elapsed})"
        case _:
            return None
