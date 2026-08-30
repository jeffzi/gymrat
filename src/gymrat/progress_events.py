"""Typed progress events emitted during a sampling run.

Each event is a frozen dataclass stamped with ``at_ms`` — a monotonic
millisecond timestamp provided by the emitter's clock. The ``ProgressEvent``
union and ``ProgressCallback`` alias give consumers a typed contract without
coupling them to the sampler's internals.
"""

import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

logger = logging.getLogger(__name__)


def default_clock() -> float:
    """Return a monotonic millisecond timestamp.

    ``perf_counter`` is monotonic, so an NTP correction or DST shift on the wall
    clock cannot make a gap appear negative or inflate an estimate.
    """
    return time.perf_counter() * 1000


@dataclass(frozen=True, slots=True)
class PrepareStarted:
    """Emitted before a prepare command runs for a target."""

    label: str
    at_ms: float


@dataclass(frozen=True, slots=True)
class PrepareFinished:
    """Emitted after a prepare command completes for a target."""

    label: str
    at_ms: float


@dataclass(frozen=True, slots=True)
class PassStarted:
    """Emitted before a bench command runs for one round against a target.

    Attributes:
        round: 1-based round number.
        total_rounds: Total number of rounds in the schedule.
        target_count: Number of targets in the schedule.
        label: The target's display label.
        phase: Whether this pass is a measurement or confirmation run.
        at_ms: Monotonic millisecond timestamp from the emitter's clock.
    """

    round: int
    total_rounds: int
    target_count: int
    label: str
    at_ms: float
    phase: Literal["measure", "confirm"] = "measure"


@dataclass(frozen=True, slots=True)
class PassFinished:
    """Emitted after a bench command completes for one round against a target."""

    round: int
    total_rounds: int
    target_count: int
    label: str
    at_ms: float
    phase: Literal["measure", "confirm"] = "measure"


@dataclass(frozen=True, slots=True)
class HookStarted:
    """Emitted before a lifecycle hook runs."""

    stage: Literal["before", "after"]
    at_ms: float


@dataclass(frozen=True, slots=True)
class HookFinished:
    """Emitted after a lifecycle hook completes."""

    stage: Literal["before", "after"]
    at_ms: float


@dataclass(frozen=True, slots=True)
class JudgeStarted:
    """Emitted before the judge evaluates an iteration's samples."""

    at_ms: float


@dataclass(frozen=True, slots=True)
class JudgeFinished:
    """Emitted after the judge evaluates an iteration's samples."""

    primary_delta_pct: float | None
    regressed: tuple[str, ...]
    metric_count: int
    at_ms: float


@dataclass(frozen=True, slots=True)
class ConfirmStarted:
    """Emitted before a confirmation pass begins."""

    filtered_metrics: tuple[str, ...] | None
    at_ms: float


@dataclass(frozen=True, slots=True)
class ConfirmFinished:
    """Emitted after a confirmation pass completes."""

    reproduced: bool
    at_ms: float


@dataclass(frozen=True, slots=True)
class IterationRecorded:
    """Emitted when an iteration's outcome is recorded."""

    seq: int
    outcome: str
    at_ms: float


type ProgressEvent = (
    PrepareStarted
    | PrepareFinished
    | PassStarted
    | PassFinished
    | HookStarted
    | HookFinished
    | JudgeStarted
    | JudgeFinished
    | ConfirmStarted
    | ConfirmFinished
    | IterationRecorded
)

type ProgressCallback = Callable[[ProgressEvent], None]


def emit_progress(on_progress: ProgressCallback | None, event: ProgressEvent) -> None:
    """Fire ``on_progress`` with ``event`` when a callback is registered."""
    if on_progress is not None:
        on_progress(event)


def create_fan_out(subscribers: Sequence[ProgressCallback]) -> ProgressCallback:
    """Fan out each event to every subscriber, isolating failures.

    One subscriber failing (raising an exception) never silences the others:
    exceptions are logged and swallowed so the remaining subscribers always run.
    """
    subs = list(subscribers)

    def fan_out(event: ProgressEvent) -> None:
        for subscriber in subs:
            try:
                subscriber(event)
            except Exception:
                logger.exception("fan-out subscriber failed")

    return fan_out
