"""Typed progress events emitted during a sampling run.

Each event is a frozen dataclass stamped with ``at_ms`` — a monotonic
millisecond timestamp provided by the emitter's clock. The ``ProgressEvent``
union and ``ProgressCallback`` alias give consumers a typed contract without
coupling them to the sampler's internals.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal


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
    """Emitted after a bench command completes for one round against a target.

    Attributes:
        round: 1-based round number.
        total_rounds: Total number of rounds in the schedule.
        target_count: Number of targets in the schedule.
        label: The target's display label.
        phase: Whether this pass was a measurement or confirmation run.
        at_ms: Monotonic millisecond timestamp from the emitter's clock.
    """

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
    """Emitted after the judge evaluates an iteration's samples.

    Attributes:
        primary_delta_pct: The primary metric's delta as a percentage, or
            ``None`` when the iteration has no primary metric.
        regressed: Names of the metrics that regressed this iteration.
        metric_count: Total number of metrics the judge evaluated.
        at_ms: Monotonic millisecond timestamp from the emitter's clock.
    """

    primary_delta_pct: float | None
    regressed: tuple[str, ...]
    metric_count: int
    at_ms: float


@dataclass(frozen=True, slots=True)
class ConfirmStarted:
    """Emitted before a confirmation pass begins.

    Attributes:
        filtered_metrics: The metric names the confirmation pass is restricted
            to, or ``None`` when the full suite is rerun.
        at_ms: Monotonic millisecond timestamp from the emitter's clock.
    """

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
"""The union of progress events a sampling run can emit."""

type ProgressCallback = Callable[[ProgressEvent], None]
"""A callback a run notifies with each emitted :data:`ProgressEvent`."""


def emit_progress(on_progress: ProgressCallback | None, event: ProgressEvent) -> None:
    """Fire ``on_progress`` with ``event`` when a callback is registered."""
    if on_progress is not None:
        on_progress(event)
