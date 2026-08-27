"""Typed progress events emitted during a sampling run.

Each event is a frozen dataclass stamped with ``at_ms`` — a monotonic
millisecond timestamp provided by the emitter's clock. The ``ProgressEvent``
union and ``ProgressCallback`` alias give consumers a typed contract without
coupling them to the sampler's internals.
"""

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal


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
        target_index: 0-based position of the target in the target list.
        target_count: Number of targets in the schedule.
        label: The target's display label.
        phase: Whether this pass is a measurement or confirmation run.
        at_ms: Monotonic millisecond timestamp from the emitter's clock.
    """

    round: int
    total_rounds: int
    target_index: int
    target_count: int
    label: str
    at_ms: float
    phase: Literal["measure", "confirm"] = "measure"


@dataclass(frozen=True, slots=True)
class PassFinished:
    """Emitted after a bench command completes for one round against a target."""

    round: int
    total_rounds: int
    target_index: int
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
class JudgeFinished:
    """Emitted after the judge evaluates an iteration's samples."""

    primary_delta_pct: float | None
    regressed: tuple[str, ...]
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
    | JudgeFinished
    | ConfirmStarted
    | ConfirmFinished
    | IterationRecorded
)

type ProgressCallback = Callable[[ProgressEvent], None]
