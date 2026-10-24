"""The refusals that end the loop: a configured stop condition met, or a spent budget.

The iteration driver raises them, the supervisor and the supervise pre-flight
read the same stop conditions, and the command boundary routes them as a gate
trip rather than as a tool failure.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from gymrat.errors import GymratError

if TYPE_CHECKING:
    from gymrat.config import BenchlessConfig
    from gymrat.session.schema import CommandReason
    from gymrat.session.store import SessionState


_STOP_HINT = "The loop is done. Report what the session measured instead of measuring again."


class LoopStopError(GymratError):
    """A configured stop condition refusing another iteration.

    Separate from a plain :class:`GymratError` because nothing failed: the loop
    ran to the end it was configured for, which the CLI reports as a gate trip
    rather than as a tool failure.
    """

    def __init__(
        self,
        *args: object,
        hint: str | None = None,
        reason: CommandReason | None = "stop-condition",
    ) -> None:
        super().__init__(*args, hint=hint, reason=reason)


class BudgetExceededError(LoopStopError):
    """The session's time budget cannot afford another iteration.

    Raised when a live budget's remaining time is shorter than the estimated
    iterate duration, so the CLI routes it through the same gate-exit path as
    any other stop condition.
    """

    def __init__(
        self,
        *args: object,
        hint: str | None = None,
        reason: CommandReason | None = "budget-exceeded",
    ) -> None:
        super().__init__(*args, hint=hint, reason=reason)


def stop_reason(config: BenchlessConfig, state: SessionState) -> str | None:
    """Which configured stop condition this session has already met, if any.

    Read off the folded log alone, so it settles before a bench command runs: an
    iteration measured past the end of the loop is one the agent would have to
    throw away. ``target_value`` stops the loop only once the target-reaching
    iteration is *kept* — discarding it puts the target back out of reach.

    Args:
        config: The resolved config, carrying the configured stop conditions.
        state: The session's folded state, read for iteration count and
            whether the target has been reached and kept.

    Returns:
        The condition that fired, such as ``max iterations (3 of 3)``, or
        ``None`` when no condition is met yet.
    """
    stop = config.stop
    if stop is None:
        return None
    if stop.max_iterations is not None and state.iteration_count >= stop.max_iterations:
        return f"max iterations ({state.iteration_count} of {stop.max_iterations})"
    if stop.target_value is not None and state.target_reached_and_kept:
        return "target reached and kept"
    return None


def stop_condition(config: BenchlessConfig, state: SessionState) -> LoopStopError | None:
    """The refusal for a configured stop condition this session has already met.

    Args:
        config: The resolved config, carrying the configured stop conditions.
        state: The session's folded state.

    Returns:
        The stop error naming the condition :func:`stop_reason` reports, or
        ``None`` when no condition is met yet.
    """
    reason = stop_reason(config, state)
    if reason is None:
        return None
    return LoopStopError(f"Stop condition met: {reason}", hint=_STOP_HINT)
