"""Run one measurement and build a baseline record from it.

The helper keeps the loop layer free of CLI imports: callers wire the progress
reporter and config resolution themselves, hand in a ``RunOptions``, and get
back both the measurement result and a ready-to-append baseline record.
Appending to the session log is the caller's responsibility so a measurement
without ``--record`` never touches the log.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from gymrat import clock as _clock
from gymrat.clock import now_ns
from gymrat.session.records import BaselineRecord

if TYPE_CHECKING:
    from gymrat.report.types import MeasurementResult
    from gymrat.sampling import RunOptions, TargetSpec


async def measure_baseline(
    target: TargetSpec,
    run_options: RunOptions,
) -> tuple[MeasurementResult, BaselineRecord]:
    """Measure ``target`` once and return the result with a baseline record.

    The engine is imported lazily so importing this module never pulls the
    heavy measurement stack.

    Args:
        target: The revision or directory to measure.
        run_options: Bench command, adapter, sample count, and config overrides.

    Returns:
        A ``(result, record)`` pair. The record carries the measurement's label,
        every round it collected, and the wall-clock duration of the engine call.
        Nothing is appended to a session log.

    Raises:
        GymratError: When the adapter is unknown or the target is neither a
            directory nor a resolvable ref. When the worktree cleanup also
            failed, the message carries its diagnostics.
        CommandError: When a prepare or bench command times out or exits
            non-zero.
    """
    from gymrat import (  # noqa: PLC0415 -- lazy import keeps CLI startup off the heavy measurement stack
        measure as engine,
    )

    options = engine.MeasureOptions(run=run_options, target=target)

    start = _clock.monotonic_ms()
    result = await engine.measure(options)
    duration_ms = int(_clock.monotonic_ms() - start)

    record = BaselineRecord(
        type="baseline",
        at=now_ns(),
        label=result.label,
        samples=tuple(result.rounds),
        duration_ms=duration_ms,
    )
    return result, record
