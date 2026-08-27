"""Estimate remaining sampling time and format durations for progress display.

The :class:`EtaTracker` observes the stream of progress steps the sampler emits
and, once it has measured at least one gap between consecutive sample steps,
projects how long the outstanding steps will take. Gaps are pooled into a single
mean across every target, because the machine's per-run cost is shared rather
than per-target. Gaps that touch a prepare step are excluded: a prepare command
runs on a different cost profile and would skew the sample-run estimate.

The two formatters are deliberately not unified. :func:`format_duration` reports
elapsed time and floors to whole seconds while always keeping a zero remainder
(``"1m 0s"``); :func:`format_eta` reports a forward estimate, rounds to whole
seconds, clamps to at least one second, and drops a zero remainder (``"~1m
left"``).
"""

import math
from collections.abc import Callable

from gymrat_py.progress_events import PassStarted, PrepareStarted, ProgressEvent, default_clock

_SECONDS_PER_MINUTE = 60
_SECONDS_PER_HOUR = 3600


class EtaTracker:
    """Project remaining sampling time from observed inter-step gaps.

    Args:
        target_count: The number of targets being sampled. Supplied by the
            caller rather than inferred, since early steps do not reveal it.
        clock: A source of millisecond timestamps, injectable for tests.
            Defaults to a monotonic ``perf_counter``-based clock.
    """

    def __init__(self, target_count: int, clock: Callable[[], float] | None = None) -> None:
        self._target_count = target_count
        self._clock = clock if clock is not None else default_clock
        self._duration_sum = 0.0
        self._duration_count = 0
        self._completed_samples = 0
        self._prev_time: float | None = None
        self._prev_was_prepare = False

    def record(self, event: ProgressEvent) -> float | None:
        """Record a progress event and return an ETA in milliseconds, if known.

        Only ``PassStarted`` events contribute to the estimate. ``PrepareStarted``
        events mark a gap boundary (the next gap is excluded). All other event
        types are ignored — they return ``None`` without affecting gap tracking.
        """
        is_prepare = isinstance(event, PrepareStarted)
        is_pass = isinstance(event, PassStarted)

        if not is_prepare and not is_pass:
            return None

        now = self._clock()

        if self._prev_time is not None and not self._prev_was_prepare and not is_prepare:
            gap = now - self._prev_time
            if gap >= 0:
                self._duration_sum += gap
                self._duration_count += 1

        self._prev_was_prepare = is_prepare
        self._prev_time = now

        if not isinstance(event, PassStarted):
            return None

        remaining = event.total_rounds * self._target_count - self._completed_samples
        self._completed_samples += 1

        if self._duration_count == 0:
            return None

        return (self._duration_sum / self._duration_count) * remaining


def format_duration(ms: float) -> str:
    """Format an elapsed duration, flooring to whole seconds.

    Uses at most two tiers and always shows a zero remainder in the lower tier
    (``60_000`` renders ``"1m 0s"``, ``3_600_000`` renders ``"1h 0m"``).
    """
    ms = max(0.0, ms)
    total_seconds = math.floor(ms / 1000)
    hours = total_seconds // _SECONDS_PER_HOUR
    minutes = (total_seconds % _SECONDS_PER_HOUR) // _SECONDS_PER_MINUTE
    seconds = total_seconds % _SECONDS_PER_MINUTE

    if total_seconds < _SECONDS_PER_MINUTE:
        return f"{seconds}s"
    if total_seconds < _SECONDS_PER_HOUR:
        return f"{minutes}m {seconds}s"
    return f"{hours}h {minutes}m"


def format_eta(ms: float) -> str:
    """Format a forward time estimate, rounding to whole seconds.

    Clamps to at least one second and drops a zero remainder in the lower tier
    (``60_000`` renders ``"~1m left"``, not ``"~1m 0s left"``).
    """
    total_seconds = max(1, round(ms / 1000))
    hours = total_seconds // _SECONDS_PER_HOUR
    minutes = (total_seconds % _SECONDS_PER_HOUR) // _SECONDS_PER_MINUTE
    seconds = total_seconds % _SECONDS_PER_MINUTE

    if total_seconds < _SECONDS_PER_MINUTE:
        return f"~{seconds}s left"
    if total_seconds < _SECONDS_PER_HOUR:
        if seconds > 0:
            return f"~{minutes}m {seconds}s left"
        return f"~{minutes}m left"
    if minutes > 0:
        return f"~{hours}h {minutes}m left"
    return f"~{hours}h left"
