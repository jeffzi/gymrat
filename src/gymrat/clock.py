"""Clocks for session-log timestamps and duration measurement.

:func:`now_ns` is the unit every log record and event stamps with.
:func:`now_ms` and :func:`monotonic_ms` serve budgets, dashboards, and durations.
"""

import time

from gymrat.utils import MS_PER_SECOND


def now_ns() -> int:
    """Nanoseconds since the epoch, the unit every log record and event stamps with."""
    return time.time_ns()


def now_ms() -> int:
    """Milliseconds since the epoch, for budgets, deadlines, and log-file names."""
    return int(time.time() * MS_PER_SECOND)


def monotonic_ms() -> float:
    """Milliseconds from an arbitrary, ever-increasing reference point.

    Unaffected by system clock adjustments (NTP corrections, DST shifts), so
    bracketing a measurement's start and end with this instead of
    :func:`now_ms` cannot yield a skewed or negative duration.

    Returns:
        Milliseconds as a float, comparable only with other
        :func:`monotonic_ms` readings.
    """
    return time.perf_counter() * MS_PER_SECOND
