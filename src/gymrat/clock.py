"""Clocks for session-log timestamps and duration measurement.

:func:`now_ns` is the unit every log record and event stamps with.
:func:`now_iso` stays only for the lock-holder record.
:func:`now_ms` and :func:`monotonic_ms` serve budgets, dashboards, and durations.
"""

import time
from datetime import UTC, datetime

from gymrat.eta import MS_PER_SECOND


def now_iso() -> str:
    """The current UTC time as ISO-8601 with millisecond precision and a ``Z`` suffix."""
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


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
        Wall-clock-independent milliseconds suitable for elapsed-time
        measurement.
    """
    return time.perf_counter() * MS_PER_SECOND
