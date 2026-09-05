"""The single timestamp format every session log record is stamped with.

Also provides a monotonic clock for measuring durations.
"""

import time
from datetime import UTC, datetime


def format_iso(dt: datetime) -> str:
    """A UTC-aware ``dt`` as ISO-8601 with millisecond precision and a ``Z`` suffix."""
    return dt.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def now_iso() -> str:
    """The current UTC time as ISO-8601 with millisecond precision and a ``Z`` suffix."""
    return format_iso(datetime.now(UTC))


def now_ms() -> int:
    """Milliseconds since the epoch, the unit every session event stamps with."""
    return int(time.time() * 1000)


def monotonic_ms() -> float:
    """Milliseconds from an arbitrary, ever-increasing reference point.

    Unaffected by system clock adjustments (NTP corrections, DST shifts), so
    bracketing a measurement's start and end with this instead of
    :func:`now_ms` cannot yield a skewed or negative duration.
    """
    return time.perf_counter() * 1000
