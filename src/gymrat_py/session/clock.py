"""The single timestamp format every session log record is stamped with."""

import time
from datetime import UTC, datetime


def format_iso(dt: datetime) -> str:
    """``dt`` as ISO-8601 with millisecond precision and a ``Z`` suffix."""
    return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{dt.microsecond // 1000:03d}Z"


def now_iso() -> str:
    """The current UTC time as ISO-8601 with millisecond precision and a ``Z`` suffix."""
    return format_iso(datetime.now(UTC))


def now_ms() -> int:
    """Milliseconds since the epoch, the unit every session event stamps with."""
    return int(time.time() * 1000)
