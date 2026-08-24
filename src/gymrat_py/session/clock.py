"""The single timestamp format every session log record is stamped with."""

from datetime import UTC, datetime


def now_iso() -> str:
    """The current UTC time as ISO-8601 with millisecond precision and a ``Z`` suffix."""
    now = datetime.now(UTC)
    return now.strftime("%Y-%m-%dT%H:%M:%S.") + f"{now.microsecond // 1000:03d}Z"
