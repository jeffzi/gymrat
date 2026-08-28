"""Format durations and ETAs for progress display.

The two formatters are deliberately not unified. :func:`format_duration` reports
elapsed time and floors to whole seconds while always keeping a zero remainder
(``"1m 0s"``); :func:`format_eta` reports a forward estimate, rounds to whole
seconds, clamps to at least one second, and drops a zero remainder (``"~1m
left"``).
"""

import math

_SECONDS_PER_MINUTE = 60
_SECONDS_PER_HOUR = 3600


def _hours_minutes_seconds(total_seconds: int) -> tuple[int, int, int]:
    """Split a whole-second count into an hours/minutes/seconds tier."""
    hours = total_seconds // _SECONDS_PER_HOUR
    minutes = (total_seconds % _SECONDS_PER_HOUR) // _SECONDS_PER_MINUTE
    seconds = total_seconds % _SECONDS_PER_MINUTE
    return hours, minutes, seconds


def format_duration(ms: float) -> str:
    """Format an elapsed duration, flooring to whole seconds.

    Uses at most two tiers and always shows a zero remainder in the lower tier
    (``60_000`` renders ``"1m 0s"``, ``3_600_000`` renders ``"1h 00m"``).  The
    hour tier zero-pads the minute remainder to two digits.
    """
    total_seconds = math.floor(max(0.0, ms) / 1000)
    hours, minutes, seconds = _hours_minutes_seconds(total_seconds)

    if total_seconds < _SECONDS_PER_MINUTE:
        return f"{seconds}s"
    if total_seconds < _SECONDS_PER_HOUR:
        return f"{minutes}m {seconds}s"
    return f"{hours}h {minutes:02d}m"


def format_eta(ms: float) -> str:
    """Format a forward time estimate, rounding to whole seconds.

    Clamps to at least one second and drops a zero remainder in the lower tier
    (``60_000`` renders ``"~1m left"``, not ``"~1m 0s left"``).
    """
    total_seconds = max(1, round(ms / 1000))
    hours, minutes, seconds = _hours_minutes_seconds(total_seconds)

    if total_seconds < _SECONDS_PER_MINUTE:
        return f"~{seconds}s left"
    if total_seconds < _SECONDS_PER_HOUR:
        if seconds > 0:
            return f"~{minutes}m {seconds}s left"
        return f"~{minutes}m left"
    if minutes > 0:
        return f"~{hours}h {minutes:02d}m left"
    return f"~{hours}h left"
