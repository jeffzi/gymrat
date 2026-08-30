"""Format durations and ETAs for progress display.

The formatters are deliberately not unified. :func:`format_duration` reports
elapsed time and floors to whole seconds while always keeping a zero remainder
(``"1m 0s"``); :func:`format_eta` reports a forward estimate, rounds to whole
seconds, clamps to at least one second, and drops a zero remainder (``"~1m
left"``); :func:`format_clock` renders the media-player clock a progress bar
ticks through (``"07:45"``, ``"1:07:45"``).
"""

import math

#: Shared by every module that converts between milliseconds and clock tiers,
#: so the conversion factors are declared once.
SECONDS_PER_MINUTE = 60
SECONDS_PER_HOUR = 3600
MS_PER_SECOND = 1000


def _hours_minutes_seconds(total_seconds: int) -> tuple[int, int, int]:
    """Split a whole-second count into an hours/minutes/seconds tier."""
    hours = total_seconds // SECONDS_PER_HOUR
    minutes = (total_seconds % SECONDS_PER_HOUR) // SECONDS_PER_MINUTE
    seconds = total_seconds % SECONDS_PER_MINUTE
    return hours, minutes, seconds


def _floored_whole_seconds(ms: float) -> int:
    """Clamp a negative elapsed input to zero, then floor to whole seconds."""
    return math.floor(max(0.0, ms) / MS_PER_SECOND)


def format_duration(ms: float) -> str:
    """Format an elapsed duration, flooring to whole seconds.

    Uses at most two tiers and always shows a zero remainder in the lower tier
    (``60_000`` renders ``"1m 0s"``, ``3_600_000`` renders ``"1h 00m"``).  The
    hour tier zero-pads the minute remainder to two digits.
    """
    total_seconds = _floored_whole_seconds(ms)
    hours, minutes, seconds = _hours_minutes_seconds(total_seconds)

    if total_seconds < SECONDS_PER_MINUTE:
        return f"{seconds}s"
    if total_seconds < SECONDS_PER_HOUR:
        return f"{minutes}m {seconds}s"
    return f"{hours}h {minutes:02d}m"


def format_timestamp(at_ms: float, run_start_ms: float | None) -> str:
    """Format an elapsed timestamp as ``[HH:MM:SS]`` since ``run_start_ms``.

    Falls back to zero elapsed when ``run_start_ms`` is ``None`` (the run has not
    yet been anchored), matching how each caller anchors its own run start.
    """
    start_ms = at_ms if run_start_ms is None else run_start_ms
    elapsed_ms = at_ms - start_ms
    total_seconds = _floored_whole_seconds(elapsed_ms)
    hours, minutes, seconds = _hours_minutes_seconds(total_seconds)
    return f"[{hours:02d}:{minutes:02d}:{seconds:02d}]"


def format_clock(ms: float) -> str:
    """Format a duration as a media-player clock, flooring to whole seconds.

    Minutes always take two digits (``"00:09"``, ``"07:45"``); the hour tier
    appears only when there are whole hours (``"1:07:45"``).
    """
    total_seconds = _floored_whole_seconds(ms)
    hours, minutes, seconds = _hours_minutes_seconds(total_seconds)

    if hours == 0:
        return f"{minutes:02d}:{seconds:02d}"
    return f"{hours}:{minutes:02d}:{seconds:02d}"


def format_eta(ms: float) -> str:
    """Format a forward time estimate, rounding to whole seconds.

    Clamps to at least one second and drops a zero remainder in the lower tier
    (``60_000`` renders ``"~1m left"``, not ``"~1m 0s left"``).
    """
    total_seconds = max(1, round(ms / 1000))
    hours, minutes, seconds = _hours_minutes_seconds(total_seconds)

    if total_seconds < SECONDS_PER_MINUTE:
        return f"~{seconds}s left"
    if total_seconds < SECONDS_PER_HOUR:
        if seconds > 0:
            return f"~{minutes}m {seconds}s left"
        return f"~{minutes}m left"
    if minutes > 0:
        return f"~{hours}h {minutes:02d}m left"
    return f"~{hours}h left"
