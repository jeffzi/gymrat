"""A hand-advanced monotonic clock for the tests that pin elapsed milliseconds."""

from unittest.mock import create_autospec

import pytest

from gymrat.clock import monotonic_ms
from tests._rich import Clock


def install_monotonic_clock(
    monkeypatch: pytest.MonkeyPatch, start_ms: float = 1_000.0
) -> Clock[float]:
    """Stand a clock still in place of ``gymrat.clock.monotonic_ms``; it moves only on ``tick``.

    Args:
        monkeypatch: The fixture the clock is patched through.
        start_ms: The reading the clock starts at, in milliseconds.

    Returns:
        The installed clock, for the test to advance.
    """
    clock = Clock(start_ms)
    monkeypatch.setattr(
        "gymrat.clock.monotonic_ms", create_autospec(monotonic_ms, side_effect=clock)
    )
    return clock
