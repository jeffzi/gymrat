"""Behavioral tests for the package clock: wall-clock stamps and the duration clock."""

import re
import time
from collections.abc import Callable
from datetime import UTC, datetime

import pytest

from gymrat.clock import format_iso, monotonic_ms, now_iso, now_ms, now_ns

ISO_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$")


def test_format_iso_when_given_utc_datetime_does_render_milliseconds_with_z_suffix():
    moment = datetime(2026, 1, 2, 3, 4, 5, 678_901, tzinfo=UTC)

    result = format_iso(moment)

    assert result == "2026-01-02T03:04:05.678Z"


def test_now_iso_when_called_does_render_millisecond_utc_timestamp():
    result = now_iso()

    assert ISO_PATTERN.match(result)


@pytest.mark.parametrize(
    ("clock", "ns_per_unit"),
    [
        pytest.param(now_ns, 1, id="now_ns"),
        pytest.param(now_ms, 1_000_000, id="now_ms"),
    ],
)
def test_wall_clock_when_called_does_return_epoch_time_in_its_unit(
    clock: Callable[[], int], ns_per_unit: int
):
    before = time.time_ns() // ns_per_unit

    result = clock()

    after = time.time_ns() // ns_per_unit
    assert isinstance(result, int)
    assert before - 1 <= result <= after + 1


def test_monotonic_ms_when_called_does_return_perf_counter_in_milliseconds(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(time, "perf_counter", lambda: 1.5)

    result = monotonic_ms()

    assert result == 1500.0
