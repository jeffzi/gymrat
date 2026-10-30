"""Behavioral tests for the package clock: wall-clock stamps and the duration clock."""

import time

import pytest

from gymrat.utils import monotonic_ms, now_ms


def test_now_ms_when_called_does_return_epoch_milliseconds(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(time, "time", lambda: 1.5)

    result = now_ms()

    assert result == 1500


def test_monotonic_ms_when_called_does_return_perf_counter_in_milliseconds(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(time, "perf_counter", lambda: 1.5)

    result = monotonic_ms()

    assert result == 1500.0
