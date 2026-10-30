"""Tests for the baseline-measurement helper.

``measure_baseline`` runs one measurement through the engine and returns both
the measurement result and a baseline record built from it, without appending
anything to a session log.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING
from unittest.mock import create_autospec

from gymrat.loop.baseline import measure_baseline
from gymrat.sampling import TargetSpec
from gymrat.session.records import BaselineRecord
from gymrat.utils import now_ns
from tests._clock import install_monotonic_clock
from tests._pipeline import run_options
from tests.loop._probe import install_measure
from tests.report._measurements import create_measurement_result

if TYPE_CHECKING:
    import pytest


def test_measure_baseline_when_target_label_differs_does_label_the_record_by_the_measurement(
    monkeypatch: pytest.MonkeyPatch,
):
    rounds: list[dict[str, float]] = [{"latency": 41}, {"latency": 43}]
    handed_back = create_measurement_result(label="build", rounds=rounds)
    clock = install_monotonic_clock(monkeypatch)
    install_measure(monkeypatch, handed_back, on_call=lambda: clock.tick(500.0))
    stamp_ns = 1_700_000_000_123_456_789
    monkeypatch.setattr("gymrat.utils.now_ns", create_autospec(now_ns, return_value=stamp_ns))
    target = TargetSpec(label="release", target="main")

    _, record = asyncio.run(measure_baseline(target, run_options(samples=5)))

    assert record == BaselineRecord(
        type="baseline", at=stamp_ns, label="build", samples=tuple(rounds), duration_ms=500
    )
