"""Tests for the baseline-measurement helper.

``measure_baseline`` runs one measurement through the engine and returns both
the measurement result and a baseline record built from it, without appending
anything to a session log. The engine is replaced with a canned result, so the
record is checked against what the measurement reported, not the target string.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING
from unittest.mock import create_autospec

from gymrat.loop.baseline import measure_baseline
from gymrat.sampling import TargetSpec
from gymrat.session.records import BaselineRecord
from gymrat.utils import monotonic_ms, now_ns
from tests._pipeline import run_options
from tests.loop._probe import install_measure
from tests.report._measurements import create_measurement_result

if TYPE_CHECKING:
    import pytest

_ROUNDS = ({"latency": 5.0}, {"latency": 6.0})


def test_measure_baseline_when_measured_does_build_the_record_from_the_measurement(
    monkeypatch: pytest.MonkeyPatch,
):
    result = create_measurement_result(label="v1.2.0", rounds=_ROUNDS)
    install_measure(monkeypatch, result)
    monkeypatch.setattr(
        "gymrat.utils.monotonic_ms", create_autospec(monotonic_ms, side_effect=[1_000.0, 1_500.0])
    )
    stamp_ns = 1_700_000_000_123_456_789
    monkeypatch.setattr("gymrat.utils.now_ns", create_autospec(now_ns, return_value=stamp_ns))

    measured, record = asyncio.run(
        measure_baseline(TargetSpec(label=None, target="main"), run_options(samples=2))
    )

    assert measured is result
    assert record == BaselineRecord(
        type="baseline", at=stamp_ns, label="v1.2.0", samples=_ROUNDS, duration_ms=500
    )
