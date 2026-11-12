"""Tests for the baseline-measurement helper.

``measure_baseline`` runs one measurement through the engine and returns both
the measurement result and a baseline record built from it, without appending
anything to a session log. The test runs the real engine against a scratch
repository's committed bench, faking only the clocks.
"""

from __future__ import annotations

import asyncio
import sys
from typing import TYPE_CHECKING
from unittest.mock import create_autospec

from gymrat.loop.baseline import measure_baseline
from gymrat.sampling import TargetSpec
from gymrat.session.records import BaselineRecord
from gymrat.utils import monotonic_ms, now_ns
from tests._pipeline import run_options
from tests.loop._bench import BASELINE_LATENCY, BENCH_FILE, commit_project

if TYPE_CHECKING:
    import pytest

_SAMPLES = 2


def test_measure_baseline_when_target_unlabelled_does_label_the_record_by_the_resolved_ref(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    commit_project(repo, samples=_SAMPLES)
    monkeypatch.setattr(
        "gymrat.utils.monotonic_ms", create_autospec(monotonic_ms, side_effect=[1_000.0, 1_500.0])
    )
    stamp_ns = 1_700_000_000_123_456_789
    monkeypatch.setattr("gymrat.utils.now_ns", create_autospec(now_ns, return_value=stamp_ns))
    options = run_options(
        samples=_SAMPLES, bench=f"{sys.executable} {BENCH_FILE}", prepare=None, timeout_seconds=60
    )

    _, record = asyncio.run(measure_baseline(TargetSpec(label=None, target="main"), options))

    assert record == BaselineRecord(
        type="baseline",
        at=stamp_ns,
        label="main",
        samples=({"latency": BASELINE_LATENCY},) * _SAMPLES,
        duration_ms=500,
    )
