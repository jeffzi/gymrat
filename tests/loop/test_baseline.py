"""Tests for the baseline-measurement helper.

``measure_baseline`` runs one measurement through the engine and returns both
the measurement result and a baseline record built from it, without appending
anything to a session log.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from gymrat.config import KindEntry, MetricEntry
from gymrat.loop.baseline import measure_baseline
from gymrat.sampling import TargetSpec
from gymrat.session.records import BaselineRecord
from tests._clock import install_monotonic_clock
from tests._pipeline import run_options
from tests.loop._probe import install_measure, only_call
from tests.report._measurements import create_measurement_result

if TYPE_CHECKING:
    import pytest

    from gymrat.progress_events import ProgressEvent
    from gymrat.sampling import RunOptions

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _run_options() -> RunOptions:
    """Run options the baseline must hand on to the measurement engine unchanged."""
    warnings: list[str] = []
    events: list[ProgressEvent] = []
    return run_options(
        samples=5,
        bench="sh bench.sh",
        prepare="sh prepare.sh",
        timeout_seconds=30,
        on_progress=events.append,
        warn=warnings.append,
        config_metrics={"decode/time": MetricEntry(direction="higher")},
        config_kinds={"memory": KindEntry(gating=False)},
    )


# ---------------------------------------------------------------------------
# measure_baseline returns result and record
# ---------------------------------------------------------------------------


def test_measure_baseline_when_target_label_differs_does_return_the_measurement_with_a_record_labeled_by_it(
    monkeypatch: pytest.MonkeyPatch,
):
    rounds: list[dict[str, float]] = [{"latency": 41}, {"latency": 43}]
    handed_back = create_measurement_result(label="build", rounds=rounds)
    clock = install_monotonic_clock(monkeypatch)
    recorder = install_measure(monkeypatch, handed_back, on_call=lambda: clock.tick(500.0))
    stamp_ns = 1_700_000_000_123_456_789
    monkeypatch.setattr("gymrat.loop.baseline.now_ns", lambda: stamp_ns)
    target = TargetSpec(label="release", target="main")
    run_options = _run_options()

    result, record = asyncio.run(measure_baseline(target, run_options))

    forwarded = only_call(recorder)
    assert forwarded.target == target
    assert forwarded.run == run_options
    assert result is handed_back
    assert record == BaselineRecord(
        type="baseline", at=stamp_ns, label="build", samples=tuple(rounds), duration_ms=500
    )
