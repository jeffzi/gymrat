"""Tests for the baseline-measurement helper.

``measure_baseline`` runs one measurement through the engine and returns both
the measurement result and a baseline record built from it, without appending
anything to a session log.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import pytest

from gymrat.config import KindEntry, MetricEntry
from gymrat.loop.baseline import measure_baseline
from gymrat.sampling import RunOptions, SamplingOptions, TargetSpec
from gymrat.session.records import BaselineRecord
from tests.loop._probe import install_measure, only_call
from tests.report._measurements import create_measurement_result

if TYPE_CHECKING:
    from gymrat.progress_events import ProgressEvent

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _run_options() -> RunOptions:
    """Run options with every field set away from its default, so a dropped field shows."""
    warnings: list[str] = []
    events: list[ProgressEvent] = []
    return RunOptions(
        sampling=SamplingOptions(
            bench="sh bench.sh",
            prepare="sh prepare.sh",
            samples=5,
            timeout_seconds=30,
            on_progress=events.append,
            warn=warnings.append,
        ),
        adapter="metric-lines",
        config_metrics={"decode/time": MetricEntry(direction="higher")},
        config_kinds={"memory": KindEntry(gating=False)},
    )


# ---------------------------------------------------------------------------
# measure_baseline returns result and record
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "target_label",
    [
        pytest.param("build", id="target-label-matches-measurement"),
        pytest.param("release", id="target-label-differs-from-measurement"),
    ],
)
def test_measure_baseline_when_target_given_does_return_the_measurement_with_its_baseline_record(
    monkeypatch: pytest.MonkeyPatch,
    target_label: str,
):
    rounds: list[dict[str, float]] = [{"latency": 41}, {"latency": 43}]
    handed_back = create_measurement_result(label="build", rounds=rounds)
    recorder = install_measure(monkeypatch, handed_back)
    ticks = iter([1_000.0, 1_500.0])
    monkeypatch.setattr("gymrat.clock.monotonic_ms", lambda: next(ticks))
    stamp_ns = 1_700_000_000_123_456_789
    monkeypatch.setattr("gymrat.loop.baseline.now_ns", lambda: stamp_ns)
    target = TargetSpec(label=target_label, target="main")
    run_options = _run_options()

    result, record = asyncio.run(measure_baseline(target, run_options))

    forwarded = only_call(recorder)
    assert forwarded.target == target
    assert forwarded.run is run_options
    assert result is handed_back
    assert record == BaselineRecord(
        type="baseline", at=stamp_ns, label="build", samples=tuple(rounds), duration_ms=500
    )
