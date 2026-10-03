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
from tests.report._measurements import create_measurement_result

if TYPE_CHECKING:
    from gymrat.measure import MeasureOptions
    from gymrat.progress_events import ProgressEvent
    from gymrat.report.types import MeasurementResult

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


def _fake_engine(result: MeasurementResult):
    async def fake_measure(options: object) -> MeasurementResult:
        return result

    return fake_measure


# ---------------------------------------------------------------------------
# measure_baseline returns result and record
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "target_label",
    [
        pytest.param("build", id="target-label-matches-measurement"),
        pytest.param(None, id="target-label-differs-from-measurement"),
    ],
)
def test_measure_baseline_when_called_does_return_result_and_record_with_matching_fields(
    monkeypatch: pytest.MonkeyPatch,
    target_label: str | None,
):
    rounds: list[dict[str, float]] = [{"latency": 41}, {"latency": 43}]
    handed_back = create_measurement_result(label="build", rounds=rounds)
    monkeypatch.setattr("gymrat.measure.measure", _fake_engine(handed_back))
    ticks = iter([1_000.0, 1_500.0])
    monkeypatch.setattr("gymrat.clock.monotonic_ms", lambda: next(ticks))

    result, record = asyncio.run(
        measure_baseline(TargetSpec(label=target_label, target="main"), _run_options())
    )

    assert result is handed_back
    assert isinstance(record, BaselineRecord)
    assert record.label == "build"
    assert record.samples == tuple(rounds)
    assert record.duration_ms == 500


def test_measure_baseline_when_called_does_hand_the_engine_the_target_and_run_options(
    monkeypatch: pytest.MonkeyPatch,
):
    captured: list[MeasureOptions] = []

    async def spy_measure(options: MeasureOptions) -> MeasurementResult:
        captured.append(options)
        return create_measurement_result(label="build")

    monkeypatch.setattr("gymrat.measure.measure", spy_measure)
    target = TargetSpec(label="build", target="main")
    run_options = _run_options()

    asyncio.run(measure_baseline(target, run_options))

    assert len(captured) == 1
    assert captured[0].target == target
    assert captured[0].run is run_options
