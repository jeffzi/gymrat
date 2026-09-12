"""Shared probe test doubles: the recorded baseline, the canned measurement, and the engine stand-in.

Both the ``probe_session`` engine tests and the ``gymrat probe`` command tests
replace the same boundary — the measurement engine that shells out to the
consumer's bench script — and pair the same recorded baseline against the same
canned measurement, so the pieces live here once.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from gymrat.session import BaselineRecord
from tests.report._inputs import create_measurement_result, measured_metric
from tests.session.records._fixtures import AT

if TYPE_CHECKING:
    from collections.abc import Sequence

    import pytest

    from gymrat.measure import MeasureOptions
    from gymrat.progress_events import ProgressEvent
    from gymrat.report.types import MeasurementResult, MetricMeasurement

#: The rounds the recorded baseline reports, whose ``total_ms`` median is 100.
BASELINE_SAMPLES: tuple[dict[str, float], ...] = ({"total_ms": 98.0}, {"total_ms": 102.0})


def baseline_of(samples: tuple[dict[str, float], ...] = BASELINE_SAMPLES) -> BaselineRecord:
    """A recorded baseline of the experiment worktree over ``samples``."""
    return BaselineRecord(type="baseline", at=AT, label="experiment", samples=samples)


def measurement(
    metrics: dict[str, MetricMeasurement] | None = None, *, adapter: str = "mitata"
) -> MeasurementResult:
    """A measurement of the experiment worktree, reporting ``total_ms`` at 90 by default."""
    return create_measurement_result(
        label="experiment",
        adapter=adapter,
        metrics=metrics
        if metrics is not None
        else {"total_ms": measured_metric(median=90.0, spread=2.0, short_name="total_ms")},
    )


class MeasureRecorder:
    """A stand-in for the measurement engine that records every call it answers.

    The engine is the one boundary a probe crosses into the consumer's bench
    script, so it is replaced wholesale: the recorder hands back a canned
    :class:`MeasurementResult` and keeps the options it was called with, which is
    how a test reads the target, bench command, and sample count a probe asked
    for. When ``progress`` events are given, each call reports them through the
    progress callback it was handed, so a test can read what the caller wired
    that callback to.
    """

    def __init__(self, result: MeasurementResult, progress: Sequence[ProgressEvent] = ()) -> None:
        self.result = result
        self.progress = tuple(progress)
        self.calls: list[MeasureOptions] = []

    async def __call__(self, options: MeasureOptions) -> MeasurementResult:
        self.calls.append(options)
        if options.on_progress is not None:
            for event in self.progress:
                options.on_progress(event)
        return self.result


def install_measure(
    monkeypatch: pytest.MonkeyPatch,
    result: MeasurementResult,
    *,
    progress: Sequence[ProgressEvent] = (),
) -> MeasureRecorder:
    """Replace ``gymrat.measure.measure`` with a recorder answering ``result``."""
    recorder = MeasureRecorder(result, progress)
    monkeypatch.setattr("gymrat.measure.measure", recorder)
    return recorder


def only_call(recorder: MeasureRecorder) -> MeasureOptions:
    """The options of the single measure call ``recorder`` answered."""
    assert len(recorder.calls) == 1, f"expected one measure call, got {len(recorder.calls)}"
    return recorder.calls[0]
