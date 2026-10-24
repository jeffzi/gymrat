"""Shared probe test doubles: the recorded baseline, the canned measurement, and the engine stand-in.

The ``probe_session`` and ``measure_baseline`` engine tests and the CLI session
command tests replace the same boundary — the measurement engine that shells out
to the consumer's bench script — and the probe tests pair the same recorded
baseline against the same canned measurement, so the pieces live here once.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import create_autospec

from gymrat.measure import measure
from tests.report._measurements import create_measurement_result, measured_metric

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    import pytest

    from gymrat.measure import MeasureOptions
    from gymrat.progress_events import ProgressEvent
    from gymrat.report.types import MeasurementResult, MetricMeasurement

#: The rounds the recorded baseline reports, whose ``total_ms`` median is 100.
BASELINE_SAMPLES: tuple[dict[str, float], ...] = ({"total_ms": 98.0}, {"total_ms": 102.0})


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
    script, so it is replaced wholesale. The recorder keeps the options each call
    passed in ``calls``, which is how a test reads the target, bench command, and
    sample count a probe asked for, and what the caller wired its progress
    callback and warn sink to.

    Args:
        result: The measurement every call hands back.
        progress: Events each call reports through the progress callback it was handed.
        warnings: Messages each call sends through the warn sink it was handed.
        on_call: Run at the start of each call, before anything is recorded: where
            a test advances a clock to give the measurement a duration.
    """

    def __init__(
        self,
        result: MeasurementResult,
        progress: Sequence[ProgressEvent] = (),
        warnings: Sequence[str] = (),
        on_call: Callable[[], None] | None = None,
    ) -> None:
        self.result = result
        self.progress = tuple(progress)
        self.warnings = tuple(warnings)
        self.on_call = on_call
        self.calls: list[MeasureOptions] = []

    async def __call__(self, options: MeasureOptions) -> MeasurementResult:
        if self.on_call is not None:
            self.on_call()
        self.calls.append(options)
        sampling = options.run.sampling
        if sampling.on_progress is not None:
            for event in self.progress:
                sampling.on_progress(event)
        for message in self.warnings:
            sampling.warn(message)
        return self.result


def install_measure(
    monkeypatch: pytest.MonkeyPatch,
    result: MeasurementResult,
    *,
    progress: Sequence[ProgressEvent] = (),
    warnings: Sequence[str] = (),
    on_call: Callable[[], None] | None = None,
) -> MeasureRecorder:
    """Replace ``gymrat.measure.measure`` with a recorder answering ``result``.

    The stand-in keeps the real engine's signature, so a call the real
    ``measure`` would reject fails the test.

    Args:
        monkeypatch: The test's monkeypatch fixture.
        result: The measurement every call hands back.
        progress: Events each call reports through the progress callback it was handed.
        warnings: Messages each call sends through the warn sink it was handed.
        on_call: Run at the start of each call, before anything is recorded.

    Returns:
        The installed recorder.
    """
    recorder = MeasureRecorder(result, progress, warnings, on_call)
    monkeypatch.setattr(
        "gymrat.measure.measure", create_autospec(measure, side_effect=recorder.__call__)
    )
    return recorder


def only_call(recorder: MeasureRecorder) -> MeasureOptions:
    """The options of the single measure call ``recorder`` answered."""
    assert len(recorder.calls) == 1, f"expected one measure call, got {len(recorder.calls)}"
    return recorder.calls[0]
