"""Shared probe test doubles: the recorded baseline, the canned measurement, and the engine stand-in.

The CLI command tests replace the measurement engine that shells out to the
consumer's bench script, and the probe engine tests read the same recorded
baseline, so the pieces live here once.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import create_autospec

from gymrat.loop.baseline import measure
from tests.report._measurements import create_measurement_result, measured_metric

if TYPE_CHECKING:
    import pytest

    from gymrat.loop.baseline import MeasureOptions
    from gymrat.report.types import MeasurementResult

#: The rounds the recorded baseline reports, whose ``total_ms`` median is 100.
BASELINE_SAMPLES: tuple[dict[str, float], ...] = ({"total_ms": 98.0}, {"total_ms": 102.0})


def measurement(*, adapter: str = "mitata") -> MeasurementResult:
    """A measurement of the experiment worktree, reporting ``total_ms`` at 90."""
    return create_measurement_result(
        label="experiment",
        adapter=adapter,
        metrics={"total_ms": measured_metric(median=90.0, spread=2.0, short_name="total_ms")},
    )


class MeasureRecorder:
    """A stand-in for the measurement engine that records every call it answers.

    The engine is the one boundary a probe crosses into the consumer's bench
    script, so it is replaced wholesale. The recorder keeps the options each call
    passed in ``calls``, which is how a test reads the target, bench command, and
    sample count a probe asked for, and what the caller wired its progress
    callback to.

    Args:
        result: The measurement every call hands back.
    """

    def __init__(self, result: MeasurementResult) -> None:
        self.result = result
        self.calls: list[MeasureOptions] = []

    async def __call__(self, options: MeasureOptions) -> MeasurementResult:
        self.calls.append(options)
        return self.result


def install_measure(
    monkeypatch: pytest.MonkeyPatch,
    result: MeasurementResult,
) -> MeasureRecorder:
    """Replace ``gymrat.loop.baseline.measure`` with a recorder answering ``result``.

    The stand-in keeps the real engine's signature, so a call the real
    ``measure`` would reject fails the test.

    Args:
        monkeypatch: The test's monkeypatch fixture.
        result: The measurement every call hands back.

    Returns:
        The installed recorder.
    """
    recorder = MeasureRecorder(result)
    monkeypatch.setattr(
        "gymrat.loop.baseline.measure", create_autospec(measure, side_effect=recorder.__call__)
    )
    return recorder
