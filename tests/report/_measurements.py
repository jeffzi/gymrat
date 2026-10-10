"""Measurement builders for report formatting tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

from gymrat.config import KindEntry
from gymrat.report.types import (
    MeasurementResult,
    MetricMeasurement,
)
from tests._pipeline import CLEAN_RESULT
from tests.report._verdicts import metric_meta

if TYPE_CHECKING:
    from collections.abc import Sequence

    from gymrat.model import MetricUnit
    from gymrat.sampling import CleanupResult


def measured_metric(
    *,
    median: float = 100.0,
    spread: float | None = 1.0,
    short_name: str = "time",
    kind: str = "other",
    unit: MetricUnit | None = None,
    gating: bool = True,
) -> MetricMeasurement:
    """One metric of a single-target run: what it measured, and how steady it was.

    Args:
        median: The metric's median.
        spread: The run-to-run spread as a percentage of the median. ``None``
            pins the single-sample case, where there is no jitter to report.
        short_name: The metric's display name.
        kind: The metric's kind.
        unit: The metric's unit, if any.
        gating: Whether the metric gates the run.

    Returns:
        The metric measurement.
    """
    return MetricMeasurement(
        median=median,
        spread=spread,
        meta=metric_meta(short_name, kind=kind, unit=unit, gating=gating),
    )


def create_measurement_result(
    *,
    label: str = "main",
    samples: int = 10,
    adapter: str = "mitata",
    metrics: dict[str, MetricMeasurement] | None = None,
    rounds: Sequence[dict[str, float]] = (),
    config_kinds: dict[str, KindEntry] | None = None,
    cleanup: CleanupResult = CLEAN_RESULT,
) -> MeasurementResult:
    """A measurement of a single-target run, clean and without metrics unless overridden.

    Args:
        label: The target's display label.
        samples: How many samples the run collected.
        adapter: The adapter that parsed the bench output.
        metrics: The measured metrics keyed by name; ``None`` means none.
        rounds: What each round reported, as metric name to value, in run order.
        config_kinds: The config's ``kinds`` section, when it has one.
        cleanup: What the worktree cleanup removed and left behind.

    Returns:
        The measurement result.
    """
    return MeasurementResult(
        label=label,
        samples=samples,
        adapter=adapter,
        metrics=dict(metrics) if metrics is not None else {},
        rounds=tuple(rounds),
        config_kinds=config_kinds,
        cleanup=cleanup,
    )


def entity_time_measurements() -> dict[str, MetricMeasurement]:
    """The two ``time`` metrics of the ``entity`` group, keyed by full metric name."""
    return {
        "entity/alive_check#time": measured_metric(
            kind="time",
            short_name="entity.alive_check",
            unit="ns",
        ),
        "entity/spawn#time": measured_metric(
            kind="time",
            short_name="entity.spawn",
            median=104,
            unit="ns",
        ),
    }


def two_kind_measurement() -> MeasurementResult:
    """A measurement spanning a gating ``time`` kind and an informational ``memory`` kind."""
    return create_measurement_result(
        metrics={
            **entity_time_measurements(),
            "warmup#time": measured_metric(kind="time", short_name="warmup", unit="ns"),
            "encode#memory": measured_metric(
                kind="memory",
                short_name="encode",
                median=93,
                unit="bytes",
                gating=False,
            ),
        },
        config_kinds={"memory": KindEntry(gating=False)},
    )
