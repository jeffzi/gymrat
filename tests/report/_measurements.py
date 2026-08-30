"""Measurement builders for report formatting tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

from gymrat.config import KindEntry
from gymrat.report.types import (
    MeasurementResult,
    MetricMeasurement,
)
from tests.report._comparisons import metric_meta

if TYPE_CHECKING:
    from collections.abc import Sequence

    from gymrat.model import MetricUnit
    from gymrat.targets import WorktreeRemovalFailure


# ---------------------------------------------------------------------------
# Measurement builders
# ---------------------------------------------------------------------------


def measured_metric(
    *,
    median: float | None = 100.0,
    spread: float | None = 1.0,
    short_name: str = "time",
    kind: str = "other",
    unit: MetricUnit | None = None,
    gating: bool = True,
) -> MetricMeasurement:
    """One metric of a single-target run: what it measured, and how steady it was.

    ``spread`` is a percentage of the median. Passing ``None`` is how a caller
    pins the single-sample case, where there is no run-to-run jitter to report.
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
    worktrees_removed: int = 0,
    worktrees_left_behind: Sequence[WorktreeRemovalFailure] = (),
    worktree_prune_error: str | None = None,
) -> MeasurementResult:
    """A measurement of a clean single-target run with no metrics."""
    return MeasurementResult(
        label=label,
        samples=samples,
        adapter=adapter,
        metrics=dict(metrics) if metrics is not None else {},
        rounds=tuple(rounds),
        config_kinds=config_kinds,
        worktrees_removed=worktrees_removed,
        worktrees_left_behind=tuple(worktrees_left_behind),
        worktree_prune_error=worktree_prune_error,
    )


def two_kind_measurement(
    *,
    worktrees_removed: int = 0,
    worktrees_left_behind: Sequence[WorktreeRemovalFailure] = (),
    worktree_prune_error: str | None = None,
) -> MeasurementResult:
    """A measurement spanning a gating ``time`` kind and an informational ``memory`` kind."""
    return create_measurement_result(
        metrics={
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
        worktrees_removed=worktrees_removed,
        worktrees_left_behind=worktrees_left_behind,
        worktree_prune_error=worktree_prune_error,
    )
