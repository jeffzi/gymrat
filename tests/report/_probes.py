"""Probe result builders for report formatting tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

from gymrat.loop.probe import ProbeMetric, ProbeResult
from tests.report._comparisons import metric_meta

if TYPE_CHECKING:
    from collections.abc import Sequence

    from gymrat.model import Direction, MetricUnit


def probe_metric(
    name: str = "total_ms",
    *,
    median: float | None = 90.0,
    spread: float | None = 2.0,
    reference_median: float | None = 100.0,
    delta_pct: float | None = -10.0,
    direction: Direction = "lower",
    unit: MetricUnit | None = None,
    kind: str = "other",
    gating: bool = True,
) -> ProbeMetric:
    """One probed metric: what it measured now, what the baseline holds, and the gap.

    ``spread`` is a percentage of the median. ``reference_median`` and
    ``delta_pct`` are ``None`` together when the baseline has nothing to pair
    the metric with.
    """
    return ProbeMetric(
        name=name,
        median=median,
        spread=spread,
        reference_median=reference_median,
        delta_pct=delta_pct,
        meta=metric_meta(name, direction=direction, gating=gating, kind=kind, unit=unit),
    )


def probe_result(
    *,
    label: str = "experiment",
    samples: int = 6,
    adapter: str = "mitata",
    metrics: Sequence[ProbeMetric] = (),
    scoped: bool = False,
    names: Sequence[str] = (),
) -> ProbeResult:
    """A probe of one worktree paired against the newest recorded baseline."""
    return ProbeResult(
        label=label,
        samples=samples,
        adapter=adapter,
        metrics=tuple(metrics),
        scoped=scoped,
        names=tuple(names),
    )
