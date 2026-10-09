"""Probe result builders for report formatting tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

from gymrat.loop.probe import ProbeMetric, ProbeResult
from tests.report._verdicts import metric_meta

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
) -> ProbeMetric:
    """One probed metric: what it measured now, what the baseline holds, and the gap.

    Args:
        name: The metric name.
        median: The metric's median now. ``None`` means no round reported the
            metric.
        spread: The run-to-run spread as a percentage of the median. ``None``
            means there was no run-to-run jitter to report.
        reference_median: The baseline's median, ``None`` together with
            ``delta_pct`` when the baseline has nothing to pair the metric with.
        delta_pct: The gap to the baseline, in percent.
        direction: Which way is better for the metric.
        unit: The metric's unit, if any.
        kind: The metric's kind.

    Returns:
        The probe metric.
    """
    return ProbeMetric(
        name=name,
        median=median,
        spread=spread,
        reference_median=reference_median,
        delta_pct=delta_pct,
        meta=metric_meta(name, direction=direction, gating=True, kind=kind, unit=unit),
    )


def probe_result(
    *,
    label: str = "experiment",
    samples: int = 6,
    adapter: str = "mitata",
    metrics: Sequence[ProbeMetric] = (),
    names: Sequence[str] = (),
) -> ProbeResult:
    """A probe of one worktree paired against the newest recorded baseline.

    Args:
        label: The display label of the benched worktree.
        samples: How many rounds the probe ran.
        adapter: The adapter that parsed the bench output.
        metrics: One entry per metric the run reported, in report order.
        names: The metric names the bench was narrowed to, in the order given;
            empty means the whole bench ran. It filters the bench, not
            ``metrics``.

    Returns:
        The probe result.
    """
    return ProbeResult(
        label=label,
        samples=samples,
        adapter=adapter,
        metrics=tuple(metrics),
        names=tuple(names),
    )


def golden_probe() -> ProbeResult:
    """A probe with a paired metric, a higher-is-better metric, and one the baseline lacks."""
    return probe_result(
        metrics=[
            probe_metric("total_ns", unit="ns", kind="time"),
            probe_metric(
                "ops_per_sec",
                median=1200.0,
                reference_median=1000.0,
                delta_pct=20.0,
                direction="higher",
                kind="throughput",
            ),
            probe_metric("cold_start_ns", reference_median=None, delta_pct=None, unit="ns"),
        ]
    )
