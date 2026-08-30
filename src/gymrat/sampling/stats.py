"""Stats helpers and metric-meta resolution from collected samples."""

import math
from collections.abc import Sequence

from gymrat.adapters.types import Adapter
from gymrat.config.meta import resolve_metric_meta
from gymrat.config.types import KindEntry, MetricEntry
from gymrat.errors import GymratError
from gymrat.model import ResolvedMetricMeta
from gymrat.sampling.types import _MIN_SPREAD_SAMPLES, MetricStats
from gymrat.stats.descriptive import compute_half_range, compute_median


def compute_metric_stats(values: Sequence[float]) -> MetricStats:
    """Summarize a metric's samples as a median and relative spread.

    Args:
        values: The metric's sampled values.

    Returns:
        The median and its half-range as a percentage of ``abs(median)``. The
        spread is absent when there are fewer than two values, the median is
        zero, or the ratio is non-finite.
    """
    if not values:
        return MetricStats(median=None, spread=None)

    median = compute_median(values)
    if len(values) < _MIN_SPREAD_SAMPLES or median == 0:
        return MetricStats(median=median, spread=None)

    ratio = compute_half_range(values) / abs(median) * 100
    if not math.isfinite(ratio):
        return MetricStats(median=median, spread=None)
    return MetricStats(median=median, spread=ratio)


def own_values(samples: Sequence[dict[str, float]], name: str) -> list[float]:
    """Collect the values a side reported for ``name``, skipping rounds without it.

    Args:
        samples: One metric record per round.
        name: The metric to extract.

    Returns:
        The reported values for ``name``, in round order.
    """
    return [record[name] for record in samples if name in record]


def paired_or_own_values(
    paired: Sequence[float],
    samples: Sequence[dict[str, float]],
    name: str,
) -> list[float]:
    """Prefer already-paired values, falling back to a side's own values.

    Args:
        paired: Values paired across sides; used as-is when non-empty.
        samples: One metric record per round, used only for the fallback.
        name: The metric to extract when falling back.

    Returns:
        ``paired`` when it holds any values, otherwise ``own_values(samples, name)``.
    """
    if paired:
        return list(paired)
    return own_values(samples, name)


def resolve_metric_meta_from_samples(
    sample_sets: Sequence[list[dict[str, float]]],
    config_metrics: dict[str, MetricEntry] | None,
    adapter: Adapter,
    config_kinds: dict[str, KindEntry] | None = None,
) -> dict[str, ResolvedMetricMeta]:
    """Collect every metric name across the sample sets and resolve its metadata.

    The union of names is taken in first-appearance order across the flattened
    samples, so the resolved metadata — and every report drawn from it — reads in
    the order the run first reported each metric.

    Args:
        sample_sets: One list of per-round metric records per target.
        config_metrics: Per-metric overrides from config, or ``None``.
        adapter: The adapter whose defaults seed each metric's metadata.
        config_kinds: Per-kind overrides from config, or ``None``.

    Returns:
        The resolved metadata for each metric, keyed by metric name.

    Raises:
        GymratError: No sample set reported any metric. Adapters reject empty
            output themselves, so this guards the otherwise-unreachable case.
    """
    names = dict.fromkeys(name for samples in sample_sets for sample in samples for name in sample)
    if not names:
        message = "No metrics found in benchmark output"
        raise GymratError(message)

    return resolve_metric_meta(list(names), config_metrics, adapter, config_kinds)
