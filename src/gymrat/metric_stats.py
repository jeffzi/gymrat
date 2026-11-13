"""Summarizing a metric's samples as a median and a relative spread."""

import statistics
from collections.abc import Sequence
from dataclasses import dataclass

from gymrat.errors import GymratError
from gymrat.stats import compute_half_range
from gymrat.utils import fraction_of_median


@dataclass(frozen=True, slots=True)
class MetricStats:
    """A metric's central value and relative spread.

    Attributes:
        median: The metric's median.
        spread: The half-range as a percentage of the median's magnitude, or
            ``None`` when it is undefined (fewer than two values, a zero median,
            or a non-finite ratio).
    """

    median: float
    spread: float | None


_MIN_SPREAD_SAMPLES = 2


def compute_metric_stats(name: str, values: Sequence[float]) -> MetricStats:
    """Summarize a metric's samples as a median and relative spread.

    Args:
        name: The metric's full name, which the error for an empty sample names.
        values: The metric's sampled values.

    Returns:
        The median and its half-range as a percentage of ``abs(median)``. The
        spread is absent when there are fewer than two values, the median is
        zero, or the ratio is non-finite.

    Raises:
        GymratError: ``values`` is empty, so the metric has no median.
    """
    if not values:
        message = f"Metric {name!r} has no samples to summarize"
        raise GymratError(message)

    median = statistics.median(values)
    if len(values) < _MIN_SPREAD_SAMPLES:
        return MetricStats(median=median, spread=None)
    return MetricStats(
        median=median, spread=fraction_of_median(compute_half_range(values), median, 100)
    )
