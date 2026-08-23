"""Metric metadata slice and shared metric type aliases."""

from dataclasses import dataclass
from typing import Literal

Direction = Literal["lower", "higher"]
"""Whether a lower or higher raw value is the better outcome for a metric."""

MetricUnit = Literal["ns", "bytes"]
"""Physical unit a metric is measured in."""


@dataclass(frozen=True, slots=True)
class MetricMeta:
    """Static metadata describing how a single metric is interpreted.

    Attributes:
        direction: Whether lower or higher is better.
        gating: Whether the metric can gate (fail) a comparison.
        exact: Whether the metric is compared exactly rather than statistically.
        unit: The metric's physical unit, or ``None`` when unitless.
    """

    direction: Direction
    gating: bool
    exact: bool
    unit: MetricUnit | None
