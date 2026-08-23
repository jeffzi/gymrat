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


@dataclass(frozen=True, slots=True)
class ResolvedMetricMeta(MetricMeta):
    """A :class:`MetricMeta` resolved against a concrete metric, with display metadata.

    Extends the static metadata with the metric's classification and label. Being a subclass, a
    ``ResolvedMetricMeta`` is usable anywhere a :class:`MetricMeta` is.

    Attributes:
        kind: The metric's classification (e.g. ``"time"``, ``"mem"``).
        short_name: The metric's display label.
    """

    kind: str
    short_name: str
