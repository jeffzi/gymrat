"""Exclusion taxonomy and the geomean result record."""

from dataclasses import dataclass
from typing import Literal

ExclusionReason = Literal["no-verdict", "unstable", "undefined-ratio", "infinite-rho"]
"""Why a metric was excluded from an aggregate."""


@dataclass(frozen=True, slots=True)
class Exclusion:
    """A metric excluded from an aggregate, paired with the reason.

    Attributes:
        metric: The excluded metric's name.
        reason: Why it was excluded.
    """

    metric: str
    reason: ExclusionReason


@dataclass(frozen=True, slots=True)
class GeomeanResult:
    """Geometric-mean aggregate over many metrics.

    Attributes:
        value: The geometric-mean value.
        n: Number of metrics contributing to ``value``.
        band: The instability band around ``value``.
        excluded: Metrics left out of the aggregate, with reasons.
    """

    value: float
    n: int
    band: float
    excluded: tuple[Exclusion, ...]
