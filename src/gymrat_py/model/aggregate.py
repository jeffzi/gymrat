"""Aggregate protocol, exclusion taxonomy, and the geomean result record."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

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


class Aggregate(Protocol):
    """Structural contract for an aggregate summary over many metrics.

    Members are read-only so that frozen dataclasses (whose fields are read-only) satisfy the
    protocol; a read-write attribute declaration would reject them.

    Attributes:
        value: The aggregate value.
        n: Number of metrics contributing to ``value``.
        band: The instability band around ``value``.
        excluded: Metrics left out of the aggregate, with reasons.
    """

    @property
    def value(self) -> float:
        """The aggregate value."""

    @property
    def n(self) -> int:
        """Number of metrics contributing to ``value``."""

    @property
    def band(self) -> float:
        """The instability band around ``value``."""

    @property
    def excluded(self) -> Sequence[Exclusion]:
        """Metrics left out of the aggregate, with reasons."""


@dataclass(frozen=True, slots=True)
class GeomeanResult:
    """Geometric-mean aggregate satisfying :class:`Aggregate`.

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
