"""Pure descriptive-statistics helpers.

Every helper here is a total function over its inputs: it performs no I/O, no
formatting, reads no configuration, and imports only the standard library
(``math`` and ``statistics``). Empty input to a reduction is a programming error
— callers are expected to have samples — so it raises a plain :class:`ValueError`
rather than a domain error.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from statistics import fmean, median
from typing import Literal

from gymrat.model.metrics import Direction

type RatioExclusion = Literal["undefined-ratio", "infinite-rho"]
"""Why a percent delta could not be normalized into a usable ratio rho."""


@dataclass(frozen=True, slots=True)
class RatioOutcome:
    """A normalized ratio rho, or the reason it was excluded.

    Exactly one of two states holds. A usable rho has ``rho`` set and ``reason``
    ``None``; an exclusion has ``rho`` ``None`` and ``reason`` naming the cause.
    A caller distinguishes them by testing ``rho is not None``.

    Attributes:
        rho: The usable ratio, or ``None`` when the delta was excluded.
        reason: The exclusion cause, or ``None`` when ``rho`` is usable.
    """

    rho: float | None
    reason: RatioExclusion | None


@dataclass(frozen=True, slots=True)
class GeomeanCombination:
    """The geometric-mean combination of a set of included entries.

    This is a pure math result local to this module — not the richer
    model-layer geomean result.

    Attributes:
        value: The combined percent change, ``(exp(mean(ln rho)) - 1) * 100``.
        n: The number of entries combined.
        band: The quadrature-combined noise band,
            ``sqrt(sum(noise_pct**2)) / n``.
    """

    value: float
    n: int
    band: float


def compute_median(values: Sequence[float]) -> float:
    """Return the median of ``values`` without mutating the input.

    Args:
        values: The samples to summarize. Must be non-empty.

    Returns:
        The middle element for odd-length input, or the mean of the two middle
        elements for even-length input.

    Raises:
        ValueError: If ``values`` is empty.
    """
    if not values:
        msg = "compute_median requires at least one sample; got an empty sequence"
        raise ValueError(msg)
    return float(median(values))


def compute_half_range(values: Sequence[float]) -> float:
    """Return half the span ``(max - min) / 2`` of ``values``.

    A single non-finite sample poisons the whole result: it never drops a
    sample and never returns ±infinity, so any NaN or ±infinity among the
    samples yields the NaN undefined-measurement sentinel.

    Args:
        values: The samples to summarize. Must be non-empty.

    Returns:
        Half the span over finite samples, or ``float("nan")`` when any sample
        is non-finite.

    Raises:
        ValueError: If ``values`` is empty.
    """
    if not values:
        msg = "compute_half_range requires at least one sample; got an empty sequence"
        raise ValueError(msg)
    if any(not math.isfinite(sample) for sample in values):
        return math.nan
    return (max(values) - min(values)) / 2.0


def normalize_ratio(delta: float, direction: Direction) -> RatioOutcome:
    """Normalize a percent ``delta`` into a ratio rho for the given direction.

    For ``"lower"``, ``rho = 1 + delta / 100``. For ``"higher"``,
    ``rho = 1 / (1 + delta / 100)``. A rho that is not strictly positive and
    finite cannot be combined in log space, so it is reported as an exclusion.

    Args:
        delta: The percent delta to normalize.
        direction: Whether a lower or higher raw value is the improvement.

    Returns:
        A :class:`RatioOutcome` carrying a usable rho, or ``"undefined-ratio"``
        when ``delta`` is NaN, or ``"infinite-rho"`` when the resulting rho is
        non-positive or non-finite.
    """
    if math.isnan(delta):
        return RatioOutcome(rho=None, reason="undefined-ratio")
    factor = 1.0 + delta / 100.0
    rho = factor
    if direction == "higher":
        # A zero factor would divide by zero; route it to the infinite-rho
        # exclusion below instead of raising.
        rho = 1.0 / factor if factor != 0.0 else math.inf
    if rho <= 0.0 or not math.isfinite(rho):
        return RatioOutcome(rho=None, reason="infinite-rho")
    return RatioOutcome(rho=rho, reason=None)


def combine_geomean(entries: Sequence[tuple[float, float]]) -> GeomeanCombination:
    """Combine included ``(rho, noise_pct)`` entries via the geometric mean.

    The combined value is the geometric mean of the ratios expressed as a
    percent change; the band combines per-entry noise in quadrature.

    Args:
        entries: The included entries, each a ``(rho, noise_pct)`` pair. Every
            ``rho`` must be strictly positive.

    Returns:
        A :class:`GeomeanCombination`. Empty input yields all-zero fields; a
        single entry yields that rho as a percent change with its own noise as
        the band.
    """
    n = len(entries)
    if n == 0:
        return GeomeanCombination(value=0.0, n=0, band=0.0)
    mean_log = fmean(math.log(rho) for rho, _ in entries)
    value = (math.exp(mean_log) - 1.0) * 100.0
    band = math.sqrt(sum(noise_pct**2 for _, noise_pct in entries)) / n
    return GeomeanCombination(value=value, n=n, band=band)
