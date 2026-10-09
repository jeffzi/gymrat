"""Standalone verdict and metric builders for report formatting tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

from gymrat.model import (
    ApproximateVerdict,
    BandVerdict,
    Direction,
    ExactVerdict,
    Exclusion,
    GeomeanResult,
    MetricUnit,
    PermutationVerdict,
    ResolvedMetricMeta,
    Verdict,
)
from gymrat.report.types import (
    CandidateMetric,
    MetricComparison,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


# ---------------------------------------------------------------------------
# Standalone verdict builders
# ---------------------------------------------------------------------------


def metric_meta(
    short_name: str,
    *,
    direction: Direction = "lower",
    gating: bool = True,
    exact: bool = False,
    kind: str = "other",
    unit: MetricUnit | None = None,
) -> ResolvedMetricMeta:
    """A metric meta block, defaulting to a lower-is-better, gating, non-exact "other" metric."""
    return ResolvedMetricMeta(
        direction=direction,
        gating=gating,
        exact=exact,
        unit=unit,
        kind=kind,
        short_name=short_name,
    )


def band_verdict(
    *,
    verdict: ApproximateVerdict = "no-signal",
    delta: float = -0.5,
    n: int = 10,
    usable_n: int = 3,
    noise_pct: float = 2.5,
    noise_abs: float = 2.5,
) -> BandVerdict:
    """A noise-band verdict, the method used when too few pairs remain for a permutation test.

    Args:
        verdict: The approximate outcome.
        delta: The percentage delta between the paired medians.
        n: The total pair count.
        usable_n: How many pairs survived tie-dropping; below 6 the
            permutation test is starved and the band decides.
        noise_pct: The noise band, in percent of the baseline median.
        noise_abs: The noise band, in the metric's own units.

    Returns:
        The band verdict.
    """
    return BandVerdict(
        method="band",
        verdict=verdict,
        usable_n=usable_n,
        noise_pct=noise_pct,
        noise_abs=noise_abs,
        delta=delta,
        n=n,
    )


def permutation_verdict(
    *,
    verdict: ApproximateVerdict = "no-signal",
    delta: float = 0.2,
    n: int = 10,
    p: float = 0.49,
    noise_pct: float = 2.5,
    noise_abs: float = 2.5,
) -> PermutationVerdict:
    """A verdict the sign-flip permutation test produced.

    Args:
        verdict: The approximate outcome.
        delta: The percentage delta between the paired medians.
        n: The paired sample count.
        p: The permutation test's p-value; below 0.05 the shift is significant.
        noise_pct: The noise band, in percent of the baseline median.
        noise_abs: The noise band, in the metric's own units.

    Returns:
        The permutation verdict.
    """
    return PermutationVerdict(
        method="permutation",
        verdict=verdict,
        p=p,
        noise_pct=noise_pct,
        noise_abs=noise_abs,
        delta=delta,
        n=n,
    )


def exact_verdict(
    *,
    verdict: Verdict | None = None,
    delta: float = 0.0,
    n: int = 10,
) -> ExactVerdict:
    """A verdict read straight off a counted metric, with no statistics behind it.

    Args:
        verdict: The verdict to carry, or None to derive it from the sign of
            ``delta`` as a lower-is-better metric reads it: negative improves,
            positive regresses, zero or NaN is no signal.
        delta: Percentage delta against the baseline.
        n: Pair count behind the verdict.

    Returns:
        The exact verdict.
    """
    if verdict is None:
        if delta < 0:
            verdict = "improved"
        elif delta > 0:
            verdict = "regressed"
        else:
            verdict = "no-signal"
    return ExactVerdict(
        method="exact",
        verdict=verdict,
        delta=delta,
        n=n,
    )


def geomean_of(
    value: float = 0.0,
    n: int = 2,
    *,
    band: float = 0.0,
    excluded: Sequence[Exclusion] = (),
) -> GeomeanResult:
    """A geomean over ``n`` metrics, with no exclusions and no band unless overridden.

    Args:
        value: The geometric-mean delta, in percent.
        n: How many metrics contribute to ``value``.
        band: The instability band around ``value``, in percent; 0 means no band.
        excluded: The metrics left out of the aggregate, with their reasons.

    Returns:
        The geomean result.
    """
    return GeomeanResult(value=value, n=n, band=band, excluded=tuple(excluded))


# ---------------------------------------------------------------------------
# Metric builders
# ---------------------------------------------------------------------------


def band_metric(
    *,
    verdict: ApproximateVerdict = "no-signal",
    delta: float = -1.0,
    noise_pct: float = 2.5,
    n: int = 4,
    usable_n: int | None = None,
    unit: MetricUnit | None = None,
) -> MetricComparison:
    """A two-sided metric whose verdict fell back to the noise band.

    ``n < 6`` means the run was too short for the permutation test; ``n >= 6``
    with ``usable_n < 6`` means ties starved it.

    Args:
        verdict: The band verdict.
        delta: The candidate's delta, in percent.
        noise_pct: The noise band, in percent.
        n: The total pair count.
        usable_n: How many pairs survived tie-dropping; ``None`` means all ``n``.
        unit: The metric's unit, if any.

    Returns:
        The metric comparison, judged for a single candidate.
    """
    return MetricComparison(
        baseline_median=100.0,
        baseline_spread=5.0,
        candidates=(
            CandidateMetric(
                median=100.0 + delta,
                spread=4.0,
                verdict=band_verdict(
                    verdict=verdict,
                    delta=delta,
                    n=n,
                    usable_n=n if usable_n is None else usable_n,
                    noise_pct=noise_pct,
                    noise_abs=3.5,
                ),
            ),
        ),
        meta=metric_meta("time", unit=unit),
    )


def one_sided_metric() -> MetricComparison:
    """A metric the candidate never reported, so no verdict could be computed."""
    return MetricComparison(
        baseline_median=100.0,
        baseline_spread=1.0,
        candidates=(CandidateMetric(),),
        meta=metric_meta("time", gating=False),
    )
