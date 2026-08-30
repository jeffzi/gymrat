"""Standalone verdict and metric builders for report formatting tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from gymrat.model import (
    ApproximateVerdict,
    BandVerdict,
    Direction,
    Effect,
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


def _percent(value: float) -> Effect:
    """A percentage effect — the only unit the model's deltas carry today."""
    return Effect(value=value, unit="percent")


# ---------------------------------------------------------------------------
# Standalone verdict builders
# ---------------------------------------------------------------------------


def band_verdict(
    *,
    verdict: ApproximateVerdict = "no-signal",
    delta: float = -0.5,
    n: int = 10,
    usable_n: int = 3,
    noise_pct: float = 2.5,
    noise_abs: float = 2.5,
) -> BandVerdict:
    """A noise-band verdict, tied pairs and all."""
    return BandVerdict(
        method="band",
        verdict=verdict,
        usable_n=usable_n,
        noise_pct=noise_pct,
        noise_abs=noise_abs,
        delta=_percent(delta),
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
    """A verdict the sign-flip permutation test test produced."""
    return PermutationVerdict(
        method="permutation",
        verdict=verdict,
        p=p,
        noise_pct=noise_pct,
        noise_abs=noise_abs,
        delta=_percent(delta),
        n=n,
    )


def exact_verdict(
    *,
    verdict: Verdict = "no-signal",
    delta: float = 0.0,
    n: int = 10,
) -> ExactVerdict:
    """A verdict read straight off a counted metric, with no statistics behind it."""
    return ExactVerdict(
        method="exact",
        verdict=verdict,
        delta=_percent(delta),
        n=n,
    )


def geomean_of(
    value: float = 0.0,
    n: int = 2,
    *,
    band: float = 0.0,
    excluded: Sequence[Exclusion] = (),
) -> GeomeanResult:
    """A geomean over ``n`` metrics, with no exclusions and no band unless overridden."""
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
    direction: Direction = "lower",
    unit: MetricUnit | None = None,
) -> MetricComparison:
    """A two-sided metric whose verdict fell back to the noise band.

    ``n`` is the total pair count and ``usable_n`` how many of those pairs
    survived tie-dropping. ``n < 6`` means the run was too short for the
    permutation test; ``n >= 6`` with ``usable_n < 6`` means ties starved it.
    """
    resolved_usable = n if usable_n is None else usable_n
    return MetricComparison(
        baseline_median=100.0,
        baseline_spread=5.0,
        candidates=(
            CandidateMetric(
                median=100.0 + delta,
                spread=4.0,
                verdict=BandVerdict(
                    method="band",
                    verdict=verdict,
                    usable_n=resolved_usable,
                    noise_pct=noise_pct,
                    noise_abs=3.5,
                    delta=_percent(delta),
                    n=n,
                ),
            ),
        ),
        meta=ResolvedMetricMeta(
            direction=direction,
            gating=True,
            exact=False,
            unit=unit,
            kind="other",
            short_name="time",
        ),
    )


@dataclass(frozen=True, slots=True)
class CandidateSpec:
    """One candidate's permutation outcome against the shared baseline."""

    verdict: ApproximateVerdict
    delta: float
    noise_pct: float = 2.5


def metric_for(
    candidates: Sequence[CandidateSpec],
    direction: Direction = "lower",
) -> MetricComparison:
    """A metric judged once per candidate against the shared baseline.

    The baseline median and spread are carried once, so the candidate entries
    differ only in what the pairwise verdict engine returned for each of them.
    """
    return MetricComparison(
        baseline_median=100.0,
        baseline_spread=1.0,
        candidates=tuple(
            CandidateMetric(
                median=100.0 + candidate.delta,
                spread=1.0,
                verdict=PermutationVerdict(
                    method="permutation",
                    verdict=candidate.verdict,
                    p=0.01,
                    noise_pct=candidate.noise_pct,
                    noise_abs=candidate.noise_pct,
                    delta=_percent(candidate.delta),
                    n=10,
                ),
            )
            for candidate in candidates
        ),
        meta=ResolvedMetricMeta(
            direction=direction,
            gating=True,
            exact=False,
            unit=None,
            kind="other",
            short_name="time",
        ),
    )


def approximate_metric(
    *,
    verdict: ApproximateVerdict,
    delta: float,
    noise_pct: float = 2.5,
    direction: Direction = "lower",
) -> MetricComparison:
    """A single-candidate metric whose verdict came from the permutation method."""
    return metric_for(
        [CandidateSpec(verdict=verdict, delta=delta, noise_pct=noise_pct)],
        direction,
    )


def one_sided_metric() -> MetricComparison:
    """A metric the candidate never reported, so no verdict could be computed."""
    return MetricComparison(
        baseline_median=100.0,
        baseline_spread=1.0,
        candidates=(CandidateMetric(),),
        meta=ResolvedMetricMeta(
            direction="lower",
            gating=False,
            exact=False,
            unit=None,
            kind="other",
            short_name="time",
        ),
    )
