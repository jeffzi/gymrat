"""Shared comparison/verdict builders for the report formatting tests.

These port the fixtures ``format_test`` needs from the TypeScript
``comparison-result`` fixture module plus the ``metricFor`` / ``approximateMetric``
/ ``oneSidedMetric`` helpers that were defined inline in ``format.test.ts``.

The builders return the frozen dataclasses declared in
:mod:`gymrat_py.report.types`, so a test writes one metric's shape once and lets
the builder wire the baseline, per-candidate slices, and metadata.

This is test-support code, not a test module: ``test_format`` imports it. It
carries no test functions of its own.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from gymrat_py.model import (
    ApproximateVerdict,
    BandVerdict,
    Direction,
    Effect,
    ExactVerdict,
    Exclusion,
    GeomeanResult,
    MetricUnit,
    ResolvedMetricMeta,
    SignedRankVerdict,
    Verdict,
)
from gymrat_py.report.types import (
    CandidateMetric,
    MetricComparison,
    MetricComparisons,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

# The name-keyed map of every metric compared in a run.
Metrics = MetricComparisons
# One metric's comparison data: baseline, per-candidate results, and meta.
MetricEntry = MetricComparison


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


def signed_rank_verdict(
    *,
    verdict: ApproximateVerdict = "no-signal",
    delta: float = 0.2,
    n: int = 10,
    p: float = 0.49,
    noise_pct: float = 2.5,
    noise_abs: float = 2.5,
) -> SignedRankVerdict:
    """A verdict the Wilcoxon signed-rank test produced."""
    return SignedRankVerdict(
        method="signed-rank",
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
) -> MetricEntry:
    """A two-sided metric whose verdict fell back to the noise band.

    ``n`` is the total pair count and ``usable_n`` how many of those pairs
    survived tie-dropping. ``n < 6`` means the run was too short for the
    signed-rank test; ``n >= 6`` with ``usable_n < 6`` means ties starved it.
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


@dataclass(frozen=True)
class CandidateSpec:
    """One candidate's signed-rank outcome against the shared baseline."""

    verdict: ApproximateVerdict
    delta: float
    noise_pct: float = 2.5


def metric_for(
    candidates: Sequence[CandidateSpec],
    direction: Direction = "lower",
) -> MetricEntry:
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
                verdict=SignedRankVerdict(
                    method="signed-rank",
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
) -> MetricEntry:
    """A single-candidate metric whose verdict came from the signed-rank method."""
    return metric_for(
        [CandidateSpec(verdict=verdict, delta=delta, noise_pct=noise_pct)],
        direction,
    )


def one_sided_metric() -> MetricEntry:
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


__all__ = [
    "CandidateSpec",
    "MetricEntry",
    "Metrics",
    "approximate_metric",
    "band_metric",
    "band_verdict",
    "exact_verdict",
    "geomean_of",
    "metric_for",
    "one_sided_metric",
    "signed_rank_verdict",
]
