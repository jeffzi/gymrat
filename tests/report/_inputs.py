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

from dataclasses import dataclass, replace
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
    CandidateComparison,
    CandidateMetric,
    ComparisonResult,
    MeasurementResult,
    MetricComparison,
    MetricComparisons,
    MetricMeasurement,
)
from gymrat_py.verdict import KindAggregate

if TYPE_CHECKING:
    from collections.abc import Sequence

    from gymrat_py.targets import WorktreeRemovalFailure

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


# ---------------------------------------------------------------------------
# Metadata and structural builders
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


def other_kind(
    value: float,
    n: int,
    *,
    band: float = 0.0,
    excluded: Sequence[Exclusion] = (),
) -> KindAggregate:
    """The single-kind ``other`` aggregate every default here describes.

    It gates, holds no groups, and shares one aggregate between its section and
    gated geomeans, so a caller excluding metrics or widening the band writes
    that only once.
    """
    geomean = geomean_of(value, n, band=band, excluded=excluded)
    return KindAggregate(kind="other", geomean=geomean, groups=(), gated_geomean=geomean)


def create_candidate(
    *,
    label: str = "perf/faster-decode",
    kinds: Sequence[KindAggregate] | None = None,
) -> CandidateComparison:
    """One candidate's run-level results, judged against the shared baseline.

    ``kinds`` defaults to the single-kind run every other default here
    describes: one ``other`` kind, no groups, whose section and gated geomeans
    share the same default aggregate.
    """
    return CandidateComparison(
        label=label,
        kinds=tuple(kinds) if kinds is not None else (other_kind(-5.8, 10),),
    )


def create_comparison_result(
    *,
    baseline_label: str = "main",
    candidates: Sequence[CandidateComparison] | None = None,
    samples: int = 10,
    adapter: str = "mitata",
    metrics: MetricComparisons | None = None,
    worktrees_removed: int = 0,
    worktrees_left_behind: Sequence[WorktreeRemovalFailure] = (),
    worktree_prune_error: str | None = None,
) -> ComparisonResult:
    """A comparison result with a clean baseline-plus-one-candidate run and no metrics."""
    return ComparisonResult(
        baseline_label=baseline_label,
        candidates=tuple(candidates) if candidates is not None else (create_candidate(),),
        samples=samples,
        adapter=adapter,
        metrics=dict(metrics) if metrics is not None else {},
        worktrees_removed=worktrees_removed,
        worktrees_left_behind=tuple(worktrees_left_behind),
        worktree_prune_error=worktree_prune_error,
    )


# ---------------------------------------------------------------------------
# Comparison metric builders
# ---------------------------------------------------------------------------


def signed_rank_metric(
    *,
    verdict: ApproximateVerdict,
    delta: float,
    baseline_median: float = 100.0,
    baseline_spread: float = 1.0,
    candidate_median: float | None = None,
    candidate_spread: float = 1.0,
    p: float = 0.01,
    noise_pct: float = 2.5,
    noise_abs: float = 3.5,
    unit: MetricUnit | None = None,
    gating: bool = True,
    direction: Direction = "lower",
    n: int = 10,
) -> MetricEntry:
    """A two-sided metric whose verdict came from the signed-rank method."""
    resolved_median = (
        baseline_median * (1 + delta / 100) if candidate_median is None else candidate_median
    )
    return MetricComparison(
        baseline_median=baseline_median,
        baseline_spread=baseline_spread,
        candidates=(
            CandidateMetric(
                median=resolved_median,
                spread=candidate_spread,
                verdict=SignedRankVerdict(
                    method="signed-rank",
                    verdict=verdict,
                    p=p,
                    noise_pct=noise_pct,
                    noise_abs=noise_abs,
                    delta=_percent(delta),
                    n=n,
                ),
            ),
        ),
        meta=ResolvedMetricMeta(
            direction=direction,
            gating=gating,
            exact=False,
            unit=unit,
            kind="other",
            short_name="time",
        ),
    )


def exact_metric(
    *,
    delta: float,
    baseline_median: float = 1000.0,
    candidate_median: float | None = None,
    n: int = 10,
    unit: MetricUnit | None = "bytes",
) -> MetricEntry:
    """A counted metric, compared exactly rather than statistically."""
    resolved_median = (
        baseline_median * (1 + delta / 100) if candidate_median is None else candidate_median
    )
    verdict: Verdict = "improved" if delta < 0 else "regressed"
    return MetricComparison(
        baseline_median=baseline_median,
        baseline_spread=None,
        candidates=(
            CandidateMetric(
                median=resolved_median,
                verdict=ExactVerdict(
                    method="exact",
                    verdict=verdict,
                    delta=_percent(delta),
                    n=n,
                ),
            ),
        ),
        meta=ResolvedMetricMeta(
            direction="lower",
            gating=True,
            exact=True,
            unit=unit,
            kind="other",
            short_name="heap",
        ),
    )


@dataclass(frozen=True)
class NWayCandidate:
    """One candidate's signed-rank outcome, carrying its own measured median."""

    verdict: ApproximateVerdict
    delta: float
    median: float


def n_way_metric(candidates: Sequence[NWayCandidate]) -> MetricEntry:
    """One metric judged for several candidates against a single shared baseline."""
    return MetricComparison(
        baseline_median=100.0,
        baseline_spread=1.0,
        candidates=tuple(
            CandidateMetric(
                median=candidate.median,
                spread=1.0,
                verdict=SignedRankVerdict(
                    method="signed-rank",
                    verdict=candidate.verdict,
                    p=0.01,
                    noise_pct=2.5,
                    noise_abs=3.5,
                    delta=_percent(candidate.delta),
                    n=10,
                ),
            )
            for candidate in candidates
        ),
        meta=ResolvedMetricMeta(
            direction="lower",
            gating=True,
            exact=False,
            unit="ns",
            kind="other",
            short_name="time",
        ),
    )


def kind_metric(
    *,
    kind: str,
    short_name: str,
    verdict: ApproximateVerdict,
    delta: float,
    gating: bool = True,
    unit: MetricUnit | None = "ns",
) -> MetricEntry:
    """A metric of ``kind``, displayed under ``short_name``, judged by the signed-rank test."""
    metric = signed_rank_metric(verdict=verdict, delta=delta, gating=gating, unit=unit)
    return replace(metric, meta=replace(metric.meta, kind=kind, short_name=short_name))


def single_sample_result() -> ComparisonResult:
    """A run of one paired sample, where every verdict rests on a single pair.

    One pair leaves the band method no spread to measure, so it collapses to
    the noise floor and reports no signal whatever the deltas were.
    """
    return create_comparison_result(
        samples=1,
        metrics={
            "decode/time": band_metric(delta=-0.4, noise_pct=0.5, n=1, unit="ns"),
            "encode/time": band_metric(delta=0.2, noise_pct=0.5, n=1, unit="ns"),
        },
        candidates=[create_candidate(kinds=[other_kind(-0.1, 2)])],
    )


# ---------------------------------------------------------------------------
# Measurement builders
# ---------------------------------------------------------------------------


def measured_metric(
    *,
    median: float | None = 100.0,
    spread: float | None = 1.0,
    short_name: str = "time",
    kind: str = "other",
    unit: MetricUnit | None = None,
    gating: bool = True,
) -> MetricMeasurement:
    """One metric of a single-target run: what it measured, and how steady it was.

    ``spread`` is a percentage of the median. Passing ``None`` is how a caller
    pins the single-sample case, where there is no run-to-run jitter to report.
    """
    return MetricMeasurement(
        median=median,
        spread=spread,
        meta=metric_meta(short_name, kind=kind, unit=unit, gating=gating),
    )


def create_measurement_result(
    *,
    label: str = "main",
    samples: int = 10,
    adapter: str = "mitata",
    metrics: dict[str, MetricMeasurement] | None = None,
    rounds: Sequence[dict[str, float]] = (),
    worktrees_removed: int = 0,
    worktrees_left_behind: Sequence[WorktreeRemovalFailure] = (),
    worktree_prune_error: str | None = None,
) -> MeasurementResult:
    """A measurement of a clean single-target run with no metrics."""
    return MeasurementResult(
        label=label,
        samples=samples,
        adapter=adapter,
        metrics=dict(metrics) if metrics is not None else {},
        rounds=tuple(rounds),
        worktrees_removed=worktrees_removed,
        worktrees_left_behind=tuple(worktrees_left_behind),
        worktree_prune_error=worktree_prune_error,
    )


def two_kind_measurement(
    *,
    worktrees_removed: int = 0,
    worktrees_left_behind: Sequence[WorktreeRemovalFailure] = (),
    worktree_prune_error: str | None = None,
) -> MeasurementResult:
    """A measurement spanning a gating ``time`` kind and an informational ``memory`` kind."""
    return create_measurement_result(
        metrics={
            "entity.alive_check/time": measured_metric(
                kind="time",
                short_name="entity.alive_check",
                unit="ns",
            ),
            "entity.spawn/time": measured_metric(
                kind="time",
                short_name="entity.spawn",
                median=104,
                unit="ns",
            ),
            "warmup/time": measured_metric(kind="time", short_name="warmup", unit="ns"),
            "encode/heap": measured_metric(
                kind="memory",
                short_name="encode",
                median=93,
                unit="bytes",
                gating=False,
            ),
        },
        worktrees_removed=worktrees_removed,
        worktrees_left_behind=worktrees_left_behind,
        worktree_prune_error=worktree_prune_error,
    )


__all__ = [
    "CandidateSpec",
    "MetricEntry",
    "Metrics",
    "NWayCandidate",
    "approximate_metric",
    "band_metric",
    "band_verdict",
    "create_candidate",
    "create_comparison_result",
    "create_measurement_result",
    "exact_metric",
    "exact_verdict",
    "geomean_of",
    "kind_metric",
    "measured_metric",
    "metric_for",
    "metric_meta",
    "n_way_metric",
    "one_sided_metric",
    "other_kind",
    "signed_rank_metric",
    "signed_rank_verdict",
    "single_sample_result",
    "two_kind_measurement",
]
