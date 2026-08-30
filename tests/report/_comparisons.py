"""Metadata, structural, and comparison metric builders for report formatting tests."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from gymrat.config import KindEntry
from gymrat.model import (
    ApproximateVerdict,
    BandVerdict,
    Direction,
    Exclusion,
    MetricUnit,
    PermutationVerdict,
    ResolvedMetricMeta,
)
from gymrat.report.types import (
    CandidateComparison,
    CandidateMetric,
    ComparisonResult,
    MetricComparison,
    MetricComparisons,
)
from gymrat.verdict import GroupAggregate, KindAggregate
from tests.report._verdicts import (
    _percent,
    band_metric,
    geomean_of,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from gymrat.targets import WorktreeRemovalFailure


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
    config_kinds: dict[str, KindEntry] | None = None,
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
        config_kinds=config_kinds,
        worktrees_removed=worktrees_removed,
        worktrees_left_behind=tuple(worktrees_left_behind),
        worktree_prune_error=worktree_prune_error,
    )


# ---------------------------------------------------------------------------
# Comparison metric builders
# ---------------------------------------------------------------------------


def permutation_metric(
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
) -> MetricComparison:
    """A two-sided metric whose verdict came from the permutation method."""
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
                verdict=PermutationVerdict(
                    method="permutation",
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
) -> MetricComparison:
    """A counted metric, compared exactly rather than statistically."""
    from gymrat.model import ExactVerdict, Verdict

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


@dataclass(frozen=True, slots=True)
class NWayCandidate:
    """One candidate's permutation outcome, carrying its own measured median."""

    verdict: ApproximateVerdict
    delta: float
    median: float


def n_way_metric(candidates: Sequence[NWayCandidate]) -> MetricComparison:
    """One metric judged for several candidates against a single shared baseline."""
    return MetricComparison(
        baseline_median=100.0,
        baseline_spread=1.0,
        candidates=tuple(
            CandidateMetric(
                median=candidate.median,
                spread=1.0,
                verdict=PermutationVerdict(
                    method="permutation",
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


def multi_candidate_result(candidate_count: int = 3) -> ComparisonResult:
    """A multi-candidate comparison with one metric judged per candidate.

    With three candidates (the default): ``candidate-a`` improved,
    ``candidate-b`` regressed, ``candidate-c`` unstable (band method). With two:
    the first pair alone.
    """
    candidates = [
        create_candidate(label="candidate-a", kinds=[other_kind(-10, 1)]),
        create_candidate(label="candidate-b", kinds=[other_kind(4, 1)]),
    ]
    metric_candidates = [
        CandidateMetric(
            median=90.0,
            spread=1.0,
            verdict=PermutationVerdict(
                method="permutation",
                verdict="improved",
                p=0.002,
                noise_pct=2.5,
                noise_abs=2.5,
                delta=_percent(-10),
                n=10,
            ),
        ),
        CandidateMetric(
            median=104.0,
            spread=1.0,
            verdict=PermutationVerdict(
                method="permutation",
                verdict="regressed",
                p=0.002,
                noise_pct=2.5,
                noise_abs=2.5,
                delta=_percent(4),
                n=10,
            ),
        ),
    ]
    if candidate_count == 3:
        candidates.append(create_candidate(label="candidate-c", kinds=[other_kind(0, 1)]))
        metric_candidates.append(
            CandidateMetric(
                median=150.0,
                spread=3.0,
                verdict=BandVerdict(
                    method="band",
                    verdict="unstable",
                    usable_n=3,
                    noise_pct=30,
                    noise_abs=30,
                    delta=_percent(50),
                    n=10,
                ),
            )
        )
    return create_comparison_result(
        baseline_label="main",
        candidates=candidates,
        metrics={
            "decode/time": MetricComparison(
                baseline_median=100.0,
                baseline_spread=1.0,
                candidates=tuple(metric_candidates),
                meta=ResolvedMetricMeta(
                    direction="lower",
                    gating=True,
                    exact=False,
                    unit="ns",
                    kind="other",
                    short_name="decode/time",
                ),
            ),
        },
    )


def n_way_kind_metric(
    *,
    kind: str,
    short_name: str,
    candidates: Sequence[NWayCandidate],
    gating: bool = True,
) -> MetricComparison:
    """A metric of ``kind``, displayed under ``short_name``, judged once per candidate."""
    metric = n_way_metric(candidates)
    return replace(
        metric,
        meta=replace(metric.meta, kind=kind, short_name=short_name, gating=gating),
    )


def kind_metric(
    *,
    kind: str,
    short_name: str,
    verdict: ApproximateVerdict,
    delta: float,
    gating: bool = True,
    unit: MetricUnit | None = "ns",
) -> MetricComparison:
    """A metric of ``kind``, displayed under ``short_name``, judged by the permutation test."""
    metric = permutation_metric(verdict=verdict, delta=delta, gating=gating, unit=unit)
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


def two_kind_metrics() -> MetricComparisons:
    """A gating ``time`` kind (a grouped pair plus a bare row) and an informational ``memory`` kind.

    ``time`` holds a two-metric ``entity`` group beside an ungrouped ``warmup``,
    so its rendered section carries both a group block and a bare row; ``memory``
    holds one ungrouped metric, so its rendered section carries no group rows.
    """
    return {
        "entity/alive_check#time": kind_metric(
            kind="time", short_name="entity.alive_check", verdict="improved", delta=-10
        ),
        "entity/spawn#time": kind_metric(
            kind="time", short_name="entity.spawn", verdict="regressed", delta=4
        ),
        "warmup#time": kind_metric(
            kind="time", short_name="warmup", verdict="no-signal", delta=0.3
        ),
        "encode#memory": kind_metric(
            kind="memory",
            short_name="encode",
            verdict="improved",
            delta=-7,
            gating=False,
            unit="bytes",
        ),
    }


def time_kind() -> KindAggregate:
    """The gating ``time`` aggregate: a grouped pair and an ungrouped metric.

    Both its geomeans carry the band propagated from the metrics behind them, and
    both sit outside it, so a section rendered from this aggregate shows a band
    beside every figure it prints.
    """
    geomean = geomean_of(-3.2, 3, band=2)
    return KindAggregate(
        kind="time",
        geomean=geomean,
        groups=(GroupAggregate(group="entity", geomean=geomean_of(-3.1, 2, band=1.5)),),
        gated_geomean=geomean,
    )


def memory_kind() -> KindAggregate:
    """The informational ``memory`` aggregate: one ungrouped metric, nothing gated.

    Its geomean keeps the default zero band, so a section rendered from this
    aggregate shows the figure alone.
    """
    return KindAggregate(kind="memory", geomean=geomean_of(-7, 1), groups=(), gated_geomean=None)


def two_kind_result() -> ComparisonResult:
    """A single-candidate comparison spanning the gating ``time`` and informational ``memory`` kinds."""
    return create_comparison_result(
        metrics=two_kind_metrics(),
        candidates=[create_candidate(kinds=[time_kind(), memory_kind()])],
        config_kinds={"memory": KindEntry(gating=False)},
    )


def without_gated_geomean(kind: KindAggregate) -> KindAggregate:
    """The kind with its gated geomean cleared, as a non-gating kind carries."""
    return replace(kind, gated_geomean=None)


def grouped_comparison() -> ComparisonResult:
    """A two-candidate run spanning a grouped ``time`` kind and a ``memory`` kind.

    A run of a single kind renders flat and drops its group rows, so the second
    kind is what makes the ``entity`` group render at all.
    """
    return create_comparison_result(
        metrics={
            "entity/alive_check#time": n_way_kind_metric(
                kind="time",
                short_name="entity.alive_check",
                candidates=[
                    NWayCandidate(verdict="improved", delta=-10, median=90),
                    NWayCandidate(verdict="regressed", delta=4, median=104),
                ],
            ),
            "encode#memory": n_way_kind_metric(
                kind="memory",
                short_name="encode",
                gating=False,
                candidates=[
                    NWayCandidate(verdict="improved", delta=-7, median=93),
                    NWayCandidate(verdict="improved", delta=-2, median=98),
                ],
            ),
        },
        candidates=[
            create_candidate(
                label="candidate-a",
                kinds=[
                    KindAggregate(
                        kind="time",
                        geomean=geomean_of(-10, 1),
                        groups=(GroupAggregate(group="entity", geomean=geomean_of(-10, 1)),),
                        gated_geomean=geomean_of(-10, 1),
                    ),
                    memory_kind(),
                ],
            ),
            create_candidate(
                label="candidate-b",
                kinds=[
                    KindAggregate(
                        kind="time",
                        geomean=geomean_of(4, 1),
                        groups=(GroupAggregate(group="entity", geomean=geomean_of(4, 1)),),
                        gated_geomean=geomean_of(4, 1),
                    ),
                    KindAggregate(
                        kind="memory",
                        geomean=geomean_of(-2, 1),
                        groups=(),
                        gated_geomean=None,
                    ),
                ],
            ),
        ],
        config_kinds={"memory": KindEntry(gating=False)},
    )
