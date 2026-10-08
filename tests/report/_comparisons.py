"""Metadata, structural, and comparison metric builders for report formatting tests."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from gymrat.config import KindEntry
from gymrat.report.types import (
    CandidateComparison,
    CandidateMetric,
    ComparisonResult,
    MetricComparison,
    MetricComparisons,
)
from gymrat.verdict import GroupAggregate, KindAggregate
from tests.report._verdicts import (
    band_metric,
    band_verdict,
    exact_verdict,
    geomean_of,
    metric_meta,
    permutation_verdict,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from gymrat.model import (
        ApproximateVerdict,
        Direction,
        Exclusion,
        GeomeanResult,
        MetricUnit,
    )
    from gymrat.targets import WorktreeRemovalFailure


# ---------------------------------------------------------------------------
# Metadata and structural builders
# ---------------------------------------------------------------------------


def gating_kind(
    kind: str,
    geomean: GeomeanResult,
    groups: Mapping[str, GeomeanResult] | None = None,
) -> KindAggregate:
    """A gating kind aggregate whose gated geomean is its section geomean.

    Args:
        kind: The kind's name.
        geomean: The section geomean, which the kind also gates on.
        groups: Each group's geomean keyed by group name, in display order;
            ``None`` gives a kind with no groups.

    Returns:
        The kind aggregate.
    """
    return KindAggregate(
        kind=kind,
        geomean=geomean,
        groups=tuple(
            GroupAggregate(group=group, geomean=group_geomean)
            for group, group_geomean in (groups or {}).items()
        ),
        gated_geomean=geomean,
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

    Args:
        value: The geomean delta, in percent.
        n: How many metrics the geomean covers.
        band: The geomean's noise band, in percent.
        excluded: The metrics the geomean left out, with their reasons.

    Returns:
        The kind aggregate, its gated geomean equal to its section geomean.
    """
    return gating_kind("other", geomean_of(value, n, band=band, excluded=excluded))


def create_candidate(
    *,
    label: str = "perf/faster-decode",
    kinds: Sequence[KindAggregate] | None = None,
) -> CandidateComparison:
    """One candidate's run-level results, judged against the shared baseline.

    Args:
        label: The candidate's target label.
        kinds: The candidate's kind aggregates. ``None`` gives the single-kind
            run every other default here describes: one ``other`` kind, no
            groups, whose section and gated geomeans share the same default
            aggregate.

    Returns:
        The candidate comparison.
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
    p: float = 0.01,
    noise_pct: float = 2.5,
    noise_abs: float = 3.5,
    unit: MetricUnit | None = None,
    gating: bool = True,
    direction: Direction = "lower",
    n: int = 10,
) -> MetricComparison:
    """A two-sided metric whose verdict came from the permutation method."""
    return MetricComparison(
        baseline_median=baseline_median,
        baseline_spread=baseline_spread,
        candidates=(
            CandidateMetric(
                median=baseline_median * (1 + delta / 100),
                spread=1.0,
                verdict=permutation_verdict(
                    verdict=verdict,
                    delta=delta,
                    n=n,
                    p=p,
                    noise_pct=noise_pct,
                    noise_abs=noise_abs,
                ),
            ),
        ),
        meta=metric_meta("time", direction=direction, gating=gating, unit=unit),
    )


def exact_metric(
    *,
    delta: float,
    n: int = 10,
    unit: MetricUnit | None = "bytes",
) -> MetricComparison:
    """A counted metric with a 1000 baseline, compared exactly rather than statistically."""
    baseline_median = 1000.0
    return MetricComparison(
        baseline_median=baseline_median,
        baseline_spread=None,
        candidates=(
            CandidateMetric(
                median=baseline_median * (1 + delta / 100),
                verdict=exact_verdict(delta=delta, n=n),
            ),
        ),
        meta=metric_meta("heap", exact=True, unit=unit),
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
                verdict=permutation_verdict(
                    verdict=candidate.verdict,
                    delta=candidate.delta,
                    p=0.01,
                    noise_abs=3.5,
                ),
            )
            for candidate in candidates
        ),
        meta=metric_meta("time", unit="ns"),
    )


def multi_candidate_result(candidate_count: int = 3) -> ComparisonResult:
    """A multi-candidate comparison with one metric judged per candidate.

    Args:
        candidate_count: ``3`` gives ``candidate-a`` improved, ``candidate-b``
            regressed and ``candidate-c`` unstable (band method); ``2`` gives
            the first pair alone.

    Returns:
        The comparison result.
    """
    candidates = [
        create_candidate(label="candidate-a", kinds=[other_kind(-10, 1)]),
        create_candidate(label="candidate-b", kinds=[other_kind(4, 1)]),
    ]
    metric_candidates = [
        CandidateMetric(
            median=90.0,
            spread=1.0,
            verdict=permutation_verdict(verdict="improved", delta=-10, p=0.002),
        ),
        CandidateMetric(
            median=104.0,
            spread=1.0,
            verdict=permutation_verdict(verdict="regressed", delta=4, p=0.002),
        ),
    ]
    if candidate_count == 3:
        candidates.append(create_candidate(label="candidate-c", kinds=[other_kind(0, 1)]))
        metric_candidates.append(
            CandidateMetric(
                median=150.0,
                spread=3.0,
                verdict=band_verdict(
                    verdict="unstable", delta=50, usable_n=3, noise_pct=30, noise_abs=30
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
                meta=metric_meta("decode/time", unit="ns"),
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

    Returns:
        The single-sample comparison result.
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

    Returns:
        The metrics keyed by full metric name.
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

    Returns:
        The ``time`` kind aggregate.
    """
    return gating_kind(
        "time", geomean_of(-3.2, 3, band=2), {"entity": geomean_of(-3.1, 2, band=1.5)}
    )


def memory_kind() -> KindAggregate:
    """The informational ``memory`` aggregate: one ungrouped metric, nothing gated.

    Its geomean keeps the default zero band, so a section rendered from this
    aggregate shows the figure alone.

    Returns:
        The ``memory`` kind aggregate.
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

    Returns:
        The two-candidate comparison result.
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
                    gating_kind("time", geomean_of(-10, 1), {"entity": geomean_of(-10, 1)}),
                    memory_kind(),
                ],
            ),
            create_candidate(
                label="candidate-b",
                kinds=[
                    gating_kind("time", geomean_of(4, 1), {"entity": geomean_of(4, 1)}),
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
