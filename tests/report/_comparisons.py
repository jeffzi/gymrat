"""Metadata, structural, and comparison metric builders for report formatting tests."""

from __future__ import annotations

import math
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


def informational_kind(kind: str, geomean: GeomeanResult) -> KindAggregate:
    """A kind aggregate that gates nothing and holds no groups.

    Args:
        kind: The kind's name.
        geomean: The section geomean.

    Returns:
        The kind aggregate, with no gated geomean.
    """
    return KindAggregate(kind=kind, geomean=geomean, groups=())


def other_kind(
    value: float,
    n: int,
    *,
    band: float = 0.0,
    excluded: Sequence[Exclusion] = (),
) -> KindAggregate:
    """The single-kind ``other`` aggregate every default here describes.

    It gates and holds no groups, so a caller excluding metrics or widening the
    band writes that only once.

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
    """A two-sided metric whose verdict came from the permutation method.

    Args:
        verdict: The verdict the permutation test reached.
        delta: The candidate's delta against the baseline, in percent.
        baseline_median: The baseline's measured value.
        baseline_spread: The baseline's spread, in percent of its median.
        p: The permutation test's p-value.
        noise_pct: The noise band, in percent.
        noise_abs: The noise band, in the metric's own unit.
        unit: The metric's unit, or ``None`` for a unitless count.
        gating: Whether the metric counts toward the gated geomean.
        direction: Which way is better for the metric.
        n: The pair count behind the verdict.

    Returns:
        The metric comparison.
    """
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
    baseline_median: float = 1000.0,
    median: float | None = None,
    short_name: str = "heap",
) -> MetricComparison:
    """A counted metric with no spread, compared exactly rather than statistically.

    Args:
        delta: The candidate's delta against the baseline, in percent; NaN
            stands for an undefined ratio.
        n: The pair count behind the verdict.
        unit: The metric's unit, or ``None`` for a unitless count.
        baseline_median: The baseline's measured value.
        median: The candidate's measured value; ``None`` derives it from
            ``baseline_median`` and ``delta``.
        short_name: The name the metric displays under.

    Returns:
        The metric comparison.
    """
    return MetricComparison(
        baseline_median=baseline_median,
        baseline_spread=None,
        candidates=(
            CandidateMetric(
                median=baseline_median * (1 + delta / 100) if median is None else median,
                verdict=exact_verdict(delta=delta, n=n),
            ),
        ),
        meta=metric_meta(short_name, exact=True, unit=unit),
    )


def undefined_ratio_metric(short_name: str, *, n: int) -> MetricComparison:
    """A unitless exact metric whose zero baseline leaves its delta undefined.

    Args:
        short_name: The name the metric displays under.
        n: The pair count behind the verdict.

    Returns:
        The metric comparison, its candidate at 120 against a baseline of 0.
    """
    return exact_metric(
        delta=math.nan, n=n, unit=None, baseline_median=0, median=120, short_name=short_name
    )


@dataclass(frozen=True, slots=True)
class NWayCandidate:
    """One candidate's permutation outcome, carrying its own measured median."""

    verdict: ApproximateVerdict
    delta: float
    median: float


def permutation_candidate(
    *, verdict: ApproximateVerdict, delta: float, median: float
) -> CandidateMetric:
    """One candidate's slice of a metric, judged by the permutation test.

    Args:
        verdict: The verdict the permutation test reached.
        delta: The candidate's delta against the baseline, in percent.
        median: The candidate's measured value.

    Returns:
        The candidate's metric slice.
    """
    return CandidateMetric(
        median=median,
        spread=1.0,
        verdict=permutation_verdict(verdict=verdict, delta=delta, p=0.01, noise_abs=3.5),
    )


def shared_baseline_metric(
    candidates: Sequence[CandidateMetric], *, name: str = "time"
) -> MetricComparison:
    """A nanosecond metric holding the given candidate slices against one shared baseline.

    Args:
        candidates: Each candidate's slice of the metric, in candidate order.
        name: The name the metric displays under.

    Returns:
        The metric comparison, its baseline at 100 with a 1% spread.
    """
    return MetricComparison(
        baseline_median=100.0,
        baseline_spread=1.0,
        candidates=tuple(candidates),
        meta=metric_meta(name, unit="ns"),
    )


def n_way_metric(candidates: Sequence[NWayCandidate]) -> MetricComparison:
    """One metric judged for several candidates against a single shared baseline."""
    return shared_baseline_metric([
        permutation_candidate(
            verdict=candidate.verdict, delta=candidate.delta, median=candidate.median
        )
        for candidate in candidates
    ])


def multi_candidate_result(
    candidate_count: int = 3,
    *,
    labels: Sequence[str] = ("candidate-a", "candidate-b", "candidate-c"),
    name: str = "decode/time",
) -> ComparisonResult:
    """A multi-candidate comparison with one metric judged per candidate.

    Args:
        candidate_count: ``3`` gives the first candidate improved, the second
            regressed and the third unstable (band method); ``2`` gives the
            first pair alone.
        labels: The candidate labels, in order; only the first
            ``candidate_count`` are used.
        name: The name of the one metric every candidate is judged on.

    Returns:
        The comparison result.
    """
    candidates = [
        create_candidate(label=labels[0], kinds=[other_kind(-10, 1)]),
        create_candidate(label=labels[1], kinds=[other_kind(4, 1)]),
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
        candidates.append(create_candidate(label=labels[2], kinds=[other_kind(0, 1)]))
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
            name: shared_baseline_metric(metric_candidates, name=name),
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
    """A metric of ``kind``, displayed under ``short_name``, judged by the permutation test.

    Args:
        kind: The kind the metric belongs to.
        short_name: The name the metric displays under.
        verdict: The verdict the permutation test reached.
        delta: The candidate's delta against the baseline, in percent.
        gating: Whether the metric counts toward the gated geomean.
        unit: The metric's unit, or ``None`` for a unitless count.

    Returns:
        The metric comparison.
    """
    metric = permutation_metric(verdict=verdict, delta=delta, gating=gating, unit=unit)
    return replace(metric, meta=replace(metric.meta, kind=kind, short_name=short_name))


def mixed_methods_result(*, n: int) -> ComparisonResult:
    """Banded, exact and unstable rows sharing one verdict column.

    Args:
        n: The pair count behind every row's verdict.

    Returns:
        The comparison result.
    """
    return create_comparison_result(
        metrics={
            "latency#other": permutation_metric(verdict="improved", delta=-10, n=n),
            "heap#other": exact_metric(delta=-5, n=n),
            "flaky#other": permutation_metric(verdict="unstable", delta=50, n=n),
        },
    )


def every_class_metrics() -> MetricComparisons:
    """One metric per display class: improved, regressed, within noise, identical, inconclusive, unstable."""
    return {
        "faster/time": permutation_metric(verdict="improved", delta=-17.5, unit="ns"),
        "slower/time": permutation_metric(verdict="regressed", delta=2.4, unit="ns"),
        "flat/time": permutation_metric(verdict="no-signal", delta=0.3, unit="ns"),
        "tied/heap": band_metric(verdict="no-signal", delta=-0.5, n=10, usable_n=0),
        "single-pair/time": band_metric(delta=-0.4, noise_pct=0.5, n=1, unit="ns"),
        "jittery/time": permutation_metric(verdict="unstable", delta=-50, noise_pct=30),
    }


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


def entity_time_metrics() -> MetricComparisons:
    """The ``entity`` group's two ``time`` metrics: ``alive_check`` improved, ``spawn`` regressed."""
    return {
        "entity/alive_check#time": kind_metric(
            kind="time", short_name="entity.alive_check", verdict="improved", delta=-10
        ),
        "entity/spawn#time": kind_metric(
            kind="time", short_name="entity.spawn", verdict="regressed", delta=4
        ),
    }


def two_kind_metrics() -> MetricComparisons:
    """A gating ``time`` kind (a grouped pair plus a bare row) and an informational ``memory`` kind.

    ``time`` holds a two-metric ``entity`` group beside an ungrouped ``warmup``,
    so its rendered section carries both a group block and a bare row; ``memory``
    holds one ungrouped metric, so its rendered section carries no group rows.

    Returns:
        The metrics keyed by full metric name.
    """
    return {
        **entity_time_metrics(),
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
    return informational_kind("memory", geomean_of(-7, 1))


def two_kind_result(kinds: Sequence[KindAggregate] | None = None) -> ComparisonResult:
    """A single-candidate comparison spanning the gating ``time`` and informational ``memory`` kinds.

    Args:
        kinds: The candidate's kind aggregates; ``None`` gives the ``time`` and
            ``memory`` aggregates the metrics describe.

    Returns:
        The comparison result.
    """
    return create_comparison_result(
        metrics=two_kind_metrics(),
        candidates=[
            create_candidate(kinds=kinds if kinds is not None else [time_kind(), memory_kind()])
        ],
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
                    informational_kind("memory", geomean_of(-2, 1)),
                ],
            ),
        ],
        config_kinds={"memory": KindEntry(gating=False)},
    )
