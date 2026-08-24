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

import re
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from gymrat_py.config import KindEntry
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
from gymrat_py.verdict import GroupAggregate, KindAggregate

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


@dataclass(frozen=True, slots=True)
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


@dataclass(frozen=True, slots=True)
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
            verdict=SignedRankVerdict(
                method="signed-rank",
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
            verdict=SignedRankVerdict(
                method="signed-rank",
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
) -> MetricEntry:
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


def two_kind_metrics() -> MetricComparisons:
    """A gating ``time`` kind (a grouped pair plus a bare row) and an informational ``memory`` kind.

    ``time`` holds a two-metric ``entity`` group beside an ungrouped ``warmup``,
    so its rendered section carries both a group block and a bare row; ``memory``
    holds one ungrouped metric, so its rendered section carries no group rows.
    """
    return {
        "entity.alive_check/time": kind_metric(
            kind="time", short_name="entity.alive_check", verdict="improved", delta=-10
        ),
        "entity.spawn/time": kind_metric(
            kind="time", short_name="entity.spawn", verdict="regressed", delta=4
        ),
        "warmup/time": kind_metric(
            kind="time", short_name="warmup", verdict="no-signal", delta=0.3
        ),
        "encode/heap": kind_metric(
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
            "entity.alive_check/time": n_way_kind_metric(
                kind="time",
                short_name="entity.alive_check",
                candidates=[
                    NWayCandidate(verdict="improved", delta=-10, median=90),
                    NWayCandidate(verdict="regressed", delta=4, median=104),
                ],
            ),
            "encode/heap": n_way_kind_metric(
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
    config_kinds: dict[str, KindEntry] | None = None,
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
        config_kinds=config_kinds,
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
        config_kinds={"memory": KindEntry(gating=False)},
        worktrees_removed=worktrees_removed,
        worktrees_left_behind=worktrees_left_behind,
        worktree_prune_error=worktree_prune_error,
    )


# ---------------------------------------------------------------------------
# Rendered-text assertion helpers
# ---------------------------------------------------------------------------

# One SGR escape sequence (rich packs several parameters into one, e.g. ``1;4``).
_SGR = re.compile(r"\x1b\[([0-9;]*)m")
# The column separator every rendered table row is split on.
_SEPARATOR = "│"
# A trailing run of SGR escapes with nothing but escapes between them and the end.
_TRAILING_SGR_RUN = re.compile(r"(?:\x1b\[[0-9;]*m)*$")
# A table rule: dashes meeting the first column separator at a crossing junction.
_RULE = re.compile(r"^─+┼")
# A section border: only dashes and top-T junctions, edge to edge.
_BORDER = re.compile(r"^[─┬]+$")
# A line dimmed end to end: opens with SGR 2 and closes with a reset. Rich closes
# a dim span with a full reset (SGR 0) rather than the incremental dim-off SGR 22.
DIMMED_LINE = re.compile(r"^\x1b\[2m.*\x1b\[0m$")


def strip_ansi(text: str) -> str:
    """Every SGR escape removed, leaving the visible text a terminal would show."""
    return _SGR.sub("", text)


def table_rows(report: str) -> list[str]:
    """Every rendered table row of ``report``, styling stripped, in report order."""
    return [strip_ansi(line) for line in report.split("\n") if _SEPARATOR in line]


def cells_of(line: str) -> list[str]:
    """The cells of a rendered table line, padding included."""
    return line.split(_SEPARATOR)


def delta_cell(line: str) -> str:
    """The last cell of a rendered table line — the delta/verdict column."""
    return cells_of(line)[-1]


def last_table_row(report: str) -> str:
    """The last rendered table row of a report — the row the table closes on."""
    return table_rows(report)[-1]


def line_starting_with(report: str, prefix: str) -> str:
    """The single rendered line starting with ``prefix``, or a failure naming the report."""
    for candidate in report.split("\n"):
        if candidate.startswith(prefix):
            return candidate
    msg = f"no line starting with {prefix!r} in report:\n{report}"
    raise AssertionError(msg)


def line_containing(report: str, needle: str) -> str:
    """The first rendered line containing ``needle``, or a failure naming the report.

    A colored line starts with escape codes rather than its text, so the color
    tests match on content instead of a prefix.
    """
    for candidate in report.split("\n"):
        if needle in candidate:
            return candidate
    msg = f"no line containing {needle!r} in report:\n{report}"
    raise AssertionError(msg)


def styles_at(line: str, marker: str, *, last: bool = False) -> list[str]:
    r"""The SGR parameters opened immediately before ``marker`` in ``line``.

    Only the unbroken run of escape sequences touching the marker counts, so a
    style opened at the start of the line does not leak into the result. A reset
    (``0`` or an empty parameter list) is dropped: it closes styles rather than
    opening one. Pass ``last`` to read the trailing occurrence of a repeated
    marker instead of the leading one.

    Rich packs several parameters into one escape (``\\x1b[1;4m``), so each run is
    split on both the escape boundaries and the ``;`` inside them.
    """
    index = line.rfind(marker) if last else line.find(marker)
    if index == -1:
        msg = f"no {marker!r} in line: {line!r}"
        raise AssertionError(msg)
    run = _TRAILING_SGR_RUN.search(line[:index])
    opened = run.group(0) if run is not None else ""
    params: list[str] = []
    for escape in _SGR.finditer(opened):
        params.extend(param for param in escape.group(1).split(";") if param not in {"", "0"})
    return params


def offsets_of(line: str, glyph: str) -> list[int]:
    """Character offsets of every occurrence of ``glyph`` in a rendered line."""
    offsets: list[int] = []
    start = line.find(glyph)
    while start != -1:
        offsets.append(start)
        start = line.find(glyph, start + 1)
    return offsets


def separator_offsets(line: str) -> list[int]:
    """Character offsets of every column separator in a rendered table line.

    Two lines whose separators sit at the same offsets have aligned columns.
    """
    return [index for index, char in enumerate(line) if char == _SEPARATOR]


def separator_styles(line: str) -> list[list[str]]:
    """The SGR parameters still open at each column separator of ``line``.

    A separator that inherits its row's style reports that style here; one left in
    the terminal's default color reports nothing.
    """
    closers: dict[str, re.Pattern[str]] = {
        "0": re.compile(r"^\d+$"),
        "22": re.compile(r"^[12]$"),
        "23": re.compile(r"^3$"),
        "24": re.compile(r"^4$"),
        "39": re.compile(r"^(?:3[0-7]|9[0-7])$"),
        "49": re.compile(r"^(?:4[0-7]|10[0-7])$"),
    }
    open_params: list[str] = []
    styles: list[list[str]] = []
    for token in re.finditer(r"\x1b\[([0-9;]*)m|│", line):
        if token.group(0) == _SEPARATOR:
            styles.append(list(open_params))
            continue
        for param in token.group(1).split(";"):
            if param == "":
                continue
            closes = closers.get(param)
            if closes is None:
                open_params.append(param)
            else:
                open_params = [p for p in open_params if not closes.match(p)]
    return styles


def table_shape(report: str) -> list[str]:
    """One entry per report line, coarse enough to read as a layout.

    A table row collapses to its first cell, a header rule collapses to
    ``"<rule>"``, a section's top border to ``"<border>"``, and every other line
    stays as its plain text.
    """
    shape: list[str] = []
    for line in report.split("\n"):
        bare = strip_ansi(line)
        if _RULE.match(bare):
            shape.append("<rule>")
        elif _BORDER.match(bare):
            shape.append("<border>")
        elif _SEPARATOR not in bare:
            shape.append(bare.rstrip())
        else:
            shape.append(cells_of(bare)[0].strip())
    return shape


def table_region(report: str) -> list[str]:
    """The table region of a report: its shape down to the last table row."""
    lines = report.split("\n")
    last = -1
    for index, line in enumerate(lines):
        if _SEPARATOR in strip_ansi(line):
            last = index
    if last == -1:
        msg = f"no table rows in report:\n{report}"
        raise AssertionError(msg)
    return table_shape(report)[: last + 1]


def highlight_lines(report: str) -> list[str]:
    """The lines of the ``highlights`` block, its heading excluded.

    The block runs from the line after the ``highlights`` heading down to the
    next blank line (or the end of the report). Lines keep their styling, so the
    color tests can read the SGR parameters off a highlight entry. An absent
    block yields an empty list.
    """
    lines = report.split("\n")
    start = next(
        (index for index, line in enumerate(lines) if strip_ansi(line) == "highlights"), -1
    )
    if start == -1:
        return []
    rest = lines[start + 1 :]
    try:
        end = rest.index("")
    except ValueError:
        return rest
    return rest[:end]


__all__ = [
    "DIMMED_LINE",
    "CandidateSpec",
    "MetricEntry",
    "Metrics",
    "NWayCandidate",
    "approximate_metric",
    "band_metric",
    "band_verdict",
    "cells_of",
    "create_candidate",
    "create_comparison_result",
    "create_measurement_result",
    "delta_cell",
    "exact_metric",
    "exact_verdict",
    "geomean_of",
    "grouped_comparison",
    "highlight_lines",
    "kind_metric",
    "last_table_row",
    "line_containing",
    "line_starting_with",
    "measured_metric",
    "memory_kind",
    "metric_for",
    "metric_meta",
    "multi_candidate_result",
    "n_way_kind_metric",
    "n_way_metric",
    "offsets_of",
    "one_sided_metric",
    "other_kind",
    "separator_offsets",
    "separator_styles",
    "signed_rank_metric",
    "signed_rank_verdict",
    "single_sample_result",
    "strip_ansi",
    "styles_at",
    "table_region",
    "table_rows",
    "table_shape",
    "time_kind",
    "two_kind_measurement",
    "two_kind_metrics",
    "two_kind_result",
    "without_gated_geomean",
]
