"""Compare one baseline revision against one or more candidate revisions.

The topology is a star: every candidate is judged against the same baseline
samples and never against another candidate. Reusing one set of baseline samples
is what keeps that affordable, and it is also why the resulting verdicts are
statistically correlated — a baseline round that ran slow inflates every
candidate's delta at once. Each verdict is still sound evidence about its own
candidate; the gap between two candidates' deltas is not a quantity this test
measured.

Rendering is the caller's job. Worktree cleanup and signal handling belong to
:func:`~gymrat.sampling.run_with_worktrees`.
"""

import asyncio
from dataclasses import dataclass
from functools import partial

from gymrat.adapters import get_adapter
from gymrat.config.types import KindEntry
from gymrat.model import MetricVerdict, ResolvedMetricMeta, pair_metric
from gymrat.report.types import (
    CandidateComparison,
    CandidateMetric,
    ComparisonResult,
    MetricComparison,
)
from gymrat.sampling import (
    RunOptions,
    TargetSamples,
    TargetSpec,
    collect_samples,
    compute_metric_stats,
    own_values,
    resolve_metric_meta_from_samples,
    run_with_worktrees,
    to_context,
)
from gymrat.targets import CleanupResult, WorktreeInfo, resolve_target
from gymrat.verdict import KindAggregate, compute_kind_aggregates, compute_verdicts
from gymrat.warn import WarnSink


@dataclass(frozen=True, slots=True, kw_only=True)
class CompareOptions:
    """Caller-facing configuration for a single comparison run.

    One baseline, one or more candidates: every candidate is compared with the
    baseline and never with another candidate.

    Attributes:
        run: The bench, sampling, adapter, and config-override settings of the run.
        baseline: The revision every candidate is judged against.
        candidates: The revisions judged against ``baseline``, reported in order.
        unstable_noise_pct: Noise band width, in percent, above which a metric is
            reported unstable.
    """

    run: RunOptions
    baseline: TargetSpec
    candidates: list[TargetSpec]
    unstable_noise_pct: float


@dataclass(frozen=True, slots=True)
class CandidateMeasurement:
    """One candidate's samples and the verdicts they earned against the baseline."""

    label: str
    samples: list[dict[str, float]]
    verdicts: dict[str, MetricVerdict]
    kinds: list[KindAggregate]


@dataclass(frozen=True, slots=True)
class ComparisonMeasurement:
    """Everything the measurement phase produces that the report is built from.

    Bundling these lets the whole set outlive the phase, so the report can be
    assembled after worktree cleanup has already run.

    Attributes:
        baseline_label: Display label for the baseline target.
        baseline_samples: Per-round metric samples collected for the baseline.
        candidates: Measured candidates, each with its own verdicts and kinds.
        metric_meta: Resolved metric metadata keyed by metric name.
    """

    baseline_label: str
    baseline_samples: list[dict[str, float]]
    candidates: list[CandidateMeasurement]
    metric_meta: dict[str, ResolvedMetricMeta]


def _baseline_paired_values(
    baseline_samples: list[dict[str, float]],
    candidate_sample_sets: list[list[dict[str, float]]],
    metric_name: str,
) -> list[float]:
    """The baseline's values for a metric, restricted to candidate-paired rounds.

    A round counts only when at least one candidate also reported the metric at
    the same index — the same rounds a verdict's delta can be drawn from for at
    least one candidate. When no candidate ever reported the metric, falls back to
    every round the baseline reported it in: a baseline-only metric has no verdict
    to stay consistent with, so its displayed median is the baseline's own.

    Args:
        baseline_samples: The baseline's per-round samples.
        candidate_sample_sets: Each candidate's per-round samples.
        metric_name: The metric whose values to collect.

    Returns:
        The baseline float values paired with at least one candidate, or all
        baseline values when no candidate reported the metric.
    """
    paired = [
        sample[metric_name]
        for index, sample in enumerate(baseline_samples)
        if metric_name in sample
        and any(
            index < len(samples) and metric_name in samples[index]
            for samples in candidate_sample_sets
        )
    ]
    return paired or own_values(baseline_samples, metric_name)


def _measure_candidates(
    baseline_samples: list[dict[str, float]],
    candidates: list[TargetSamples],
    metric_meta: dict[str, ResolvedMetricMeta],
    unstable_noise_pct: float,
    warn: WarnSink,
) -> list[CandidateMeasurement]:
    """Judge every candidate against the same baseline samples, one comparison each."""
    measured: list[CandidateMeasurement] = []
    for candidate in candidates:
        verdicts = compute_verdicts(
            baseline_samples,
            candidate.samples,
            metric_meta,
            unstable_noise_pct=unstable_noise_pct,
            warn=warn,
        )
        measured.append(
            CandidateMeasurement(
                label=candidate.ctx.label,
                samples=candidate.samples,
                verdicts=verdicts,
                kinds=compute_kind_aggregates(verdicts, metric_meta),
            )
        )
    return measured


def build_comparison_result(
    measurement: ComparisonMeasurement,
    cleanup: CleanupResult,
    *,
    samples: int,
    adapter: str,
    config_kinds: dict[str, KindEntry] | None,
) -> ComparisonResult:
    """Build a comparison result from measured candidates and a cleanup outcome.

    Both ``compare`` (multi-candidate, real cleanup) and the loop engine
    (single candidate, zeroed cleanup) call this.

    Args:
        measurement: The baseline samples, the measured candidates, and the
            resolved metric metadata.
        cleanup: Outcome of the worktree cleanup performed after sampling.
        samples: Number of samples requested per target.
        adapter: Name of the adapter used to parse bench output.
        config_kinds: Kind entries from the config, or ``None`` when not set.

    Returns:
        The assembled :class:`ComparisonResult` with per-metric baselines,
        candidate verdicts, and worktree cleanup status.
    """
    baseline_samples = measurement.baseline_samples
    candidates = measurement.candidates
    candidate_sample_sets = [c.samples for c in candidates]

    metrics: dict[str, MetricComparison] = {}
    for metric_name, meta in measurement.metric_meta.items():
        baseline_stats = compute_metric_stats(
            _baseline_paired_values(baseline_samples, candidate_sample_sets, metric_name)
        )
        candidate_metrics: list[CandidateMetric] = []
        for candidate in candidates:
            paired = pair_metric(baseline_samples, candidate.samples, metric_name).right
            stats = compute_metric_stats(list(paired) or own_values(candidate.samples, metric_name))
            candidate_metrics.append(
                CandidateMetric(
                    median=stats.median,
                    spread=stats.spread,
                    verdict=candidate.verdicts.get(metric_name),
                )
            )
        metrics[metric_name] = MetricComparison(
            baseline_median=baseline_stats.median,
            baseline_spread=baseline_stats.spread,
            candidates=tuple(candidate_metrics),
            meta=meta,
        )

    return ComparisonResult(
        worktrees_removed=cleanup.removed,
        worktrees_left_behind=tuple(cleanup.failures),
        worktree_prune_error=cleanup.prune_error,
        baseline_label=measurement.baseline_label,
        candidates=tuple(
            CandidateComparison(label=c.label, kinds=tuple(c.kinds)) for c in candidates
        ),
        samples=samples,
        adapter=adapter,
        metrics=metrics,
        config_kinds=config_kinds,
    )


async def _compare_phase(
    options: CompareOptions,
    repo_dir: str,
    worktrees: list[WorktreeInfo],
    abort: asyncio.Event,
) -> ComparisonMeasurement:
    run = options.run
    adapter = get_adapter(run.adapter)

    baseline_target = resolve_target(options.baseline.target, repo_dir)
    candidate_targets = [
        (spec, resolve_target(spec.target, repo_dir)) for spec in options.candidates
    ]

    baseline_context = to_context(options.baseline, baseline_target, repo_dir, worktrees, "old")
    candidate_contexts = [
        to_context(spec, target, repo_dir, worktrees, "new") for spec, target in candidate_targets
    ]

    baseline, *candidates = await collect_samples(
        adapter,
        [baseline_context, *candidate_contexts],
        run.sampling,
        abort,
    )

    metric_meta = resolve_metric_meta_from_samples(
        [baseline.samples, *(candidate.samples for candidate in candidates)],
        run.config_metrics,
        adapter,
        run.config_kinds,
    )

    return ComparisonMeasurement(
        baseline_label=baseline.ctx.label,
        baseline_samples=baseline.samples,
        candidates=_measure_candidates(
            baseline.samples,
            candidates,
            metric_meta,
            options.unstable_noise_pct,
            run.sampling.warn,
        ),
        metric_meta=metric_meta,
    )


async def compare(options: CompareOptions) -> ComparisonResult:
    """Compare one baseline revision against one or more candidate revisions.

    Resolves every target's directory or ref, runs the bench round-robin across
    all of them, parses each run with the configured adapter, and computes each
    candidate's verdicts against the shared baseline.

    Args:
        options: Fully resolved comparison options, including baseline and
            candidate targets, sampling settings, and the adapter to use.

    Returns:
        The :class:`ComparisonResult` containing every candidate's verdicts
        against the shared baseline.

    Raises:
        GymratError: When the adapter is unknown or a target is neither a
            directory nor a resolvable ref.
        CommandError: When a prepare or bench command times out or exits
            non-zero.
    """
    run = options.run
    return await run_with_worktrees(
        partial(_compare_phase, options),
        partial(
            build_comparison_result,
            samples=run.sampling.samples,
            adapter=run.adapter,
            config_kinds=run.config_kinds,
        ),
    )
