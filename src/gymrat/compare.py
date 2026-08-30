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
from typing import Literal

from gymrat.adapters import get_adapter
from gymrat.adapters.types import WarnSink, warn_to_stderr
from gymrat.config import KindEntry
from gymrat.model import (
    DEFAULT_UNSTABLE_NOISE_PCT,
    MetricVerdict,
    Observations,
    ResolvedMetricMeta,
    pair_metric,
)
from gymrat.report.types import (
    CandidateComparison,
    CandidateMetric,
    ComparisonResult,
    MetricComparison,
)
from gymrat.sampling import (
    RunOptions,
    SamplingOptions,
    TargetContext,
    TargetSamples,
    TargetSpec,
    collect_samples,
    compute_metric_stats,
    paired_or_own_values,
    resolve_dir,
    resolve_label,
    resolve_metric_meta_from_samples,
    run_with_worktrees,
)
from gymrat.targets import CleanupResult, Target, WorktreeInfo, resolve_target
from gymrat.verdict import KindAggregate, compute_kind_aggregates, compute_verdicts


@dataclass(frozen=True, slots=True, kw_only=True)
class CompareOptions(RunOptions):
    """Caller-facing configuration for a single comparison run.

    One baseline, one or more candidates: every candidate is compared with the
    baseline and never with another candidate.

    Attributes:
        baseline: The revision every candidate is judged against.
        candidates: The revisions judged against ``baseline``, reported in order.
        unstable_noise_pct: Noise band width, in percent, above which a metric is
            reported unstable. ``None`` defers to the verdict engine's own default
            — the CLI always supplies the resolved config value, so only direct
            callers see the fallback.
    """

    baseline: TargetSpec
    candidates: list[TargetSpec]
    unstable_noise_pct: float | None = None


@dataclass(frozen=True, slots=True)
class CandidateMeasurement:
    """One candidate's samples and the verdicts they earned against the baseline."""

    label: str
    samples: list[dict[str, float]]
    verdicts: dict[str, MetricVerdict]
    kinds: list[KindAggregate]


@dataclass(frozen=True, slots=True)
class _Measurement:
    """Everything the measurement phase produces that the report is built from.

    Bundling these lets the whole set outlive the phase, so the report can be
    assembled after worktree cleanup has already run.
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
    """
    paired: list[float] = []
    for index, sample in enumerate(baseline_samples):
        if metric_name not in sample:
            continue
        has_candidate = any(
            index < len(samples) and metric_name in samples[index]
            for samples in candidate_sample_sets
        )
        if has_candidate:
            paired.append(sample[metric_name])
    return paired_or_own_values(paired, baseline_samples, metric_name)


def _measure_candidates(
    baseline_samples: list[dict[str, float]],
    candidates: list[TargetSamples],
    metric_meta: dict[str, ResolvedMetricMeta],
    unstable_noise_pct: float | None,
    warn: WarnSink | None,
) -> list[CandidateMeasurement]:
    """Judge every candidate against the same baseline samples, one comparison each."""
    resolved_noise_pct = (
        unstable_noise_pct if unstable_noise_pct is not None else DEFAULT_UNSTABLE_NOISE_PCT
    )
    resolved_warn = warn if warn is not None else warn_to_stderr

    baseline_obs = Observations.from_rounds(baseline_samples)
    measured: list[CandidateMeasurement] = []
    for candidate in candidates:
        verdicts = compute_verdicts(
            baseline_obs,
            Observations.from_rounds(candidate.samples),
            metric_meta,
            unstable_noise_pct=resolved_noise_pct,
            warn=resolved_warn,
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


def build_comparison_result(  # noqa: PLR0913 -- flat parameter list avoids an intermediate dataclass
    baseline_label: str,
    baseline_samples: list[dict[str, float]],
    candidates: list[CandidateMeasurement],
    metric_meta: dict[str, ResolvedMetricMeta],
    *,
    samples: int,
    adapter: str,
    config_kinds: dict[str, KindEntry] | None,
    cleanup: CleanupResult,
) -> ComparisonResult:
    """Build a comparison result from measured candidates and a cleanup outcome.

    Both ``compare`` (multi-candidate, real cleanup) and the loop engine
    (single candidate, zeroed cleanup) call this.
    """
    candidate_sample_sets = [c.samples for c in candidates]
    baseline_obs = Observations.from_rounds(baseline_samples)

    metrics: dict[str, MetricComparison] = {}
    for metric_name, meta in metric_meta.items():
        baseline_stats = compute_metric_stats(
            _baseline_paired_values(baseline_samples, candidate_sample_sets, metric_name)
        )
        candidate_metrics: list[CandidateMetric] = []
        for candidate in candidates:
            paired = pair_metric(
                baseline_obs, Observations.from_rounds(candidate.samples), metric_name
            ).right
            stats = compute_metric_stats(
                paired_or_own_values(paired, candidate.samples, metric_name)
            )
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
        baseline_label=baseline_label,
        candidates=tuple(
            CandidateComparison(label=c.label, kinds=tuple(c.kinds)) for c in candidates
        ),
        samples=samples,
        adapter=adapter,
        metrics=metrics,
        config_kinds=config_kinds,
    )


def _to_context(
    spec: TargetSpec,
    target: Target,
    position: Literal["old", "new"],
    repo_dir: str,
    worktrees: list[WorktreeInfo],
) -> TargetContext:
    return TargetContext(
        target=target,
        dir=resolve_dir(target, repo_dir, worktrees),
        label=resolve_label(spec.label, target),
        position=position,
    )


async def _compare_phase(
    options: CompareOptions,
    repo_dir: str,
    worktrees: list[WorktreeInfo],
    abort: asyncio.Event,
) -> _Measurement:
    adapter = get_adapter(options.adapter)

    baseline_target = resolve_target(options.baseline.target, repo_dir)
    candidate_targets = [
        (spec, resolve_target(spec.target, repo_dir)) for spec in options.candidates
    ]

    baseline_context = _to_context(options.baseline, baseline_target, "old", repo_dir, worktrees)
    candidate_contexts = [
        _to_context(spec, target, "new", repo_dir, worktrees) for spec, target in candidate_targets
    ]

    baseline, *candidates = await collect_samples(
        adapter,
        [baseline_context, *candidate_contexts],
        SamplingOptions(
            bench=options.bench,
            prepare=options.prepare,
            samples=options.samples,
            timeout_seconds=options.timeout_seconds,
            on_progress=options.on_progress,
            warn=options.warn,
        ),
        abort,
    )

    metric_meta = resolve_metric_meta_from_samples(
        [baseline.samples, *(candidate.samples for candidate in candidates)],
        options.config_metrics,
        adapter,
        options.config_kinds,
    )

    return _Measurement(
        baseline_label=baseline.ctx.label,
        baseline_samples=baseline.samples,
        candidates=_measure_candidates(
            baseline.samples,
            candidates,
            metric_meta,
            options.unstable_noise_pct,
            options.warn,
        ),
        metric_meta=metric_meta,
    )


async def compare(options: CompareOptions) -> ComparisonResult:
    """Compare one baseline revision against one or more candidate revisions.

    Resolves every target's directory or ref, runs the bench round-robin across
    all of them, parses each run with the configured adapter, and computes each
    candidate's verdicts against the shared baseline.
    """
    return await run_with_worktrees(
        lambda repo_dir, worktrees, abort: _compare_phase(options, repo_dir, worktrees, abort),
        lambda measurement, cleanup: build_comparison_result(
            measurement.baseline_label,
            measurement.baseline_samples,
            measurement.candidates,
            measurement.metric_meta,
            samples=options.samples,
            adapter=options.adapter,
            config_kinds=options.config_kinds,
            cleanup=cleanup,
        ),
    )
