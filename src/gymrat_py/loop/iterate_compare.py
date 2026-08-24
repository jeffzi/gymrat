"""Assemble the comparison a single iteration's report is drawn from.

An iteration measures one candidate — the experiment — against one baseline, and
its two worktrees outlive the run, so the cleanup fields state a sweep that never
happened rather than one that found nothing to do. That single-candidate,
no-cleanup shape is the only way this differs from :func:`gymrat_py.compare`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from gymrat_py.model import MetricVerdict, Observations, ResolvedMetricMeta, pair_metric
from gymrat_py.report.types import (
    CandidateComparison,
    CandidateMetric,
    ComparisonResult,
    MetricComparison,
)
from gymrat_py.sampling import TargetSamples, compute_metric_stats, paired_or_own_values
from gymrat_py.verdict import compute_kind_aggregates

if TYPE_CHECKING:
    from gymrat_py.config import ResolvedConfig


@dataclass(frozen=True, slots=True)
class BenchRunOutputs:
    """One bench run's measurement outputs, shared by the record and the report.

    Attributes:
        baseline: The baseline worktree's samples and context.
        experiment: The experiment worktree's samples and context.
        verdicts: The per-metric verdicts drawn from the two sides.
        metric_meta: The metric metadata resolved from the run's samples.
    """

    baseline: TargetSamples
    experiment: TargetSamples
    verdicts: dict[str, MetricVerdict]
    metric_meta: dict[str, ResolvedMetricMeta]


def _compare_metric(
    metric_name: str,
    baseline_obs: Observations,
    experiment_obs: Observations,
    run: BenchRunOutputs,
    meta: ResolvedMetricMeta,
) -> MetricComparison:
    """One metric's two sides, each measured over the rounds the verdict was drawn from.

    Displaying a median off exactly the verdict's paired rounds keeps it from ever
    disagreeing with the delta beside it. A metric only one side reported has no
    pairs and therefore no verdict to stay consistent with, so that side falls
    back to every round it did report.
    """
    pair = pair_metric(baseline_obs, experiment_obs, metric_name)
    baseline_stats = compute_metric_stats(
        paired_or_own_values(pair.left, run.baseline.samples, metric_name)
    )
    experiment_stats = compute_metric_stats(
        paired_or_own_values(pair.right, run.experiment.samples, metric_name)
    )
    return MetricComparison(
        baseline_median=baseline_stats.median,
        baseline_spread=baseline_stats.spread,
        candidates=(
            CandidateMetric(
                median=experiment_stats.median,
                spread=experiment_stats.spread,
                verdict=run.verdicts.get(metric_name),
            ),
        ),
        meta=meta,
    )


def build_comparison_result(run: BenchRunOutputs, config: ResolvedConfig) -> ComparisonResult:
    """The comparison one iteration's report is drawn from: one baseline, one candidate.

    Args:
        run: The bench run's samples, verdicts, and metric metadata.
        config: The resolved run configuration, read for the adapter and kinds.

    Returns:
        A comparison result with no worktree cleanup to account for.
    """
    baseline_obs = Observations.from_rounds(run.baseline.samples)
    experiment_obs = Observations.from_rounds(run.experiment.samples)

    metrics = {
        metric_name: _compare_metric(metric_name, baseline_obs, experiment_obs, run, meta)
        for metric_name, meta in run.metric_meta.items()
    }

    return ComparisonResult(
        worktrees_removed=0,
        worktrees_left_behind=(),
        worktree_prune_error=None,
        baseline_label=run.baseline.ctx.label,
        candidates=(
            CandidateComparison(
                label=run.experiment.ctx.label,
                kinds=tuple(compute_kind_aggregates(run.verdicts, run.metric_meta)),
            ),
        ),
        samples=min(len(run.baseline.samples), len(run.experiment.samples)),
        adapter=config.adapter,
        metrics=metrics,
        config_kinds=config.kinds,
    )
