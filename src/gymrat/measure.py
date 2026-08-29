"""Measure one revision or directory, with nothing to compare it against.

Mirrors :func:`~gymrat.compare.compare`'s worktree and signal discipline for a
single target: the target is resolved (a ref into a throwaway worktree, a
directory benched where it sits), ``prepare`` runs once, ``bench`` runs
``samples`` times, and the adapter turns each run's stdout into metrics.

Nothing is written to disk beyond that worktree — recording a run is the caller's
business. Worktree cleanup and signal handling belong to
:func:`~gymrat.sampling.run_with_worktrees`.
"""

import asyncio
from dataclasses import dataclass

from gymrat.adapters import get_adapter
from gymrat.model import ResolvedMetricMeta
from gymrat.report.types import MeasurementResult, MetricMeasurement
from gymrat.sampling import (
    RunOptions,
    SamplingOptions,
    TargetContext,
    TargetSpec,
    collect_samples,
    compute_metric_stats,
    own_values,
    resolve_dir,
    resolve_label,
    resolve_metric_meta_from_samples,
    run_with_worktrees,
)
from gymrat.targets import CleanupResult, WorktreeInfo, resolve_target


@dataclass(frozen=True, slots=True, kw_only=True)
class MeasureOptions(RunOptions):
    """Caller-facing configuration for a single measurement run.

    One target, no baseline: nothing is judged, so there is no noise band to set
    and no verdict to gate on.

    Attributes:
        target: The revision or directory to measure.
    """

    target: TargetSpec


@dataclass(frozen=True, slots=True)
class _Measurement:
    """Everything the measurement phase produced that the result is built from.

    Bundling these lets the whole set outlive the phase, so the result can be
    assembled after worktree cleanup has already run.
    """

    label: str
    samples: list[dict[str, float]]
    metric_meta: dict[str, ResolvedMetricMeta]


def _build_measurement_result(
    measurement: _Measurement,
    options: MeasureOptions,
    cleanup: CleanupResult,
) -> MeasurementResult:
    """Assemble the rendered result from the measurement and the cleanup outcome."""
    metrics: dict[str, MetricMeasurement] = {}
    for metric_name, meta in measurement.metric_meta.items():
        stats = compute_metric_stats(own_values(measurement.samples, metric_name))
        metrics[metric_name] = MetricMeasurement(
            median=stats.median, spread=stats.spread, meta=meta
        )

    return MeasurementResult(
        worktrees_removed=cleanup.removed,
        worktrees_left_behind=tuple(cleanup.failures),
        worktree_prune_error=cleanup.prune_error,
        label=measurement.label,
        samples=options.samples,
        adapter=options.adapter,
        metrics=metrics,
        rounds=tuple(measurement.samples),
        config_kinds=options.config_kinds,
    )


async def measure(options: MeasureOptions) -> MeasurementResult:
    """Measure one revision or directory.

    Resolves the target's directory or ref, runs the bench ``samples`` times,
    and parses each run with the configured adapter. No verdicts are computed and
    nothing is recorded — that is the caller's job.

    Args:
        options: The target, bench/prepare commands, adapter, and config
            overrides for the run.

    Returns:
        The assembled measurement, ready for a renderer.
    """

    async def phase(
        repo_dir: str,
        worktrees: list[WorktreeInfo],
        abort: asyncio.Event,
    ) -> _Measurement:
        adapter = get_adapter(options.adapter)
        target = resolve_target(options.target.target, repo_dir)

        ctx = TargetContext(
            target=target,
            dir=resolve_dir(target, repo_dir, worktrees),
            label=resolve_label(options.target.label, target),
        )

        (collected,) = await collect_samples(
            adapter,
            [ctx],
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

        return _Measurement(
            label=ctx.label,
            samples=collected.samples,
            metric_meta=resolve_metric_meta_from_samples(
                [collected.samples],
                options.config_metrics,
                adapter,
                options.config_kinds,
            ),
        )

    return await run_with_worktrees(
        phase,
        lambda measurement, cleanup: _build_measurement_result(measurement, options, cleanup),
    )
