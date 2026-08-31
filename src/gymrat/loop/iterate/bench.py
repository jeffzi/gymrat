"""Bench one session's worktree pair and judge the resulting samples.

The bench primitives — sampling both sides, computing verdicts, resolving the
primary figure — live here so the orchestrator in ``iterate`` stays short and
the confirm module can re-use the same judge without a circular import.
"""

from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from gymrat.adapters import get_adapter
from gymrat.config import GEOMEAN_PRIMARY, ResolvedConfig
from gymrat.model import MetricVerdict, Observations, ResolvedMetricMeta
from gymrat.progress_events import JudgeStarted, default_clock, emit_progress
from gymrat.report.loop import (
    EXPERIMENT_INDEX,
    GeomeanPrimary,
    LoopPrimary,
    MetricPrimary,
)
from gymrat.sampling import (
    SamplingOptions,
    TargetContext,
    TargetSamples,
    collect_samples,
    resolve_metric_meta_from_samples,
)
from gymrat.session import PairedSamples, SessionRecord
from gymrat.targets import InPlaceTarget
from gymrat.verdict import compute_geomean, compute_kind_aggregates, compute_verdicts

if TYPE_CHECKING:
    from gymrat.config import KindEntry
    from gymrat.loop.iterate.confirm import Confirmation
    from gymrat.loop.iterate.run import IterateOptions
    from gymrat.report.types import ComparisonResult, MetricComparisons


@dataclass(frozen=True, slots=True)
class IterationContext:
    """The session, config, caller options, and log path that every iteration step shares."""

    session: SessionRecord
    config: ResolvedConfig
    options: IterateOptions
    jsonl_path: str


@dataclass(frozen=True, slots=True)
class _BenchRun:
    """One bench-and-judge pass: both sides' samples, the verdicts, and the metric metadata."""

    baseline: TargetSamples
    experiment: TargetSamples
    metric_meta: dict[str, ResolvedMetricMeta]
    verdicts: dict[str, MetricVerdict]
    samples: PairedSamples


@dataclass(frozen=True, slots=True)
class BenchRunOutputs:
    """One bench run's measurement outputs, shared by the record and the report."""

    baseline: TargetSamples
    experiment: TargetSamples
    verdicts: dict[str, MetricVerdict]
    metric_meta: dict[str, ResolvedMetricMeta]


@dataclass(frozen=True, slots=True)
class Judged:
    """The first run, judged and confirmed: the outputs, the comparison, and the rerun."""

    run: BenchRunOutputs
    result: ComparisonResult
    confirmation: Confirmation | None
    samples: PairedSamples


def build_iteration_comparison(
    run: BenchRunOutputs,
    adapter: str,
    config_kinds: dict[str, KindEntry] | None,
) -> ComparisonResult:
    """Build a comparison result for a single iteration: one baseline, one candidate, no cleanup."""
    from gymrat.compare import (  # noqa: PLC0415 -- deferred to keep compare out of the CLI import chain
        CandidateMeasurement,
        build_comparison_result,
    )
    from gymrat.targets import CleanupResult  # noqa: PLC0415 -- same deferral as above

    candidate = CandidateMeasurement(
        label=run.experiment.ctx.label,
        samples=run.experiment.samples,
        verdicts=run.verdicts,
        kinds=compute_kind_aggregates(run.verdicts, run.metric_meta),
    )
    return build_comparison_result(
        run.baseline.ctx.label,
        run.baseline.samples,
        [candidate],
        run.metric_meta,
        samples=min(len(run.baseline.samples), len(run.experiment.samples)),
        adapter=adapter,
        config_kinds=config_kinds,
        cleanup=CleanupResult(removed=0, failures=(), prune_error=None),
    )


async def bench_and_judge(
    ctx: IterationContext,
    bench: str,
    metric_meta: dict[str, ResolvedMetricMeta] | None = None,
    *,
    announce_judging: bool = False,
) -> _BenchRun:
    """Bench a session's worktrees and judge the resulting samples, in one call.

    ``metric_meta`` is optional because the first run does not know the metric set
    until it has samples to read it from; the confirmation rerun already has one
    from the first run and passes it through unchanged.

    ``announce_judging`` opens the judge phase once the bench has stopped
    reporting passes, so a progress renderer shows judging as running only while
    the verdicts are actually being computed. The confirmation rerun leaves it
    off: its judging belongs to the confirm phase, which reports itself.
    """
    baseline, experiment = await _measure(ctx.session, ctx.config, ctx.options, bench)
    if announce_judging:
        emit_progress(ctx.options.on_progress, JudgeStarted(at_ms=default_clock()))
    adapter = get_adapter(ctx.config.adapter)
    resolved_meta = (
        metric_meta
        if metric_meta is not None
        else resolve_metric_meta_from_samples(
            [baseline.samples, experiment.samples],
            ctx.config.metrics,
            adapter,
            ctx.config.kinds,
        )
    )
    verdicts = compute_verdicts(
        Observations.from_rounds(baseline.samples),
        Observations.from_rounds(experiment.samples),
        resolved_meta,
        unstable_noise_pct=ctx.config.unstable_noise_pct,
    )
    return _BenchRun(
        baseline=baseline,
        experiment=experiment,
        metric_meta=resolved_meta,
        verdicts=verdicts,
        samples=PairedSamples(
            experiment=tuple(experiment.samples), baseline=tuple(baseline.samples)
        ),
    )


async def _measure(
    session: SessionRecord,
    config: ResolvedConfig,
    options: IterateOptions,
    bench: str,
) -> tuple[TargetSamples, TargetSamples]:
    """Bench both of the session's worktrees, baseline first.

    The order is the one :func:`gymrat.compare.compare` samples in — old side
    first — so a round of the loop perturbs the two sides in the same sequence a
    plain comparison would. ``bench`` is a parameter because a confirmation rerun
    narrows the command while sampling the same pair of worktrees the same way.
    """
    contexts: list[TargetContext] = [
        _worktree_context(session.worktrees.baseline, "baseline", "old"),
        _worktree_context(session.worktrees.experiment, "experiment", "new"),
    ]
    sampling_options = SamplingOptions(
        bench=bench,
        prepare=config.prepare,
        samples=config.samples,
        timeout_seconds=config.timeout_seconds,
        on_progress=options.on_progress,
    )
    adapter = get_adapter(config.adapter)
    abort = options.abort if options.abort is not None else asyncio.Event()
    baseline, experiment = await collect_samples(adapter, contexts, sampling_options, abort)
    return baseline, experiment


def _worktree_context(directory: str, label: str, position: Literal["old", "new"]) -> TargetContext:
    """A session worktree, benched where it sits: it is checked out for the whole session."""
    return TargetContext(
        target=InPlaceTarget(dir=directory), dir=directory, label=label, position=position
    )


def resolve_primary(
    primary: str,
    verdicts: dict[str, MetricVerdict],
    metric_meta: dict[str, ResolvedMetricMeta],
) -> LoopPrimary:
    """The figure the iteration is read on: a gating geomean, or the named metric.

    A named metric the run never measured yields a primary with no delta at all —
    ``None``, the form a figure that has no value takes everywhere in the record.
    A zero must never stand there: a zero is a measurement, and it would have the
    report, the log, and the keep commit all claim the run held its ground.
    """
    if primary == GEOMEAN_PRIMARY:
        gating = {name: meta for name, meta in metric_meta.items() if meta.gating}
        geomean = compute_geomean(verdicts, gating)
        return GeomeanPrimary(delta_pct=None if geomean.n == 0 else recorded_delta(geomean.value))

    measured = verdicts.get(primary)
    return MetricPrimary(
        name=primary,
        delta_pct=None if measured is None else recorded_delta(measured.delta.value),
    )


def recorded_delta(delta: float) -> float | None:
    """A delta in the form the log keeps it: ``None`` where the ratio had no value.

    The engine answers a degenerate ratio — a baseline median of zero — with
    ``NaN``, and JSON serialization writes that as ``null`` whatever the writer
    intended. Making the substitution here keeps the record a caller holds
    identical to the one read back off the log.
    """
    return None if math.isnan(delta) else delta


def target_reached(
    config: ResolvedConfig,
    primary: LoopPrimary,
    metrics: MetricComparisons,
) -> bool:
    """Whether the experiment has reached the value the loop was told to stop at.

    The target is read in the primary metric's own direction, so it needs a named
    primary — which config validation already demands of a ``stop.target_value``.
    """
    target = config.stop.target_value if config.stop is not None else None
    if target is None or not isinstance(primary, MetricPrimary):
        return False

    metric = metrics.get(primary.name)
    if metric is None:
        return False
    median = metric.candidates[EXPERIMENT_INDEX].median
    if median is None:
        return False
    return median >= target if metric.meta.direction == "higher" else median <= target
