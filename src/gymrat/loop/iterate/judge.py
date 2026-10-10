"""Bench one session's worktree pair, judge the samples, and confirm regressions by a rerun.

The bench primitives — sampling both sides, computing verdicts, resolving the
primary figure — serve the first run and the confirmation rerun alike.

Sampling is driven here rather than through :func:`gymrat.compare.compare`
because a session's worktrees are persistent: there is nothing to check out and
nothing to sweep afterwards, and the raw samples have to survive the run to reach
the log.

A confirmation re-measures only the gating metrics the first run called
regressed.  ``exact`` metrics never take part: one differing sample is already
their whole signal, so a rerun could only add noise to a decision that has none.
"""

from __future__ import annotations

import asyncio
import shlex
import subprocess
import sys
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Literal

from gymrat.adapters import get_adapter
from gymrat.clock import monotonic_ms
from gymrat.config import FILTER_PLACEHOLDER, GEOMEAN_PRIMARY, ResolvedConfig
from gymrat.loop.gating import is_gating_regression
from gymrat.progress_events import (
    ConfirmFinished,
    ConfirmSkipped,
    ConfirmStarted,
    JudgeStarted,
    PassFinished,
    PassStarted,
    ProgressEvent,
    emit_progress,
)
from gymrat.report.loop import (
    GeomeanPrimary,
    LoopPrimary,
    MetricPrimary,
)
from gymrat.sampling import (
    CleanupResult,
    RunOptions,
    TargetContext,
    TargetSamples,
    collect_samples,
    resolve_metric_meta_from_samples,
)
from gymrat.session.records import PairedSamples, SessionRecord
from gymrat.targets import InPlaceTarget
from gymrat.utils import finite_or_none
from gymrat.verdict import compute_geomean, compute_verdicts

if TYPE_CHECKING:
    from collections.abc import Sequence

    from gymrat.adapters import Adapter
    from gymrat.config import KindEntry
    from gymrat.loop.iterate.run import IterateOptions
    from gymrat.model import MetricVerdict, ResolvedMetricMeta
    from gymrat.report.loop import RerunAnswer
    from gymrat.report.types import ComparisonResult, MetricComparisons

# ---------------------------------------------------------------------------
# Bench and judge
# ---------------------------------------------------------------------------

EXPERIMENT_INDEX = 0
"""The candidate an iteration measures: the experiment, judged against the baseline."""

EXPERIMENT_LABEL = "experiment"
"""The label the experiment worktree's target carries in progress and reports."""


@dataclass(frozen=True, slots=True)
class IterationContext:
    """The session, config, caller options, and log path that every iteration step shares."""

    session: SessionRecord
    config: ResolvedConfig
    options: IterateOptions
    jsonl_path: str


@dataclass(frozen=True, slots=True)
class BenchRunOutputs:
    """One bench-and-judge pass: both sides' samples, the verdicts, and the metric metadata.

    Attributes:
        baseline: The baseline worktree's samples.
        experiment: The experiment worktree's samples.
        verdicts: The verdict computed for each measured metric, by name.
        metric_meta: The resolved metadata for each measured metric, by name.
        samples: Both sides' rounds in the form the log stores them.
    """

    baseline: TargetSamples
    experiment: TargetSamples
    verdicts: dict[str, MetricVerdict]
    metric_meta: dict[str, ResolvedMetricMeta]
    samples: PairedSamples


@dataclass(frozen=True, slots=True)
class Judged:
    """The first run, judged and confirmed: the outputs, the comparison, and the rerun.

    The rerun only demotes a regression to ``no-signal``, and neither word moves
    the primary's delta, so the primary is resolved from the first run alone.

    Attributes:
        run: The first run's bench outputs, its verdicts already
            confirmation-applied.
        result: The comparison built from those outputs.
        confirmation: What the confirmation rerun found, or ``None`` when no
            rerun was needed.
        primary: The primary figure, resolved from the first run's verdicts.
    """

    run: BenchRunOutputs
    result: ComparisonResult
    confirmation: Confirmation | None
    primary: LoopPrimary


def build_iteration_comparison(
    run: BenchRunOutputs,
    adapter: str,
    config_kinds: dict[str, KindEntry] | None,
) -> ComparisonResult:
    """Build a comparison result for a single iteration: one baseline, one candidate, no cleanup.

    Args:
        run: The bench run's measurement outputs — baseline, experiment,
            verdicts, and metric metadata.
        adapter: The adapter name used to parse bench output.
        config_kinds: Per-kind configuration entries for aggregation, or
            ``None`` when no kind overrides are configured.

    Returns:
        The comparison result built from the single iteration's pair.
    """
    from gymrat.compare import (  # noqa: PLC0415 -- deferred to keep compare out of the CLI import chain
        CandidateMeasurement,
        ComparisonMeasurement,
        build_comparison_result,
    )

    candidate = CandidateMeasurement(
        label=run.experiment.ctx.label,
        samples=run.experiment.samples,
        verdicts=run.verdicts,
    )
    measurement = ComparisonMeasurement(
        baseline_label=run.baseline.ctx.label,
        baseline_samples=run.baseline.samples,
        candidates=[candidate],
        metric_meta=run.metric_meta,
    )
    return build_comparison_result(
        measurement,
        CleanupResult(removed=0, failures=(), prune_error=None),
        samples=min(len(run.baseline.samples), len(run.experiment.samples)),
        adapter=adapter,
        config_kinds=config_kinds,
    )


async def bench_and_judge(
    ctx: IterationContext,
    bench: str,
    metric_meta: dict[str, ResolvedMetricMeta] | None = None,
    *,
    announce_judging: bool = False,
) -> BenchRunOutputs:
    """Bench a session's worktrees and judge the resulting samples, in one call.

    Args:
        ctx: The iteration context, carrying the session, config, and options.
        bench: The bench command to run against both worktrees.
        metric_meta: Previously resolved metric metadata to reuse, or ``None``
            to resolve it fresh from the collected samples.  Optional because
            the first run does not know the metric set until it has samples to
            read it from; the confirmation rerun already has one from the first
            run and passes it through unchanged.
        announce_judging: Whether to emit a judge-started progress event once
            benching finishes, so a progress renderer shows judging as running
            only while verdicts are actually being computed.  The confirmation
            rerun leaves it off: its judging belongs to the confirm phase,
            which reports itself.

    Returns:
        The bench run with baseline/experiment samples, resolved metric metadata,
        and computed verdicts.

    Raises:
        GymratError: When the configured adapter is unknown; nothing is sampled.
        CommandError: When a prepare or bench command times out or exits
            non-zero.
    """
    adapter = get_adapter(ctx.config.adapter)
    baseline, experiment = await _measure(ctx, adapter, bench)
    if announce_judging:
        emit_progress(ctx.options.on_progress, JudgeStarted(at_ms=monotonic_ms()))
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
        baseline.samples,
        experiment.samples,
        resolved_meta,
        unstable_noise_pct=ctx.config.unstable_noise_pct,
    )
    return BenchRunOutputs(
        baseline=baseline,
        experiment=experiment,
        verdicts=verdicts,
        metric_meta=resolved_meta,
        samples=PairedSamples(
            experiment=tuple(experiment.samples), baseline=tuple(baseline.samples)
        ),
    )


async def _measure(
    ctx: IterationContext, adapter: Adapter, bench: str
) -> tuple[TargetSamples, TargetSamples]:
    """Bench both of the session's worktrees, baseline first.

    The order is the one :func:`gymrat.compare.compare` samples in — old side
    first — so a round of the loop perturbs the two sides in the same sequence a
    plain comparison would.

    Args:
        ctx: The iteration context: the session whose baseline and experiment
            worktrees are benched, the configuration supplying prepare, samples,
            and timeout, and the options supplying the progress callback and the
            warning sink.
        adapter: Parses a bench run's stdout into a metric record.
        bench: The bench command to run. A parameter because a confirmation rerun
            narrows the command while sampling the same pair of worktrees the same
            way.

    Returns:
        The baseline and experiment target samples, in that order.

    Raises:
        CommandError: When a prepare or bench command times out or exits
            non-zero.
    """
    worktrees = ctx.session.worktrees
    options = ctx.options
    contexts: list[TargetContext] = [
        _worktree_context(worktrees.baseline, "baseline", "old"),
        _worktree_context(worktrees.experiment, EXPERIMENT_LABEL, "new"),
    ]
    sampling_options = RunOptions.from_config(
        ctx.config, bench=bench, on_progress=options.on_progress, warn=options.warn
    ).sampling
    abort = options.abort if options.abort is not None else asyncio.Event()
    baseline, experiment = await collect_samples(adapter, contexts, sampling_options, abort)
    return baseline, experiment


def _worktree_context(directory: str, label: str, position: Literal["old", "new"]) -> TargetContext:
    """A session worktree, benched where it sits: it is checked out for the whole session."""
    return TargetContext(
        target=InPlaceTarget(dir=directory), dir=directory, label=label, position=position
    )


# ---------------------------------------------------------------------------
# Primary and target resolution
# ---------------------------------------------------------------------------


def resolve_primary(
    primary: str,
    verdicts: dict[str, MetricVerdict],
    metric_meta: dict[str, ResolvedMetricMeta],
) -> LoopPrimary:
    """The figure the iteration is read on: a gating geomean, or the named metric.

    Args:
        primary: The configured primary — a gating geomean marker or a metric name.
        verdicts: The computed verdict for each measured metric, by name.
        metric_meta: The resolved metadata for each measured metric, by name.

    Returns:
        The resolved primary — a :class:`GeomeanPrimary` or :class:`MetricPrimary`
        carrying the recorded delta. Its ``delta_pct`` is ``None`` when the named
        metric has no verdict, when no gating metric feeds the geomean, or when
        the ratio is not finite. A zero must never stand there: a zero is a
        measurement, and it would have the report, the log, and the keep commit
        all claim the run held its ground.
    """
    if primary == GEOMEAN_PRIMARY:
        gating = {name: meta for name, meta in metric_meta.items() if meta.gating}
        geomean = compute_geomean(verdicts, gating)
        return GeomeanPrimary(delta_pct=None if geomean.n == 0 else finite_or_none(geomean.value))

    measured = verdicts.get(primary)
    return MetricPrimary(
        name=primary,
        delta_pct=None if measured is None else finite_or_none(measured.delta),
    )


def target_reached(
    config: ResolvedConfig,
    primary: LoopPrimary,
    metrics: MetricComparisons,
) -> bool:
    """Whether the experiment has reached the value the loop was told to stop at.

    The target is read in the primary metric's own direction, so it needs a named
    primary — which config validation already demands of a ``stop.target_value``.

    Args:
        config: The resolved config, carrying the configured stop target.
        primary: The resolved primary the run was judged on.
        metrics: The comparison figures for every measured metric.

    Returns:
        Whether the primary metric's delta meets or exceeds the configured target.
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


# ---------------------------------------------------------------------------
# Confirmation rerun
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Confirmation:
    """What a confirmation rerun measured, and which regressions it stood behind.

    Attributes:
        filtered: The metrics the rerun re-measured, in the order the run
            reported them.
        samples: The rerun's own rounds, kept raw so a later statistics change
            can re-read them.
        confirmed: The subset of ``filtered`` the rerun also gated as regressed.
        absent: The subset of ``filtered`` the rerun produced no verdict for at
            all — disjoint from ``confirmed``, and not the complement of it.
    """

    filtered: tuple[str, ...]
    samples: PairedSamples
    confirmed: frozenset[str]
    absent: frozenset[str]


def rerun_answer(confirmation: Confirmation, metric: str) -> RerunAnswer:
    """What the confirmation rerun answered about one metric it re-measured.

    Args:
        confirmation: The rerun's findings.
        metric: A metric in ``confirmation.filtered``.

    Returns:
        ``"absent"`` when the rerun produced no verdict for it, ``"confirmed"``
        when the rerun also gated it as regressed, and ``"disagreed"`` otherwise.
    """
    if metric in confirmation.absent:
        return "absent"
    return "confirmed" if metric in confirmation.confirmed else "disagreed"


def with_confirm_phase(ctx: IterationContext) -> IterationContext:
    """Return a context whose ``on_progress`` tags pass events as confirmation runs."""
    original = ctx.options.on_progress
    if original is None:
        return ctx

    def wrapper(event: ProgressEvent) -> None:
        if isinstance(event, PassStarted | PassFinished):
            original(replace(event, phase="confirm"))
        else:
            original(event)

    return replace(ctx, options=replace(ctx.options, on_progress=wrapper))


def regresses_gating(meta: ResolvedMetricMeta, verdict: MetricVerdict | None) -> bool:
    """Whether ``meta``'s metric gates the run and ``verdict`` calls it regressed.

    Args:
        meta: The metric's resolved metadata.
        verdict: The metric's verdict, or ``None`` when it has none.

    Returns:
        ``True`` for a gating metric whose verdict is ``regressed``.
    """
    return is_gating_regression(
        gating=meta.gating, verdict=None if verdict is None else verdict.verdict
    )


def gating_regressions(
    verdicts: dict[str, MetricVerdict], metric_meta: dict[str, ResolvedMetricMeta]
) -> tuple[str, ...]:
    """Names of the gating metrics ``verdicts`` calls regressed, in ``metric_meta`` order.

    Args:
        verdicts: Per-metric verdicts, by name.
        metric_meta: The resolved metadata for each measured metric, by name.

    Returns:
        The regressed gating metrics' names.
    """
    return tuple(
        name for name, meta in metric_meta.items() if regresses_gating(meta, verdicts.get(name))
    )


async def confirm_regressions(
    ctx: IterationContext,
    verdicts: dict[str, MetricVerdict],
    metric_meta: dict[str, ResolvedMetricMeta],
) -> Confirmation | None:
    """Re-measure the gating metrics the first run called regressed, once.

    ``exact`` metrics never take part (see the module docstring). A ``filter``
    template benches just the named metrics; without one the whole bench re-runs
    and the same metrics are read out of it.

    Args:
        ctx: The iteration context, carrying the session, config, and options.
        verdicts: The first run's per-metric verdicts, by name.
        metric_meta: The resolved metadata for each measured metric, by name.

    Returns:
        What the rerun found, or ``None`` when nothing called for one.

    Raises:
        GymratError: When the rerun's bench command fails — an iteration nobody
            could confirm is not recorded.
    """
    filtered = tuple(
        name for name in gating_regressions(verdicts, metric_meta) if not metric_meta[name].exact
    )
    if not filtered:
        emit_progress(ctx.options.on_progress, ConfirmSkipped(at_ms=monotonic_ms()))
        return None

    bench = scoped_bench(ctx.config, filtered)
    emit_progress(
        ctx.options.on_progress,
        ConfirmStarted(
            filtered_metrics=None if ctx.config.filter is None else filtered,
            at_ms=monotonic_ms(),
        ),
    )

    confirm_ctx = with_confirm_phase(ctx)
    rerun = await bench_and_judge(confirm_ctx, bench, metric_meta)
    confirmed = frozenset(
        name for name in filtered if regresses_gating(metric_meta[name], rerun.verdicts.get(name))
    )
    absent = frozenset(name for name in filtered if rerun.verdicts.get(name) is None)

    emit_progress(
        ctx.options.on_progress,
        ConfirmFinished(reproduced=bool(confirmed), at_ms=monotonic_ms()),
    )

    return Confirmation(
        filtered=filtered,
        samples=rerun.samples,
        confirmed=confirmed,
        absent=absent,
    )


def apply_confirmation(
    verdicts: dict[str, MetricVerdict],
    confirmation: Confirmation | None,
) -> dict[str, MetricVerdict]:
    """The verdicts as finally read, with every regression the rerun disowned demoted.

    A metric the rerun never reported is left regressed — the rerun's job is to
    disprove a regression, and silence disproves nothing. The asymmetry is
    deliberate: a false alarm costs the agent an edit it did not need, while a
    missed regression is caught by the next iteration's baseline.

    Only the verdict word moves; the delta, noise, and p-value stay the first
    run's, because they describe the first run's samples — the ones the record
    stores and the table draws its medians from. The rerun's own rounds are kept
    separately under ``confirm``.

    Args:
        verdicts: The first run's per-metric verdicts, by name.
        confirmation: What the confirmation rerun found, or ``None`` when no
            rerun was needed.

    Returns:
        The verdicts with unconfirmed regressions demoted to ``no-signal``.
    """
    if confirmation is None:
        return verdicts

    settled: dict[str, MetricVerdict] = {}
    for name, verdict in verdicts.items():
        disagreed = (
            name in confirmation.filtered and rerun_answer(confirmation, name) == "disagreed"
        )
        settled[name] = replace(verdict, verdict="no-signal") if disagreed else verdict
    return settled


# ---------------------------------------------------------------------------
# Bench scoping and shell quoting
# ---------------------------------------------------------------------------


def scoped_bench(config: ResolvedConfig, names: Sequence[str]) -> str:
    """The bench command narrowed to ``names``, or the whole bench when it cannot be.

    Args:
        config: The run configuration, carrying the bench command and the
            optional ``filter`` template.
        names: The metric names to narrow the bench to. An empty sequence
            narrows nothing.

    Returns:
        The ``filter`` template with its placeholder replaced by the shell-quoted
        names joined by single spaces, or ``config.bench`` when there are no
        names to scope to or no template to scope with.
    """
    if not names or config.filter is None:
        return config.bench
    quoted = " ".join(shell_quote_name(name) for name in names)
    return config.filter.replace(FILTER_PLACEHOLDER, quoted)


def shell_quote_name(value: str) -> str:
    """``value`` as a single shell-safe word, platform-aware.

    Metric names are the bench's to choose, and mitata's ``sort(n=1000)/time``
    alias shape is an ordinary one: spliced into the filter template raw, the
    shell either splits the name across arguments or refuses the command as a
    syntax error — and a rerun that cannot run demotes a real regression to no
    signal.

    On POSIX, ``shlex.quote`` handles safe-word detection and single-quote
    escaping. On win32, ``cmd.exe`` uses double quotes, and
    ``subprocess.list2cmdline`` produces the correct escaping.

    Args:
        value: The metric name to shell-quote for the filter template.

    Returns:
        The shell-quoted string safe for interpolation into a command.
    """
    if sys.platform == "win32":
        return subprocess.list2cmdline([value])
    return shlex.quote(value)
