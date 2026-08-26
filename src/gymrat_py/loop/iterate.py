"""Measure one edit of an open session, record it, and phrase it for the agent.

Sampling is driven here rather than through :func:`gymrat_py.compare.compare`
because a session's worktrees are persistent: there is nothing to check out and
nothing to sweep afterwards, and the raw samples have to survive the run to reach
the log. Holding the repository lock across the call is the caller's job — two
concurrent sessions' bench runs would perturb each other's measurements.

Three asymmetries drive the shape of this module:

- **The confirmation rerun is one-sided.** Only a rerun that *also* gates a
  metric makes its regression stand; a rerun that stays silent about a metric
  disproves nothing, so that metric is left regressed. A false alarm costs the
  agent an edit it did not need; a missed regression is caught by the next
  iteration's baseline.
- **Only the verdict word moves.** When a rerun disagrees, the metric is demoted
  to no-signal but its delta, noise, and p-value stay the first run's — they
  describe the first run's samples, the ones the record stores and the table
  draws its medians from. The rerun's own rounds are kept separately under
  ``confirm``.
- **A ratio with no value is recorded as ``None``.** A degenerate ratio — a
  baseline median of zero — yields ``NaN``, which JSON writes as ``null``. Making
  that substitution here keeps the record a caller holds identical to the one
  read back off the log, and never lets a zero stand where there was no
  measurement.

The loop header lands last, replacing the comparison table's own header, so the
table opens on the loop's terms rather than on ``gymrat compare``'s.
"""

from __future__ import annotations

import asyncio
import math
import re
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Literal

from gymrat_py.adapters import get_adapter
from gymrat_py.config import FILTER_PLACEHOLDER, GEOMEAN_PRIMARY, KindEntry, ResolvedConfig
from gymrat_py.errors import GymratError
from gymrat_py.loop.hooks import HookInvocation, run_hook
from gymrat_py.model import (
    ExactVerdict,
    MetricVerdict,
    Observations,
    PermutationVerdict,
    ResolvedMetricMeta,
)
from gymrat_py.report.loop import (
    EXPERIMENT_INDEX,
    GeomeanPrimary,
    LoopOutcome,
    LoopPrimary,
    MetricPrimary,
    RerunAnswer,
    RerunConfirmation,
    derive_outcome,
    format_loop_header,
    format_verdict_block,
)
from gymrat_py.report.style import RENDER_WIDTH, render_lines
from gymrat_py.report.text import render_report
from gymrat_py.report.types import (
    ComparisonResult,
    MetricComparisons,
    ReportOptions,
)
from gymrat_py.sampling import (
    SamplingOptions,
    TargetContext,
    TargetSamples,
    collect_samples,
    resolve_metric_meta_from_samples,
)
from gymrat_py.session import (
    Confirm,
    IterationPrimary,
    IterationRecord,
    PairedSamples,
    SessionRecord,
    SessionState,
    append_record,
    require_open_session,
)
from gymrat_py.session import MetricVerdict as RecordMetricVerdict
from gymrat_py.session.clock import now_iso
from gymrat_py.targets import InPlaceTarget
from gymrat_py.verdict import compute_geomean, compute_kind_aggregates, compute_verdicts

if TYPE_CHECKING:
    from collections.abc import Callable

    from gymrat_py.sampling import ProgressStep

#: What the loop tells the agent to do next, one per outcome.
_NEXT_STEPS: dict[LoopOutcome, str] = {
    "improved": "gymrat keep",
    "regressed": "fix or gymrat discard",
    "no-signal": "gymrat keep or gymrat discard",
}

#: What every stop condition tells the agent to do once the loop is over.
_STOP_HINT = "The loop is done. Report what the session measured instead of measuring again."

#: Characters a POSIX shell passes through untouched, so a word of them needs no quoting.
_SHELL_SAFE_WORD = re.compile(r"[\w@%+=:,./-]+", re.ASCII)


class LoopStopError(GymratError):
    """A configured stop condition refusing another iteration.

    Separate from a plain :class:`GymratError` because nothing failed: the loop
    ran to the end it was configured for, which the CLI reports as a gate trip
    rather than as a tool failure.
    """


@dataclass(frozen=True, slots=True)
class IterateOptions:
    """What a caller can hand an iteration beyond its configuration.

    Attributes:
        on_progress: Fire-and-forget callback invoked at the start of each
            prepare or sample step.
        abort: Setting it kills the in-flight bench command. When ``None``, a
            fresh event is used and nothing can interrupt the run.
    """

    on_progress: Callable[[ProgressStep], None] | None = None
    abort: asyncio.Event | None = None


@dataclass(frozen=True, slots=True)
class IterateResult:
    """One measured iteration: what was written to the log, and what to print.

    Attributes:
        record: The record appended to the session log.
        report: The iteration as the agent reads it — header, comparison table,
            verdict block.
    """

    record: IterationRecord
    report: str


@dataclass(frozen=True, slots=True)
class _Confirmation:
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


@dataclass(frozen=True, slots=True)
class _IterationContext:
    """The session, config, and caller options that every iteration step shares."""

    session: SessionRecord
    config: ResolvedConfig
    options: IterateOptions


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


def build_iteration_comparison(
    run: BenchRunOutputs,
    adapter: str,
    config_kinds: dict[str, KindEntry] | None,
) -> ComparisonResult:
    """Build a comparison result for a single iteration: one baseline, one candidate, no cleanup."""
    from gymrat_py.compare import (  # noqa: PLC0415 -- deferred to keep compare out of the CLI import chain
        CandidateMeasurement,
        build_comparison_result,
    )
    from gymrat_py.targets import CleanupResult  # noqa: PLC0415

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


@dataclass(frozen=True, slots=True)
class _Judged:
    """The first run, judged and confirmed: the outputs, the comparison, and the rerun."""

    run: BenchRunOutputs
    result: ComparisonResult
    confirmation: _Confirmation | None
    samples: PairedSamples


@dataclass(frozen=True, slots=True)
class _IterationJudgment:
    """The iteration's judgment: what happened, what drove it, and whether a target was met."""

    outcome: LoopOutcome
    primary: LoopPrimary
    confirmation: _Confirmation | None
    reached_target: bool


async def iterate_session(
    root: str,
    config: ResolvedConfig,
    options: IterateOptions | None = None,
) -> IterateResult:
    """Measure the experiment worktree against the baseline worktree and record it.

    Args:
        root: The repository whose open session is measured.
        config: The resolved run configuration.
        options: Progress and abort hooks; a fresh set is used when ``None``.

    Returns:
        The appended record and the report to print for it.

    Raises:
        GymratError: When no session has been started, when the last iteration is
            still unsettled, or when the bench command fails.
        LoopStopError: When a configured stop condition has already been met,
            before anything is measured or recorded.
    """
    opts = options if options is not None else IterateOptions()
    required = require_open_session(root, "measuring an edit")
    session, state, jsonl_path = required.session, required.state, required.jsonl_path

    if state.unsettled:
        message = f"Iteration {state.last_seq} has not been settled"
        raise GymratError(
            message,
            hint="Run gymrat keep or gymrat discard before measuring the next edit.",
        )

    stop = _stop_condition(config, state)
    if stop is not None:
        raise stop

    seq = state.last_seq + 1
    before_command = config.hooks.before if config.hooks is not None else None
    before_report = await _fire_hook(
        jsonl_path,
        HookInvocation(
            command=before_command,
            stage="before",
            seq=seq,
            session=session,
            last_iteration=state.last_iteration,
            iteration_count=state.iteration_count,
            abort=opts.abort,
        )
        if before_command is not None
        else None,
    )

    ctx = _IterationContext(session=session, config=config, options=opts)
    judged = await _measure_and_judge(ctx)
    primary = _resolve_primary(config.primary, judged.run.verdicts, judged.run.metric_meta)
    judgment = _IterationJudgment(
        outcome=derive_outcome(judged.result.metrics, primary),
        primary=primary,
        confirmation=judged.confirmation,
        reached_target=_target_reached(config, primary, judged.result.metrics),
    )

    record = _build_iteration_record(judged, seq, judgment)
    append_record(jsonl_path, record)

    after_command = config.hooks.after if config.hooks is not None else None
    after_report = await _fire_hook(
        jsonl_path,
        HookInvocation(
            command=after_command,
            stage="after",
            seq=seq,
            session=session,
            last_iteration=record,
            iteration_count=state.iteration_count + 1,
            abort=opts.abort,
        )
        if after_command is not None
        else None,
    )

    iteration_report = _render_iteration(judged.result, seq, judgment)
    report = "\n".join(
        part for part in (before_report, iteration_report, after_report) if part != ""
    )
    return IterateResult(record=record, report=report)


async def _measure_and_judge(ctx: _IterationContext) -> _Judged:
    """Bench the pair, confirm any gating regression, and assemble the comparison."""
    first = await _bench_and_judge(ctx, ctx.config.bench)
    confirmation = await _confirm_regressions(ctx, first.verdicts, first.metric_meta)
    verdicts = _apply_confirmation(first.verdicts, confirmation)
    run = BenchRunOutputs(
        baseline=first.baseline,
        experiment=first.experiment,
        verdicts=verdicts,
        metric_meta=first.metric_meta,
    )
    return _Judged(
        run=run,
        result=build_iteration_comparison(run, ctx.config.adapter, ctx.config.kinds),
        confirmation=confirmation,
        samples=first.samples,
    )


async def _fire_hook(jsonl_path: str, invocation: HookInvocation | None) -> str:
    """Run the consumer's hook, logging what it did.

    A stage the config leaves out passes ``None`` and runs nothing at all: no
    process, no record, no line in the report. Returns what to print for the hook
    — empty when there was no hook or it said nothing.
    """
    if invocation is None:
        return ""
    run = await run_hook(invocation)
    append_record(jsonl_path, run.record)
    return run.report


def _stop_condition(config: ResolvedConfig, state: SessionState) -> LoopStopError | None:
    """The configured stop condition this session has already met, if any.

    Read off the folded log alone, so it settles before a bench command runs: an
    iteration measured past the end of the loop is one the agent would have to
    throw away. ``target_value`` stops the loop only once the target-reaching
    iteration is *kept* — discarding it puts the target back out of reach.
    """
    stop = config.stop
    if stop is None:
        return None

    if stop.max_iterations is not None and state.iteration_count >= stop.max_iterations:
        message = (
            f"Stop condition met: max iterations ({state.iteration_count} of {stop.max_iterations})"
        )
        return LoopStopError(message, hint=_STOP_HINT)

    if stop.target_value is not None and state.target_reached_and_kept:
        return LoopStopError("Stop condition met: target reached and kept", hint=_STOP_HINT)
    return None


async def _confirm_regressions(
    ctx: _IterationContext,
    verdicts: dict[str, MetricVerdict],
    metric_meta: dict[str, ResolvedMetricMeta],
) -> _Confirmation | None:
    """Re-measure the gating metrics the first run called regressed, once.

    ``exact`` metrics never take part: one differing sample is already their whole
    signal, so a rerun could only add noise to a decision that has none. A
    ``filter`` template benches just the named metrics; without one the whole
    bench re-runs and the same metrics are read out of it.

    Returns what the rerun found, or ``None`` when nothing called for one.

    Raises:
        GymratError: When the rerun's bench command fails — an iteration nobody
            could confirm is not recorded.
    """
    filtered = [
        name
        for name, meta in metric_meta.items()
        if meta.gating
        and not meta.exact
        and (verdict := verdicts.get(name)) is not None
        and verdict.verdict == "regressed"
    ]
    if not filtered:
        return None

    names = " ".join(_shell_quote(name) for name in filtered)
    # str.replace is literal, so a metric name carrying a ``$&``-like sequence is
    # its own text rather than a substitution pattern.
    bench = (
        ctx.config.bench
        if ctx.config.filter is None
        else ctx.config.filter.replace(FILTER_PLACEHOLDER, names)
    )
    rerun = await _bench_and_judge(ctx, bench, metric_meta)
    confirmed = frozenset(
        name
        for name in filtered
        if (verdict := rerun.verdicts.get(name)) is not None and verdict.verdict == "regressed"
    )
    absent = frozenset(name for name in filtered if rerun.verdicts.get(name) is None)
    return _Confirmation(
        filtered=tuple(filtered),
        samples=rerun.samples,
        confirmed=confirmed,
        absent=absent,
    )


def _shell_quote(value: str) -> str:
    """``value`` as a single word of a POSIX shell command.

    Metric names are the bench's to choose, and mitata's ``sort(n=1000)/time``
    alias shape is an ordinary one: spliced into the filter template raw, the
    shell either splits the name across arguments or refuses the command as a
    syntax error — and a rerun that cannot run demotes a real regression to no
    signal. Single quotes are the only POSIX quoting that suspends every
    expansion, so a name that is not a plain word is wrapped in them, with each
    single quote inside closed, escaped and reopened.
    """
    if _SHELL_SAFE_WORD.fullmatch(value):
        return value
    escaped = value.replace("'", "'\\''")
    return f"'{escaped}'"


async def _bench_and_judge(
    ctx: _IterationContext,
    bench: str,
    metric_meta: dict[str, ResolvedMetricMeta] | None = None,
) -> _BenchRun:
    """Bench a session's worktrees and judge the resulting samples, in one call.

    ``metric_meta`` is optional because the first run does not know the metric set
    until it has samples to read it from; the confirmation rerun already has one
    from the first run and passes it through unchanged.
    """
    baseline, experiment = await _measure(ctx.session, ctx.config, ctx.options, bench)
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


def _apply_confirmation(
    verdicts: dict[str, MetricVerdict],
    confirmation: _Confirmation | None,
) -> dict[str, MetricVerdict]:
    """The verdicts as finally read, with every regression the rerun disowned demoted.

    A metric the rerun never reported is left regressed — the rerun's job is to
    disprove a regression, and silence disproves nothing. Only the verdict word
    moves; the delta, noise, and p-value stay the first run's.
    """
    if confirmation is None:
        return verdicts

    settled: dict[str, MetricVerdict] = {}
    for name, verdict in verdicts.items():
        disagreed = (
            name in confirmation.filtered
            and name not in confirmation.confirmed
            and name not in confirmation.absent
        )
        settled[name] = replace(verdict, verdict="no-signal") if disagreed else verdict
    return settled


async def _measure(
    session: SessionRecord,
    config: ResolvedConfig,
    options: IterateOptions,
    bench: str,
) -> tuple[TargetSamples, TargetSamples]:
    """Bench both of the session's worktrees, baseline first.

    The order is the one :func:`gymrat_py.compare.compare` samples in — old side
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


def _resolve_primary(
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
        return GeomeanPrimary(delta_pct=None if geomean.n == 0 else _recorded_delta(geomean.value))

    measured = verdicts.get(primary)
    return MetricPrimary(
        name=primary,
        delta_pct=None if measured is None else _recorded_delta(measured.delta.value),
    )


def _recorded_delta(delta: float) -> float | None:
    """A delta in the form the log keeps it: ``None`` where the ratio had no value.

    The engine answers a degenerate ratio — a baseline median of zero — with
    ``NaN``, and JSON serialization writes that as ``null`` whatever the writer
    intended. Making the substitution here keeps the record a caller holds
    identical to the one read back off the log.
    """
    return None if math.isnan(delta) else delta


def _target_reached(
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


def _recorded_verdicts(
    verdicts: dict[str, MetricVerdict],
    metric_meta: dict[str, ResolvedMetricMeta],
    confirmation: _Confirmation | None,
) -> dict[str, RecordMetricVerdict]:
    """The per-metric verdicts as the log keeps them, flattened out of the method shapes.

    A key whose value is absent is left out — ``p`` off a non-permutation verdict,
    ``noise_pct`` off an exact one — so a record handed to a caller matches the one
    read back off the log.
    """
    recorded: dict[str, RecordMetricVerdict] = {}
    for name, verdict in verdicts.items():
        meta = metric_meta.get(name)
        recorded[name] = RecordMetricVerdict(
            delta_pct=_recorded_delta(verdict.delta.value),
            verdict=verdict.verdict,
            method=verdict.method,
            gating=meta.gating if meta is not None else True,
            confirmed=name in confirmation.confirmed if confirmation is not None else False,
            p=verdict.p if isinstance(verdict, PermutationVerdict) else None,
            noise_pct=None if isinstance(verdict, ExactVerdict) else verdict.noise_pct,
        )
    return recorded


def _build_iteration_record(
    judged: _Judged,
    seq: int,
    judgment: _IterationJudgment,
) -> IterationRecord:
    """Assemble the iteration record from the judged run."""
    confirmation = judgment.confirmation
    confirm: Confirm | None = None
    if confirmation is not None:
        absent = tuple(name for name in confirmation.filtered if name in confirmation.absent)
        confirm = Confirm(
            ran=True,
            filtered=tuple(confirmation.filtered),
            samples=confirmation.samples,
            absent=absent or None,
        )
    primary = judgment.primary
    return IterationRecord(
        type="iteration",
        seq=seq,
        at=now_iso(),
        samples=judged.samples,
        metrics=_recorded_verdicts(judged.run.verdicts, judged.run.metric_meta, confirmation),
        primary=IterationPrimary(
            kind=primary.kind,
            delta_pct=primary.delta_pct,
            name=primary.name if isinstance(primary, MetricPrimary) else None,
        ),
        outcome=judgment.outcome,
        target_reached=judgment.reached_target,
        confirm=confirm,
    )


def _rerun_answer(confirmation: _Confirmation, metric: str) -> RerunAnswer:
    """What the rerun answered about ``metric``, as the report words it."""
    if metric in confirmation.absent:
        return "absent"
    return "confirmed" if metric in confirmation.confirmed else "disagreed"


def _render_iteration(
    result: ComparisonResult,
    seq: int,
    judgment: _IterationJudgment,
) -> str:
    """The iteration as it prints: the loop's header, the comparison table, the verdict."""
    confirmation = judgment.confirmation
    reruns: list[RerunConfirmation] = (
        [
            RerunConfirmation(metric=name, answer=_rerun_answer(confirmation, name))
            for name in confirmation.filtered
        ]
        if confirmation is not None
        else []
    )
    # render_report places its header override verbatim and resolves only its own
    # body markup, so the loop header and the verdict block are rendered here the
    # same way — deferring color to the environment, at the report's own width.
    header = render_lines(format_loop_header(seq, result.samples), width=RENDER_WIDTH)
    report = render_report(result, ReportOptions(header=header))
    verdict = render_lines(
        *format_verdict_block(
            outcome=judgment.outcome,
            primary=judgment.primary,
            next_step=_NEXT_STEPS[judgment.outcome],
            reruns=reruns,
            target_reached=judgment.reached_target,
        ),
        width=RENDER_WIDTH,
    )
    return f"{report}\n\n{verdict}"
