"""Measure one edit of an open session, record it, and phrase it for the agent.

Sampling is driven here rather than through :func:`gymrat.compare.compare`
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

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from gymrat.errors import GymratError
from gymrat.loop.hooks import HookInvocation, run_hook_stage
from gymrat.loop.iterate.bench import (
    BenchRunOutputs,
    IterationContext,
    Judged,
    bench_and_judge,
    build_iteration_comparison,
    resolve_primary,
    target_reached,
)
from gymrat.loop.iterate.confirm import Confirmation, apply_confirmation, confirm_regressions
from gymrat.loop.iterate.record import IterationJudgment, build_iteration_record
from gymrat.progress_events import (
    IterationRecorded,
    JudgeFinished,
    default_clock,
    emit_progress,
)
from gymrat.report.loop import (
    LoopOutcome,
    RerunAnswer,
    RerunConfirmation,
    derive_outcome,
    format_loop_header,
    format_verdict_block,
)
from gymrat.report.style import RENDER_WIDTH, render_lines
from gymrat.report.text import render_report
from gymrat.report.types import ComparisonResult, ReportOptions
from gymrat.session import (
    IterationRecord,
    SessionState,
    append_record,
    require_open_session,
)
from gymrat.session import budget as _budget
from gymrat.session import clock as _clock
from gymrat.session import workspace as _workspace
from gymrat.warn import warn_to_stderr

if TYPE_CHECKING:
    import asyncio
    from collections.abc import Sequence

    from gymrat.config import ResolvedConfig
    from gymrat.progress_events import ProgressCallback
    from gymrat.session import SessionLogRecord

__all__ = [
    "BenchRunOutputs",
    "BudgetExceededError",
    "IterateOptions",
    "IterateResult",
    "LoopStopError",
    "build_iteration_comparison",
    "iterate_session",
]


@dataclass(frozen=True, slots=True)
class IterateOptions:
    """What a caller can hand an iteration beyond its configuration.

    Attributes:
        on_progress: Fire-and-forget callback invoked for every progress
            event the iteration emits — prepare and sample steps from sampling,
            plus hook, judge, confirm, and record events from the loop itself.
        abort: Setting it kills the in-flight bench command. When ``None``, a
            fresh event is used and nothing can interrupt the run.
    """

    on_progress: ProgressCallback | None = None
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


def _judge(config: ResolvedConfig, judged: Judged) -> IterationJudgment:
    """Resolve the primary, derive the outcome, and bundle the judgment."""
    primary = resolve_primary(config.primary, judged.run.verdicts, judged.run.metric_meta)
    return IterationJudgment(
        outcome=derive_outcome(judged.result.metrics, primary),
        primary=primary,
        confirmation=judged.confirmation,
        reached_target=target_reached(config, primary, judged.result.metrics),
    )


def _append_iteration(ctx: IterationContext, record: IterationRecord, *, seq: int) -> None:
    append_record(ctx.jsonl_path, record)
    emit_progress(
        ctx.options.on_progress,
        IterationRecorded(seq=seq, outcome=record.outcome, at_ms=default_clock()),
    )


def _guard_budget(root: str, records: Sequence[SessionLogRecord]) -> None:
    """Refuse when a live budget cannot afford another iteration."""
    current_ms = _clock.now_ms()
    budget = _budget.read_budget(root, now_ms=current_ms)
    if budget is None:
        return
    estimate = _budget.estimate_iterate_duration(records)
    if estimate is None:
        return
    remaining = budget.remaining_ms(current_ms)
    if estimate.duration_ms > remaining:
        remaining_minutes = int(remaining / 60_000)
        estimate_minutes = int(estimate.duration_ms / 60_000)
        message = (
            f"{remaining_minutes}m left; the last {estimate.source} took "
            f"{estimate_minutes}m and the cap would cut this one off."
        )
        raise BudgetExceededError(
            message,
            hint="Report what the session measured instead of measuring again.",
        )


def _guard_ready(
    config: ResolvedConfig, state: SessionState, root: str, records: Sequence[SessionLogRecord]
) -> None:
    """Refuse another iteration when the session is not ready for one."""
    if state.unsettled:
        message = f"Iteration {state.last_seq} has not been settled"
        raise GymratError(
            message,
            hint="Run gymrat keep or gymrat discard before measuring the next edit.",
        )
    stop = stop_condition(config, state)
    if stop is not None:
        raise stop
    _guard_budget(root, records)


async def iterate_session(
    root: str,
    config: ResolvedConfig,
    options: IterateOptions | None = None,
    *,
    color: bool | None = None,
) -> IterateResult:
    """Measure the experiment worktree against the baseline worktree and record it.

    Args:
        root: The repository whose open session is measured.
        config: The resolved run configuration.
        options: Progress and abort hooks; a fresh set is used when ``None``.
        color: Explicit color choice for the iteration report — ``True``
            forces ANSI, ``False`` suppresses it, ``None`` defers to the
            environment and TTY.

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
    _guard_ready(config, state, root, required.records)

    start_ms = _clock.now_ms()
    seq = state.last_seq + 1
    ctx = IterationContext(session=session, config=config, options=opts, jsonl_path=jsonl_path)
    before_report = await _hook_stage(
        ctx,
        seq,
        stage="before",
        last_iteration=state.last_iteration,
        iteration_count=state.iteration_count,
    )

    judged = await _measure_and_judge(ctx)
    judgment = _judge(config, judged)

    duration_ms = _clock.now_ms() - start_ms
    measured_tree = _workspace.worktree_fingerprint(Path(session.worktrees.experiment))
    if measured_tree is None:
        warn_to_stderr("Could not fingerprint the experiment worktree; measured_tree will be null.")

    record = build_iteration_record(
        judged, seq, judgment, duration_ms=duration_ms, measured_tree=measured_tree
    )
    _append_iteration(ctx, record, seq=seq)

    after_report = await _hook_stage(
        ctx,
        seq,
        stage="after",
        last_iteration=record,
        iteration_count=state.iteration_count + 1,
    )

    iteration_report = render_iteration(judged.result, seq, judgment, color=color)
    report = "\n".join(
        part for part in (before_report, iteration_report, after_report) if part != ""
    )
    return IterateResult(record=record, report=report)


async def _hook_stage(
    ctx: IterationContext,
    seq: int,
    *,
    stage: Literal["before", "after"],
    last_iteration: IterationRecord | None,
    iteration_count: int,
) -> str:
    config, opts = ctx.config, ctx.options
    command = (
        (config.hooks.before if stage == "before" else config.hooks.after)
        if config.hooks is not None
        else None
    )
    invocation = (
        HookInvocation(
            command=command,
            stage=stage,
            seq=seq,
            session=ctx.session,
            last_iteration=last_iteration,
            iteration_count=iteration_count,
            abort=opts.abort,
        )
        if command is not None
        else None
    )
    return await run_hook_stage(
        ctx.jsonl_path,
        opts.on_progress,
        invocation=invocation,
    )


async def _measure_and_judge(ctx: IterationContext) -> Judged:
    """Bench the pair, confirm any gating regression, and assemble the comparison."""
    first = await bench_and_judge(ctx, ctx.config.bench, announce_judging=True)

    primary = resolve_primary(ctx.config.primary, first.verdicts, first.metric_meta)
    regressed_names = tuple(
        name
        for name, meta in first.metric_meta.items()
        if meta.gating and (v := first.verdicts.get(name)) is not None and v.verdict == "regressed"
    )
    emit_progress(
        ctx.options.on_progress,
        JudgeFinished(
            primary_delta_pct=primary.delta_pct,
            regressed=regressed_names,
            metric_count=len(first.metric_meta),
            at_ms=default_clock(),
        ),
    )

    confirmation = await confirm_regressions(ctx, first.verdicts, first.metric_meta)
    verdicts = apply_confirmation(first.verdicts, confirmation)
    run = BenchRunOutputs(
        baseline=first.baseline,
        experiment=first.experiment,
        verdicts=verdicts,
        metric_meta=first.metric_meta,
    )
    return Judged(
        run=run,
        result=build_iteration_comparison(run, ctx.config.adapter, ctx.config.kinds),
        confirmation=confirmation,
        samples=first.samples,
    )


_STOP_HINT = "The loop is done. Report what the session measured instead of measuring again."


class LoopStopError(GymratError):
    """A configured stop condition refusing another iteration.

    Separate from a plain :class:`GymratError` because nothing failed: the loop
    ran to the end it was configured for, which the CLI reports as a gate trip
    rather than as a tool failure.
    """


class BudgetExceededError(LoopStopError):
    """The session's time budget cannot afford another iteration.

    Raised when a live budget's remaining time is shorter than the estimated
    iterate duration, so the CLI routes it through the same gate-exit path as
    any other stop condition.
    """


def stop_condition(config: ResolvedConfig, state: SessionState) -> LoopStopError | None:
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


_NEXT_STEPS: dict[LoopOutcome, str] = {
    "improved": "`gymrat keep`",
    "regressed": "fix or run `gymrat discard`",
    "no-signal": "`gymrat keep` or `gymrat discard`",
}


def _rerun_answer(confirmation: Confirmation, metric: str) -> RerunAnswer:
    """What the rerun answered about ``metric``, as the report words it."""
    if metric in confirmation.absent:
        return "absent"
    return "confirmed" if metric in confirmation.confirmed else "disagreed"


def render_iteration(
    result: ComparisonResult,
    seq: int,
    judgment: IterationJudgment,
    *,
    color: bool | None = None,
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
    header = render_lines(format_loop_header(seq, result.samples), color=color, width=RENDER_WIDTH)
    report = render_report(result, ReportOptions(header=header, color=color, command="iterate"))
    verdict = render_lines(
        *format_verdict_block(
            outcome=judgment.outcome,
            primary=judgment.primary,
            next_step=_NEXT_STEPS[judgment.outcome],
            reruns=reruns,
            target_reached=judgment.reached_target,
        ),
        color=color,
        width=RENDER_WIDTH,
    )
    return f"{report}\n\n{verdict}"
