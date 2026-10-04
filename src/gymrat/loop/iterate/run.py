"""Measure one edit of an open session, record it, and phrase it for the agent.

Sampling is driven here rather than through :func:`gymrat.compare.compare`
because a session's worktrees are persistent: there is nothing to check out and
nothing to sweep afterwards, and the raw samples have to survive the run to reach
the log. Holding the repository lock across the call is the caller's job — two
concurrent sessions' bench runs would perturb each other's measurements.

How a confirmation rerun rewrites the verdicts is
:func:`gymrat.loop.iterate.confirm.apply_confirmation`'s contract, and how a
degenerate delta is recorded is :func:`gymrat.loop.iterate.bench.recorded_delta`'s.

The loop header lands last, replacing the comparison table's own header, so the
table opens on the loop's terms rather than on ``gymrat compare``'s.

The before and after commands a consumer hangs off each measurement run here too.
Hooks steer the loop; they cannot brick it. Every invocation that reaches
:func:`run_hook` runs its command -- deciding whether a stage has a command at
all is the caller's job. A command that fails, overruns its timeout, or never
starts at all comes back as a report and a record, never as a raised exception:
there is no hook failure worth throwing away a measurement over.

Two shaping choices are worth spelling out:

- The record's byte counts are what the command *wrote*, not what was relayed.
  A figure above the relay limit is how a reader of the log learns the report
  was cut, so the pre-relay totals from ``exec`` are what land in the record.
- A successful hook's stderr is kept out of the report. Commands write progress
  there routinely, and repeating it would drown the measurement the hook was
  annotating. A *failing* hook's stderr is shown -- and held to the same byte
  cap as its stdout, since a build log buries a measurement as easily on one
  channel as on the other.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from gymrat import clock as _clock
from gymrat.clock import monotonic_ms, now_ns
from gymrat.errors import GymratError

# Bound at module scope under the builtin's name so a test can substitute the
# subprocess boundary via ``monkeypatch.setattr`` on this module.
from gymrat.exec import (
    FAILURE_EXIT_CODE,
    ExecOptions,
    ExecResult,
    ExecTimeoutError,
    exec,  # noqa: A004 -- names the subprocess executor `exec`
)
from gymrat.loop.iterate.bench import (
    IterationContext,
    Judged,
    bench_and_judge,
    build_iteration_comparison,
    resolve_primary,
    target_reached,
)
from gymrat.loop.iterate.confirm import (
    Confirmation,
    apply_confirmation,
    confirm_regressions,
    is_gating_regression,
)
from gymrat.loop.iterate.record import IterationJudgment, build_iteration_record
from gymrat.loop.output_limit import limit_output
from gymrat.model import is_improvement
from gymrat.progress_events import (
    HookFinished,
    HookStarted,
    IterationRecorded,
    JudgeFinished,
    emit_progress,
)
from gymrat.report.loop import (
    EXPERIMENT_INDEX,
    GeomeanPrimary,
    LoopOutcome,
    LoopPrimary,
    RerunAnswer,
    RerunConfirmation,
    format_loop_header,
    format_verdict_block,
)
from gymrat.report.style import render_lines
from gymrat.report.text.render import render_report
from gymrat.report.types import ComparisonResult, ReportOptions, candidate_at
from gymrat.session import budget as _budget
from gymrat.session import workspace as _workspace
from gymrat.session.records import HookRecord, record_to_wire
from gymrat.session.store import (
    SessionState,
    append_record,
    require_open_session,
    require_settled,
)
from gymrat.utils import warn_to_stderr

if TYPE_CHECKING:
    import asyncio
    from collections.abc import Sequence

    from gymrat.config import BenchlessConfig, ResolvedConfig
    from gymrat.progress_events import ProgressCallback
    from gymrat.report.types import MetricComparisons
    from gymrat.session.records import IterationRecord, SessionLogRecord, SessionRecord
    from gymrat.session.schema import CommandReason, HookStage
    from gymrat.utils import WarnSink

__all__ = [
    "BudgetExceededError",
    "HookInvocation",
    "IterateOptions",
    "IterateResult",
    "LoopStopError",
    "derive_outcome",
    "iterate_session",
    "run_hook",
    "stop_condition",
    "stop_reason",
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
        warn: Where the adapter reports bench output it could not read.
    """

    on_progress: ProgressCallback | None = None
    abort: asyncio.Event | None = None
    warn: WarnSink = warn_to_stderr


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


def _has_gating_regression(metrics: MetricComparisons) -> bool:
    """Whether any metric the run is gated on came back regressed for the experiment."""
    for metric in metrics.values():
        experiment = candidate_at(metric, EXPERIMENT_INDEX)
        if is_gating_regression(metric.meta, None if experiment is None else experiment.verdict):
            return True
    return False


def _primary_improved(metrics: MetricComparisons, primary: LoopPrimary) -> bool:
    """Whether the primary figure moved the way its direction calls an improvement.

    A figure whose ratio had no value moved in no direction at all, so it
    improves nothing. The geomean is normalized so that lower is better; a named
    metric is judged in its own direction.

    Args:
        metrics: The run's metric comparisons, used to look up the primary's
            direction when it names a metric rather than the geomean.
        primary: The one figure the iteration is read on.

    Returns:
        Whether the primary figure moved in the direction its metric calls an
        improvement.
    """
    if primary.delta_pct is None:
        return False
    if isinstance(primary, GeomeanPrimary):
        return is_improvement(primary.delta_pct, "lower")
    metric = metrics.get(primary.name)
    if metric is None:
        return False
    return is_improvement(primary.delta_pct, metric.meta.direction)


def derive_outcome(metrics: MetricComparisons, primary: LoopPrimary) -> LoopOutcome:
    """What an iteration amounted to, read off its metrics and its primary figure.

    A gating regression settles it whatever the primary did: the run is judged on
    every metric it gates, so a headline that improved while a gate broke is still
    an iteration to fix rather than one to keep.

    Everything that is neither a gating regression nor an improvement in the
    primary's own direction reads ``no-signal`` — including a primary the run
    never measured, which reports nothing rather than reporting zero.

    Args:
        metrics: The run's per-metric comparisons.
        primary: The one figure the iteration is read on.

    Returns:
        The iteration's outcome.
    """
    if _has_gating_regression(metrics):
        return "regressed"
    return "improved" if _primary_improved(metrics, primary) else "no-signal"


def _judge(config: ResolvedConfig, judged: Judged) -> IterationJudgment:
    """Derive the outcome and bundle the judgment."""
    return IterationJudgment(
        outcome=derive_outcome(judged.result.metrics, judged.primary),
        primary=judged.primary,
        confirmation=judged.confirmation,
        reached_target=target_reached(config, judged.primary, judged.result.metrics),
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
        remaining_minutes = int(_budget.ms_to_minutes(remaining))
        estimate_minutes = int(_budget.ms_to_minutes(estimate.duration_ms))
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
    require_settled(state, "Run gymrat keep or gymrat discard before measuring the next edit.")
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
        options: Progress, abort, and warning hooks; a fresh set is used when
            ``None``.
        color: Explicit color choice for the iteration report — ``True``
            forces ANSI, ``False`` suppresses it, ``None`` defers to the
            environment and TTY.

    Returns:
        The appended record and the report to print for it.

    Raises:
        GymratError: When no session has been started, when the last iteration is
            still unsettled, or when the bench command fails.
        LoopStopError: When a configured stop condition has already been met,
            or (as :class:`BudgetExceededError`) the session's time budget is
            spent, before anything is measured or recorded.
    """
    opts = options if options is not None else IterateOptions()
    required = require_open_session(root, "measuring an edit")
    session, state, jsonl_path = required.session, required.state, required.jsonl_path
    _guard_ready(config, state, root, required.records)

    start_ms = _clock.monotonic_ms()
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

    duration_ms = int(_clock.monotonic_ms() - start_ms)
    measured_tree = _workspace.worktree_fingerprint(Path(session.worktrees.experiment))
    if measured_tree is None:
        warn_to_stderr(
            "Could not fingerprint the experiment worktree; "
            "measured_tree is omitted from the iteration record."
        )

    record = build_iteration_record(
        judged, seq, judgment, duration_ms=duration_ms, measured_tree=measured_tree
    )
    append_record(ctx.jsonl_path, record)
    emit_progress(
        ctx.options.on_progress,
        IterationRecorded(seq=record.seq, outcome=record.outcome, at_ms=monotonic_ms()),
    )

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


#: How long a hook may run before it is killed. Long enough to build, short
#: enough to notice.
HOOK_TIMEOUT_MS = 30_000


@dataclass(frozen=True, slots=True)
class HookInvocation:
    """Which command to run, and everything the payload tells it about the loop so far.

    Attributes:
        command: The command line the consumer configured for this stage.
        stage: Which side of a measurement the hook runs on.
        seq: The iteration the hook brackets -- about to be measured, or just recorded.
        session: The session header, source of the worktree, baseline, and branch.
        last_iteration: The iteration the hook can read, ``None`` while the
            session has measured nothing.
        iteration_count: How many iterations the log holds as of this invocation.
        abort: Event whose setting kills the hook's process group; ``None``
            leaves the hook uninterruptible.
    """

    command: str
    stage: HookStage
    seq: int
    session: SessionRecord
    last_iteration: IterationRecord | None
    iteration_count: int
    abort: asyncio.Event | None = None


@dataclass(frozen=True, slots=True)
class HookRun:
    """What one fired hook leaves behind: a record for the log, a report for the agent.

    Attributes:
        record: The record to append to the session log.
        report: The hook's stdout, truncated and labeled with its stage, then a
            note naming the exit code or timeout when the command did not
            succeed, and the truncated stderr under it. Empty when a successful
            hook printed nothing.
    """

    record: HookRecord
    report: str


async def run_hook(invocation: HookInvocation) -> HookRun:
    """Run the stage's command, handing it the loop as JSON on stdin.

    The command runs in the experiment worktree with the payload on its stdin,
    and is killed after :data:`HOOK_TIMEOUT_MS`. Nothing here raises, so a hook
    that fails, times out, or cannot start never aborts the loop. The record is
    not appended to any log: the caller owns that.

    Args:
        invocation: The stage, command, session, and payload data for the run.

    Returns:
        The hook run result containing the log record and the formatted report,
        whether the command succeeded, failed, timed out, or failed to start.
    """
    payload = json.dumps(_build_payload(invocation))

    started_at = _clock.monotonic_ms()
    result = await exec(
        invocation.command,
        ExecOptions(
            cwd=invocation.session.worktrees.experiment,
            timeout_ms=HOOK_TIMEOUT_MS,
            abort=invocation.abort,
            stdin=f"{payload}\n",
        ),
    )
    duration_ms = _clock.monotonic_ms() - started_at

    timed_out = isinstance(result, ExecTimeoutError)
    record = HookRecord(
        type="hook",
        at=now_ns(),
        stage=invocation.stage,
        seq=invocation.seq,
        # A timeout carries no exit code of its own -- the process was killed
        # before it had one -- so the shared failure code stands in for it.
        exit_code=FAILURE_EXIT_CODE if isinstance(result, ExecTimeoutError) else result.exit_code,
        duration_ms=duration_ms,
        # What the command wrote, not what was relayed: a figure above the relay
        # limit is how a reader of the log learns the report was cut.
        stdout_bytes=result.stdout_bytes,
        stderr_bytes=result.stderr_bytes,
        timed_out=timed_out,
    )
    return HookRun(record=record, report=_format_report(invocation.stage, result))


def _build_payload(invocation: HookInvocation) -> dict[str, object]:
    """The loop as the hook reads it: where the edit lives, which iteration, whose session."""
    session = invocation.session
    last_iteration = invocation.last_iteration
    return {
        "stage": invocation.stage,
        "experiment_dir": session.worktrees.experiment,
        "seq": invocation.seq,
        "last_iteration": record_to_wire(last_iteration) if last_iteration is not None else None,
        "session": {
            "session_id": session.session_id,
            "baseline": {"ref": session.baseline.ref, "sha": session.baseline.sha},
            "branch": session.branch,
            "iteration_count": invocation.iteration_count,
        },
    }


def _format_report(stage: HookStage, result: ExecResult | ExecTimeoutError) -> str:
    """Every stdout line labeled with the stage, then a failing hook's note and stderr under it."""
    lines = _split_lines(limit_output(result.stdout))
    note = _failure_note(result)

    if note is not None:
        lines.append(note)
        lines.extend(_split_lines(limit_output(result.stderr)))

    return "\n".join(f"[{stage}] {line}" for line in lines)


def _failure_note(result: ExecResult | ExecTimeoutError) -> str | None:
    """What to tell the reader about a hook that did not succeed, or ``None`` if it did."""
    if isinstance(result, ExecTimeoutError):
        return f"hook timed out after {result.timeout_ms}ms"
    if result.exit_code != 0:
        return f"hook exited {result.exit_code}"
    return None


def _split_lines(text: str) -> list[str]:
    """``text`` as lines, with the trailing newline a command leaves behind dropped."""
    trimmed = text.removesuffix("\n")
    return [] if trimmed == "" else trimmed.split("\n")


async def _hook_stage(
    ctx: IterationContext,
    seq: int,
    *,
    stage: Literal["before", "after"],
    last_iteration: IterationRecord | None,
    iteration_count: int,
) -> str:
    """Run one lifecycle hook stage, bracketed by progress events and recorded in the log.

    A stage with no configured command runs no process, appends no record, and
    emits no event.

    Args:
        ctx: The iteration context, carrying the session, config, and options.
        seq: The iteration the hook brackets.
        stage: Which side of the measurement the hook runs on.
        last_iteration: The iteration the hook can read, or ``None``.
        iteration_count: How many iterations the log holds as of this stage.

    Returns:
        The text to print for the hook — empty when there was no hook or it
        said nothing.
    """
    hooks = ctx.config.hooks
    command = None if hooks is None else (hooks.before if stage == "before" else hooks.after)
    if command is None:
        return ""
    on_progress = ctx.options.on_progress
    emit_progress(on_progress, HookStarted(stage=stage, at_ms=monotonic_ms()))
    run = await run_hook(
        HookInvocation(
            command=command,
            stage=stage,
            seq=seq,
            session=ctx.session,
            last_iteration=last_iteration,
            iteration_count=iteration_count,
            abort=ctx.options.abort,
        )
    )
    append_record(ctx.jsonl_path, run.record)
    emit_progress(on_progress, HookFinished(stage=stage, at_ms=monotonic_ms()))
    return run.report


async def _measure_and_judge(ctx: IterationContext) -> Judged:
    """Bench the pair, confirm any gating regression, and assemble the comparison."""
    first = await bench_and_judge(ctx, ctx.config.bench, announce_judging=True)

    primary = resolve_primary(ctx.config.primary, first.verdicts, first.metric_meta)
    regressed_names = tuple(
        name
        for name, meta in first.metric_meta.items()
        if is_gating_regression(meta, first.verdicts.get(name))
    )
    emit_progress(
        ctx.options.on_progress,
        JudgeFinished(
            primary_delta_pct=primary.delta_pct,
            regressed=regressed_names,
            metric_count=len(first.metric_meta),
            at_ms=monotonic_ms(),
        ),
    )

    confirmation = await confirm_regressions(ctx, first.verdicts, first.metric_meta)
    run = replace(first, verdicts=apply_confirmation(first.verdicts, confirmation))
    return Judged(
        run=run,
        result=build_iteration_comparison(run, ctx.config.adapter, ctx.config.kinds),
        confirmation=confirmation,
        primary=primary,
    )


_STOP_HINT = "The loop is done. Report what the session measured instead of measuring again."


class LoopStopError(GymratError):
    """A configured stop condition refusing another iteration.

    Separate from a plain :class:`GymratError` because nothing failed: the loop
    ran to the end it was configured for, which the CLI reports as a gate trip
    rather than as a tool failure.
    """

    def __init__(
        self,
        *args: object,
        hint: str | None = None,
        reason: CommandReason | None = "stop-condition",
    ) -> None:
        super().__init__(*args, hint=hint, reason=reason)


class BudgetExceededError(LoopStopError):
    """The session's time budget cannot afford another iteration.

    Raised when a live budget's remaining time is shorter than the estimated
    iterate duration, so the CLI routes it through the same gate-exit path as
    any other stop condition.
    """

    def __init__(
        self,
        *args: object,
        hint: str | None = None,
        reason: CommandReason | None = "budget-exceeded",
    ) -> None:
        super().__init__(*args, hint=hint, reason=reason)


def stop_reason(config: BenchlessConfig, state: SessionState) -> str | None:
    """Which configured stop condition this session has already met, if any.

    Read off the folded log alone, so it settles before a bench command runs: an
    iteration measured past the end of the loop is one the agent would have to
    throw away. ``target_value`` stops the loop only once the target-reaching
    iteration is *kept* — discarding it puts the target back out of reach.

    Args:
        config: The resolved config, carrying the configured stop conditions.
        state: The session's folded state, read for iteration count and
            whether the target has been reached and kept.

    Returns:
        The condition that fired, such as ``max iterations (3 of 3)``, or
        ``None`` when no condition is met yet.
    """
    stop = config.stop
    if stop is None:
        return None
    if stop.max_iterations is not None and state.iteration_count >= stop.max_iterations:
        return f"max iterations ({state.iteration_count} of {stop.max_iterations})"
    if stop.target_value is not None and state.target_reached_and_kept:
        return "target reached and kept"
    return None


def stop_condition(config: BenchlessConfig, state: SessionState) -> LoopStopError | None:
    """The refusal for a configured stop condition this session has already met.

    Args:
        config: The resolved config, carrying the configured stop conditions.
        state: The session's folded state.

    Returns:
        The stop error naming the condition :func:`stop_reason` reports, or
        ``None`` when no condition is met yet.
    """
    reason = stop_reason(config, state)
    if reason is None:
        return None
    return LoopStopError(f"Stop condition met: {reason}", hint=_STOP_HINT)


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
    """The iteration as it prints: the loop's header, the comparison table, the verdict.

    Args:
        result: The comparison result for this iteration.
        seq: The 1-based iteration sequence number, used for the header.
        judgment: The outcome, primary, and optional confirmation.
        color: Explicit ANSI color choice — ``True`` forces color, ``False``
            suppresses it, ``None`` defers to the environment.

    Returns:
        The fully rendered iteration report as a single string.
    """
    confirmation = judgment.confirmation
    reruns: list[RerunConfirmation] = (
        [
            RerunConfirmation(metric=name, answer=_rerun_answer(confirmation, name))
            for name in confirmation.filtered
        ]
        if confirmation is not None
        else []
    )
    header = render_lines(format_loop_header(seq, result.samples), color=color)
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
    )
    return f"{report}\n\n{verdict}"
