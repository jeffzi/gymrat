"""Settle a measured edit: keep it into the baseline, or discard it.

A keep passes three gates — something measured, no standing gating regression,
and the configured checks — and each gate that trips is *recorded* rather than
thrown. A blocked keep is history the agent and ``gymrat status`` can read back,
which a raised error would leave nowhere. The caller turns a blocked record into
an exit code; every other failure here is a :class:`GymratError`.

Holding the repository lock across either call is the caller's job: the baseline
worktree moves in the middle of a keep, and a concurrent iterate must not sample
it mid-advance.
"""

from collections.abc import Callable
from dataclasses import dataclass, replace

from rich.markup import escape

from gymrat.config import BenchlessConfig
from gymrat.loop.settle.checks import (
    ChecksRun,
    gating_refusal,
    has_standing_gating_regression,
    run_checks,
)
from gymrat.model import Effect
from gymrat.report.format import format_delta
from gymrat.report.style import RENDER_WIDTH, format_hint, render_lines
from gymrat.session.clock import now_iso
from gymrat.session.records import (
    IterationRecord,
    KeepChecks,
    KeepRecord,
)
from gymrat.session.schema import KeepReason
from gymrat.session.store import append_record, last_kept_position, require_open_session
from gymrat.session.workspace import (
    advance_baseline,
    commit_workspace,
    is_worktree_dirty,
    worktree_head,
)

__all__ = [
    "KeepOptions",
    "KeepResult",
    "keep_session",
]


@dataclass(frozen=True, slots=True)
class KeepOptions:
    """What a caller can hand a keep beyond its configuration."""

    message: str | None = None


@dataclass(frozen=True, slots=True)
class KeepResult:
    """One settled — or refused — keep: what was logged, and what to print about it."""

    record: KeepRecord
    report: str


@dataclass(frozen=True, slots=True)
class _KeepContext:
    """The settle context both keep paths thread through.

    Bundling the shared inputs keeps the gate helpers to a couple of parameters
    each and makes it plain that both paths settle against the same worktree,
    baseline, and iteration.
    """

    jsonl_path: str
    config: BenchlessConfig
    experiment_dir: str
    baseline_dir: str
    iteration: IterationRecord
    message: str | None


async def keep_session(
    root: str,
    config: BenchlessConfig,
    options: KeepOptions | None = None,
    *,
    color: bool | None = None,
) -> KeepResult:
    """Commit the measured edit standing in the experiment worktree, if it may be kept.

    Args:
        root: The repository the session was started in.
        config: The configuration the checks gate is read from.
        options: What the caller hands the keep beyond its configuration.
        color: Whether the report carries color, or ``None`` to defer to the
            environment.

    Returns:
        The keep as it was logged, with its report rendered for the terminal.

    Raises:
        GymratError: When no session has been started, or when git refuses to
            commit the worktree or to advance the baseline.
    """
    settled = await _settle_keep(root, config, options or KeepOptions())
    return replace(settled, report=render_lines(settled.report, color=color, width=RENDER_WIDTH))


async def _settle_keep(root: str, config: BenchlessConfig, options: KeepOptions) -> KeepResult:
    """Run the keep gates, leaving the report as the markup :func:`keep_session` renders.

    Every gate phrases its refusal as markup and none of them renders it, so the
    color choice is resolved once for the whole report however the keep settled.
    """
    required = require_open_session(root, "settling an edit")
    session, state, jsonl_path = required.session, required.state, required.jsonl_path
    configured = config.checks is not None

    iteration = state.last_iteration if state.unsettled else None
    if iteration is None:
        return _blocked_keep(
            jsonl_path=jsonl_path,
            seq=state.last_seq + 1,
            reason="nothing-measured",
            checks=KeepChecks(configured=configured),
            report=(
                "Keep refused: nothing has been measured since the last keep or discard.\n"
                + format_hint(
                    "run `gymrat iterate` first — an unmeasured commit is one the loop "
                    "cannot account for."
                )
            ),
        )

    if has_standing_gating_regression(iteration):
        return _blocked_keep(
            jsonl_path=jsonl_path,
            seq=iteration.seq,
            reason="gating-regression",
            checks=KeepChecks(configured=configured),
            report=gating_refusal(iteration),
        )

    experiment_dir = session.worktrees.experiment
    context = _KeepContext(
        jsonl_path=jsonl_path,
        config=config,
        experiment_dir=experiment_dir,
        baseline_dir=session.worktrees.baseline,
        iteration=iteration,
        message=options.message,
    )

    if not is_worktree_dirty(experiment_dir):
        return await _keep_clean_worktree(
            context, baseline_position=last_kept_position(state, session.baseline.sha)
        )

    return await _gated_keep(
        context, commit=lambda message: commit_workspace(experiment_dir, message)
    )


async def _keep_clean_worktree(context: _KeepContext, *, baseline_position: str) -> KeepResult:
    """Settle a keep against a worktree that has nothing left to commit.

    Either nothing was measured (the agent never edited the tree) or the work is
    already committed and only the baseline advance is outstanding, in which case
    the commit already made is gated and picked up rather than repeated.
    """
    head = worktree_head(context.experiment_dir)

    if head == baseline_position:
        return _blocked_keep(
            jsonl_path=context.jsonl_path,
            seq=context.iteration.seq,
            reason="nothing-to-commit",
            checks=KeepChecks(configured=context.config.checks is not None),
            report=(
                "Keep refused: the experiment worktree has nothing to commit.\n"
                + format_hint(
                    "edit the code in the experiment worktree, then run `gymrat iterate` again."
                )
            ),
        )

    # HEAD is ahead of the baseline: a prior call committed the work and failed at
    # advance_baseline or append_record, or something ran git commit in the
    # worktree outside gymrat. The gate runs on the commit standing there rather
    # than assuming anything ever examined it.
    return await _gated_keep(context, commit=lambda _message: head)


async def _gated_keep(context: _KeepContext, *, commit: Callable[[str], str]) -> KeepResult:
    """Gate the experiment worktree on the checks, then keep what ``commit`` returns.

    Both keep paths settle through here, so the gate cannot be skipped by whichever
    of them produced the commit: ``commit`` is called only once the checks have
    passed, and it either makes the commit from the worktree's uncommitted work or
    hands back the one already standing at HEAD.
    """
    checks = await run_checks(context.config, context.experiment_dir)
    if checks is not None and not checks.passed:
        return _checks_failed_keep(context.jsonl_path, context.iteration.seq, checks)

    resolved_message = (
        context.message if context.message is not None else _generated_message(context.iteration)
    )

    return _commit_keep(
        context,
        commit=commit(resolved_message),
        message=resolved_message,
        checks=_passed_checks_field(checks),
    )


def _checks_failed_keep(jsonl_path: str, seq: int, checks: ChecksRun) -> KeepResult:
    """Record the refusal a failing checks run earns, phrased for the agent."""
    return _blocked_keep(
        jsonl_path=jsonl_path,
        seq=seq,
        reason="checks-failed",
        checks=KeepChecks(
            configured=True,
            passed=False,
            stdout_bytes=checks.stdout_bytes,
            stderr_bytes=checks.stderr_bytes,
        ),
        report=(
            f"Keep refused: the checks command failed.\n\n{escape(checks.output)}\n"
            + format_hint("fix the failures and run `gymrat keep` again.")
        ),
    )


def _passed_checks_field(checks: ChecksRun | None) -> KeepChecks:
    if checks is None:
        return KeepChecks(configured=False)
    return KeepChecks(configured=True, passed=True)


def _commit_keep(
    context: _KeepContext, *, commit: str, message: str, checks: KeepChecks
) -> KeepResult:
    record = KeepRecord(
        type="keep",
        seq=context.iteration.seq,
        at=now_iso(),
        status="committed",
        checks=checks,
        commit=commit,
        message=message,
    )
    # Move the baseline before recording the keep: a record written first would
    # settle the iteration even when git refuses the advance, leaving the loop
    # sampling a baseline the log says it has already left behind.
    advance_baseline(context.baseline_dir, commit)
    append_record(context.jsonl_path, record)

    return KeepResult(
        record=record,
        report=(
            f"Kept iteration {context.iteration.seq} as {commit}\n"
            f"  message: {escape(message)}\n"
            "  the baseline now measures against this commit"
        ),
    )


def _blocked_keep(
    *,
    jsonl_path: str,
    seq: int,
    reason: KeepReason,
    checks: KeepChecks,
    report: str,
) -> KeepResult:
    """Record the refusal so the log carries it, and phrase it for the agent."""
    record = KeepRecord(
        type="keep",
        seq=seq,
        at=now_iso(),
        status="blocked",
        checks=checks,
        reason=reason,
    )
    append_record(jsonl_path, record)
    return KeepResult(record=record, report=report)


def _generated_message(iteration: IterationRecord) -> str:
    primary = iteration.primary
    moved = (
        "delta undefined"
        if primary.delta_pct is None
        else format_delta(Effect(value=primary.delta_pct, unit="percent"))
    )
    return f"iteration {iteration.seq}: {primary.name or primary.kind} {moved}"
