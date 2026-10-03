"""Keep a measured edit: commit it and advance the baseline worktree to it.

A keep passes three gates — something measured, no standing gating regression,
and the configured checks — and each gate that trips is *recorded* rather than
thrown. A blocked keep is history the agent and ``gymrat status`` can read back,
which a raised error would leave nowhere. The caller turns a blocked record into
an exit code; every other failure here is a :class:`GymratError`.

The checks gate runs the configured checks command and shapes its output for the
keep record; the gating gate decides from the session's verdicts whether a
standing gating regression blocks the keep.

Holding the repository lock across a keep is the caller's job: the baseline
worktree moves in the middle of it, and a concurrent iterate must not sample it
mid-advance.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from rich.markup import escape

from gymrat.clock import now_ns
from gymrat.eta import MS_PER_SECOND
from gymrat.exec import (
    ExecOptions,
    ExecTimeoutError,
    exec,  # noqa: A004 -- names the subprocess executor `exec`
)
from gymrat.git import SHORT_SHA_LENGTH
from gymrat.loop.output_limit import limit_output
from gymrat.report.format import format_percent_delta
from gymrat.report.style import RENDER_WIDTH, color_from_env, format_hint, render_lines
from gymrat.session.records import (
    BaselineRecord,
    IterationRecord,
    KeepChecks,
    KeepRecord,
)
from gymrat.session.store import append_record, last_kept_position, require_open_session
from gymrat.session.workspace import (
    advance_baseline,
    commit_workspace,
    is_worktree_dirty,
    worktree_head,
)
from gymrat.warn import WarnSink, warn_to_stderr

if TYPE_CHECKING:
    from collections.abc import Callable

    from gymrat.config.types import BenchlessConfig
    from gymrat.session.records import MetricVerdict
    from gymrat.session.schema import KeepReason


# ---------------------------------------------------------------------------
# checks and gating gates
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ChecksRun:
    """What the checks command answered, once it has run."""

    passed: bool
    output: str
    stdout_bytes: int
    stderr_bytes: int


def _stderr_color() -> bool:
    """Whether the warning gymrat writes to stderr carries color.

    :func:`color_from_env` owns the ``FORCE_COLOR`` / ``NO_COLOR`` precedence
    every color surface shares; with neither declared, stderr's own TTY state
    decides, so a warning piped into a file stays plain.

    Returns:
        Whether stderr output should carry ANSI color escapes.
    """
    declared = color_from_env()
    return declared if declared is not None else sys.stderr.isatty()


def _gate_off_warning(*, color: bool) -> str:
    """The warning a keep emits when no checks command gates it.

    Args:
        color: Whether the hint line carries ANSI color escapes.

    Returns:
        The warning and its hint, without a trailing newline.
    """
    hint = render_lines(
        format_hint(
            "set `checks` in `gymrat.toml` to the command that must pass before an edit is kept."
        ),
        color=color,
        width=RENDER_WIDTH,
    )
    return (
        "Warning: no checks command is configured, so gymrat keep is committing "
        f"with the gate off.\n{hint}"
    )


async def run_checks(
    config: BenchlessConfig,
    experiment_dir: str,
    warn: WarnSink | None = None,
    *,
    color: bool | None = None,
) -> ChecksRun | None:
    """Run the configured checks in the experiment worktree.

    A timeout counts as a failure with whatever the command managed to write: the
    gate asks whether the tree is provably good, and a run that never finished has
    not answered. Each stream is cut to the relay limit on its own, so a suite that
    writes its failures to stderr is as readable as one that writes them to stdout.

    Args:
        config: The resolved config, carrying the checks command and timeout.
        experiment_dir: The experiment worktree to run the checks command in.
        warn: Where the gate-off warning goes, or ``None`` to write it to stderr.
            A caller-supplied sink owns its own presentation — a CLI interleaving
            the warning with a progress line, for one — so it is handed plain
            text, while the stderr default keeps the color it renders with.
        color: Whether the warning written to stderr carries color, or ``None``
            to defer to ``FORCE_COLOR``, ``NO_COLOR`` and stderr's TTY state.

    Returns:
        What the command answered, or ``None`` when no checks are configured — in
        which case the missing gate is warned about instead.
    """
    command = config.checks
    if command is None:
        if warn is None:
            stderr_color = _stderr_color() if color is None else color
            warn_to_stderr(_gate_off_warning(color=stderr_color))
        else:
            warn(_gate_off_warning(color=False))
        return None

    result = await exec(
        command,
        ExecOptions(cwd=experiment_dir, timeout_ms=config.timeout_seconds * MS_PER_SECOND),
    )

    if isinstance(result, ExecTimeoutError):
        passed = False
        lead = [f"{command} timed out after {result.timeout_ms}ms"]
    else:
        passed = result.exit_code == 0
        lead: list[str] = []

    output = "\n".join(
        part
        for part in (*lead, limit_output(result.stdout), limit_output(result.stderr))
        if part.strip() != ""
    )

    return ChecksRun(
        passed=passed,
        output=output,
        stdout_bytes=result.stdout_bytes,
        stderr_bytes=result.stderr_bytes,
    )


def has_standing_gating_regression(iteration: IterationRecord) -> bool:
    """Whether the iteration carries a regression the loop refuses to commit over.

    Both halves are required: the outcome is what the agent was shown, and a gating
    metric standing behind the regression is what makes it real. A noisy metric
    earns that standing from the confirmation rerun — a regression the rerun would
    not repeat leaves the iteration keepable. An exact metric is deterministic, so
    the rerun skips it and its ``confirmed`` stays ``False``; gating on ``confirmed``
    alone would let every exact regression through.

    Silence earns the same standing as disagreement: a metric the rerun was asked
    about and never reported back on lands in ``confirm.absent``, its ``confirmed``
    still ``False`` because nothing re-measured it. The gate fails closed on those —
    a rerun that cannot see the metric is not evidence the regression went away.

    Args:
        iteration: The iteration record to check for a standing gating regression.

    Returns:
        Whether the iteration carries a confirmed or exact gating regression.
    """
    if iteration.outcome != "regressed":
        return False
    if _unmeasured_gating_regressions(iteration):
        return True
    return any(
        _is_gating_regression(metric) and (metric.confirmed or metric.method == "exact")
        for metric in iteration.metrics.values()
    )


def _unmeasured_gating_regressions(iteration: IterationRecord) -> list[str]:
    confirm = iteration.confirm
    absent = set(confirm.absent) if confirm is not None and confirm.absent is not None else set()
    return [
        name
        for name, metric in iteration.metrics.items()
        if _is_gating_regression(metric) and name in absent
    ]


def _is_gating_regression(metric: MetricVerdict) -> bool:
    return metric.gating and metric.verdict == "regressed"


def gating_refusal(iteration: IterationRecord) -> str:
    """How the refusal reads to the agent that has to act on it, as markup.

    A regression the rerun stood behind needs no explaining beyond the number the
    iteration already reported. One the rerun never re-measured does: the agent is
    looking at a metric its own report called regressed and unconfirmed, and without
    the missing measurement named, the block reads as gymrat contradicting itself.
    The extra hint points at the likeliest cause — a filter template that narrows
    the rerun to a subset the bench does not answer with.

    Args:
        iteration: The iteration record whose gating regression is refused.

    Returns:
        The refusal message as markup, including a hint for the agent.
    """
    refusal = f"Keep refused: iteration {iteration.seq} regressed a gating metric."
    settle_hint = "fix the regression and run `iterate` again, or run `discard`"

    unmeasured = _unmeasured_gating_regressions(iteration)
    if not unmeasured:
        return f"{refusal}\n{format_hint(f'{settle_hint}.')}"

    named = "\n".join(
        f"  {escape(name)}: not measured on the confirmation rerun, so the regression stands"
        for name in unmeasured
    )
    reported = ", ".join(unmeasured)
    return f"{refusal}\n{named}\n" + format_hint(
        f"check that the filter template (or the bench itself) reports {reported}, "
        f"then {settle_hint}."
    )


# ---------------------------------------------------------------------------
# keep
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class KeepOptions:
    """What a caller can hand a keep beyond its configuration.

    Attributes:
        message: Message to attach to the keep record.
        allow_unimproved: Whether to keep a result the loop did not measure as an
            improvement. Defaults to refusing: an automated caller that passes no
            options keeps only what the loop measured as an improvement.
        warn: Sink for the checks gate's warnings. Defaults to ``None``, which
            leaves those warnings on stderr.
        warn_color: Whether a warning left on stderr carries color. Defaults to
            ``None``, which defers to ``FORCE_COLOR``, ``NO_COLOR`` and stderr's
            TTY state. A ``warn`` sink always receives plain text, so it ignores
            this.
    """

    message: str | None = None
    allow_unimproved: bool = False
    warn: WarnSink | None = None
    warn_color: bool | None = None


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
    warn: WarnSink | None
    warn_color: bool | None


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

    Args:
        root: The repository root.
        config: The bench-less configuration governing the keep's checks.
        options: What the caller hands the keep beyond its configuration.

    Returns:
        The keep result with the report still in markup form.

    Raises:
        GymratError: When no session is open, or git or the session log refuses
            the commit, the record append, or the baseline advance.
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
                    "run `iterate` first — an unmeasured commit is one the loop cannot account for."
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

    # After the gating gate, so a confirmed regression keeps reading as one, and
    # before the checks run, so a refused keep never spends the consumer's suite.
    if iteration.outcome != "improved" and not options.allow_unimproved:
        return _blocked_keep(
            jsonl_path=jsonl_path,
            seq=iteration.seq,
            reason="not-improved",
            checks=KeepChecks(configured=configured),
            report=(
                f"Keep refused: the iteration was {iteration.outcome}, not improved.\n"
                + format_hint("discard it, or pass --allow-unimproved to keep it anyway.")
            ),
        )

    experiment_dir = session.worktrees.experiment
    context = _KeepContext(
        jsonl_path=jsonl_path,
        config=config,
        experiment_dir=experiment_dir,
        baseline_dir=session.worktrees.baseline,
        iteration=iteration,
        message=options.message,
        warn=options.warn,
        warn_color=options.warn_color,
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

    Args:
        context: The keep context carrying the worktree, config, and iteration.
        baseline_position: The commit the experiment worktree is expected to be
            at when there is nothing new to commit.

    Returns:
        The keep result — blocked if the worktree has nothing new, or gated
        against the standing commit.

    Raises:
        GymratError: When git cannot read the worktree HEAD, or the record append
            or baseline advance fails.
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
                + format_hint("edit the code in the experiment worktree, then run `iterate` again.")
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
    of them produced the commit.

    Args:
        context: The keep context carrying the worktree, config, and iteration.
        commit: Produces the commit SHA to keep, given the commit message. Called
            only once the checks have passed; it either makes the commit from the
            worktree's uncommitted work or hands back the one already standing at
            HEAD.

    Returns:
        The keep result — committed if the checks passed, blocked otherwise.

    Raises:
        GymratError: When ``commit``, the baseline advance, or the record append
            fails.
    """
    checks = await run_checks(
        context.config, context.experiment_dir, context.warn, color=context.warn_color
    )
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
            + format_hint("fix the failures and run `keep` again.")
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
        at=now_ns(),
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
    # Bookkeeping, so it follows the settlement: the kept commit is the baseline
    # from here on, and the iteration already measured it. Nothing is benched.
    # A failure between the two appends leaves a committed keep whose baseline
    # record is missing, which readers tolerate by falling back to the older one.
    append_record(
        context.jsonl_path,
        BaselineRecord(
            type="baseline",
            at=now_ns(),
            label=commit[:SHORT_SHA_LENGTH],
            samples=context.iteration.samples.experiment,
        ),
    )

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
        at=now_ns(),
        status="blocked",
        checks=checks,
        reason=reason,
    )
    append_record(jsonl_path, record)
    return KeepResult(record=record, report=report)


def _generated_message(iteration: IterationRecord) -> str:
    primary = iteration.primary
    moved = (
        "delta undefined" if primary.delta_pct is None else format_percent_delta(primary.delta_pct)
    )
    return f"iteration {iteration.seq}: {primary.name or primary.kind} {moved}"
