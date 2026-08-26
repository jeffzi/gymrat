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

import sys
from collections.abc import Callable
from dataclasses import dataclass

from gymrat_py.config import BenchlessConfig
from gymrat_py.errors import GymratError
from gymrat_py.exec import (
    ExecOptions,
    ExecTimeoutError,
    exec,  # noqa: A004 -- names the subprocess executor `exec`
)
from gymrat_py.loop.output_limit import limit_output
from gymrat_py.model import Effect
from gymrat_py.report.format import format_delta
from gymrat_py.session.clock import now_iso
from gymrat_py.session.records import (
    DiscardRecord,
    IterationRecord,
    KeepChecks,
    KeepReason,
    KeepRecord,
    MetricVerdict,
)
from gymrat_py.session.store import append_record, require_open_session
from gymrat_py.session.workspace import (
    advance_baseline,
    commit_workspace,
    is_worktree_dirty,
    revert_workspace,
    worktree_head,
)

MS_PER_SECOND = 1000


@dataclass(frozen=True, slots=True)
class ChecksRun:
    """What the checks command answered, once it has run."""

    passed: bool
    #: Both streams as the agent needs to read them, each cut to the relay limit.
    output: str
    #: What the command wrote on stdout, however much of it ``output`` carries.
    stdout_bytes: int
    #: What the command wrote on stderr, however much of it ``output`` carries.
    stderr_bytes: int


@dataclass(frozen=True, slots=True)
class KeepOptions:
    """What a caller can hand a keep beyond its configuration."""

    #: The commit message; absent, one is generated from the iteration being kept.
    message: str | None = None


@dataclass(frozen=True, slots=True)
class KeepResult:
    """One settled — or refused — keep: what was logged, and what to print about it."""

    #: The record appended to the session log, committed or blocked.
    record: KeepRecord
    #: The keep as the agent reads it: the commit, or the reason there is none.
    report: str


@dataclass(frozen=True, slots=True)
class DiscardResult:
    """One reverted iteration: what was logged, and what to print about it."""

    #: The record appended to the session log.
    record: DiscardRecord
    #: The discard as the agent reads it.
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
) -> KeepResult:
    """Commit the measured edit standing in the experiment worktree, if it may be kept.

    Raises:
        GymratError: When no session has been started, or when git refuses to
            commit the worktree or to advance the baseline.
    """
    options = options or KeepOptions()
    required = require_open_session(root, "settling an edit")
    session, state, jsonl_path = required.session, required.state, required.jsonl_path
    configured = config.checks is not None

    iteration = state.last_iteration if state.unsettled else None
    if iteration is None:
        return _blocked_keep(
            jsonl_path=jsonl_path,
            # The refusal settles nothing, so it takes the number no iteration has
            # used yet: numbering it ``last_seq`` would leave the log with two
            # settlement records against an already kept or discarded iteration.
            seq=state.last_seq + 1,
            reason="nothing-measured",
            checks=KeepChecks(configured=configured),
            report=(
                "Keep refused: nothing has been measured since the last keep or discard.\n"
                "Hint: run gymrat iterate first — an unmeasured commit is one the loop "
                "cannot account for."
            ),
        )

    if _has_standing_gating_regression(iteration):
        return _blocked_keep(
            jsonl_path=jsonl_path,
            seq=iteration.seq,
            reason="gating-regression",
            checks=KeepChecks(configured=configured),
            report=_gating_refusal(iteration),
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
        # The worktree is clean — either the agent made no changes (nothing to
        # commit), or the work is already committed and the baseline has yet to
        # move over it. The baseline's current position distinguishes them.
        return await _keep_clean_worktree(
            context, baseline_position=state.last_kept_commit or session.baseline.sha
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
        # HEAD matches the baseline: the agent measured an iteration but never
        # edited the worktree. There is nothing to commit, and running checks or
        # git-commit would waste time on a tree that has nothing to give.
        return _blocked_keep(
            jsonl_path=context.jsonl_path,
            seq=context.iteration.seq,
            reason="nothing-to-commit",
            checks=KeepChecks(configured=context.config.checks is not None),
            report=(
                "Keep refused: the experiment worktree has nothing to commit.\n"
                "Hint: edit the code in the experiment worktree, then run gymrat keep again."
            ),
        )

    # HEAD is ahead of the baseline: a prior call committed the work and failed at
    # advance_baseline or append_record, or something ran git commit in the
    # worktree outside gymrat. Nothing here distinguishes the two, so the gate runs
    # on the commit standing there rather than assuming anything ever examined it —
    # trusting the ahead-ness alone would let a direct commit walk work the checks
    # never saw into the baseline. A genuine retry passed these checks once and
    # passes them again.
    return await _gated_keep(context, commit=lambda _message: head)


async def _gated_keep(context: _KeepContext, *, commit: Callable[[str], str]) -> KeepResult:
    """Gate the experiment worktree on the checks, then keep what ``commit`` returns.

    Both keep paths settle through here, so the gate cannot be skipped by whichever
    of them produced the commit: ``commit`` is called only once the checks have
    passed, and it either makes the commit from the worktree's uncommitted work or
    hands back the one already standing at HEAD.
    """
    checks = await _run_checks(context.config, context.experiment_dir)
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
    """Record the refusal a failing checks run earns, phrased for the agent.

    Both keep paths gate on the same run, so both refuse in the same words and with
    the same record: an agent must not have to tell from the wording whether the
    tree it has to fix was committed before the gate ran.
    """
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
            f"Keep refused: the checks command failed.\n\n{checks.output}\n"
            "Hint: fix the failures and run gymrat keep again."
        ),
    )


def _passed_checks_field(checks: ChecksRun | None) -> KeepChecks:
    """The ``checks`` a keep records once the gate let it through, gate off included."""
    if checks is None:
        return KeepChecks(configured=False)
    return KeepChecks(configured=True, passed=True)


def _commit_keep(
    context: _KeepContext, *, commit: str, message: str, checks: KeepChecks
) -> KeepResult:
    """Record a keep that committed, advance the baseline to it, and phrase the report.

    Both the fresh commit made from the worktree's uncommitted work and the one
    :func:`_keep_clean_worktree` finds already standing at HEAD settle through here,
    so the record shape and the report wording stay identical whichever path
    produced the commit.
    """
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
    # sampling a baseline the log says it has already left behind. Failing with the
    # iteration still unsettled lets the agent retry the keep.
    advance_baseline(context.baseline_dir, commit)
    append_record(context.jsonl_path, record)

    return KeepResult(
        record=record,
        report=(
            f"Kept iteration {context.iteration.seq} as {commit}\n  message: {message}\n"
            "  the baseline now measures against this commit"
        ),
    )


def discard_session(root: str, expected_session_id: str | None = None) -> DiscardResult:
    """Throw the experiment worktree's uncommitted work away and record that it went.

    A clean worktree is discarded just as loudly as a dirty one: the record is what
    settles the iteration, and gymrat does not guess whether an agent that changed
    nothing meant to. What there must be is an edit to throw away — either an
    unsettled iteration, or the one a gating regression refused to commit, which is
    settled in the log yet still standing in the worktree. Anywhere else the discard
    would number itself after an iteration the log already settled, and history
    would read as two settlements of a single iteration.

    Raises:
        GymratError: When no session has been started, when nothing has been
            measured since the last keep or discard, or when git refuses to revert
            the worktree.
    """
    required = require_open_session(root, "settling an edit")
    session, state, jsonl_path = required.session, required.state, required.jsonl_path

    if expected_session_id is not None and session.session_id != expected_session_id:
        stale_message = (
            f"Discard refused: the session on disk ({session.session_id}) is not the "
            f"one the prompt named ({expected_session_id})."
        )
        raise GymratError(
            stale_message,
            hint=(
                "Another process started a new session between the confirmation and the "
                "lock. Run gymrat discard again to confirm against the current session."
            ),
        )

    if not state.unsettled and not state.ends_on_gating_block:
        nothing_message = (
            "Discard refused: nothing has been measured since the last keep or discard."
        )
        raise GymratError(
            nothing_message,
            hint="Run gymrat iterate to measure an edit before settling it.",
        )

    revert_workspace(session.worktrees.experiment)

    record = DiscardRecord(
        type="discard",
        # The block already settled the iteration it refused, so the discard behind
        # it takes the number no iteration has used yet — the same number a refused
        # keep takes. Reusing the iteration's own seq would make the discard the last
        # settling record to carry it, and ``gymrat status`` would render it in place
        # of the block instead of alongside it.
        seq=state.last_seq + 1 if state.ends_on_gating_block else state.last_seq,
        at=now_iso(),
    )
    append_record(jsonl_path, record)

    return DiscardResult(
        record=record,
        report=(
            f"Discarded iteration {state.last_seq}: the experiment worktree is back "
            "at its last commit"
        ),
    )


def _has_standing_gating_regression(iteration: IterationRecord) -> bool:
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
    """The gating metrics that regressed and the confirmation rerun never reported on."""
    confirm = iteration.confirm
    absent = set(confirm.absent) if confirm is not None and confirm.absent is not None else set()
    return [
        name
        for name, metric in iteration.metrics.items()
        if _is_gating_regression(metric) and name in absent
    ]


def _is_gating_regression(metric: MetricVerdict) -> bool:
    """Whether a metric is a gating metric that the checks called a regression."""
    return metric.gating and metric.verdict == "regressed"


def _gating_refusal(iteration: IterationRecord) -> str:
    """How the refusal reads to the agent that has to act on it.

    A regression the rerun stood behind needs no explaining beyond the number the
    iteration already reported. One the rerun never re-measured does: the agent is
    looking at a metric its own report called regressed and unconfirmed, and without
    the missing measurement named, the block reads as gymrat contradicting itself.
    The extra hint points at the likeliest cause — a filter template that narrows
    the rerun to a subset the bench does not answer with.
    """
    refusal = f"Keep refused: iteration {iteration.seq} regressed a gating metric."
    settle_hint = "fix the regression and run gymrat iterate again, or run gymrat discard"

    unmeasured = _unmeasured_gating_regressions(iteration)
    if not unmeasured:
        return f"{refusal}\nHint: {settle_hint}."

    named = "\n".join(
        f"  {name}: not measured on the confirmation rerun, so the regression stands"
        for name in unmeasured
    )
    reported = ", ".join(unmeasured)
    return (
        f"{refusal}\n{named}\n"
        f"Hint: check that the filter template (or the bench itself) reports {reported}, "
        f"then {settle_hint}."
    )


async def _run_checks(config: BenchlessConfig, experiment_dir: str) -> ChecksRun | None:
    """Run the configured checks in the experiment worktree.

    A timeout counts as a failure with whatever the command managed to write: the
    gate asks whether the tree is provably good, and a run that never finished has
    not answered. Each stream is cut to the relay limit on its own, so a suite that
    writes its failures to stderr is as readable as one that writes them to stdout.

    Returns:
        What the command answered, or ``None`` when no checks are configured — in
        which case the missing gate is warned about instead.
    """
    command = config.checks
    if command is None:
        sys.stderr.write(
            "Warning: no checks command is configured, so gymrat keep is committing "
            "with the gate off.\n"
            'Hint: set "checks" in gymrat.toml to the command that must pass before an '
            "edit is kept.\n"
        )
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
        # What the command wrote, not what was relayed: a figure above the relay
        # limit is how a reader of the log learns the report was cut short.
        stdout_bytes=result.stdout_bytes,
        stderr_bytes=result.stderr_bytes,
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
    """The commit message a keep writes when the agent supplied none.

    It names the iteration and the figure it was read on, so the branch's history
    reads back as the loop that produced it. A figure whose ratio had no value says
    so in words: a report can print a blank percentage and let the glyph beside it
    carry the news, but a commit subject would trail off on the figure's bare name.
    """
    primary = iteration.primary
    moved = (
        "delta undefined"
        if primary.delta_pct is None
        else format_delta(Effect(value=primary.delta_pct, unit="percent"))
    )
    return f"iteration {iteration.seq}: {primary.name or primary.kind} {moved}"
