"""The settle and finalize decisions the exit sequence records.

:mod:`gymrat.supervisor.exit_sequence` owns the frame — the lock wait, the
repository lock, the error boundary, the report — and delegates every decision
that reads the session's own state to this module: whether an unsettled
iteration is kept, discarded, or left for a person, and whether the session is
finalized once nothing needs one.

One gate guards every automatic keep and every automatic discard: the iteration
measured the tree that still stands, and no hook of that iteration failed. Work
that fails it is left exactly as it is, named in the step so a person can pick it
up; a run that settles nothing by hand is worth less than one that never touched
what it could not vouch for.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from gymrat.errors import GymratError
from gymrat.git import SHORT_SHA_LENGTH
from gymrat.loop.finalize import finalize_session
from gymrat.loop.settle import KeepOptions, discard_session, keep_session
from gymrat.session.paths import session_jsonl_path
from gymrat.session.records import HookRecord
from gymrat.session.store import fold_session, last_kept_position, read_records
from gymrat.session.workspace import changed_file_count, worktree_fingerprint

if TYPE_CHECKING:
    from collections.abc import Callable

    from gymrat.cli.lock import CommandTrace
    from gymrat.session.records import IterationRecord
    from gymrat.session.store import SessionState
    from gymrat.supervisor.context import SupervisedSession
    from gymrat.warn import WarnSink

#: What a tree that no longer matches the fingerprint the iteration measured
#: reads as. The interrupted discard is named because a ``discard`` killed
#: between the revert and its record leaves an unsettled iteration whose fresh
#: fingerprint is the kept tree, and the summary must not misattribute it.
_TREE_CHANGED = (
    "tree changed after measuring — an after hook, a later edit, or an interrupted discard"
)

_NO_FINGERPRINT = "fingerprint unavailable"

_BY_HAND = "keep or discard it by hand"


type ExitStepKind = Literal["skipped", "settled", "left", "finalized", "refused", "nothing"]


@dataclass(frozen=True, slots=True)
class ExitStep:
    """One decision the sequence took, as the caller reports it.

    Attributes:
        kind: What the decision did with the session.
        text: The summary wording, rendered verbatim.
    """

    kind: ExitStepKind
    text: str


@dataclass(frozen=True, slots=True)
class ExitSinks:
    """Everywhere a decision writes as it is taken.

    Attributes:
        record: Called with each step the moment it is decided, so the steps that
            completed before a later one raises are still on the report.
        warn: Where a keep's hint about a missing checks command goes, so it
            never reaches stderr while a dashboard is live.
        trace: The trace the repository-lock seam turns into the session log's
            command record once the sequence settles.
    """

    record: Callable[[ExitStep], None]
    warn: WarnSink
    trace: CommandTrace


async def decide_exit_steps(
    context: SupervisedSession,
    state: SessionState,
    sinks: ExitSinks,
    *,
    finalize: bool,
) -> None:
    """Record the settle and finalize steps ``state`` calls for, in decision order.

    Args:
        context: The supervised session whose repository is being settled.
        state: The folded session log every decision reads.
        sinks: Where the steps, the keep's warnings, and the command trace go.
        finalize: Whether the session may be closed once nothing needs a person.
    """
    settled = await _decide_settle(context, state, sinks)
    if settled is not None:
        sinks.record(settled)
    _decide_finalize(context, settled, sinks, finalize=finalize)


def _decide_finalize(
    context: SupervisedSession,
    settled: ExitStep | None,
    sinks: ExitSinks,
    *,
    finalize: bool,
) -> None:
    """Record the finalize step the settled session calls for, if it calls for one.

    A session the finalize would refuse anyway — nothing kept, or something still
    unsettled once the settle step has run — records no step at all, so neither a
    refusal nor the ``--no-finalize`` note stands in for work that was never
    going to close.

    Args:
        context: The supervised session whose repository is being settled.
        settled: What the settle decision did, or ``None`` when it did nothing. A
            ``left`` step means work still needs a person, so no finalize may
            close over it.
        sinks: Where the step goes once it is decided.
        finalize: Whether the session may be closed once nothing needs a person.
    """
    if settled is not None and settled.kind == "left":
        return
    after = fold_session(read_records(session_jsonl_path(context.root)))
    if after.keep_count == 0 or after.unsettled:
        return
    if not finalize:
        sinks.record(ExitStep(kind="nothing", text="session left open (--no-finalize)"))
        return
    try:
        closed = finalize_session(context.root)
    except GymratError as refusal:
        sinks.record(ExitStep(kind="refused", text=str(refusal)))
        return
    short_sha = closed.record.commit[:SHORT_SHA_LENGTH]
    sinks.record(
        ExitStep(kind="finalized", text=f"finalized: {closed.record.branch} at {short_sha}")
    )


async def _decide_settle(
    context: SupervisedSession, state: SessionState, sinks: ExitSinks
) -> ExitStep | None:
    """The one settle step the session calls for, if it calls for one.

    The iteration a step settles or leaves is the one the command record names,
    so its seq lands on the trace before the step itself is decided.

    Args:
        context: The supervised session whose repository is being settled.
        state: The folded session log the decision reads.
        sinks: Where the keep's warnings and the command trace go.

    Returns:
        The step the session calls for, or ``None`` when it calls for none.
    """
    session = state.session
    if session is None:
        return None

    experiment = session.worktrees.experiment
    iteration = state.last_iteration
    if iteration is not None and state.unsettled:
        sinks.trace.seq = iteration.seq
        return await _settle_iteration(context, iteration, sinks, experiment=experiment)
    if iteration is not None and state.ends_on_gating_block:
        sinks.trace.seq = iteration.seq
        return _settle_gating_block(context, iteration, experiment=experiment)

    unmeasured = changed_file_count(experiment, last_kept_position(state, session.baseline.sha))
    if unmeasured > 0:
        return ExitStep(
            kind="left",
            text=f"left in worktree: {unmeasured} unmeasured edit(s) — measure or discard by hand",
        )
    return None


async def _settle_iteration(
    context: SupervisedSession,
    iteration: IterationRecord,
    sinks: ExitSinks,
    *,
    experiment: str,
) -> ExitStep:
    """Keep an improved iteration, discard one that did not improve, or leave it."""
    reason = _gate_reason(context.root, iteration, experiment=experiment)
    if iteration.outcome == "improved":
        if reason is None:
            return await _keep_iteration(context, iteration, sinks, experiment=experiment)
        return ExitStep(
            kind="left",
            text=(
                f"left unsettled: iteration {iteration.seq} improved but {reason} — {_BY_HAND} "
                f"(git -C {experiment} diff first; keep commits the tree as it stands)"
            ),
        )
    if reason is not None:
        return ExitStep(
            kind="left",
            text=(
                f"left unsettled: iteration {iteration.seq} {iteration.outcome} "
                f"but {reason} — {_BY_HAND}"
            ),
        )
    return _discard(context, iteration, label=iteration.outcome)


async def _keep_iteration(
    context: SupervisedSession,
    iteration: IterationRecord,
    sinks: ExitSinks,
    *,
    experiment: str,
) -> ExitStep:
    """Commit the iteration the gate cleared, reporting what the keep made of it.

    A keep the loop blocked is written to the trace, so the command record the
    sequence leaves behind reads as the refusal it was rather than a clean run.

    Args:
        context: The supervised session whose repository is being settled.
        iteration: The improved iteration standing in the experiment worktree.
        sinks: Where the keep's warnings and its refusal go.
        experiment: The experiment worktree the keep commits.

    Returns:
        The step naming what the keep committed, or what it refused to.

    Raises:
        GymratError: When the keep was blocked for a reason the sequence has no
            wording for — every gate it can trip is decided before the call.
    """
    kept = await keep_session(
        context.root,
        context.config,
        KeepOptions(message=f"supervised: iteration {iteration.seq}", warn=sinks.warn),
    )
    record = kept.record
    if record.status == "committed":
        checks = "checks passed" if record.checks.configured else "checks not configured"
        return ExitStep(
            kind="settled",
            text=(
                f"settled: kept iteration {iteration.seq} ({checks})"
                f"{_rewritten_note(iteration, experiment=experiment)}"
            ),
        )
    if record.reason == "checks-failed":
        sinks.trace.gate = True
        sinks.trace.reason = record.reason
        return ExitStep(
            kind="left",
            text=(
                f"left unsettled: iteration {iteration.seq} improved but the checks failed "
                "— fix and keep, or discard, by hand"
            ),
        )
    if record.reason == "nothing-to-commit":
        return ExitStep(
            kind="settled", text=f"settled: iteration {iteration.seq} had nothing to commit"
        )
    message = f"Keep of iteration {iteration.seq} was blocked: {record.reason}."
    raise GymratError(message, hint="Keep or discard the iteration by hand.", reason=record.reason)


def _rewritten_note(iteration: IterationRecord, *, experiment: str) -> str:
    """What to add to a kept step when the commit carries a tree nobody measured.

    The checks command runs before the keep stages, so a formatter or a lockfile
    rewrite legitimately commits a tree that differs from the measured one,
    exactly as a manual keep does.

    Args:
        iteration: The iteration the keep committed.
        experiment: The experiment worktree the keep left at the kept commit.

    Returns:
        The note, or the empty string when the kept tree is the measured one.
    """
    kept_tree = worktree_fingerprint(Path(experiment))
    if kept_tree is None or kept_tree == iteration.measured_tree:
        return ""
    return "; tree changed during checks"


def _settle_gating_block(
    context: SupervisedSession, iteration: IterationRecord, *, experiment: str
) -> ExitStep:
    """Discard the edit a gating regression refused to commit, or leave it standing."""
    reason = _gate_reason(context.root, iteration, experiment=experiment)
    if reason is not None:
        return ExitStep(
            kind="left",
            text=(
                f"left in worktree: iteration {iteration.seq} blocked by a gating regression "
                f"but {reason} — discard by hand"
            ),
        )
    return _discard(context, iteration, label="gating regression")


def _discard(context: SupervisedSession, iteration: IterationRecord, *, label: str) -> ExitStep:
    """Discard the iteration's edits and report the settled step, labeled ``label``."""
    discard_session(context.root)
    return ExitStep(kind="settled", text=f"settled: discarded iteration {iteration.seq} ({label})")


def _gate_reason(root: str, iteration: IterationRecord, *, experiment: str) -> str | None:
    """Why the iteration may not be settled without a person, or ``None`` when it may.

    A hook that failed makes the verdict indefensible; a tree that no longer
    matches the fingerprint holds work nobody measured. Either way the sequence
    keeps its hands off and says which it was.

    Args:
        root: The repository the session was started in.
        iteration: The iteration the settle step would act on.
        experiment: The experiment worktree the iteration was measured in.

    Returns:
        The reason wording the step reads, or ``None`` when the gate passes.
    """
    for record in read_records(session_jsonl_path(root)):
        if (
            isinstance(record, HookRecord)
            and record.seq == iteration.seq
            and (record.timed_out or record.exit_code != 0)
        ):
            return f"{record.stage} hook failed"

    if iteration.measured_tree is None:
        return _NO_FINGERPRINT
    standing = worktree_fingerprint(Path(experiment))
    if standing is None:
        return _NO_FINGERPRINT
    return None if standing == iteration.measured_tree else _TREE_CHANGED
