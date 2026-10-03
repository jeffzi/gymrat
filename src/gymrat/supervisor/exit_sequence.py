"""The exit sequence a supervised run ends on.

Once the agent stops, the session it worked in is still open: an iteration may be
unsettled, edits may stand in the experiment worktree, the session may be ready to
close. The sequence waits out any ``gymrat`` command still holding the repository
lock, takes the lock itself, and records one step per decision — the wording a
caller renders verbatim as its summary rows.

It never raises: one error boundary covers everything from the first lock probe
to the last decision — the wait, the skip the wait can end in, and the steps
taken under the lock — so a failure anywhere in it, including one raised by the
caller's own progress or event sink, ends the sequence with the message on the
report, next to the steps that completed before it.

Every step also lands on the supervisor event log as a ``FollowUpEvent``, so the
log replays the sequence in the order it was decided; a sequence that failed adds
one closing event naming the failure, after the events of the steps that
completed.

The decisions that read the session's own state are taken here too: whether an
unsettled iteration is kept, discarded, or left for a person, and whether the
session is finalized once nothing needs one. One gate guards every automatic keep
and every automatic discard: the iteration measured the tree that still stands,
and no hook of that iteration failed. Work that fails it is left exactly as it is,
named in the step so a person can pick it up; a run that settles nothing by hand
is worth less than one that never touched what it could not vouch for.
"""

from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from gymrat.clock import monotonic_ms, now_ns
from gymrat.command_run import with_repo_lock
from gymrat.errors import GymratError
from gymrat.eta import MS_PER_SECOND
from gymrat.git import SHORT_SHA_LENGTH
from gymrat.loop.discard import discard_session
from gymrat.loop.finalize import finalize_session
from gymrat.loop.keep import KeepOptions, keep_session
from gymrat.session.lock import LockContentionError, is_held, read_holder
from gymrat.session.paths import session_jsonl_path
from gymrat.session.records import HookRecord
from gymrat.session.store import fold_session, last_kept_position, read_records
from gymrat.session.workspace import changed_file_count, worktree_fingerprint
from gymrat.supervisor.events import FollowUpEvent

if TYPE_CHECKING:
    from collections.abc import Callable

    from gymrat.command_run import CommandTrace
    from gymrat.session.records import IterationRecord
    from gymrat.session.store import SessionState
    from gymrat.supervisor.context import SupervisedSession
    from gymrat.supervisor.events import SessionObserver
    from gymrat.supervisor.supervise import EndedBy
    from gymrat.warn import WarnSink

EXIT_LOCK_POLL_MS = 1000
"""How often the sequence re-probes a repository lock it is waiting out."""

_SKIP_TEXT = "exit sequence skipped: gymrat is still running (PID {pid})"

_CAP_NOTE = " — expected after a cap trip: the agent's last command outlives the cap"

_FAILED_TEXT = "exit sequence failed: {message}"

#: The run endings whose skip line closes on :data:`_CAP_NOTE`: a cap cuts the
#: agent off mid-command, so the command it was running is expected to outlive it
#: and hold the lock on the way out.
_CAP_ENDS = frozenset({"wall-clock", "spend-cap"})


#: What a tree that no longer matches the fingerprint the iteration measured
#: reads as. The interrupted discard is named because a ``discard`` killed
#: between the revert and its record leaves an unsettled iteration whose fresh
#: fingerprint is the kept tree, and the summary must not misattribute it.
_TREE_CHANGED = (
    "tree changed after measuring — an after hook, a later edit, or an interrupted discard"
)

_NO_FINGERPRINT = "fingerprint unavailable"

_BY_HAND = "keep or discard it by hand"


@dataclass(frozen=True, slots=True)
class ExitStep:
    """One decision the sequence took, as the caller reports it.

    Attributes:
        kind: What the decision did with the session.
        text: The summary wording, rendered verbatim.
    """

    kind: Literal["skipped", "settled", "left", "finalized", "refused", "nothing"]
    text: str


@dataclass(frozen=True, slots=True)
class ExitReport:
    """Everything the sequence decided, in decision order.

    Attributes:
        steps: The steps taken, one per decision. Never empty unless the sequence
            failed before deciding anything.
        error: The message of the failure that ended the sequence, absent when it
            ran to the end.
    """

    steps: tuple[ExitStep, ...]
    error: str | None = None


@dataclass(frozen=True, slots=True)
class ExitPhase:
    """Where the sequence currently is, for a caller rendering progress.

    Attributes:
        kind: The phase the sequence entered.
        pid: The process holding the repository lock while waiting on it, absent
            when its holder record cannot be read.
    """

    kind: Literal["waiting-lock", "settling"]
    pid: int | None


def _skip_step(lock_path: str, ended_by: EndedBy) -> ExitStep:
    """The single step a sequence that never got the lock reports."""
    holder = read_holder(lock_path)
    text = _SKIP_TEXT.format(pid="unknown" if holder is None else holder.pid)
    return ExitStep(kind="skipped", text=text + (_CAP_NOTE if ended_by in _CAP_ENDS else ""))


def _emit_failure(log: SessionObserver, message: str) -> None:
    """Emit the event closing a failed sequence, tolerating an observer that is down.

    Goes straight to ``log`` rather than through the step recorder, because a step
    would render as a second summary row next to the ``exit  error`` one the report
    already produces.

    Args:
        log: Observer the closing ``FollowUpEvent`` is emitted to.
        message: The failure the event's reason names.
    """
    # The emit is the last act of the error boundary: an observer that raises here
    # would turn the boundary back into the raise it exists to prevent.
    with contextlib.suppress(Exception):
        log(FollowUpEvent(at=now_ns(), action="ended", reason=_FAILED_TEXT.format(message=message)))


async def _held_past_wait(
    probe: Callable[[], bool],
    lock_path: str,
    progress: Callable[[ExitPhase], None],
    *,
    poll_ms: int,
    bound_ms: int,
) -> bool:
    """Wait out a held lock until it frees or the wait bound elapses.

    The bound is elapsed time since just before the first probe, never a count of
    polls, so a slow probe cannot stretch the wait past what the caller asked
    for. A bound of zero returns without sleeping.

    Args:
        probe: Reports whether the lock is still held.
        lock_path: The lock whose holder the waiting phase names, when its
            record can be read.
        progress: Called with the waiting phase once the first probe finds the
            lock held.
        poll_ms: How long to wait between probes.
        bound_ms: How long to keep probing, measured from the first probe.

    Returns:
        ``True`` when the lock was still held at the bound.
    """
    started_ms = monotonic_ms()
    if not probe():
        return False
    holder = read_holder(lock_path)
    progress(ExitPhase(kind="waiting-lock", pid=None if holder is None else holder.pid))
    while monotonic_ms() - started_ms < bound_ms:
        await asyncio.sleep(poll_ms / MS_PER_SECOND)
        if not probe():
            return False
    return True


async def run_exit_sequence(  # noqa: PLR0913 -- one parameter per exit knob
    context: SupervisedSession,
    *,
    ended_by: EndedBy,
    finalize: bool,
    progress: Callable[[ExitPhase], None],
    log: SessionObserver,
    warn: WarnSink,
    lock_poll_ms: int = EXIT_LOCK_POLL_MS,
    lock_wait_ms: int | None = None,
    is_lock_held: Callable[[], bool] | None = None,
) -> ExitReport:
    """Settle the session a supervised run leaves behind.

    Waits out a repository lock another ``gymrat`` command holds, then runs every
    decision under that lock, so the torn log tail is repaired and the sequence is
    recorded as a ``supervise`` command against ``context.root``. A lock still held
    at the wait bound — or taken again before the sequence could — skips the whole
    sequence rather than fighting the run that holds it.

    Nothing propagates to the caller. One error boundary covers the lock wait, the
    skip either the wait or the contention path can end in, and every step taken
    under the lock, so a failure anywhere there — including one raised by
    ``progress`` or ``log`` — ends the sequence with its message on the report and
    one closing ``FollowUpEvent`` naming it.

    Args:
        context: The supervised session to settle.
        ended_by: What ended the run; only the skip wording reads it.
        finalize: Whether the session may be closed once nothing needs a person.
        progress: Called as the sequence enters each phase.
        log: Observer every decision is emitted to as a ``FollowUpEvent``, plus one
            closing event when the sequence fails.
        warn: Where a keep's hint about a missing checks command goes, so it
            never reaches stderr while a dashboard is live.
        lock_poll_ms: How long to wait between probes of a held lock.
        lock_wait_ms: How long to wait for a held lock, measured from the first
            probe; ``None`` waits the config's ``timeout_seconds``.
        is_lock_held: Probe for the repository lock; ``None`` probes
            ``context.lock_path``.

    Returns:
        The steps the sequence took and the failure that ended it, if any.
    """
    probe = is_lock_held or partial(is_held, Path(context.lock_path))
    bound_ms = (
        context.config.timeout_seconds * MS_PER_SECOND if lock_wait_ms is None else lock_wait_ms
    )
    steps: list[ExitStep] = []

    def record(step: ExitStep) -> None:
        steps.append(step)
        log(FollowUpEvent(at=now_ns(), action="ended", reason=step.text))

    async def body(trace: CommandTrace) -> None:
        progress(ExitPhase(kind="settling", pid=None))
        state = fold_session(read_records(session_jsonl_path(context.root)))
        if state.finalized is not None:
            record(ExitStep(kind="nothing", text="session already finalized"))
            return
        # Each step is recorded the moment it is decided, so the steps that
        # completed before a later decision raises are still on the report.
        settled = await _decide_settle(context, state, warn=warn, trace=trace)
        if settled is not None:
            record(settled)
        closed = _decide_finalize(context, settled, finalize=finalize)
        if closed is not None:
            record(closed)
        if not steps:
            record(ExitStep(kind="nothing", text="nothing to settle"))

    try:
        if await _held_past_wait(
            probe, context.lock_path, progress, poll_ms=lock_poll_ms, bound_ms=bound_ms
        ):
            record(_skip_step(context.lock_path, ended_by))
        else:
            try:
                await with_repo_lock("supervise", body, args={"stage": "exit"}, root=context.root)
            except LockContentionError:
                record(_skip_step(context.lock_path, ended_by))
    except Exception as error:  # noqa: BLE001 -- the report is the caller's error channel
        message = str(error)
        _emit_failure(log, message)
        return ExitReport(steps=tuple(steps), error=message)
    return ExitReport(steps=tuple(steps))


def _decide_finalize(
    context: SupervisedSession, settled: ExitStep | None, *, finalize: bool
) -> ExitStep | None:
    """The finalize step the settled session calls for, if it calls for one.

    A session the finalize would refuse anyway — nothing kept, or something still
    unsettled once the settle step has run — calls for no step at all, so neither
    a refusal nor the ``--no-finalize`` note stands in for work that was never
    going to close.

    Args:
        context: The supervised session whose repository is being settled.
        settled: What the settle decision did, or ``None`` when it did nothing. A
            ``left`` step means work still needs a person, so no finalize may
            close over it.
        finalize: Whether the session may be closed once nothing needs a person.

    Returns:
        The step the session calls for, or ``None`` when it calls for none.
    """
    if settled is not None and settled.kind == "left":
        return None
    after = fold_session(read_records(session_jsonl_path(context.root)))
    if after.keep_count == 0 or after.unsettled:
        return None
    if not finalize:
        return ExitStep(kind="nothing", text="session left open (--no-finalize)")
    try:
        closed = finalize_session(context.root)
    except GymratError as refusal:
        return ExitStep(kind="refused", text=str(refusal))
    short_sha = closed.record.commit[:SHORT_SHA_LENGTH]
    return ExitStep(kind="finalized", text=f"finalized: {closed.record.branch} at {short_sha}")


async def _decide_settle(
    context: SupervisedSession, state: SessionState, *, warn: WarnSink, trace: CommandTrace
) -> ExitStep | None:
    """The one settle step the session calls for, if it calls for one.

    The iteration a step settles or leaves is the one the command record names,
    so its seq lands on the trace before the step itself is decided.

    Args:
        context: The supervised session whose repository is being settled.
        state: The folded session log the decision reads.
        warn: Where a keep's hint about a missing checks command goes.
        trace: The trace the repository-lock seam turns into the session log's
            command record once the sequence settles.

    Returns:
        The step the session calls for, or ``None`` when it calls for none.
    """
    session = state.session
    if session is None:
        return None

    experiment = session.worktrees.experiment
    iteration = state.last_iteration
    if iteration is not None and state.unsettled:
        trace.seq = iteration.seq
        return await _settle_iteration(
            context, iteration, experiment=experiment, warn=warn, trace=trace
        )
    if iteration is not None and state.ends_on_gating_block:
        trace.seq = iteration.seq
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
    *,
    experiment: str,
    warn: WarnSink,
    trace: CommandTrace,
) -> ExitStep:
    """Keep an improved iteration, discard one that did not improve, or leave it."""
    reason = _gate_reason(context.root, iteration, experiment=experiment)
    if iteration.outcome == "improved":
        if reason is None:
            return await _keep_iteration(
                context, iteration, experiment=experiment, warn=warn, trace=trace
            )
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
    *,
    experiment: str,
    warn: WarnSink,
    trace: CommandTrace,
) -> ExitStep:
    """Commit the iteration the gate cleared, reporting what the keep made of it.

    A keep the loop blocked is written to the trace, so the command record the
    sequence leaves behind reads as the refusal it was rather than a clean run.

    Args:
        context: The supervised session whose repository is being settled.
        iteration: The improved iteration standing in the experiment worktree.
        experiment: The experiment worktree the keep commits.
        warn: Where the keep's hint about a missing checks command goes.
        trace: The command trace the keep's refusal is written to.

    Returns:
        The step naming what the keep committed, or what it refused to.

    Raises:
        GymratError: When the keep was blocked for a reason the sequence has no
            wording for — every gate it can trip is decided before the call.
    """
    kept = await keep_session(
        context.root,
        context.config,
        KeepOptions(message=f"supervised: iteration {iteration.seq}", warn=warn),
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
        trace.gate = True
        trace.reason = record.reason
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

    if (
        iteration.measured_tree is None
        or (standing := worktree_fingerprint(Path(experiment))) is None
    ):
        return _NO_FINGERPRINT
    return None if standing == iteration.measured_tree else _TREE_CHANGED
