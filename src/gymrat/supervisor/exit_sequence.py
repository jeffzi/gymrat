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
"""

from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from gymrat.cli.lock import with_repo_lock
from gymrat.clock import monotonic_ms, now_ns
from gymrat.session.lock import LockContentionError, is_held, read_holder
from gymrat.session.paths import session_jsonl_path
from gymrat.session.store import fold_session, read_records
from gymrat.supervisor.events import FollowUpEvent
from gymrat.supervisor.exit_settle import ExitSinks, ExitStep, decide_exit_steps

if TYPE_CHECKING:
    from collections.abc import Callable

    from gymrat.cli.lock import CommandTrace
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


def _lock_probe(lock_path: str) -> Callable[[], bool]:
    """A probe reporting whether another run holds the repository lock."""
    path = Path(lock_path)

    def probe() -> bool:
        return is_held(path)

    return probe


def _waiting_phase(lock_path: str) -> ExitPhase:
    """The waiting-lock phase, naming the holder when its record can be read."""
    holder = read_holder(lock_path)
    return ExitPhase(kind="waiting-lock", pid=None if holder is None else holder.pid)


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


async def _still_held(
    probe: Callable[[], bool], *, started_ms: float, poll_ms: int, bound_ms: int
) -> bool:
    """Poll a held lock until it frees or the wait bound elapses.

    The bound is elapsed time since ``started_ms``, never a count of polls, so a
    slow probe cannot stretch the wait past what the caller asked for. A bound of
    zero returns without sleeping.

    Args:
        probe: Reports whether the lock is still held.
        started_ms: Monotonic reading from just before the first probe.
        poll_ms: How long to wait between probes.
        bound_ms: How long to keep probing, measured from ``started_ms``.

    Returns:
        ``True`` when the lock was still held at the bound.
    """
    while monotonic_ms() - started_ms < bound_ms:
        await asyncio.sleep(poll_ms / 1000)
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
    probe = _lock_probe(context.lock_path) if is_lock_held is None else is_lock_held
    bound_ms = context.config.timeout_seconds * 1000 if lock_wait_ms is None else lock_wait_ms
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
        sinks = ExitSinks(record=record, warn=warn, trace=trace)
        await decide_exit_steps(context, state, sinks, finalize=finalize)
        if not steps:
            record(ExitStep(kind="nothing", text="nothing to settle"))

    started_ms = monotonic_ms()
    try:
        held = probe()
        if held:
            progress(_waiting_phase(context.lock_path))
            held = await _still_held(
                probe, started_ms=started_ms, poll_ms=lock_poll_ms, bound_ms=bound_ms
            )
        if held:
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
