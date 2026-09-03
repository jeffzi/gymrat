"""Discard an edit: revert the experiment worktree, recording the action when measured.

Three states accept a discard:

1. **Unsettled iteration** — a measured iteration no keep or discard has settled yet.
   The worktree is reverted, a ``discard`` record is appended, and the result
   carries that record.

2. **Standing gating block** — a gating regression refused the keep, but the edit is
   still in the worktree.  Same as (1), with the discard numbered past the block.

3. **Unmeasured edit** — nothing has been measured since the last settle (or since the
   session started), but the worktree is dirty.  The worktree is reverted silently:
   no record is appended, because there is no iteration to settle, and the log
   should not accumulate bookkeeping entries that carry no measurement.  The result
   carries ``record=None``.

A clean worktree with nothing measured is refused — there is genuinely nothing to
throw away.
"""

from __future__ import annotations

from dataclasses import dataclass

from gymrat.errors import GymratError
from gymrat.git import try_git
from gymrat.plural import pluralize
from gymrat.report.loop import SHORT_SHA_LENGTH
from gymrat.session.clock import now_iso
from gymrat.session.records import DiscardRecord, SessionRecord
from gymrat.session.store import (
    SessionState,
    append_record,
    last_kept_position,
    require_open_session,
)
from gymrat.session.workspace import (
    changed_file_count,
    dirty_file_count,
    revert_workspace,
    worktree_head,
)


@dataclass(frozen=True, slots=True)
class DiscardResult:
    """What a discard did: an optional log record, a timestamp, and a human report.

    ``at`` is the instant the discard happened. When a record is present, it
    equals the record's timestamp; on the unmeasured path the caller supplies it
    directly.
    """

    record: DiscardRecord | None
    report: str
    at: str


def discard_session(root: str, expected_session_id: str | None = None) -> DiscardResult:
    """Revert the experiment worktree, recording the action only when an iteration is settled.

    See the module docstring for the three accepted states and when a record is
    appended.

    Raises:
        GymratError: When no session has been started, when neither a measured
            iteration nor a dirty worktree gives something to throw away, or when
            git refuses to revert the worktree.
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

    has_measured = state.unsettled or state.ends_on_gating_block

    if not has_measured:
        return _discard_unmeasured(session, state)

    target = last_kept_position(state, session.baseline.sha)
    revert_workspace(session.worktrees.experiment, target=target)

    reverted_seq = state.last_iteration.seq if state.last_iteration is not None else state.last_seq

    at = now_iso()
    record = DiscardRecord(
        type="discard",
        # The block already settled the iteration it refused, so the discard behind
        # it takes the number no iteration has used yet — the same number a refused
        # keep takes. Reusing the iteration's own seq would make the discard the last
        # settling record to carry it, and ``gymrat status`` would render it in place
        # of the block instead of alongside it.
        seq=state.last_seq + 1 if state.ends_on_gating_block else state.last_seq,
        at=at,
    )
    append_record(jsonl_path, record)

    return DiscardResult(
        record=record,
        at=at,
        report=(
            f"Discarded iteration {reverted_seq}: the experiment worktree is back at "
            f"{target[:SHORT_SHA_LENGTH]}"
        ),
    )


def _discard_unmeasured(session: SessionRecord, state: SessionState) -> DiscardResult:
    """Revert a dirty experiment worktree that has no measured iteration to settle."""
    experiment = session.worktrees.experiment
    if dirty_file_count(experiment) == 0:
        nothing_message = (
            "Discard refused: nothing has been measured since the last keep or discard."
        )
        raise GymratError(
            nothing_message,
            hint="Run gymrat iterate to measure an edit before settling it.",
        )

    target = _revert_target(state, session.baseline.sha, experiment)
    n_changed = changed_file_count(experiment, target)
    revert_workspace(experiment, target=target)

    return DiscardResult(
        record=None,
        at=now_iso(),
        report=(
            f"Reverted {pluralize(n_changed, 'unmeasured edit')}: "
            f"the experiment worktree is back at {target[:SHORT_SHA_LENGTH]}"
        ),
    )


def _revert_target(
    state: SessionState,
    baseline_sha: str,
    experiment_dir: str,
) -> str:
    """The commit the unmeasured revert should land on.

    Normally this is the last kept commit (or the baseline SHA if nothing was
    kept). When the kept commit is no longer reachable — a corruption edge case
    that should not happen in practice — the current HEAD is already the best
    the worktree can offer.
    """
    target = last_kept_position(state, baseline_sha)
    unreachable = try_git(["cat-file", "-t", target], experiment_dir) is not None
    if unreachable:
        return worktree_head(experiment_dir)
    return target
