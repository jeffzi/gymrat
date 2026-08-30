"""Discard a measured edit: revert the experiment worktree and record that it went."""

from __future__ import annotations

from dataclasses import dataclass

from gymrat.errors import GymratError
from gymrat.report.loop import SHORT_SHA_LENGTH
from gymrat.session.clock import now_iso
from gymrat.session.records import DiscardRecord
from gymrat.session.store import append_record, last_kept_position, require_open_session
from gymrat.session.workspace import revert_workspace


@dataclass(frozen=True, slots=True)
class DiscardResult:
    """One reverted iteration: what was logged, and what to print about it."""

    record: DiscardRecord
    report: str


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

    target = last_kept_position(state, session.baseline.sha)
    revert_workspace(session.worktrees.experiment, target=target)

    reverted_seq = state.last_iteration.seq if state.last_iteration is not None else state.last_seq

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
            f"Discarded iteration {reverted_seq}: the experiment worktree is back at "
            f"{target[:SHORT_SHA_LENGTH]}"
        ),
    )
