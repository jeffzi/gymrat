"""Stop the current session: append a stop record without reverting or committing anything.

A stop tells the loop to pause: no iteration runs while the log ends on a stop,
but the session is still open, the worktree is untouched, and ``iterate``,
``keep``, ``discard``, and ``finalize`` all remain available.

Five states refuse a stop:

1. **No session** — ``require_open_session`` raises.
2. **Finalized** — ``require_open_session`` raises.
3. **Unsettled iteration** — the edit must be kept or discarded first.
4. **Standing gating block** — same: the blocked edit must be settled first.
5. **Already stopped** — the log already ends on a stop; iterate, keep, or
   discard to continue.
"""

from __future__ import annotations

from dataclasses import dataclass

from gymrat.errors import GymratError
from gymrat.report.loop import first_line
from gymrat.session.clock import now_iso
from gymrat.session.records import StopRecord
from gymrat.session.store import append_record, require_open_session

#: The hint a refusal points at whenever the fix is to settle the last iteration.
_SETTLE_FIRST_HINT = "Run gymrat keep or gymrat discard before stopping."


@dataclass(frozen=True, slots=True)
class StopResult:
    """What a stop produced: the log record and a human-readable report."""

    record: StopRecord
    report: str


def stop_session(root: str, message: str) -> StopResult:
    """Append a stop record to the open session's log.

    Raises:
        GymratError: When no session is open, when the session is finalized,
            when an iteration is unsettled, when a gating block stands, or
            when the log already ends on a stop.
    """
    required = require_open_session(root, "stopping the session")
    state = required.state

    if state.unsettled:
        msg = f"Iteration {state.last_seq} has not been settled"
        raise GymratError(
            msg,
            hint=_SETTLE_FIRST_HINT,
        )

    if state.ends_on_gating_block:
        msg = f"Iteration {state.last_seq} is blocked by a gating regression"
        raise GymratError(
            msg,
            hint=_SETTLE_FIRST_HINT,
        )

    if state.ends_on_stop:
        msg = "Already stopped"
        raise GymratError(
            msg,
            hint="Run gymrat iterate, keep, or discard to continue.",
        )

    at = now_iso()
    record = StopRecord(type="stop", at=at, message=message)
    append_record(required.jsonl_path, record)

    return StopResult(
        record=record,
        report=f"Stopped: {first_line(message)}",
    )
