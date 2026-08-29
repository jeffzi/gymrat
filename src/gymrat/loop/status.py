"""What ``gymrat status`` reports: the session, rebuilt from its log and nothing else.

Every other loop command writes to the log; this one only reads it, which is why
it takes no repository lock. A status that agreed with the log only while some
process kept state in memory would be worth nothing to an agent whose every
command is a fresh process.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from gymrat.report.loop import (
    SettleDiscarded,
    SettleKeepBlocked,
    SettleKept,
    SettleUnsettled,
    StatusIteration,
    StatusSummary,
    format_status_baseline,
    format_status_finalized,
    format_status_footer,
    format_status_header,
    format_status_iteration,
    format_status_settle,
)
from gymrat.report.style import RENDER_WIDTH, render_lines
from gymrat.session import (
    BaselineRecord,
    DiscardRecord,
    IterationRecord,
    KeepRecord,
    require_session,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from gymrat.config import BenchlessConfig
    from gymrat.report.loop import SettleState
    from gymrat.session import SessionLogRecord


def _settle_state_of(record: KeepRecord | DiscardRecord) -> SettleState:
    """What a single settling record says became of the iteration it settles."""
    if isinstance(record, DiscardRecord):
        return SettleDiscarded()
    if record.status == "committed":
        return SettleKept(commit=record.commit)
    return SettleKeepBlocked(reason=record.reason)


@dataclass(frozen=True, slots=True)
class _PendingIteration:
    """The iteration a settling record is still waiting to hear about."""

    position: int
    seq: int


@dataclass(frozen=True, slots=True)
class _LastGatingBlock:
    """The last keep-blocked that settled the pending iteration.

    A discard that supersedes it can relocate the block to a standalone line and
    take the iteration's display for itself.
    """

    iteration_position: int
    block_position: int
    block_state: SettleState


def _apply_settle_record(
    states: dict[int, SettleState],
    pending: _PendingIteration | None,
    last_block: _LastGatingBlock | None,
    position: int,
    record: KeepRecord | DiscardRecord,
) -> _LastGatingBlock | None:
    """Record ``record``'s settle state and report the gating block still standing afterwards.

    A record settling the pending iteration writes there directly, and starts
    tracking the block if it is one. When that record supersedes a block already
    standing on the same iteration — a keep the checks blocked, later settled by
    another keep at the same seq — the earlier block is relocated to its own line
    first, the same way a superseding discard relocates it below.

    A discard with no pending match instead looks for a standing block: that is a
    discard that follows a gating block, which supersedes the block's claim on the
    iteration it refused — the block moves to its own position and the iteration
    shows as discarded. Anything else settles no iteration and keeps its own line.
    """
    settle = _settle_state_of(record)

    if pending is not None and pending.seq == record.seq:
        if last_block is not None and last_block.iteration_position == pending.position:
            states[last_block.block_position] = last_block.block_state
        states[pending.position] = settle
        if isinstance(settle, SettleKeepBlocked):
            return _LastGatingBlock(
                iteration_position=pending.position,
                block_position=position,
                block_state=settle,
            )
        return None

    if isinstance(record, DiscardRecord) and last_block is not None:
        states[last_block.iteration_position] = settle
        states[last_block.block_position] = last_block.block_state
        return None

    states[position] = settle
    return last_block


def _settle_states(records: Sequence[SessionLogRecord]) -> dict[int, SettleState]:
    """The settle state each record's own line states, keyed by that record's position in the log.

    A settling record settles the iteration it follows, not merely the one it
    shares a number with. ``keep`` numbers a refusal that measured nothing past
    every iteration on file, so the log can hold a blocked keep bearing the number
    an iteration was minted with only afterwards; matching on the number alone
    would hand that stale refusal to the new iteration and call it settled.

    The last settling record wins: an iteration whose keep the checks blocked, and
    which a later keep committed, is a kept iteration — the blocked attempt is
    history the log still holds, not the state the iteration rests in.

    A record that settled no iteration keeps an entry under its own position, and
    so a line of its own: a refused keep is a decision the log reads back, never
    one it erases.
    """
    states: dict[int, SettleState] = {}
    pending: _PendingIteration | None = None
    last_block: _LastGatingBlock | None = None

    for position, record in enumerate(records):
        if isinstance(record, IterationRecord):
            pending = _PendingIteration(position=position, seq=record.seq)
            last_block = None
        elif isinstance(record, (KeepRecord, DiscardRecord)):
            last_block = _apply_settle_record(states, pending, last_block, position, record)
    return states


def status_session(root: str, config: BenchlessConfig, *, color: bool | None = None) -> str:
    """The session's whole history, as the agent reads it back.

    The records are walked in file order rather than summarized, so the report
    tells the session's story in the order it happened; the folded state supplies
    only the totals. Hook records are left out — a hook is machinery around an
    iteration, not a step of the loop the agent decides on.

    The stop condition and the runbook come from live configuration rather than
    the session record's snapshot, matching every other loop command: an agent
    that raised its iteration budget mid-session must see the budget it now runs
    under.

    Args:
        root: The repository whose session is reported.
        config: The live run configuration the stop line and runbook read from.
        color: Explicit color choice — ``True`` forces ANSI, ``False``
            suppresses it, ``None`` defers to the environment and TTY.

    Returns:
        The report as a single rendered string.

    Raises:
        GymratError: When no session has been started, or when the log is corrupt
            — every parse failure names the log and the line at fault.
    """
    required = require_session(root, "asking for its status")
    session, state, records = required.session, required.state, required.records

    settled = _settle_states(records)
    history: list[str] = []
    for position, record in enumerate(records):
        if isinstance(record, BaselineRecord):
            history.append(format_status_baseline(record))
        elif isinstance(record, IterationRecord):
            settle = settled.get(position)
            history.append(
                format_status_iteration(
                    StatusIteration(
                        seq=record.seq,
                        delta_pct=record.primary.delta_pct,
                        outcome=record.outcome,
                        settle=settle if settle is not None else SettleUnsettled(),
                    )
                )
            )
        elif isinstance(record, (KeepRecord, DiscardRecord)):
            settle = settled.get(position)
            if settle is not None:
                history.append(format_status_settle(settle))

    lines: list[str] = list(format_status_header(session))
    if config.runbook is not None:
        lines.append(f"runbook {config.runbook}")
    lines.extend(history)
    lines.extend(
        format_status_footer(
            StatusSummary(
                iteration_count=state.iteration_count,
                keep_count=state.keep_count,
                discard_count=state.discard_count,
                target_reached=state.target_reached_and_kept,
                stop=config.stop,
            )
        )
    )
    if state.finalized is not None:
        lines.append(format_status_finalized(state.finalized))

    return render_lines(*lines, color=color, width=RENDER_WIDTH)
