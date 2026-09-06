"""Session-log readers for the supervise dashboard.

Folds the live session log into the summary the progress reporter displays.
"""

from __future__ import annotations

import operator
from typing import TYPE_CHECKING

from gymrat.cli.supervise.state import ReadSessionResult
from gymrat.session.paths import session_jsonl_path
from gymrat.session.records import (
    BaselineRecord,
    IterationRecord,
    KeepRecord,
    SessionRecord,
    StopRecord,
)
from gymrat.session.store import fold_session, read_records

if TYPE_CHECKING:
    from collections.abc import Callable

    from gymrat.session.records import SessionLogRecord


def _find_best_kept_iteration(
    records: list[SessionLogRecord], committed_seqs: set[int]
) -> tuple[float | None, int | None, str | None]:
    """Return ``(delta_pct, seq, primary_label)`` for the best committed keep."""
    candidates = [
        (r, delta)
        for r in records
        if isinstance(r, IterationRecord)
        and r.seq in committed_seqs
        and (delta := r.primary.delta_pct) is not None
    ]
    if not candidates:
        return None, None, None

    best, best_delta = min(candidates, key=operator.itemgetter(1))
    label = best.primary.kind if best.primary.kind != "single" else best.primary.name
    return best_delta, best.seq, label


def _find_baseline_sha(records: list[SessionLogRecord]) -> str | None:
    return next((r.baseline.sha for r in records if isinstance(r, SessionRecord)), None)


def _find_stop_message(records: list[SessionLogRecord]) -> str | None:
    """Return the newest stop record's message, or ``None`` if there is none."""
    return next((r.message for r in reversed(records) if isinstance(r, StopRecord)), None)


def make_default_read(root: str) -> Callable[[], ReadSessionResult]:
    """Build a session-reader closure that folds the live session log at ``root``."""

    def _read() -> ReadSessionResult:
        records = read_records(session_jsonl_path(root))
        state = fold_session(records)
        has_baseline = any(isinstance(r, BaselineRecord) for r in records)

        committed_seqs = {
            r.seq for r in records if isinstance(r, KeepRecord) and r.status == "committed"
        }
        best_delta_pct, best_seq, primary_label = _find_best_kept_iteration(records, committed_seqs)

        return ReadSessionResult(
            state=state,
            has_baseline=has_baseline,
            best_delta_pct=best_delta_pct,
            best_seq=best_seq,
            primary_label=primary_label,
            baseline_sha=_find_baseline_sha(records),
            stop_message=_find_stop_message(records) if state.ends_on_stop else None,
        )

    return _read
