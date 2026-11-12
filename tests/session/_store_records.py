"""Session-log records shared by the store tests and the fold-state tests."""

from gymrat.session.records import BaselineRecord, HookRecord, PairedSamples, SessionRecord
from gymrat.session.workspace import Worktrees
from tests.session.records._fixtures import (
    AT,
    baseline_record,
    finalize_record,
    hook_record,
    iteration_record,
    session_record,
)

SESSION: SessionRecord = session_record(
    worktrees=Worktrees(experiment="/repo/.gymrat/experiment", baseline="/repo/.gymrat/baseline")
)

BASELINE: BaselineRecord = baseline_record(samples=({"total_ms": 15200}, {"total_ms": 15184}))

KEPT_BASELINE: BaselineRecord = baseline_record(at=AT + 5_000)

HOOK: HookRecord = hook_record()


ITERATION_1 = iteration_record(
    seq=1,
    samples=PairedSamples(
        experiment=({"total_ms": 14100}, {"total_ms": 14088}),
        baseline=({"total_ms": 15200}, {"total_ms": 15190}),
    ),
    target_reached=False,
)

FINALIZE = finalize_record(branch=f"{SESSION.branch}-final")
