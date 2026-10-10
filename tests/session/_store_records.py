"""Session-log records shared by the store tests and the fold-state tests."""

from gymrat.session.records import BaselineRecord, HookRecord, PairedSamples, SessionRecord
from gymrat.session.workspace import Worktrees
from gymrat.utils import SHORT_SHA_LENGTH
from tests.session.records._fixtures import (
    AT,
    COMMIT,
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

_KEPT_EXPERIMENT_SAMPLES = ({"total_ms": 14100}, {"total_ms": 14088})

# The baseline a keep appends from the samples the kept iteration already
# measured: labelled with the kept commit's short sha and timing nothing.
KEPT_BASELINE: BaselineRecord = baseline_record(
    at=AT + 5_000,
    label=COMMIT[:SHORT_SHA_LENGTH],
    samples=_KEPT_EXPERIMENT_SAMPLES,
)

HOOK: HookRecord = hook_record()


ITERATION_1 = iteration_record(
    seq=1,
    samples=PairedSamples(
        experiment=_KEPT_EXPERIMENT_SAMPLES,
        baseline=({"total_ms": 15200}, {"total_ms": 15190}),
    ),
    target_reached=False,
)

FINALIZE = finalize_record(branch=f"{SESSION.branch}-final")
