"""Session-log records shared by the store, fold-state and loop-status tests.

Imported by tests/session/test_store.py, tests/session/test_store_fold.py and
tests/loop/test_status.py.
"""

from gymrat.session.records import BaselineRecord, SessionRecord
from tests.session.records._fixtures import (
    AT,
    baseline_record,
    finalize_record,
    iteration_record,
    session_record,
)

SESSION: SessionRecord = session_record()

BASELINE: BaselineRecord = baseline_record(samples=({"total_ms": 15200}, {"total_ms": 15184}))

KEPT_BASELINE: BaselineRecord = baseline_record(at=AT + 5_000)

ITERATION_1 = iteration_record()

FINALIZE = finalize_record()
