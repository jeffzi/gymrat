"""Behavioral tests for ``fold_session``, the pure replay of session-log records into state.

Records are built in memory and folded directly; nothing touches the disk.
"""

from dataclasses import replace

import pytest

from gymrat.session.records import SessionLogRecord
from gymrat.session.store import (
    SessionState,
    fold_session,
)
from tests.session._store_records import (
    FINALIZE,
    ITERATION_1,
    SESSION,
)
from tests.session.records._fixtures import (
    COMMIT,
    blocked_keep,
    command_record,
    committed_keep,
    discard_record,
    empty_session_state,
    gate_block,
    session_state,
    stop_record,
)

# ---------------------------------------------------------------------------
# fold_session
# ---------------------------------------------------------------------------

ITERATION_1_ON_TARGET = ITERATION_1.model_copy(update={"target_reached": True})
ITERATION_2 = ITERATION_1.model_copy(update={"seq": 2})

_ITERATION_1_UNSETTLED = session_state(
    session=SESSION,
    iteration_count=1,
    last_iteration=ITERATION_1,
    unsettled=True,
    last_seq=1,
)


@pytest.mark.parametrize(
    ("records", "expected"),
    [
        pytest.param([], empty_session_state(), id="an-empty-log"),
        pytest.param(
            [SESSION], session_state(session=SESSION), id="a-session-with-nothing-measured"
        ),
        pytest.param(
            [SESSION, ITERATION_1],
            _ITERATION_1_UNSETTLED,
            id="a-measured-iteration-nobody-has-settled",
        ),
        pytest.param(
            [SESSION, ITERATION_1, committed_keep(1)],
            session_state(
                session=SESSION,
                iteration_count=1,
                last_iteration=ITERATION_1,
                keep_count=1,
                last_seq=1,
                last_kept_commit=COMMIT,
            ),
            id="an-iteration-settled-by-a-keep",
        ),
        pytest.param(
            [SESSION, ITERATION_1, discard_record(1)],
            session_state(
                session=SESSION,
                iteration_count=1,
                last_iteration=ITERATION_1,
                discard_count=1,
                last_seq=1,
            ),
            id="an-iteration-settled-by-a-discard",
        ),
        pytest.param(
            [SESSION, ITERATION_1, committed_keep(1), ITERATION_2],
            session_state(
                session=SESSION,
                iteration_count=2,
                last_iteration=ITERATION_2,
                unsettled=True,
                keep_count=1,
                last_seq=2,
                last_kept_commit=COMMIT,
            ),
            id="a-fresh-iteration-after-a-settled-one",
        ),
        pytest.param(
            [SESSION, ITERATION_1_ON_TARGET, committed_keep(1)],
            session_state(
                session=SESSION,
                iteration_count=1,
                last_iteration=ITERATION_1_ON_TARGET,
                keep_count=1,
                target_reached_and_kept=True,
                last_seq=1,
                last_kept_commit=COMMIT,
            ),
            id="a-target-reaching-iteration-that-was-kept",
        ),
        pytest.param(
            [SESSION, ITERATION_1_ON_TARGET, discard_record(1)],
            session_state(
                session=SESSION,
                iteration_count=1,
                last_iteration=ITERATION_1_ON_TARGET,
                discard_count=1,
                last_seq=1,
            ),
            id="a-target-reaching-iteration-that-was-discarded",
        ),
        pytest.param(
            [SESSION, ITERATION_1_ON_TARGET, committed_keep(1), ITERATION_2, discard_record(2)],
            session_state(
                session=SESSION,
                iteration_count=2,
                last_iteration=ITERATION_2,
                keep_count=1,
                discard_count=1,
                target_reached_and_kept=True,
                last_seq=2,
                last_kept_commit=COMMIT,
            ),
            id="a-kept-target-followed-by-a-discarded-iteration",
        ),
        pytest.param(
            [SESSION, ITERATION_1, committed_keep(1), FINALIZE],
            session_state(
                session=SESSION,
                iteration_count=1,
                last_iteration=ITERATION_1,
                keep_count=1,
                last_seq=1,
                last_kept_commit=COMMIT,
                finalized=FINALIZE,
            ),
            id="a-session-closed-by-a-finalize",
        ),
        pytest.param(
            [SESSION, ITERATION_1, blocked_keep(1)],
            _ITERATION_1_UNSETTLED,
            id="a-keep-refused-because-checks-failed",
        ),
        pytest.param(
            [SESSION, ITERATION_1, blocked_keep(1, reason=None)],
            _ITERATION_1_UNSETTLED,
            id="a-keep-refused-for-no-stated-reason",
        ),
        pytest.param(
            [
                SESSION,
                ITERATION_1,
                gate_block(2, "nothing-measured"),
            ],
            session_state(
                session=SESSION,
                iteration_count=1,
                last_iteration=ITERATION_1,
                unsettled=True,
                last_seq=2,
            ),
            id="a-keep-refused-for-want-of-a-measurement",
        ),
        pytest.param(
            [
                SESSION,
                ITERATION_1,
                gate_block(1, "gating-regression"),
            ],
            session_state(
                session=SESSION,
                iteration_count=1,
                last_iteration=ITERATION_1,
                last_seq=1,
                ends_on_gating_block=True,
            ),
            id="a-keep-a-gating-regression-refused",
        ),
    ],
)
def test_fold_session_when_records_replayed_does_produce_the_summarized_state(
    records: list[SessionLogRecord], expected: SessionState
):
    state = fold_session(records)

    assert state == expected


# ---------------------------------------------------------------------------
# fold_session — ends_on_gating_block
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("records", "expected"),
    [
        pytest.param(
            [
                SESSION,
                ITERATION_1,
                gate_block(1, "gating-regression"),
                gate_block(2, "nothing-measured"),
            ],
            True,
            id="a-retried-keep-that-refused-for-want-of-a-measurement",
        ),
        pytest.param(
            [
                SESSION,
                ITERATION_1,
                gate_block(1, "gating-regression"),
                ITERATION_2,
            ],
            False,
            id="a-fresh-iteration-measured-after-the-block",
        ),
        pytest.param(
            [
                SESSION,
                ITERATION_1,
                gate_block(1, "gating-regression"),
                discard_record(2),
            ],
            False,
            id="a-discard-of-the-edit-the-block-refused",
        ),
    ],
)
def test_fold_session_when_records_replayed_does_report_ends_on_gating_block(
    records: list[SessionLogRecord], expected: bool
):
    state = fold_session(records)

    assert state.ends_on_gating_block is expected


# ---------------------------------------------------------------------------
# fold_session — ends_on_stop
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("records", "expected"),
    [
        pytest.param(
            [
                SESSION,
                ITERATION_1,
                committed_keep(1),
                stop_record(),
                gate_block(2, "nothing-measured"),
            ],
            True,
            id="a-stop-followed-by-a-nothing-measured-refusal",
        ),
        pytest.param(
            [SESSION, ITERATION_1, committed_keep(1), stop_record(), ITERATION_2],
            False,
            id="a-stop-followed-by-an-iteration",
        ),
        pytest.param(
            [SESSION, ITERATION_1, committed_keep(1), stop_record(), FINALIZE],
            False,
            id="a-stop-followed-by-a-finalize",
        ),
    ],
)
def test_fold_session_when_records_replayed_does_report_ends_on_stop(
    records: list[SessionLogRecord], expected: bool
):
    state = fold_session(records)

    assert state.ends_on_stop is expected


# ---------------------------------------------------------------------------
# fold_session — stop record changes nothing else
# ---------------------------------------------------------------------------


def test_fold_session_when_stop_appended_does_change_only_ends_on_stop():
    stop = stop_record()
    before = fold_session([SESSION, ITERATION_1, committed_keep(1), ITERATION_2])

    after = fold_session([SESSION, ITERATION_1, committed_keep(1), ITERATION_2, stop])

    assert after == replace(before, ends_on_stop=True)


# ---------------------------------------------------------------------------
# fold_session — transparent records
# ---------------------------------------------------------------------------


def test_fold_session_when_transparent_record_appended_does_leave_state_unchanged():
    base_records: list[SessionLogRecord] = [SESSION, ITERATION_1, committed_keep(1), stop_record()]
    expected = fold_session(base_records)

    state = fold_session([*base_records, command_record()])

    assert state == expected
