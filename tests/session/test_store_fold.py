"""Behavioral tests for ``fold_session``, the pure replay of session-log records into state.

Records are built in memory and folded directly; nothing touches the disk.
"""

from dataclasses import replace

import pytest

from gymrat.session.records import KeepChecks, KeepRecord, SessionLogRecord
from gymrat.session.store import (
    SessionState,
    fold_session,
)
from tests.session._store_records import (
    BASELINE,
    FINALIZE,
    HOOK,
    ITERATION_1,
    KEPT_BASELINE,
    SESSION,
)
from tests.session.records._fixtures import (
    COMMIT,
    blocked_keep,
    command_record,
    committed_keep,
    discard_record,
    empty_session_state,
    session_state,
    stop_record,
)

# ---------------------------------------------------------------------------
# Keep records
# ---------------------------------------------------------------------------


def _configured_block(seq: int, reason: str) -> KeepRecord:
    """A keep refused for ``reason``, numbered ``seq``, on a project with checks configured."""
    return blocked_keep(seq, reason=reason, checks=KeepChecks(configured=True))


def _gating_block(seq: int) -> KeepRecord:
    """The keep a gating regression refused, numbered with the iteration it refused."""
    return _configured_block(seq, "gating-regression")


def _nothing_measured_block(seq: int) -> KeepRecord:
    """The keep a retry refuses when nothing was measured since the last settle."""
    return _configured_block(seq, "nothing-measured")


# ---------------------------------------------------------------------------
# fold_session
# ---------------------------------------------------------------------------

ITERATION_1_ON_TARGET = ITERATION_1.model_copy(update={"target_reached": True})
ITERATION_2 = ITERATION_1.model_copy(update={"seq": 2})
ITERATION_2_ON_TARGET = ITERATION_2.model_copy(update={"target_reached": True})

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
            [SESSION, BASELINE, HOOK, ITERATION_1],
            _ITERATION_1_UNSETTLED,
            id="baseline-and-hook-around-a-measured-iteration",
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
            [SESSION, ITERATION_1, committed_keep(1), ITERATION_2_ON_TARGET],
            session_state(
                session=SESSION,
                iteration_count=2,
                last_iteration=ITERATION_2_ON_TARGET,
                unsettled=True,
                keep_count=1,
                last_seq=2,
                last_kept_commit=COMMIT,
            ),
            id="a-target-reaching-iteration-nobody-has-kept-yet",
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
            [SESSION, ITERATION_1, _nothing_measured_block(2)],
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
            [SESSION, ITERATION_1, _configured_block(1, "not-improved")],
            _ITERATION_1_UNSETTLED,
            id="a-keep-the-outcome-gate-refused",
        ),
        pytest.param(
            [SESSION, ITERATION_1, _gating_block(1)],
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
    assert fold_session(records) == expected


# ---------------------------------------------------------------------------
# fold_session — ends_on_gating_block
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("records", "expected"),
    [
        pytest.param(
            [SESSION, ITERATION_1, _gating_block(1), _nothing_measured_block(2)],
            True,
            id="a-retried-keep-that-refused-for-want-of-a-measurement",
        ),
        pytest.param(
            [
                SESSION,
                ITERATION_1,
                _gating_block(1),
                _nothing_measured_block(2),
                _nothing_measured_block(3),
            ],
            True,
            id="a-second-refusal-on-top-of-the-first",
        ),
        pytest.param(
            [SESSION, ITERATION_1, _gating_block(1), ITERATION_2],
            False,
            id="a-fresh-iteration-measured-after-the-block",
        ),
        pytest.param(
            [
                SESSION,
                ITERATION_1,
                _gating_block(1),
                _nothing_measured_block(2),
                ITERATION_2,
                committed_keep(2),
            ],
            False,
            id="a-keep-committed-after-a-refusal-and-a-fresh-measurement",
        ),
        pytest.param(
            [SESSION, ITERATION_1, _gating_block(1), discard_record(2)],
            False,
            id="a-discard-of-the-edit-the-block-refused",
        ),
    ],
)
def test_fold_session_when_records_replayed_does_report_ends_on_gating_block(
    records: list[SessionLogRecord], expected: bool
):
    assert fold_session(records).ends_on_gating_block is expected


# ---------------------------------------------------------------------------
# fold_session — ends_on_stop
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("records", "expected"),
    [
        pytest.param(
            [SESSION, ITERATION_1, committed_keep(1), stop_record()],
            True,
            id="a-stop-after-a-kept-iteration",
        ),
        pytest.param(
            [SESSION, ITERATION_1, committed_keep(1), stop_record(), BASELINE],
            True,
            id="a-stop-followed-only-by-a-baseline",
        ),
        pytest.param(
            [SESSION, ITERATION_1, committed_keep(1), stop_record(), HOOK],
            True,
            id="a-stop-followed-only-by-a-hook",
        ),
        pytest.param(
            [SESSION, ITERATION_1, committed_keep(1), stop_record(), BASELINE, HOOK],
            True,
            id="a-stop-followed-only-by-baseline-and-hook",
        ),
        pytest.param(
            [SESSION, ITERATION_1, committed_keep(1), stop_record(), _nothing_measured_block(2)],
            True,
            id="a-stop-followed-by-a-nothing-measured-refusal",
        ),
        pytest.param(
            [SESSION, ITERATION_1, committed_keep(1), stop_record(), ITERATION_2],
            False,
            id="a-stop-followed-by-an-iteration",
        ),
        pytest.param(
            [
                SESSION,
                ITERATION_1,
                committed_keep(1),
                stop_record(),
                ITERATION_2,
                committed_keep(2),
            ],
            False,
            id="a-stop-followed-by-an-iteration-and-keep",
        ),
        pytest.param(
            [
                SESSION,
                ITERATION_1,
                committed_keep(1),
                stop_record(),
                ITERATION_2,
                discard_record(2),
            ],
            False,
            id="a-stop-followed-by-an-iteration-and-discard",
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
    assert fold_session(records).ends_on_stop is expected


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


@pytest.mark.parametrize(
    ("base_records", "appended"),
    [
        pytest.param(
            [SESSION, ITERATION_1, committed_keep(1)],
            KEPT_BASELINE,
            id="a-baseline-a-keep-appended",
        ),
        pytest.param(
            [SESSION, ITERATION_1, committed_keep(1), stop_record()],
            command_record(),
            id="a-command-after-a-stop-preserves-ends-on-stop",
        ),
        pytest.param(
            [SESSION, ITERATION_1, _gating_block(1)],
            command_record(),
            id="a-command-after-a-gating-block-preserves-ends-on-gating-block",
        ),
        pytest.param(
            [SESSION, ITERATION_1, committed_keep(1), ITERATION_2],
            command_record(),
            id="a-command-after-an-unsettled-iteration-preserves-unsettled",
        ),
    ],
)
def test_fold_session_when_transparent_record_appended_does_leave_state_unchanged(
    base_records: list[SessionLogRecord], appended: SessionLogRecord
):
    expected = fold_session(base_records)

    state = fold_session([*base_records, appended])

    assert state == expected
