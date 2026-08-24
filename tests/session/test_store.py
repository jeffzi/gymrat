"""Behavioral tests for the session JSONL store and its fold state machine.

Records are written and read through real files in a throwaway temp root, so
the suite is order-independent and safe under ``pytest-xdist`` /
``pytest-randomly``. Nothing is mocked: the module under test is file I/O plus a
pure fold, and only real bytes on disk reveal the torn-tail recovery and the
refusal to log a record that would not read back.
"""

import json
import re
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from gymrat_py.errors import GymratError, hint_of, message_of
from gymrat_py.session import (
    BaselineRecord,
    HookRecord,
    IterationRecord,
    KeepChecks,
    KeepRecord,
    PairedSamples,
    SessionLogRecord,
    SessionRecord,
    record_to_wire,
    session_jsonl_path,
)
from gymrat_py.session.store import (
    RequiredSession,
    SessionState,
    append_record,
    fold_session,
    read_records,
    require_open_session,
    require_session,
)
from tests.session._records import (
    AT,
    COMMIT,
    Worktrees,
    blocked_keep,
    committed_keep,
    discard_record,
    finalize_record,
    hook_record,
    iteration_record,
    session_record,
    write_session_log,
)

# ---------------------------------------------------------------------------
# Fixture records
# ---------------------------------------------------------------------------

SESSION: SessionRecord = session_record(
    worktrees=Worktrees(experiment="/repo/.gymrat/experiment", baseline="/repo/.gymrat/baseline")
)

BASELINE: BaselineRecord = BaselineRecord(
    type="baseline",
    at=AT,
    label="main",
    samples=({"total_ms": 15200}, {"total_ms": 15184}),
)

HOOK: HookRecord = hook_record()


def _iteration(seq: int, *, target_reached: bool) -> IterationRecord:
    """An iteration record numbered ``seq``, reaching the target metric or not."""
    return iteration_record(
        seq=seq,
        samples=PairedSamples(
            experiment=({"total_ms": 14100}, {"total_ms": 14088}),
            baseline=({"total_ms": 15200}, {"total_ms": 15190}),
        ),
        target_reached=target_reached,
    )


def _gating_block(seq: int) -> KeepRecord:
    """The keep a gating regression refused, numbered with the iteration it refused."""
    return blocked_keep(seq, reason="gating-regression", checks=KeepChecks(configured=True))


def _nothing_measured_block(seq: int) -> KeepRecord:
    """The keep a retry refuses when nothing was measured since the last settle."""
    return blocked_keep(seq, reason="nothing-measured", checks=KeepChecks(configured=True))


ITERATION_1 = _iteration(1, target_reached=False)
ITERATION_1_ON_TARGET = _iteration(1, target_reached=True)
ITERATION_2 = _iteration(2, target_reached=False)
ITERATION_2_ON_TARGET = _iteration(2, target_reached=True)

# An iteration a NaN sample makes unreadable: it serializes to a line with a
# `null` sample that no schema accepts on read-back.
UNREADABLE_ITERATION: IterationRecord = iteration_record(
    samples=PairedSamples(experiment=({"total_ms": float("nan")},), baseline=({"total_ms": 15200},))
)

FINALIZE = finalize_record(branch=f"{SESSION.branch}-final")

EMPTY_STATE = SessionState(
    session=None,
    iteration_count=0,
    last_iteration=None,
    unsettled=False,
    keep_count=0,
    discard_count=0,
    target_reached_and_kept=False,
    last_seq=0,
    last_kept_commit=None,
    ends_on_gating_block=False,
    finalized=None,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def fresh_root(tmp_path: Path) -> str:
    """A fresh temp repo root with no ``.gymrat`` directory yet."""
    return str(tmp_path)


def _line(record: SessionLogRecord) -> str:
    """The JSON line the store writes for ``record``."""
    return json.dumps(record_to_wire(record))


def _jsonl_holding(root: str, lines: list[str]) -> str:
    """Write a session log holding exactly ``lines``, each on its own line."""
    jsonl_path = session_jsonl_path(root)
    Path(jsonl_path).parent.mkdir(parents=True, exist_ok=True)
    Path(jsonl_path).write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")
    return jsonl_path


# ---------------------------------------------------------------------------
# append_record
# ---------------------------------------------------------------------------


def test_append_record_when_directory_absent_does_create_it_and_write_one_line(fresh_root: str):
    jsonl_path = session_jsonl_path(fresh_root)

    append_record(jsonl_path, SESSION)

    content = Path(jsonl_path).read_text(encoding="utf-8")
    assert Path(jsonl_path).parent.is_dir()
    assert content.endswith("\n")
    assert content.count("\n") == 1
    assert read_records(jsonl_path) == [SESSION]


def test_append_record_when_log_holds_records_does_add_one_line_leaving_earlier_intact(
    fresh_root: str,
):
    jsonl_path = session_jsonl_path(fresh_root)
    append_record(jsonl_path, SESSION)
    first = Path(jsonl_path).read_text(encoding="utf-8")

    append_record(jsonl_path, ITERATION_1)

    content = Path(jsonl_path).read_text(encoding="utf-8")
    assert content.startswith(first)
    assert content.count("\n") == 2
    assert read_records(jsonl_path) == [SESSION, ITERATION_1]


def test_append_record_when_final_line_torn_does_truncate_it_and_append_cleanly(fresh_root: str):
    jsonl_path = session_jsonl_path(fresh_root)
    append_record(jsonl_path, SESSION)
    with Path(jsonl_path).open("a", encoding="utf-8") as handle:
        handle.write('{"type":"iter')

    append_record(jsonl_path, ITERATION_1)

    assert read_records(jsonl_path) == [SESSION, ITERATION_1]


def test_append_record_when_only_line_torn_does_recover(fresh_root: str):
    jsonl_path = session_jsonl_path(fresh_root)
    Path(jsonl_path).parent.mkdir(parents=True, exist_ok=True)
    Path(jsonl_path).write_text('{"type":"ses', encoding="utf-8")

    append_record(jsonl_path, SESSION)

    assert read_records(jsonl_path) == [SESSION]


def test_append_record_when_record_unreadable_does_raise_and_leave_the_log_byte_identical(
    fresh_root: str,
):
    jsonl_path = session_jsonl_path(fresh_root)
    append_record(jsonl_path, SESSION)
    before = Path(jsonl_path).read_bytes()

    with pytest.raises(GymratError) as excinfo:
        append_record(jsonl_path, UNREADABLE_ITERATION)

    assert re.search(r"\biteration\b", message_of(excinfo.value))
    assert Path(jsonl_path).read_bytes() == before


# ---------------------------------------------------------------------------
# read_records
# ---------------------------------------------------------------------------


def test_read_records_when_log_missing_does_read_as_no_session(fresh_root: str):
    jsonl_path = session_jsonl_path(fresh_root)

    assert read_records(jsonl_path) == []


def test_read_records_when_log_holds_appended_records_does_return_them_in_file_order(
    fresh_root: str,
):
    jsonl_path = session_jsonl_path(fresh_root)
    written: list[SessionLogRecord] = [
        SESSION,
        BASELINE,
        HOOK,
        ITERATION_1,
        committed_keep(1),
        discard_record(2),
    ]
    for record in written:
        append_record(jsonl_path, record)

    assert read_records(jsonl_path) == written


def test_read_records_when_a_line_is_not_json_does_raise_naming_the_log_and_line(fresh_root: str):
    jsonl_path = _jsonl_holding(fresh_root, [_line(SESSION), "{not json", "{}"])

    with pytest.raises(GymratError) as excinfo:
        read_records(jsonl_path)

    assert f"{jsonl_path}:2" in message_of(excinfo.value)


def test_read_records_when_a_line_matches_no_schema_does_raise_naming_line_and_field(
    fresh_root: str,
):
    without_metrics = record_to_wire(ITERATION_1)
    del without_metrics["metrics"]
    jsonl_path = _jsonl_holding(fresh_root, [_line(SESSION), json.dumps(without_metrics)])

    with pytest.raises(GymratError) as excinfo:
        read_records(jsonl_path)

    assert f"{jsonl_path}:2" in message_of(excinfo.value)
    assert re.search(r"\bmetrics\b", message_of(excinfo.value))


def test_read_records_when_first_record_not_session_does_raise_naming_the_first_line(
    fresh_root: str,
):
    jsonl_path = _jsonl_holding(fresh_root, [_line(ITERATION_1), _line(discard_record(1))])

    with pytest.raises(GymratError) as excinfo:
        read_records(jsonl_path)

    assert f"{jsonl_path}:1" in message_of(excinfo.value)
    assert re.search(r"session", message_of(excinfo.value), re.IGNORECASE)


def test_read_records_when_final_line_unterminated_does_skip_it(fresh_root: str):
    jsonl_path = _jsonl_holding(fresh_root, [_line(SESSION)])
    with Path(jsonl_path).open("a", encoding="utf-8") as handle:
        handle.write('{"type":"iter')

    assert read_records(jsonl_path) == [SESSION]


# ---------------------------------------------------------------------------
# fold_session
# ---------------------------------------------------------------------------


def _state(**changes: Any) -> SessionState:
    return replace(EMPTY_STATE, **changes)


@pytest.mark.parametrize(
    ("records", "expected"),
    [
        pytest.param([], EMPTY_STATE, id="an-empty-log"),
        pytest.param([SESSION], _state(session=SESSION), id="a-session-with-nothing-measured"),
        pytest.param(
            [SESSION, ITERATION_1],
            _state(
                session=SESSION,
                iteration_count=1,
                last_iteration=ITERATION_1,
                unsettled=True,
                last_seq=1,
            ),
            id="a-measured-iteration-nobody-has-settled",
        ),
        pytest.param(
            [SESSION, BASELINE, HOOK, ITERATION_1],
            _state(
                session=SESSION,
                iteration_count=1,
                last_iteration=ITERATION_1,
                unsettled=True,
                last_seq=1,
            ),
            id="baseline-and-hook-around-a-measured-iteration",
        ),
        pytest.param(
            [SESSION, ITERATION_1, committed_keep(1)],
            _state(
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
            _state(
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
            _state(
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
            _state(
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
            _state(
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
            _state(
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
            _state(
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
            _state(
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
    ],
)
def test_fold_session_when_records_replayed_does_produce_the_summarized_state(
    records: list[SessionLogRecord], expected: SessionState
):
    assert fold_session(records) == expected


@pytest.mark.parametrize(
    ("keep_record", "expected_keep_count", "expected_unsettled"),
    [
        pytest.param(blocked_keep(1), 0, True, id="checks-failed"),
        pytest.param(blocked_keep(1, reason=None), 0, True, id="reason-absent"),
        pytest.param(_nothing_measured_block(2), 0, True, id="nothing-measured"),
        pytest.param(_gating_block(1), 0, False, id="gating-regression"),
    ],
)
def test_fold_session_when_keep_blocked_does_set_keep_count_and_unsettled(
    keep_record: KeepRecord, expected_keep_count: int, expected_unsettled: bool
):
    state = fold_session([SESSION, ITERATION_1, keep_record])

    assert state.keep_count == expected_keep_count
    assert state.unsettled is expected_unsettled


# ---------------------------------------------------------------------------
# fold_session — ends_on_gating_block
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("records", "expected"),
    [
        pytest.param(
            [SESSION, ITERATION_1, _gating_block(1)],
            True,
            id="a-log-ending-on-the-keep-a-gating-regression-refused",
        ),
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
        pytest.param(
            [SESSION, ITERATION_1],
            False,
            id="an-iteration-nobody-has-settled",
        ),
    ],
)
def test_fold_session_when_records_replayed_does_report_ends_on_gating_block(
    records: list[SessionLogRecord], expected: bool
):
    assert fold_session(records).ends_on_gating_block is expected


# ---------------------------------------------------------------------------
# require_session
# ---------------------------------------------------------------------------


def test_require_session_when_log_holds_a_session_does_hand_back_the_full_handoff(fresh_root: str):
    jsonl_path = session_jsonl_path(fresh_root)
    append_record(jsonl_path, SESSION)
    append_record(jsonl_path, ITERATION_1)

    required = require_session(fresh_root, "measuring an edit")

    assert required == RequiredSession(
        session=SESSION,
        state=_state(
            session=SESSION,
            iteration_count=1,
            last_iteration=ITERATION_1,
            unsettled=True,
            last_seq=1,
        ),
        jsonl_path=jsonl_path,
        records=[SESSION, ITERATION_1],
    )


@pytest.mark.parametrize("verb", ["measuring an edit", "asking for its status"])
def test_require_session_when_no_session_opened_does_raise_naming_root_and_verb(
    fresh_root: str, verb: str
):
    with pytest.raises(GymratError) as excinfo:
        require_session(fresh_root, verb)

    assert fresh_root in message_of(excinfo.value)
    assert hint_of(excinfo.value) == f"Run gymrat start to open one before {verb}."


def test_require_session_when_session_finalized_does_still_hand_the_closed_session_back(
    fresh_root: str,
):
    write_session_log(fresh_root, SESSION, (ITERATION_1, committed_keep(1), FINALIZE))

    required = require_session(fresh_root, "asking for its status")

    assert required.state.finalized == FINALIZE


# ---------------------------------------------------------------------------
# require_open_session
# ---------------------------------------------------------------------------


def test_require_open_session_when_not_finalized_does_match_require_session(
    fresh_root: str,
):
    write_session_log(fresh_root, SESSION, (ITERATION_1,))

    required = require_open_session(fresh_root, "measuring an edit")

    assert required == require_session(fresh_root, "measuring an edit")


def test_require_open_session_when_session_finalized_does_raise_naming_the_closed_session(
    fresh_root: str,
):
    write_session_log(fresh_root, SESSION, (ITERATION_1, committed_keep(1), FINALIZE))

    with pytest.raises(GymratError) as excinfo:
        require_open_session(fresh_root, "measuring an edit")

    assert SESSION.session_id in message_of(excinfo.value)
    assert "gymrat start" in (hint_of(excinfo.value) or "")
