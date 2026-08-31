"""Behavioral tests for the session JSONL store and its fold state machine.

Records are written and read through real files in a throwaway temp root, so
the suite is order-independent and safe under ``pytest-xdist`` /
``pytest-randomly``. Nothing is mocked: the module under test is file I/O plus a
pure fold, and only real bytes on disk reveal the torn-tail recovery and the
refusal to log a record that would not read back.
"""

import json
import os
import re
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from gymrat.errors import GymratError, hint_of
from gymrat.session import (
    BaselineRecord,
    HookRecord,
    IterationRecord,
    KeepChecks,
    KeepRecord,
    PairedSamples,
    SessionLogRecord,
    SessionRecord,
    record_to_wire,
    recover_torn_tail,
    session_jsonl_path,
)
from gymrat.session.store import (
    RequiredSession,
    SessionState,
    append_record,
    fold_session,
    read_records,
    require_open_session,
    require_session,
)
from tests.session.records._fixtures import (
    AT,
    COMMIT,
    TORN_PREFIX,
    Worktrees,
    blocked_keep,
    committed_keep,
    discard_record,
    finalize_record,
    hook_record,
    iteration_record,
    session_record,
    tear_final_line,
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


def _jsonl_holding_bytes(root: str, raw: bytes) -> str:
    """Write a session log holding exactly ``raw``, newline-terminated or not."""
    jsonl_path = session_jsonl_path(root)
    Path(jsonl_path).parent.mkdir(parents=True, exist_ok=True)
    Path(jsonl_path).write_bytes(raw)
    return jsonl_path


SESSION_LINE: bytes = _line(SESSION).encode("utf-8") + b"\n"


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


def test_append_record_when_final_line_torn_does_add_its_line_leaving_the_torn_bytes_intact(
    fresh_root: str,
):
    # A torn tail is another writer's record still in flight. Appending must not
    # read the log or truncate it, or a concurrent append would destroy that
    # record; repairing the tail belongs to recover_torn_tail alone.
    jsonl_path = session_jsonl_path(fresh_root)
    append_record(jsonl_path, SESSION)
    tear_final_line(jsonl_path)
    before = Path(jsonl_path).read_bytes()

    append_record(jsonl_path, ITERATION_1)

    assert Path(jsonl_path).read_bytes() == before + _line(ITERATION_1).encode("utf-8") + b"\n"


def test_append_record_when_record_unreadable_does_raise_and_leave_the_log_byte_identical(
    fresh_root: str,
):
    jsonl_path = session_jsonl_path(fresh_root)
    append_record(jsonl_path, SESSION)
    before = Path(jsonl_path).read_bytes()

    with pytest.raises(GymratError) as excinfo:
        append_record(jsonl_path, UNREADABLE_ITERATION)

    assert re.search(r"\biteration\b", str(excinfo.value))
    assert Path(jsonl_path).read_bytes() == before


def test_append_record_when_os_write_short_does_retry_until_all_bytes_written(
    fresh_root: str, monkeypatch: pytest.MonkeyPatch
):
    jsonl_path = session_jsonl_path(fresh_root)
    real_write = os.write

    def short_write(fd: int, data: bytes) -> int:
        return real_write(fd, data[:1])

    monkeypatch.setattr(os, "write", short_write)

    append_record(jsonl_path, SESSION)

    monkeypatch.undo()
    assert read_records(jsonl_path) == [SESSION]


def test_append_record_when_record_written_does_fsync_before_close(
    fresh_root: str, monkeypatch: pytest.MonkeyPatch
):
    jsonl_path = session_jsonl_path(fresh_root)
    fsynced_fds: list[int] = []
    real_fsync = os.fsync

    def spy_fsync(fd: int) -> None:
        fsynced_fds.append(fd)
        real_fsync(fd)

    monkeypatch.setattr(os, "fsync", spy_fsync)

    append_record(jsonl_path, SESSION)

    assert len(fsynced_fds) >= 1


# ---------------------------------------------------------------------------
# recover_torn_tail
# ---------------------------------------------------------------------------

# \xc3 is the first byte of a 2-byte UTF-8 sequence; without the second byte
# the tail ends mid-character.
TORN_MID_UTF8: bytes = SESSION_LINE + TORN_PREFIX + b"\xc3"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        pytest.param(SESSION_LINE + TORN_PREFIX, SESSION_LINE, id="a-torn-line-after-a-whole-one"),
        pytest.param(b'{"type":"ses', b"", id="a-log-whose-only-line-is-torn"),
        pytest.param(TORN_MID_UTF8, SESSION_LINE, id="a-line-torn-mid-utf8-character"),
    ],
)
def test_recover_torn_tail_when_final_line_unterminated_does_truncate_to_the_last_newline(
    fresh_root: str, raw: bytes, expected: bytes
):
    jsonl_path = _jsonl_holding_bytes(fresh_root, raw)

    recover_torn_tail(jsonl_path)

    assert Path(jsonl_path).read_bytes() == expected


def test_recover_torn_tail_when_torn_tail_removed_does_let_a_later_append_read_back(
    fresh_root: str,
):
    jsonl_path = _jsonl_holding_bytes(fresh_root, TORN_MID_UTF8)
    recover_torn_tail(jsonl_path)

    append_record(jsonl_path, ITERATION_1)

    assert read_records(jsonl_path) == [SESSION, ITERATION_1]


def test_recover_torn_tail_when_log_ends_in_a_newline_does_leave_it_byte_identical(
    fresh_root: str,
):
    jsonl_path = session_jsonl_path(fresh_root)
    append_record(jsonl_path, SESSION)
    append_record(jsonl_path, ITERATION_1)
    before = Path(jsonl_path).read_bytes()

    recover_torn_tail(jsonl_path)

    assert Path(jsonl_path).read_bytes() == before


def test_recover_torn_tail_when_log_missing_does_leave_no_file_behind(fresh_root: str):
    jsonl_path = session_jsonl_path(fresh_root)

    recover_torn_tail(jsonl_path)

    assert not Path(jsonl_path).exists()


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

    assert f"{jsonl_path}:2" in str(excinfo.value)


def test_read_records_when_a_line_matches_no_schema_does_raise_naming_line_and_field(
    fresh_root: str,
):
    without_metrics = record_to_wire(ITERATION_1)
    del without_metrics["metrics"]
    jsonl_path = _jsonl_holding(fresh_root, [_line(SESSION), json.dumps(without_metrics)])

    with pytest.raises(GymratError) as excinfo:
        read_records(jsonl_path)

    assert f"{jsonl_path}:2" in str(excinfo.value)
    assert re.search(r"\bmetrics\b", str(excinfo.value))


def test_read_records_when_first_record_not_session_does_raise_naming_the_first_line(
    fresh_root: str,
):
    jsonl_path = _jsonl_holding(fresh_root, [_line(ITERATION_1), _line(discard_record(1))])

    with pytest.raises(GymratError) as excinfo:
        read_records(jsonl_path)

    assert f"{jsonl_path}:1" in str(excinfo.value)
    assert re.search(r"session", str(excinfo.value), re.IGNORECASE)


def test_read_records_when_final_line_unterminated_does_skip_it(fresh_root: str):
    jsonl_path = _jsonl_holding(fresh_root, [_line(SESSION)])
    tear_final_line(jsonl_path)

    assert read_records(jsonl_path) == [SESSION]


def test_read_records_when_final_line_torn_mid_utf8_does_skip_it(fresh_root: str):
    jsonl_path = session_jsonl_path(fresh_root)
    valid_line = _line(SESSION).encode("utf-8") + b"\n"
    # \xc3 is the leading byte of a 2-byte UTF-8 code point; alone at the end
    # it forms a truncated character the reader must silently discard.
    raw = valid_line + TORN_PREFIX + b"\xc3"
    Path(jsonl_path).parent.mkdir(parents=True, exist_ok=True)
    Path(jsonl_path).write_bytes(raw)

    assert read_records(jsonl_path) == [SESSION]


def test_read_records_when_complete_line_fails_to_decode_does_raise_naming_path_and_line(
    fresh_root: str,
):
    jsonl_path = session_jsonl_path(fresh_root)
    valid_line = _line(SESSION).encode("utf-8") + b"\n"
    # A newline-terminated line whose bytes are not valid UTF-8.
    corrupt_line = b"\xff\xff\n"
    raw = valid_line + corrupt_line
    Path(jsonl_path).parent.mkdir(parents=True, exist_ok=True)
    Path(jsonl_path).write_bytes(raw)

    with pytest.raises(GymratError) as excinfo:
        read_records(jsonl_path)

    assert f"{jsonl_path}:2" in str(excinfo.value)


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

    assert fresh_root in str(excinfo.value)
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

    assert SESSION.session_id in str(excinfo.value)
    assert "gymrat start" in (hint_of(excinfo.value) or "")
