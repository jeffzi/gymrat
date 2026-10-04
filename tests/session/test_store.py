"""Behavioral tests for the session JSONL store.

Records are written and read through real files in a throwaway temp root, so
the suite is order-independent and safe under ``pytest-xdist`` /
``pytest-randomly``. Nothing is mocked: the module under test is file I/O, and
only real bytes on disk reveal the torn-tail recovery and the refusal to log a
record that would not read back. The fold state machine has its own tests in
``test_store_fold.py``.
"""

import json
import os
import re
from pathlib import Path

import pytest

from gymrat.errors import GymratError
from gymrat.session.paths import session_jsonl_path
from gymrat.session.records import (
    BaselineRecord,
    IterationPrimary,
    IterationRecord,
    MetricVerdict,
    PairedSamples,
    SessionLogRecord,
)
from gymrat.session.store import (
    RequiredSession,
    append_record,
    first_line_json,
    latest_baseline,
    read_records,
    read_session_header,
    recover_torn_tail,
    require_open_session,
    require_session,
    session_header,
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
    TORN_PREFIX,
    blocked_keep,
    command_record,
    committed_keep,
    discard_record,
    iteration_record,
    session_state,
    stop_record,
    tear_final_line,
    write_session_log,
)
from tests.session.records._wire import with_raw_number

# ---------------------------------------------------------------------------
# Fixture records
# ---------------------------------------------------------------------------

# An iteration a NaN sample makes unreadable: it serializes to a line with a
# `null` sample that no schema accepts on read-back.
UNREADABLE_ITERATION: IterationRecord = iteration_record(
    samples=PairedSamples(experiment=({"total_ms": float("nan")},), baseline=({"total_ms": 15200},))
)

# The same, with an infinite sample: it serializes to `null` just as NaN does.
INFINITE_ITERATION: IterationRecord = iteration_record(
    samples=PairedSamples(experiment=({"total_ms": float("inf")},), baseline=({"total_ms": 15200},))
)

_NON_FINITE_HINT = (
    "Nothing was written. A metric that is NaN or Infinity becomes null in JSON and no longer "
    "reads back."
)
_NOT_UTF8_HINT = (
    "Nothing was written. A text field holds characters that are not valid UTF-8, for example "
    "from a non-UTF-8 argument or path."
)
_NOT_JSON_VALUE_HINT = "Nothing was written. A field holds a value JSON cannot represent."
_OFF_SCHEMA_HINT = "Nothing was written. The record does not match the session-log schema."

# Text from a non-UTF-8 argument or path: POSIX `os.fsdecode(b"feat-\xff")`
# maps the undecodable byte to the lone surrogate U+DCFF, which no UTF-8 line
# can hold. The literal stands in for that call because Windows' `os.fsdecode`
# raises on the byte instead of mapping it.
LONE_SURROGATE_TEXT = "feat-" + chr(0xDCFF)

# A baseline whose label came from a non-UTF-8 argument.
SURROGATE_LABEL_BASELINE: BaselineRecord = BASELINE.model_copy(
    update={"label": LONE_SURROGATE_TEXT}
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
    return record.model_dump_json(exclude_none=True)


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


def _log_with_session_header(root: str) -> tuple[str, int]:
    """Write a log holding only the session header; return its path and header byte length."""
    jsonl_path = session_jsonl_path(root)
    append_record(jsonl_path, SESSION)
    return jsonl_path, len(Path(jsonl_path).read_bytes())


def _appended_after_header(jsonl_path: str, header_len: int) -> bytes:
    """The bytes the log holds after its first ``header_len`` bytes."""
    return Path(jsonl_path).read_bytes()[header_len:]


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


@pytest.mark.parametrize(
    "record",
    [
        pytest.param(ITERATION_1, id="an-iteration-with-float-samples"),
        pytest.param(committed_keep(1), id="a-keep-with-no-reason"),
        pytest.param(blocked_keep(1, reason=None), id="a-blocked-keep-with-its-reason-erased"),
        pytest.param(
            command_record(exit_code=0, reason=None, seq=None), id="a-command-with-no-reason-or-seq"
        ),
    ],
)
def test_append_record_when_record_written_does_write_compact_json_without_none_keys(
    fresh_root: str, record: SessionLogRecord
):
    jsonl_path, header_len = _log_with_session_header(fresh_root)

    append_record(jsonl_path, record)

    written = _appended_after_header(jsonl_path, header_len).decode("utf-8")
    assert ", " not in written
    assert ": " not in written
    assert "null" not in written
    assert read_records(jsonl_path) == [SESSION, record]


def test_append_record_when_delta_undefined_does_write_delta_pct_as_null(fresh_root: str):
    jsonl_path, header_len = _log_with_session_header(fresh_root)
    record = iteration_record(
        metrics={
            "total_ms": MetricVerdict(
                delta_pct=None,
                verdict="no-signal",
                method="permutation",
                p=0.5,
                noise_pct=1.4,
                gating=True,
                confirmed=False,
            )
        },
        primary=IterationPrimary(kind="geomean", delta_pct=None),
        outcome="no-signal",
    )

    append_record(jsonl_path, record)

    written = json.loads(_appended_after_header(jsonl_path, header_len))
    assert written["primary"]["delta_pct"] is None
    assert written["metrics"]["total_ms"]["delta_pct"] is None
    assert read_records(jsonl_path) == [SESSION, record]


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


_UNREADABLE_RECORDS = [
    pytest.param(UNREADABLE_ITERATION, "iteration", _NON_FINITE_HINT, id="a-nan-sample"),
    pytest.param(INFINITE_ITERATION, "iteration", _NON_FINITE_HINT, id="an-infinite-sample"),
    pytest.param(
        SURROGATE_LABEL_BASELINE, "baseline", _NOT_UTF8_HINT, id="a-label-with-a-lone-surrogate"
    ),
    pytest.param(
        command_record(args={"path": LONE_SURROGATE_TEXT}),
        "command",
        _NOT_UTF8_HINT,
        id="an-argument-with-a-lone-surrogate",
    ),
    pytest.param(
        command_record(args={"value": object()}),
        "command",
        _NOT_JSON_VALUE_HINT,
        id="an-argument-json-cannot-represent",
    ),
    pytest.param(stop_record(message=""), "stop", _OFF_SCHEMA_HINT, id="an-empty-stop-message"),
]


@pytest.mark.parametrize(("record", "record_type", "hint"), _UNREADABLE_RECORDS)
def test_append_record_when_record_unreadable_does_raise_naming_its_cause_and_leave_the_log(
    fresh_root: str, record: SessionLogRecord, record_type: str, hint: str
):
    jsonl_path = session_jsonl_path(fresh_root)
    append_record(jsonl_path, SESSION)
    before = Path(jsonl_path).read_bytes()

    with pytest.raises(GymratError) as excinfo:
        append_record(jsonl_path, record)

    assert re.search(rf"\b{record_type}\b", str(excinfo.value))
    assert excinfo.value.hint == hint
    assert Path(jsonl_path).read_bytes() == before


@pytest.mark.parametrize(
    "record",
    [
        pytest.param(UNREADABLE_ITERATION, id="a-nan-sample"),
        pytest.param(INFINITE_ITERATION, id="an-infinite-sample"),
        pytest.param(SURROGATE_LABEL_BASELINE, id="a-label-with-a-lone-surrogate"),
    ],
)
def test_append_record_when_record_unreadable_and_log_absent_does_not_create_its_directory(
    tmp_path: Path, record: SessionLogRecord
):
    log_dir = tmp_path / "absent"

    with pytest.raises(GymratError):
        append_record(str(log_dir / "session.jsonl"), record)

    assert not log_dir.exists()


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


_NOT_JSON_HINT = "Line 2 is not a JSON object."
_NEVER_STORED = "Line 2 holds a number the session log never stores"


def _with_delta(literal: str) -> str:
    return with_raw_number(_line(ITERATION_1), ("primary", "delta_pct"), literal)


@pytest.mark.parametrize(
    ("bad_line", "hint"),
    [
        pytest.param("{not json", _NOT_JSON_HINT, id="malformed-json"),
        *(
            pytest.param(
                _with_delta(literal),
                f"{_NEVER_STORED}: {literal} is a non-finite number, which is not valid JSON.",
                id=f"{literal.lower()}-literal",
            )
            for literal in ("NaN", "Infinity", "-Infinity")
        ),
        pytest.param(
            _with_delta("1e999"), f"{_NEVER_STORED}: 1e999 overflows a float.", id="overflow"
        ),
    ],
)
def test_read_records_when_a_line_is_not_json_does_raise_naming_the_line_and_its_cause(
    fresh_root: str, bad_line: str, hint: str
):
    jsonl_path = _jsonl_holding(fresh_root, [_line(SESSION), bad_line, "{}"])

    with pytest.raises(GymratError) as excinfo:
        read_records(jsonl_path)

    assert str(excinfo.value) == f"Invalid JSON at {jsonl_path}:2"
    assert excinfo.value.hint == hint


def test_read_records_when_a_line_matches_no_schema_does_raise_naming_line_and_field(
    fresh_root: str,
):
    without_metrics = json.loads(_line(ITERATION_1))
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
    jsonl_path = _jsonl_holding_bytes(fresh_root, TORN_MID_UTF8)

    assert read_records(jsonl_path) == [SESSION]


def test_read_records_when_complete_line_fails_to_decode_does_raise_naming_path_and_line(
    fresh_root: str,
):
    # A newline-terminated line whose bytes are not valid UTF-8.
    corrupt_line = b"\xff\xff\n"
    jsonl_path = _jsonl_holding_bytes(fresh_root, SESSION_LINE + corrupt_line)

    with pytest.raises(GymratError) as excinfo:
        read_records(jsonl_path)

    assert f"{jsonl_path}:2" in str(excinfo.value)
    assert excinfo.value.hint == "Line 2 contains invalid UTF-8 bytes."


# ---------------------------------------------------------------------------
# read_session_header
# ---------------------------------------------------------------------------

# A newline-terminated line whose bytes are not valid UTF-8: any read past the
# first line fails on it.
UNDECODABLE_LINE: bytes = b"\xff\xff\n"


def test_read_session_header_when_first_line_is_session_does_return_it_without_reading_on(
    fresh_root: str,
):
    jsonl_path = _jsonl_holding_bytes(fresh_root, SESSION_LINE + UNDECODABLE_LINE)

    assert read_session_header(jsonl_path) == SESSION


def test_read_session_header_when_log_absent_does_return_none(fresh_root: str):
    jsonl_path = session_jsonl_path(fresh_root)

    assert read_session_header(jsonl_path) is None


@pytest.mark.parametrize(
    "raw",
    [
        pytest.param(b"", id="an-empty-log"),
        pytest.param(b"\n" + SESSION_LINE, id="an-empty-first-line"),
        pytest.param(b"   \n", id="a-whitespace-only-first-line"),
        pytest.param("　\n".encode(), id="a-non-ascii-whitespace-first-line"),
    ],
)
def test_read_session_header_when_first_line_blank_does_return_none(fresh_root: str, raw: bytes):
    jsonl_path = _jsonl_holding_bytes(fresh_root, raw)

    assert read_session_header(jsonl_path) is None


def test_read_session_header_when_first_line_not_json_does_raise_naming_the_first_line(
    fresh_root: str,
):
    jsonl_path = _jsonl_holding(fresh_root, ["{not json", _line(ITERATION_1)])

    with pytest.raises(GymratError) as excinfo:
        read_session_header(jsonl_path)

    assert str(excinfo.value) == f"Invalid JSON at {jsonl_path}:1"
    assert excinfo.value.hint == "Line 1 is not a JSON object."


@pytest.mark.parametrize(
    ("literal", "cause"),
    [
        pytest.param("NaN", "NaN is a non-finite number, which is not valid JSON", id="nan"),
        pytest.param(
            "-Infinity",
            "-Infinity is a non-finite number, which is not valid JSON",
            id="negative-infinity",
        ),
        pytest.param("1e999", "1e999 overflows a float", id="overflow"),
    ],
)
def test_read_session_header_when_first_line_holds_a_non_finite_number_does_hint_naming_it(
    fresh_root: str, literal: str, cause: str
):
    bad_header = with_raw_number(_line(SESSION), ("schema",), literal)
    jsonl_path = _jsonl_holding(fresh_root, [bad_header])

    with pytest.raises(GymratError) as excinfo:
        read_session_header(jsonl_path)

    assert (str(excinfo.value), excinfo.value.hint) == (
        f"Invalid JSON at {jsonl_path}:1",
        f"Line 1 holds a number the session log never stores: {cause}.",
    )


def test_read_session_header_when_first_record_not_session_does_raise_naming_its_type(
    fresh_root: str,
):
    jsonl_path = _jsonl_holding(fresh_root, [_line(ITERATION_1), _line(SESSION)])

    with pytest.raises(GymratError) as excinfo:
        read_session_header(jsonl_path)

    assert str(excinfo.value) == (
        f"Expected session header at {jsonl_path}:1, got a iteration record"
    )
    assert excinfo.value.hint == (
        "Line 1 is not a session header. The session log is corrupt; start a new session."
    )


def test_read_session_header_when_first_line_undecodable_does_raise_as_read_records_does(
    fresh_root: str,
):
    jsonl_path = _jsonl_holding_bytes(fresh_root, UNDECODABLE_LINE + SESSION_LINE)
    with pytest.raises(GymratError) as read_records_error:
        read_records(jsonl_path)

    with pytest.raises(GymratError) as excinfo:
        read_session_header(jsonl_path)

    assert str(excinfo.value) == f"Corrupt session log at {jsonl_path}:1"
    assert excinfo.value.hint == read_records_error.value.hint


# ---------------------------------------------------------------------------
# session_header
# ---------------------------------------------------------------------------


def test_session_header_when_first_line_is_session_does_return_it_without_reading_on(
    fresh_root: str,
):
    _jsonl_holding_bytes(fresh_root, SESSION_LINE + UNDECODABLE_LINE)

    assert session_header(fresh_root) == SESSION


def test_session_header_when_log_absent_does_return_none(fresh_root: str):
    assert session_header(fresh_root) is None


@pytest.mark.parametrize(
    "raw",
    [
        pytest.param(b"\n" + SESSION_LINE, id="a-blank-first-line"),
        pytest.param(b"{not json\n", id="a-first-line-that-is-not-json"),
        pytest.param(_line(ITERATION_1).encode("utf-8") + b"\n", id="a-first-record-not-session"),
        pytest.param(UNDECODABLE_LINE + SESSION_LINE, id="a-first-line-that-is-not-utf8"),
    ],
)
def test_session_header_when_first_line_unusable_does_return_none(fresh_root: str, raw: bytes):
    _jsonl_holding_bytes(fresh_root, raw)

    assert session_header(fresh_root) is None


# ---------------------------------------------------------------------------
# first_line_json
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw",
    [
        pytest.param(b"{not json\n" + SESSION_LINE, id="a-first-line-that-is-not-json"),
        pytest.param(UNDECODABLE_LINE + SESSION_LINE, id="a-first-line-that-is-not-utf8"),
        pytest.param(b'{"a":NaN}\n' + SESSION_LINE, id="a-first-line-holding-nan"),
    ],
)
def test_first_line_json_when_first_line_not_a_json_object_does_return_none(
    fresh_root: str, raw: bytes
):
    jsonl_path = _jsonl_holding_bytes(fresh_root, raw)

    assert first_line_json(Path(jsonl_path)) is None


# ---------------------------------------------------------------------------
# latest_baseline
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("records", "expected"),
    [
        pytest.param([], None, id="an-empty-log"),
        pytest.param(
            [SESSION, ITERATION_1, committed_keep(1)], None, id="a-log-holding-no-baseline"
        ),
        pytest.param([SESSION, BASELINE, ITERATION_1], BASELINE, id="a-baseline-measure-recorded"),
        pytest.param(
            [SESSION, ITERATION_1, committed_keep(1), KEPT_BASELINE],
            KEPT_BASELINE,
            id="a-baseline-a-keep-appended",
        ),
        pytest.param(
            [SESSION, BASELINE, ITERATION_1, committed_keep(1), KEPT_BASELINE],
            KEPT_BASELINE,
            id="two-baselines-takes-the-newest-in-file-order",
        ),
    ],
)
def test_latest_baseline_when_records_scanned_does_return_the_newest_baseline(
    records: list[SessionLogRecord], expected: BaselineRecord | None
):
    assert latest_baseline(records) == expected


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
        state=session_state(
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
    assert excinfo.value.hint == f"Run gymrat start to open one before {verb}."


def test_require_session_when_no_session_opened_does_carry_no_session_reason(fresh_root: str):
    with pytest.raises(GymratError) as excinfo:
        require_session(fresh_root, "measuring an edit")

    assert excinfo.value.reason == "no-session"


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
    assert "gymrat start" in (excinfo.value.hint or "")


def test_require_open_session_when_session_finalized_does_carry_finalized_reason(
    fresh_root: str,
):
    write_session_log(fresh_root, SESSION, (ITERATION_1, committed_keep(1), FINALIZE))

    with pytest.raises(GymratError) as excinfo:
        require_open_session(fresh_root, "measuring an edit")

    assert excinfo.value.reason == "finalized"
