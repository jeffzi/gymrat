"""Behavioral tests for the session JSONL store.

Records are written and read through real files in a throwaway temp root, so
the suite is order-independent and safe under ``pytest-xdist`` /
``pytest-randomly``. Nothing is mocked: the module under test is file I/O, and
only real bytes on disk reveal the torn-tail recovery and the refusal to log a
record that would not read back. The fold state machine has its own tests in
``test_store_fold.py``.
"""

import json
import re
from collections.abc import Callable
from pathlib import Path

import pytest

from gymrat.errors import GymratError
from gymrat.model import Direction
from gymrat.session.paths import session_jsonl_path
from gymrat.session.records import (
    BaselineRecord,
    IterationPrimary,
    IterationRecord,
    PairedSamples,
    SessionLogRecord,
)
from gymrat.session.schema import PrimaryKind
from gymrat.session.store import (
    BestIteration,
    ReadSessionResult,
    RequiredSession,
    append_record,
    fold_session,
    latest_baseline,
    read_live_session,
    read_records,
    read_session_header,
    recover_torn_tail,
    require_open_session,
    require_session,
    require_settled,
    session_header,
)
from tests._json_lines import with_raw_number
from tests._mode_bits import needs_mode_bits
from tests.session._store_records import (
    BASELINE,
    FINALIZE,
    HOOK,
    ITERATION_1,
    KEPT_BASELINE,
    SESSION,
)
from tests.session.records._fixtures import (
    BASELINE_SHA,
    COMMIT,
    TORN_PREFIX,
    append_raw,
    append_records,
    baseline_record,
    blocked_keep,
    command_record,
    committed_keep,
    discard_record,
    iteration_record,
    session_record,
    stop_record,
    tear_final_line,
    write_session_log,
)

# ---------------------------------------------------------------------------
# Fixture records
# ---------------------------------------------------------------------------

# An iteration a NaN sample makes unreadable: it serializes to a line with a
# `null` sample that no schema accepts on read-back.
UNREADABLE_ITERATION: IterationRecord = iteration_record(
    samples=PairedSamples(experiment=({"total_ms": float("nan")},), baseline=({"total_ms": 15200},))
)

# Text from a non-UTF-8 argument or path: POSIX `os.fsdecode(b"feat-\xff")`
# maps the undecodable byte to the lone surrogate U+DCFF, which no UTF-8 line
# can hold. The literal stands in for that call because Windows' `os.fsdecode`
# raises on the byte instead of mapping it.
LONE_SURROGATE_TEXT = "feat-" + chr(0xDCFF)


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
    """Write a session log holding exactly ``lines``, each ending in a bare line feed."""
    # Bytes, not text: a text-mode write on win32 turns each "\n" into "\r\n",
    # which is not what the store writes.
    return _jsonl_holding_bytes(root, "".join(f"{line}\n" for line in lines).encode())


def _jsonl_holding_bytes(root: str, raw: bytes) -> str:
    """Write a session log holding exactly ``raw``, newline-terminated or not."""
    jsonl_path = session_jsonl_path(root)
    append_raw(jsonl_path, raw)
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


def test_append_record_when_log_directory_absent_does_create_it_holding_one_line(
    fresh_root: str,
):
    jsonl_path = session_jsonl_path(fresh_root)

    append_record(jsonl_path, SESSION)

    assert Path(jsonl_path).read_bytes() == SESSION_LINE


def test_append_record_when_log_holds_records_does_add_one_line_after_them(fresh_root: str):
    jsonl_path = _jsonl_holding(fresh_root, [_line(SESSION)])

    append_record(jsonl_path, ITERATION_1)

    assert Path(jsonl_path).read_bytes() == SESSION_LINE + _line(ITERATION_1).encode() + b"\n"


def test_append_record_when_text_holds_a_unicode_line_break_does_write_one_unbroken_line(
    fresh_root: str,
):
    jsonl_path, header_len = _log_with_session_header(fresh_root)
    record = BASELINE.model_copy(update={"label": "before\N{LINE SEPARATOR}after"})

    append_record(jsonl_path, record)

    written = _appended_after_header(jsonl_path, header_len).decode("utf-8")
    assert len(written.splitlines()) == 1
    assert read_records(jsonl_path) == [SESSION, record]


# Each cause's hint is told apart by one word it alone carries.
_UNREADABLE_RECORDS = [
    pytest.param(UNREADABLE_ITERATION, "iteration", "NaN", id="a-nan-sample"),
    pytest.param(
        command_record(args={"path": LONE_SURROGATE_TEXT}),
        "command",
        "UTF-8",
        id="an-argument-with-a-lone-surrogate",
    ),
    pytest.param(
        command_record(args={"value": object()}),
        "command",
        "represent",
        id="an-argument-json-cannot-represent",
    ),
    pytest.param(stop_record(message=""), "stop", "schema", id="an-empty-stop-message"),
]


@pytest.mark.parametrize(("record", "record_type", "hint_marker"), _UNREADABLE_RECORDS)
def test_append_record_when_record_unreadable_does_raise_naming_its_cause(
    fresh_root: str, record: SessionLogRecord, record_type: str, hint_marker: str
):
    jsonl_path = session_jsonl_path(fresh_root)
    append_record(jsonl_path, SESSION)

    with pytest.raises(GymratError) as excinfo:
        append_record(jsonl_path, record)

    assert re.search(rf"\b{record_type}\b", str(excinfo.value))
    assert hint_marker in (excinfo.value.hint or "")


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
    ],
)
def test_recover_torn_tail_when_final_line_unterminated_does_truncate_to_the_last_newline(
    fresh_root: str, raw: bytes, expected: bytes
):
    jsonl_path = _jsonl_holding_bytes(fresh_root, raw)

    recover_torn_tail(jsonl_path)

    assert Path(jsonl_path).read_bytes() == expected


def test_recover_torn_tail_when_log_missing_does_leave_no_file_behind(fresh_root: str):
    jsonl_path = session_jsonl_path(fresh_root)

    recover_torn_tail(jsonl_path)

    assert not Path(jsonl_path).exists()


# ---------------------------------------------------------------------------
# read_records
# ---------------------------------------------------------------------------


def _no_log(root: str) -> str:
    return session_jsonl_path(root)


def _torn_final_line(root: str) -> str:
    jsonl_path = _jsonl_holding(root, [_line(SESSION)])
    tear_final_line(jsonl_path)
    return jsonl_path


def _torn_mid_utf8(root: str) -> str:
    return _jsonl_holding_bytes(root, TORN_MID_UTF8)


@pytest.mark.parametrize(
    ("make_log", "expected"),
    [
        pytest.param(_no_log, [], id="log-missing"),
        pytest.param(_torn_final_line, [SESSION], id="final-line-unterminated"),
        pytest.param(_torn_mid_utf8, [SESSION], id="final-line-torn-mid-utf8"),
    ],
)
def test_read_records_when_log_missing_or_final_line_unterminated_does_return_only_complete_lines(
    fresh_root: str, make_log: Callable[[str], str], expected: list[SessionLogRecord]
):
    jsonl_path = make_log(fresh_root)

    records = read_records(jsonl_path)

    assert records == expected


def _directory_at(jsonl_path: str) -> None:
    Path(jsonl_path).mkdir(parents=True)


def _without_read_permission(jsonl_path: str) -> None:
    append_record(jsonl_path, SESSION)
    Path(jsonl_path).chmod(0o000)


@pytest.mark.parametrize(
    "make_unreadable",
    [
        pytest.param(_directory_at, id="a-directory-at-the-path"),
        pytest.param(
            _without_read_permission,
            id="a-log-without-read-permission",
            marks=needs_mode_bits,
        ),
    ],
)
def test_read_records_when_log_exists_but_cannot_be_read_does_raise_naming_the_log(
    fresh_root: str, make_unreadable: Callable[[str], None]
):
    jsonl_path = session_jsonl_path(fresh_root)
    make_unreadable(jsonl_path)

    with pytest.raises(GymratError) as excinfo:
        read_records(jsonl_path)

    cause = excinfo.value.__cause__
    assert isinstance(cause, OSError)
    assert jsonl_path in str(excinfo.value)
    assert excinfo.value.hint == f"{cause.strerror}."


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
    append_records(fresh_root, *written)

    records = read_records(jsonl_path)

    assert records == written


def _names_line(hint: str | None, line_number: int) -> bool:
    """Whether ``hint`` names line ``line_number``."""
    return re.search(rf"\b{line_number}\b", hint or "") is not None


def test_read_records_when_a_line_is_not_json_does_raise_naming_the_line(
    fresh_root: str,
):
    jsonl_path = _jsonl_holding(fresh_root, [_line(SESSION), "", "{"])

    with pytest.raises(GymratError) as excinfo:
        read_records(jsonl_path)

    assert f"{jsonl_path}:3" in str(excinfo.value)
    assert _names_line(excinfo.value.hint, 3)


def test_read_records_when_a_line_matches_no_schema_does_raise_naming_the_line(
    fresh_root: str,
):
    without_metrics = json.loads(_line(ITERATION_1))
    del without_metrics["metrics"]
    jsonl_path = _jsonl_holding(fresh_root, [_line(SESSION), json.dumps(without_metrics)])

    with pytest.raises(GymratError) as excinfo:
        read_records(jsonl_path)

    assert f"{jsonl_path}:2" in str(excinfo.value)


# ---------------------------------------------------------------------------
# read_session_header
# ---------------------------------------------------------------------------

# A newline-terminated line whose bytes are not valid UTF-8: any read past the
# first line fails on it.
UNDECODABLE_LINE: bytes = b"\xff\xff\n"


@pytest.mark.parametrize(
    "leading",
    [
        pytest.param(b"", id="no-leading-lines"),
        pytest.param(b"\n \t\n\n", id="several-blank-lines"),
    ],
)
def test_read_session_header_when_first_non_blank_line_is_session_does_return_it_without_reading_on(
    fresh_root: str, leading: bytes
):
    jsonl_path = _jsonl_holding_bytes(fresh_root, leading + SESSION_LINE + UNDECODABLE_LINE)

    header = read_session_header(jsonl_path)

    assert header == SESSION


@pytest.mark.parametrize(
    "raw",
    [
        pytest.param(None, id="log-absent"),
        pytest.param(b"\n \t\n\n", id="several-blank-lines"),
    ],
)
def test_read_session_header_when_log_absent_or_holds_only_blank_lines_does_return_none(
    fresh_root: str, raw: bytes | None
):
    if raw is not None:
        _jsonl_holding_bytes(fresh_root, raw)

    header = read_session_header(session_jsonl_path(fresh_root))

    assert header is None


_NOT_JSON = b"{not json\n" + _line(ITERATION_1).encode() + b"\n"
_NON_FINITE = with_raw_number(_line(SESSION), ("schema",)).encode() + b"\n"
_NOT_A_SESSION = _line(ITERATION_1).encode() + b"\n" + SESSION_LINE


@pytest.mark.parametrize(
    ("reader", "raw", "hint_marker"),
    [
        # Each failure is told apart by one word its hint carries.
        pytest.param(read_session_header, _NOT_JSON, "JSON object", id="not-json"),
        pytest.param(read_session_header, _NON_FINITE, "NaN", id="non-finite-number"),
        pytest.param(
            read_records, UNDECODABLE_LINE + SESSION_LINE, "UTF-8", id="read-records-undecodable"
        ),
    ],
)
def test_session_log_reader_when_first_line_unusable_does_raise_naming_line_one(
    fresh_root: str, reader: Callable[[str], object], raw: bytes, hint_marker: str
):
    jsonl_path = _jsonl_holding_bytes(fresh_root, raw)

    with pytest.raises(GymratError) as excinfo:
        reader(jsonl_path)

    assert f"{jsonl_path}:1" in str(excinfo.value)
    assert hint_marker in (excinfo.value.hint or "")


def test_read_records_when_first_record_not_a_session_does_name_its_type(fresh_root: str):
    jsonl_path = _jsonl_holding_bytes(fresh_root, _NOT_A_SESSION)

    with pytest.raises(GymratError) as excinfo:
        read_records(jsonl_path)

    assert f"{jsonl_path}:1" in str(excinfo.value)
    assert "iteration" in str(excinfo.value)


def test_read_session_header_when_line_holds_only_non_ascii_whitespace_does_raise_naming_it(
    fresh_root: str,
):
    jsonl_path = _jsonl_holding_bytes(fresh_root, "\n　\n".encode() + SESSION_LINE)

    with pytest.raises(GymratError) as excinfo:
        read_session_header(jsonl_path)

    assert f"{jsonl_path}:2" in str(excinfo.value)
    assert _names_line(excinfo.value.hint, 2)


# ---------------------------------------------------------------------------
# session_header
# ---------------------------------------------------------------------------


def test_session_header_when_first_line_unusable_does_return_none(fresh_root: str):
    _jsonl_holding_bytes(fresh_root, b"{not json\n")

    header = session_header(fresh_root)

    assert header is None


# ---------------------------------------------------------------------------
# latest_baseline
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("records", "expected"),
    [
        pytest.param(
            [SESSION, ITERATION_1, committed_keep(1)], None, id="a-log-holding-no-baseline"
        ),
        pytest.param([SESSION, BASELINE, ITERATION_1], BASELINE, id="a-baseline-measure-recorded"),
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
    baseline = latest_baseline(records)

    assert baseline == expected


# ---------------------------------------------------------------------------
# require_session
# ---------------------------------------------------------------------------


def test_require_session_when_log_holds_a_session_does_hand_back_the_full_handoff(fresh_root: str):
    jsonl_path = session_jsonl_path(fresh_root)
    write_session_log(fresh_root, SESSION, (ITERATION_1,))

    required = require_session(fresh_root, "measuring an edit")

    assert required == RequiredSession(
        session=SESSION,
        state=fold_session([SESSION, ITERATION_1]),
        jsonl_path=jsonl_path,
        records=[SESSION, ITERATION_1],
    )


def test_require_session_when_no_session_opened_does_raise_naming_root_and_verb(fresh_root: str):
    with pytest.raises(GymratError) as excinfo:
        require_session(fresh_root, "measuring an edit")

    assert fresh_root in str(excinfo.value)
    assert "measuring an edit" in (excinfo.value.hint or "")


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
    assert FINALIZE.branch in str(excinfo.value)
    assert "measuring an edit" in (excinfo.value.hint or "")


# ---------------------------------------------------------------------------
# require_settled
# ---------------------------------------------------------------------------


def test_require_settled_when_last_iteration_unsettled_does_raise_naming_it_with_the_settle_hint():
    state = fold_session([SESSION, ITERATION_1])

    with pytest.raises(GymratError) as excinfo:
        require_settled(state, "finalizing")

    assert re.search(r"\b1\b", str(excinfo.value))
    assert "finalizing" in (excinfo.value.hint or "")


# ---------------------------------------------------------------------------
# read_live_session
# ---------------------------------------------------------------------------


def _read_back(
    root: str, history: tuple[SessionLogRecord, ...], direction: Direction = "lower"
) -> ReadSessionResult:
    """Write a session log under ``root`` and read it back the dashboard's way."""
    write_session_log(root, session_record(), history)
    return read_live_session(root, direction)


@pytest.mark.parametrize(
    ("kind", "name", "primary_label", "direction", "expected"),
    [
        pytest.param(
            "geomean", None, "geomean", "lower", (-9.0, 2, COMMIT), id="geomean-lower-is-better"
        ),
        pytest.param(
            "metric",
            "decode/time",
            "decode/time",
            "higher",
            (-7.2, 1, BASELINE_SHA),
            id="metric-higher-is-better",
        ),
    ],
)
def test_read_live_session_when_keeps_committed_does_report_the_best_committed_iteration(
    *,
    fresh_root: str,
    kind: PrimaryKind,
    name: str | None,
    primary_label: str,
    direction: Direction,
    expected: tuple[float, int, str],
):
    history = (
        iteration_record(seq=1, primary=IterationPrimary(kind=kind, name=name, delta_pct=-7.2)),
        committed_keep(1),
        iteration_record(seq=2, primary=IterationPrimary(kind=kind, name=name, delta_pct=-9.0)),
        committed_keep(2),
        iteration_record(seq=3, primary=IterationPrimary(kind=kind, name=name, delta_pct=-20.0)),
        blocked_keep(3),
    )

    result = _read_back(fresh_root, history, direction)

    delta_pct, seq, baseline_sha = expected
    assert result.best == BestIteration(
        delta_pct=delta_pct,
        seq=seq,
        label=primary_label,
        baseline_sha=baseline_sha,
        direction=direction,
    )


_FIRST_KEEP_SHA = "1" * 40


@pytest.mark.parametrize(
    ("deltas", "discarded", "expected"),
    [
        pytest.param(
            (-7.2, -20.0, -9.0), (2,), (3, _FIRST_KEEP_SHA), id="previous-keep-across-a-discard"
        ),
    ],
)
def test_read_live_session_when_keeps_precede_the_best_does_name_its_baseline(
    fresh_root: str,
    deltas: tuple[float, ...],
    discarded: tuple[int, ...],
    expected: tuple[int, str],
):
    history = tuple(
        record
        for seq, delta_pct in enumerate(deltas, start=1)
        for record in (
            iteration_record(
                seq=seq, primary=IterationPrimary(kind="geomean", delta_pct=delta_pct)
            ),
            discard_record(seq) if seq in discarded else committed_keep(seq, commit=str(seq) * 40),
        )
    )

    result = _read_back(fresh_root, history)

    assert result.best is not None
    assert (result.best.seq, result.best.baseline_sha) == expected


@pytest.mark.parametrize(
    "history",
    [
        pytest.param(
            (
                iteration_record(seq=1, primary=IterationPrimary(kind="geomean", delta_pct=None)),
                committed_keep(1),
            ),
            id="committed-keep-without-a-delta",
        ),
    ],
)
def test_read_live_session_when_no_committed_keep_has_a_delta_does_report_no_best(
    fresh_root: str, history: tuple[SessionLogRecord, ...]
):
    result = _read_back(fresh_root, history)

    assert result.best is None


@pytest.mark.parametrize(
    ("history", "stop_message"),
    [
        pytest.param((stop_record(message="done for today"),), "done for today", id="ends-on-stop"),
        pytest.param(
            (stop_record(message="done for today"), iteration_record(seq=1)),
            None,
            id="stop-then-iteration",
        ),
    ],
)
def test_read_live_session_when_stop_recorded_does_report_it_only_while_last(
    fresh_root: str, history: tuple[SessionLogRecord, ...], stop_message: str | None
):
    result = _read_back(fresh_root, history)

    assert result.stop_message == stop_message


@pytest.mark.parametrize(
    ("history", "has_baseline"),
    [
        pytest.param((), False, id="no-baseline"),
        pytest.param((baseline_record(),), True, id="baseline-recorded"),
    ],
)
def test_read_live_session_when_baseline_presence_varies_does_report_it(
    fresh_root: str, history: tuple[SessionLogRecord, ...], has_baseline: bool
):
    result = _read_back(fresh_root, history)

    assert result.has_baseline is has_baseline
