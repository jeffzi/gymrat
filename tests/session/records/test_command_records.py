from typing import get_args

import pytest

from gymrat.errors import GymratError
from gymrat.session import parse_record, record_to_wire
from tests.session.records.test_records import (
    COMMAND_RECORD,
    COMMAND_RECORD_SUCCESS,
    COMMAND_RECORD_WITH_TRACEPARENT,
    mentions,
    omitting,
    patching,
)

# ---------------------------------------------------------------------------
# CommandRecord — exit_code / reason consistency
# ---------------------------------------------------------------------------


def test_parse_record_when_command_exit_code_zero_with_reason_does_reject_with_rule_sentence():
    record = patching(COMMAND_RECORD_SUCCESS, {"reason": "budget-exceeded"})

    with pytest.raises(GymratError) as exc:
        parse_record(record)

    msg = str(exc.value)
    assert "a successful command must not carry a reason" in msg
    assert "expected a valid value" not in msg


def test_parse_record_when_command_exit_code_nonzero_without_reason_does_reject_with_rule_sentence():
    record = omitting(COMMAND_RECORD, "reason")

    with pytest.raises(GymratError) as exc:
        parse_record(record)

    msg = str(exc.value)
    assert "a failed command must carry a reason" in msg
    assert "expected a valid value" not in msg


# ---------------------------------------------------------------------------
# CommandRecord — field-level rejections with phrase-worded messages
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("record", "field", "phrase"),
    [
        pytest.param(
            patching(COMMAND_RECORD, {"exit_code": 3}),
            "exit_code",
            "0, 1 or 2",
            id="command-exit-code-out-of-range",
        ),
        pytest.param(
            patching(COMMAND_RECORD, {"reason": "bored"}),
            "reason",
            "one of",
            id="command-reason-invalid",
        ),
        pytest.param(
            patching(COMMAND_RECORD, {"args": "not-an-object"}),
            "args",
            "an object",
            id="command-args-not-object",
        ),
        pytest.param(
            patching(COMMAND_RECORD, {"duration_ms": -1}),
            "duration_ms",
            "a non-negative integer",
            id="command-duration-ms-negative",
        ),
        pytest.param(
            patching(COMMAND_RECORD_WITH_TRACEPARENT, {"traceparent": 42}),
            "traceparent",
            "a string",
            id="command-traceparent-not-string",
        ),
    ],
)
def test_parse_record_when_command_field_invalid_does_name_field_and_phrase(
    record: dict[str, object], field: str, phrase: str
):
    with pytest.raises(GymratError) as exc:
        parse_record(record)

    msg = str(exc.value)
    assert mentions(field).search(msg)
    assert phrase in msg


# ---------------------------------------------------------------------------
# CommandRecord — missing required fields
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("record", "field"),
    [
        pytest.param(omitting(COMMAND_RECORD, "name"), "name", id="command-no-name"),
        pytest.param(omitting(COMMAND_RECORD, "args"), "args", id="command-no-args"),
        pytest.param(omitting(COMMAND_RECORD, "exit_code"), "exit_code", id="command-no-exit-code"),
        pytest.param(
            omitting(COMMAND_RECORD, "duration_ms"), "duration_ms", id="command-no-duration-ms"
        ),
    ],
)
def test_parse_record_when_command_field_missing_does_name_field(
    record: dict[str, object], field: str
):
    with pytest.raises(GymratError) as exc:
        parse_record(record)

    assert mentions(field).search(str(exc.value))


# ---------------------------------------------------------------------------
# CommandRecord — non-integer seq reports through phrase table
# ---------------------------------------------------------------------------


def test_parse_record_when_command_seq_not_integer_does_name_field_and_phrase():
    record = patching(COMMAND_RECORD, {"seq": 3.5})

    with pytest.raises(GymratError) as exc:
        parse_record(record)

    msg = str(exc.value)
    assert mentions("seq").search(msg)
    assert "an integer" in msg


# ---------------------------------------------------------------------------
# CommandRecord — name must be non-empty
# ---------------------------------------------------------------------------


def test_parse_record_when_command_name_empty_does_reject():
    record = patching(COMMAND_RECORD, {"name": ""})

    with pytest.raises(GymratError) as exc:
        parse_record(record)

    assert mentions("name").search(str(exc.value))


# ---------------------------------------------------------------------------
# CommandRecord — reason and traceparent omitted from wire when None
# ---------------------------------------------------------------------------


def test_record_to_wire_when_command_reason_none_does_omit_from_wire():
    record = parse_record(COMMAND_RECORD_SUCCESS)

    wire = record_to_wire(record)

    assert "reason" not in wire


def test_record_to_wire_when_command_traceparent_none_does_omit_from_wire():
    record = parse_record(COMMAND_RECORD)

    wire = record_to_wire(record)

    assert "traceparent" not in wire


# ---------------------------------------------------------------------------
# CommandRecord — schema key on command records is rejected
# ---------------------------------------------------------------------------


def test_parse_record_when_schema_on_command_record_does_reject():
    record = patching(COMMAND_RECORD, {"schema": 1})

    with pytest.raises(GymratError) as exc:
        parse_record(record)

    assert mentions("schema").search(str(exc.value))


# ---------------------------------------------------------------------------
# CommandRecord — known type listing includes command
# ---------------------------------------------------------------------------


def test_parse_record_when_type_unknown_does_list_command_in_known_types():
    with pytest.raises(GymratError) as exc:
        parse_record({"type": "banana", "seq": 1})

    hint = exc.value.hint or ""
    assert "command" in hint


# ---------------------------------------------------------------------------
# CommandReason — vocabulary
# ---------------------------------------------------------------------------


def test_command_reason_when_imported_does_accept_all_defined_values():
    from gymrat.session import CommandReason

    expected = {
        "stop-condition",
        "budget-exceeded",
        "unsettled",
        "gating-block",
        "already-stopped",
        "no-session",
        "finalized",
        "nothing-measured",
        "gating-regression",
        "nothing-to-commit",
        "checks-failed",
        "nothing-to-discard",
        "stale-session",
        "nothing-kept",
        "dirty-worktree",
        "unkept-commits",
        "bad-branch",
        "branch-exists",
        "fail-on",
        "error",
    }
    actual = set(get_args(CommandReason))
    assert actual == expected


# ---------------------------------------------------------------------------
# CommandRecord — exports
# ---------------------------------------------------------------------------


def test_command_record_when_imported_from_records_package_does_exist():
    from gymrat.session.records import CommandRecord

    assert CommandRecord is not None


def test_command_reason_when_imported_from_session_package_does_exist():
    from gymrat.session import CommandReason

    assert CommandReason is not None
