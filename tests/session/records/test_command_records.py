from typing import get_args

import pytest

from gymrat.errors import GymratError
from gymrat.session import CommandReason, parse_record, record_to_wire
from tests.session.records._wire import (
    COMMAND_RECORD,
    COMMAND_RECORD_SUCCESS,
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

    assert str(exc.value) == "Unknown session record key: schema"


# ---------------------------------------------------------------------------
# CommandRecord — known type listing includes command
# ---------------------------------------------------------------------------


def test_parse_record_when_type_unknown_does_list_command_in_known_types():
    with pytest.raises(GymratError) as exc:
        parse_record({"type": "banana", "seq": 1})

    known_types = (exc.value.hint or "").removeprefix("Expected one of: ").rstrip(".").split(", ")
    assert "command" in known_types


# ---------------------------------------------------------------------------
# CommandReason — vocabulary
# ---------------------------------------------------------------------------

#: Every reason a command record may carry, as the session log documents them.
COMMAND_REASONS = (
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
    "not-improved",
    "nothing-to-discard",
    "stale-session",
    "nothing-kept",
    "dirty-worktree",
    "unkept-commits",
    "bad-branch",
    "branch-exists",
    "fail-on",
    "no-filter",
    "no-baseline",
    "supervised-use-tool",
    "error",
)


def test_command_reason_when_imported_does_accept_all_defined_values():
    actual = set(get_args(CommandReason))

    assert actual == set(COMMAND_REASONS)
