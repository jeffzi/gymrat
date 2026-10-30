"""Tests for session-log records: round-tripping and the field validation of ``parse_record``.

The validation tests cover how each rejection names the value it refused: values
that are not an object, unknown record types, missing and unknown keys, and the
field checks this module decides itself — number-before-string ordering, explicit
nulls on optional fields, integers coerced from a whole float, tuple items of the
wrong type, and the key path a nested error reports.
"""

import pytest

from gymrat.errors import GymratError
from gymrat.session.records import parse_record, record_to_wire
from tests.session.records._fixtures import hook_record
from tests.session.records._wire import (
    BASELINE_RECORD,
    COMMAND_RECORD,
    COMMAND_RECORD_SUCCESS,
    DISCARD_RECORD,
    ITERATION_RECORD,
    METRIC_VERDICT,
    SESSION_RECORD,
    field_of,
    nested_with,
    omitting,
    patching,
    verdict_with,
)


@pytest.mark.parametrize(
    "record",
    [
        pytest.param(
            patching(
                verdict_with(delta_pct=None, verdict="no-signal"),
                {"primary": {"kind": "geomean", "delta_pct": None}, "outcome": "no-signal"},
            ),
            id="iteration-nulled-deltas",
        ),
        pytest.param(COMMAND_RECORD, id="command"),
        pytest.param(COMMAND_RECORD_SUCCESS, id="command-success-no-reason"),
    ],
)
def test_parse_record_when_record_satisfies_schema_does_round_trip(record: dict[str, object]):
    wire = record_to_wire(parse_record(record))

    assert wire == record


def test_hook_record_failed_when_timed_out_with_exit_code_zero_does_report_failure():
    record = hook_record(timed_out=True, exit_code=0)

    assert record.failed is True


# ---------------------------------------------------------------------------
# parse_record — values that match no record type
# ---------------------------------------------------------------------------


def test_parse_record_when_value_not_an_object_does_raise_naming_it():
    with pytest.raises(GymratError) as exc:
        parse_record(["session"])

    assert str(exc.value).endswith('["session"]')


@pytest.mark.parametrize(
    ("value", "shown"),
    [
        pytest.param({"type": "banana", "seq": 1}, '"banana"', id="unknown-name"),
        pytest.param(omitting(DISCARD_RECORD, "type"), "undefined", id="no-type-discriminator"),
        pytest.param({"type": 42}, "42", id="type-not-a-string"),
    ],
)
def test_parse_record_when_type_unknown_does_raise_naming_the_value(value: object, shown: str):
    with pytest.raises(GymratError) as exc:
        parse_record(value)

    assert str(exc.value).endswith(shown)


def test_parse_record_when_type_unknown_does_hint_every_known_record_type():
    with pytest.raises(GymratError) as exc:
        parse_record({"type": "banana"})

    hint = exc.value.hint or ""
    assert ("session" in hint, "command" in hint) == (True, True)


# ---------------------------------------------------------------------------
# parse_record — missing keys
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "key"),
    [
        pytest.param(
            patching(
                SESSION_RECORD, {"config": omitting(field_of(SESSION_RECORD, "config"), "bench")}
            ),
            "config.bench",
            id="config-no-bench",
        ),
        pytest.param(omitting(DISCARD_RECORD, "at"), "at", id="discard-no-at"),
    ],
)
def test_parse_record_when_key_missing_does_reject_naming_key(value: object, key: str):
    with pytest.raises(GymratError) as exc:
        parse_record(value)

    message = str(exc.value)
    assert ("Missing" in message, message.endswith(f" {key}")) == (True, True)


# ---------------------------------------------------------------------------
# parse_record — unknown keys
# ---------------------------------------------------------------------------


def test_parse_record_when_key_unknown_does_reject_naming_key():
    with pytest.raises(GymratError) as exc:
        parse_record(nested_with(SESSION_RECORD, "config", retries=3))

    message = str(exc.value)
    assert ("Unknown" in message, message.endswith(" config.retries")) == (True, True)


# ---------------------------------------------------------------------------
# parse_record — invalid values: the key path, the shape this module asks
# for, and the rejected value
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "key", "shape", "shown"),
    [
        # numbers are tried before strings, so a string names the number shape
        pytest.param(
            patching(
                ITERATION_RECORD,
                {"metrics": {"decode_ms": {**METRIC_VERDICT, "delta_pct": "banana"}}},
            ),
            "metrics.decode_ms.delta_pct",
            "a number",
            '"banana"',
            id="verdict-delta-string",
        ),
        # optional fields that are not nullable reject an explicit null with their type
        pytest.param(
            nested_with(SESSION_RECORD, "config", filter=None),
            "config.filter",
            "a string",
            "null",
            id="config-filter-null",
        ),
        # tuple items stay strict
        pytest.param(
            patching(BASELINE_RECORD, {"samples": [{"total_ms": "15200"}]}),
            "samples.0.total_ms",
            "a number",
            '"15200"',
            id="baseline-sample-string",
        ),
        # integers coerced from a whole float
        pytest.param(
            patching(nested_with(SESSION_RECORD, "config", samples=0.0), {"samples": 0}),
            "config.samples",
            "at or above 1",
            "0",
            id="config-samples-zero-float-beside-equal-stray-key",
        ),
    ],
)
def test_parse_record_when_field_invalid_does_reject_naming_key_shape_and_value(
    value: object, key: str, shape: str, shown: str
):
    with pytest.raises(GymratError) as exc:
        parse_record(value)

    message = str(exc.value)
    assert (f" {key}: " in message, shape in message, message.endswith(f" {shown}")) == (
        True,
        True,
        True,
    )


# ---------------------------------------------------------------------------
# CommandRecord — exit_code / reason consistency
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "fragment"),
    [
        pytest.param(
            patching(COMMAND_RECORD_SUCCESS, {"reason": "budget-exceeded"}),
            "budget-exceeded",
            id="success-with-reason",
        ),
        pytest.param(omitting(COMMAND_RECORD, "reason"), "1", id="failure-without-reason"),
    ],
)
def test_parse_record_when_command_exit_code_and_reason_disagree_does_reject_with_the_rule(
    value: object, fragment: str
):
    with pytest.raises(GymratError) as exc:
        parse_record(value)

    assert fragment in str(exc.value)
