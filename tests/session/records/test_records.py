from collections.abc import Callable
from typing import Any

import pytest

from gymrat.session.records import (
    SESSION_LOG_ADAPTER,
    NonFiniteNumberError,
    decode_log_line,
    parse_record,
    record_to_wire,
)
from tests.session.records._wire import (
    BASELINE_RECORD,
    BLOCKED_KEEP_RECORD,
    COMMAND_RECORD,
    COMMAND_RECORD_SUCCESS,
    COMMAND_RECORD_WITH_TRACEPARENT,
    COMMITTED_KEEP_RECORD,
    CONFIRM,
    DISCARD_RECORD,
    FINALIZE_RECORD,
    HOOK_RECORD,
    ITERATION_RECORD,
    METRIC_VERDICT,
    SESSION_RECORD,
    STOP_RECORD,
    config_with,
    omitting,
    patching,
    verdict_with,
)

# ---------------------------------------------------------------------------
# parse_record — valid records round-trip through the wire
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "record",
    [
        pytest.param(SESSION_RECORD, id="session"),
        pytest.param(
            patching(
                SESSION_RECORD,
                {
                    "config": config_with(
                        prepare="npm run build",
                        filter="npm run bench -- --filter {names}",
                    )
                },
            ),
            id="session-with-prepare-and-filter",
        ),
        pytest.param(
            patching(
                SESSION_RECORD,
                {
                    "config": config_with(
                        hooks={"before": "npm run warm-cache", "after": "npm run cool-down"}
                    )
                },
            ),
            id="session-with-hooks",
        ),
        pytest.param(BASELINE_RECORD, id="baseline"),
        pytest.param(
            patching(BASELINE_RECORD, {"duration_ms": 15192.3}),
            id="baseline-with-duration",
        ),
        pytest.param(ITERATION_RECORD, id="iteration"),
        pytest.param(
            patching(
                ITERATION_RECORD,
                {
                    "metrics": {
                        "total_ms": {
                            **omitting(omitting(METRIC_VERDICT, "p"), "noise_pct"),
                            "method": "band",
                        }
                    }
                },
            ),
            id="iteration-metric-omits-optional-stats",
        ),
        pytest.param(
            patching(
                verdict_with(delta_pct=None, verdict="no-signal"),
                {"primary": {"kind": "geomean", "delta_pct": None}, "outcome": "no-signal"},
            ),
            id="iteration-nulled-deltas",
        ),
        pytest.param(
            patching(ITERATION_RECORD, {"confirm": CONFIRM}),
            id="iteration-reran-to-confirm",
        ),
        pytest.param(
            patching(ITERATION_RECORD, {"duration_ms": 4200.5, "measured_tree": "abc123def456"}),
            id="iteration-with-duration-and-tree",
        ),
        pytest.param(COMMITTED_KEEP_RECORD, id="committed-keep"),
        pytest.param(
            omitting(COMMITTED_KEEP_RECORD, "message"),
            id="committed-keep-without-message",
        ),
        pytest.param(BLOCKED_KEEP_RECORD, id="blocked-keep"),
        pytest.param(
            patching(
                BLOCKED_KEEP_RECORD,
                {"seq": 0, "reason": "nothing-measured", "checks": {"configured": False}},
            ),
            id="keep-blocked-nothing-measured",
        ),
        pytest.param(DISCARD_RECORD, id="discard"),
        pytest.param(HOOK_RECORD, id="hook"),
        pytest.param(patching(HOOK_RECORD, {"stderr_bytes": 42}), id="hook-with-stderr-bytes"),
        pytest.param(patching(HOOK_RECORD, {"exit_code": -9}), id="hook-exit-code-negative-signal"),
        pytest.param(FINALIZE_RECORD, id="finalize"),
        pytest.param(STOP_RECORD, id="stop"),
        pytest.param(COMMAND_RECORD, id="command"),
        pytest.param(COMMAND_RECORD_SUCCESS, id="command-success-no-reason"),
        pytest.param(COMMAND_RECORD_WITH_TRACEPARENT, id="command-with-traceparent"),
    ],
)
def test_parse_record_when_record_satisfies_schema_does_round_trip(record: dict[str, object]):
    wire = record_to_wire(parse_record(record))

    assert wire == record


# ---------------------------------------------------------------------------
# decode_log_line
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("line", "literal"),
    [
        pytest.param('{"a":NaN}', "NaN", id="nan-literal"),
        pytest.param('{"a":Infinity}', "Infinity", id="infinity-literal"),
        pytest.param('{"a":-Infinity}', "-Infinity", id="negative-infinity-literal"),
        pytest.param('[{"a":[NaN]}]', "NaN", id="nan-literal-nested-in-an-array"),
    ],
)
def test_decode_log_line_when_a_number_is_nan_or_infinity_does_raise_naming_the_literal(
    line: str, literal: str
):
    with pytest.raises(NonFiniteNumberError) as excinfo:
        decode_log_line(line)

    assert str(excinfo.value) == f"{literal} is a non-finite number, which is not valid JSON"


@pytest.mark.parametrize(
    ("line", "literal"),
    [
        pytest.param('{"a":1e999}', "1e999", id="overflowing-literal"),
        pytest.param('{"a":-1e999}', "-1e999", id="negative-overflowing-literal"),
        pytest.param('{"a":{"b":[1,2e999]}}', "2e999", id="overflowing-literal-nested"),
    ],
)
def test_decode_log_line_when_a_number_overflows_a_float_does_raise_naming_the_literal(
    line: str, literal: str
):
    with pytest.raises(NonFiniteNumberError) as excinfo:
        decode_log_line(line)

    assert str(excinfo.value) == f"{literal} overflows a float"


def test_decode_log_line_when_numbers_finite_does_decode_them_unchanged():
    line = '{"a":1.5,"b":[-2,0,1e308],"c":{"d":-0.25,"e":"NaN"}}'

    decoded = decode_log_line(line)

    assert decoded == {"a": 1.5, "b": [-2, 0, 1e308], "c": {"d": -0.25, "e": "NaN"}}


# ---------------------------------------------------------------------------
# JSON schema — nullability of optional fields
# ---------------------------------------------------------------------------


def _session_log_schema() -> dict[str, Any]:
    return SESSION_LOG_ADAPTER.json_schema()


def _schema_fields_where(predicate: Callable[[dict[str, Any]], bool]) -> set[tuple[str, str]]:
    return {
        (model, field)
        for model, definition in _session_log_schema()["$defs"].items()
        for field, prop in definition.get("properties", {}).items()
        if predicate(prop)
    }


def _is_nullable(prop: dict[str, Any]) -> bool:
    return prop.get("type") == "null" or {"type": "null"} in prop.get("anyOf", [])


def _lacks_description(prop: dict[str, Any]) -> bool:
    return "description" not in prop


def test_json_schema_when_generated_does_type_null_only_on_the_delta_pct_fields():
    nullable = _schema_fields_where(_is_nullable)

    assert nullable == {("IterationPrimary", "delta_pct"), ("MetricVerdict", "delta_pct")}


# ---------------------------------------------------------------------------
# JSON schema — Field(description=...) on every field
# ---------------------------------------------------------------------------


def test_json_schema_when_generated_does_carry_descriptions_on_every_field():
    missing = _schema_fields_where(_lacks_description)

    assert missing == set()


# ---------------------------------------------------------------------------
# JSON schema — SessionHooks and Confirm referenced through $defs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("model_name", "field", "definition"),
    [
        pytest.param("SessionConfig", "hooks", "SessionHooks", id="session-config-hooks"),
        pytest.param("IterationRecord", "confirm", "Confirm", id="iteration-record-confirm"),
    ],
)
def test_json_schema_when_generated_does_ref_the_nested_model_without_null(
    model_name: str, field: str, definition: str
):
    schema = _session_log_schema()

    field_schema = schema["$defs"][model_name]["properties"][field]

    assert (field_schema["$ref"], "anyOf" in field_schema) == (f"#/$defs/{definition}", False)


# ---------------------------------------------------------------------------
# JSON schema — seq distribution across record types
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "model_name",
    [
        pytest.param("SessionRecord", id="session"),
        pytest.param("BaselineRecord", id="baseline"),
        pytest.param("FinalizeRecord", id="finalize"),
        pytest.param("StopRecord", id="stop"),
    ],
)
def test_json_schema_when_generated_does_not_include_seq_on_non_sequenced_record(
    model_name: str,
):
    schema = _session_log_schema()
    model_schema = schema["$defs"][model_name]

    assert "seq" not in model_schema["properties"]
