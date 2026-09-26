from typing import Any

import pytest
from pydantic import TypeAdapter

from gymrat.session.records import SessionLogRecord

# ---------------------------------------------------------------------------
# JSON schema — WithJsonSchema overrides on optional fields
# ---------------------------------------------------------------------------


def _session_log_schema() -> dict[str, Any]:
    """The JSON schema ``TypeAdapter`` generates directly from ``SessionLogRecord``."""
    return TypeAdapter(SessionLogRecord).json_schema()


def test_json_schema_when_optional_never_null_fields_present_does_emit_non_null_types():
    schema = _session_log_schema()
    defs = schema["$defs"]

    # KeepChecks.stdout_bytes is the sentinel: if it gains a null alternative
    # the assertion fails unconditionally.
    stdout_bytes_schema = defs["KeepChecks"]["properties"]["stdout_bytes"]
    if "anyOf" in stdout_bytes_schema:
        types = [alt.get("type") for alt in stdout_bytes_schema["anyOf"]]
        assert "null" not in types, (
            f"stdout_bytes should not include null in anyOf: {stdout_bytes_schema}"
        )
    else:
        assert stdout_bytes_schema.get("type") == "integer"

    delta_pct_schema = defs["MetricVerdict"]["properties"]["delta_pct"]
    assert delta_pct_schema, "delta_pct must appear in MetricVerdict properties"


# ---------------------------------------------------------------------------
# JSON schema — Field(description=...) on every field
# ---------------------------------------------------------------------------


def test_json_schema_when_generated_does_carry_descriptions_on_every_field():
    schema = _session_log_schema()
    defs = schema.get("$defs", {})

    missing: list[str] = []
    for model_name, model_schema in defs.items():
        props = model_schema.get("properties", {})
        for field_name, field_schema in props.items():
            if "description" not in field_schema:
                missing.append(f"{model_name}.{field_name}")

    assert not missing, f"Fields without description: {missing}"


# ---------------------------------------------------------------------------
# JSON schema — SessionHooks and Confirm lifted to $defs as $ref
# ---------------------------------------------------------------------------


def test_json_schema_when_generated_does_define_session_hooks_and_confirm_under_defs():
    schema = _session_log_schema()
    defs = schema.get("$defs", {})

    assert "SessionHooks" in defs
    hooks_props = defs["SessionHooks"]["properties"]
    assert "before" in hooks_props
    assert "after" in hooks_props

    assert "Confirm" in defs
    confirm_props = defs["Confirm"]["properties"]
    assert "ran" in confirm_props
    assert "filtered" in confirm_props
    assert "samples" in confirm_props


def test_json_schema_when_generated_does_ref_hooks_from_session_config_without_null():
    schema = _session_log_schema()
    config_props = schema["$defs"]["SessionConfig"]["properties"]
    hooks_prop = config_props["hooks"]

    assert "$ref" in hooks_prop
    assert hooks_prop["$ref"] == "#/$defs/SessionHooks"
    assert "anyOf" not in hooks_prop


def test_json_schema_when_generated_does_ref_confirm_from_iteration_record_without_null():
    schema = _session_log_schema()
    iteration_props = schema["$defs"]["IterationRecord"]["properties"]
    confirm_prop = iteration_props["confirm"]

    assert "$ref" in confirm_prop
    assert confirm_prop["$ref"] == "#/$defs/Confirm"
    assert "anyOf" not in confirm_prop


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
