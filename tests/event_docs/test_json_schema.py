"""JSON Schema generation for session-log and supervisor-log event docs.

``render_json_schemas()`` builds two draft 2020-12 JSON Schema documents,
one for every ``SessionLogRecord`` member and one for every ``SessionEvent``
member, each discriminated on the ``type`` field.  The schemas carry
``description`` from each model field, enforce ``additionalProperties``
where the model does, type ``at`` as integer, and produce the correct
``$id`` / title / ``$schema`` envelope.

``session_event_adapter()`` exposes the pydantic ``TypeAdapter`` that
``event_from_wire`` uses, so callers that need the adapter directly can
get it without reaching for private state.
"""

from typing import Any, get_args

import pytest
from pydantic import TypeAdapter

from gymrat.session.records import SessionLogRecord
from gymrat.supervisor.events import SessionEvent
from tests.event_docs._imports import modules_imported_by

# ---------------------------------------------------------------------------
# render_json_schemas — envelope and structure
# ---------------------------------------------------------------------------


def _schemas() -> tuple[dict[str, Any], dict[str, Any]]:
    from gymrat.event_docs.json_schema import render_json_schemas

    return render_json_schemas()


def test_render_json_schemas_when_called_does_return_draft_2020_12_session_log_schema():
    session_log, _ = _schemas()

    assert session_log["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert session_log["title"] == "gymrat session log record"
    assert session_log["$id"] == "https://github.com/jeffzi/gymrat/schemas/session-log.schema.json"


def test_render_json_schemas_when_called_does_return_draft_2020_12_supervisor_log_schema():
    _, supervisor_log = _schemas()

    assert supervisor_log["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert supervisor_log["title"] == "gymrat supervisor log event"
    assert (
        supervisor_log["$id"]
        == "https://github.com/jeffzi/gymrat/schemas/supervisor-log.schema.json"
    )


def test_render_json_schemas_when_called_does_have_one_of_per_union_member():
    session_log, supervisor_log = _schemas()

    session_members = get_args(SessionLogRecord.__value__)
    supervisor_members = get_args(SessionEvent)

    assert "oneOf" in session_log
    assert len(session_log["oneOf"]) == len(session_members) == 9

    assert "oneOf" in supervisor_log
    assert len(supervisor_log["oneOf"]) == len(supervisor_members) == 12


def test_render_json_schemas_when_called_does_have_defs_per_record_model():
    session_log, supervisor_log = _schemas()

    assert "$defs" in session_log
    assert "IterationRecord" in session_log["$defs"]
    assert "SessionRecord" in session_log["$defs"]

    assert "$defs" in supervisor_log
    assert "LaunchEvent" in supervisor_log["$defs"]
    assert "ToolStartEvent" in supervisor_log["$defs"]


# ---------------------------------------------------------------------------
# field descriptions
# ---------------------------------------------------------------------------


def test_render_json_schemas_when_called_does_carry_description_on_iteration_outcome():
    session_log, _ = _schemas()

    iteration_schema = session_log["$defs"]["IterationRecord"]
    outcome_prop = iteration_schema["properties"]["outcome"]

    assert "description" in outcome_prop


def test_render_json_schemas_when_called_does_carry_description_on_launch_kickoff_summary():
    _, supervisor_log = _schemas()

    launch_schema = supervisor_log["$defs"]["LaunchEvent"]
    kickoff_prop = launch_schema["properties"]["kickoff_summary"]

    assert "description" in kickoff_prop


# ---------------------------------------------------------------------------
# type discriminator — const on each member's type field
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("schema_idx", "model_name", "expected_const"),
    [
        pytest.param(0, "IterationRecord", "iteration", id="session-log-iteration"),
        pytest.param(1, "ToolStartEvent", "tool_start", id="supervisor-log-tool-start"),
    ],
)
def test_render_json_schemas_when_called_does_have_const_type_discriminator(
    schema_idx: int,
    model_name: str,
    expected_const: str,
):
    schemas = _schemas()
    schema = schemas[schema_idx]

    model_schema = schema["$defs"][model_name]
    type_prop = model_schema["properties"]["type"]

    assert type_prop.get("const") == expected_const


# ---------------------------------------------------------------------------
# additionalProperties — present on record models, absent on event models
# ---------------------------------------------------------------------------


def test_render_json_schemas_when_called_does_set_additional_properties_false_on_records():
    session_log, _ = _schemas()

    iteration_schema = session_log["$defs"]["IterationRecord"]

    assert iteration_schema.get("additionalProperties") is False


def test_render_json_schemas_when_called_does_omit_additional_properties_on_events():
    _, supervisor_log = _schemas()

    tool_start_schema = supervisor_log["$defs"]["ToolStartEvent"]

    assert "additionalProperties" not in tool_start_schema


# ---------------------------------------------------------------------------
# optional-never-null fields and nullable fields
# ---------------------------------------------------------------------------


def test_render_json_schemas_when_called_does_type_optional_never_null_as_non_null():
    session_log, _ = _schemas()

    iteration_schema = session_log["$defs"]["IterationRecord"]
    duration_prop = iteration_schema["properties"]["duration_ms"]

    assert duration_prop == {"type": "number"} or duration_prop.get("type") == "number"
    assert "anyOf" not in duration_prop


def test_render_json_schemas_when_called_does_type_nullable_delta_pct_with_null_variant():
    session_log, _ = _schemas()

    metric_verdict = session_log["$defs"]["MetricVerdict"]
    delta_prop = metric_verdict["properties"]["delta_pct"]

    type_strings = set()
    if "anyOf" in delta_prop:
        for variant in delta_prop["anyOf"]:
            if "type" in variant:
                type_strings.add(variant["type"])
    elif "type" in delta_prop:
        types = delta_prop["type"]
        if isinstance(types, list):
            type_strings = set(types)

    assert "number" in type_strings
    assert "null" in type_strings


# ---------------------------------------------------------------------------
# at typed as integer
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("schema_idx", "model_name"),
    [
        pytest.param(0, "IterationRecord", id="session-log-iteration"),
        pytest.param(1, "LaunchEvent", id="supervisor-log-launch"),
    ],
)
def test_render_json_schemas_when_called_does_type_at_as_integer(
    schema_idx: int,
    model_name: str,
):
    schemas = _schemas()
    schema = schemas[schema_idx]

    model_schema = schema["$defs"][model_name]
    at_prop = model_schema["properties"]["at"]

    assert at_prop.get("type") == "integer"


# ---------------------------------------------------------------------------
# SessionHooks and Confirm lifted to $defs as $ref
# ---------------------------------------------------------------------------


def test_render_json_schemas_when_called_does_define_session_hooks_and_confirm_under_defs():
    session_log, _ = _schemas()
    defs = session_log["$defs"]

    assert "SessionHooks" in defs
    hooks_props = defs["SessionHooks"]["properties"]
    assert "before" in hooks_props
    assert "after" in hooks_props

    assert "Confirm" in defs
    confirm_props = defs["Confirm"]["properties"]
    assert "ran" in confirm_props
    assert "filtered" in confirm_props
    assert "samples" in confirm_props


def test_render_json_schemas_when_called_does_ref_hooks_from_session_config():
    session_log, _ = _schemas()
    config_schema = session_log["$defs"]["SessionConfig"]
    hooks_prop = config_schema["properties"]["hooks"]

    assert "$ref" in hooks_prop
    assert hooks_prop["$ref"] == "#/$defs/SessionHooks"
    assert "anyOf" not in hooks_prop


def test_render_json_schemas_when_called_does_ref_confirm_from_iteration_record():
    session_log, _ = _schemas()
    iteration_schema = session_log["$defs"]["IterationRecord"]
    confirm_prop = iteration_schema["properties"]["confirm"]

    assert "$ref" in confirm_prop
    assert confirm_prop["$ref"] == "#/$defs/Confirm"
    assert "anyOf" not in confirm_prop


# ---------------------------------------------------------------------------
# schema field presence
# ---------------------------------------------------------------------------


def test_render_json_schemas_when_called_does_have_schema_only_on_session_record():
    session_log, _ = _schemas()

    session_record = session_log["$defs"]["SessionRecord"]
    assert "schema" in session_record["properties"]

    iteration_record = session_log["$defs"]["IterationRecord"]
    assert "schema" not in iteration_record["properties"]


def test_render_json_schemas_when_called_does_have_schema_and_session_id_only_on_launch_event():
    _, supervisor_log = _schemas()

    launch = supervisor_log["$defs"]["LaunchEvent"]
    assert "schema" in launch["properties"]
    assert "session_id" in launch["properties"]


# ---------------------------------------------------------------------------
# schema required on header records, no default
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("schema_idx", "model_name"),
    [
        pytest.param(0, "SessionRecord", id="session-log"),
        pytest.param(1, "LaunchEvent", id="supervisor-log"),
    ],
)
def test_render_json_schemas_when_called_does_require_schema_on_header(
    schema_idx: int,
    model_name: str,
):
    schemas = _schemas()
    model_schema = schemas[schema_idx]["$defs"][model_name]

    assert "schema" in model_schema.get("required", [])


@pytest.mark.parametrize(
    ("schema_idx", "model_name"),
    [
        pytest.param(0, "SessionRecord", id="session-log"),
        pytest.param(1, "LaunchEvent", id="supervisor-log"),
    ],
)
def test_render_json_schemas_when_called_does_not_set_default_on_schema(
    schema_idx: int,
    model_name: str,
):
    schemas = _schemas()
    schema_prop = schemas[schema_idx]["$defs"][model_name]["properties"]["schema"]

    assert "default" not in schema_prop


# ---------------------------------------------------------------------------
# minimum keywords — Field(ge=…) must produce {"minimum": N}, never {"ge": N}
# ---------------------------------------------------------------------------


def test_render_json_schemas_when_called_does_use_minimum_not_ge_for_positive_ints():
    session_log, _ = _schemas()

    iteration_schema = session_log["$defs"]["IterationRecord"]
    seq_prop = iteration_schema["properties"]["seq"]
    assert seq_prop.get("minimum") == 1

    config_schema = session_log["$defs"]["SessionConfig"]
    samples_prop = config_schema["properties"]["samples"]
    assert samples_prop.get("minimum") == 1


def test_render_json_schemas_when_called_does_use_minimum_zero_for_non_negative_ints():
    session_log, _ = _schemas()

    hook_schema = session_log["$defs"]["HookRecord"]
    stdout_prop = hook_schema["properties"]["stdout_bytes"]
    assert stdout_prop.get("minimum") == 0

    command_schema = session_log["$defs"]["CommandRecord"]
    duration_prop = command_schema["properties"]["duration_ms"]
    assert duration_prop.get("minimum") == 0


def _collect_ge_keys(schema: dict[str, Any]) -> list[str]:
    """Walk the schema tree and return every path where a ``ge`` key appears."""
    found: list[str] = []

    def _walk(obj: Any, path: str) -> None:
        if isinstance(obj, dict):
            if "ge" in obj:
                found.append(path)
            for key, value in obj.items():
                _walk(value, f"{path}.{key}")
        elif isinstance(obj, list):
            for idx, item in enumerate(obj):
                _walk(item, f"{path}[{idx}]")

    _walk(schema, "$")
    return found


def test_render_json_schemas_when_called_does_not_contain_any_ge_key():
    session_log, supervisor_log = _schemas()

    session_ge = _collect_ge_keys(session_log)
    supervisor_ge = _collect_ge_keys(supervisor_log)

    assert not session_ge, f"session-log schema contains 'ge' at: {session_ge}"
    assert not supervisor_ge, f"supervisor-log schema contains 'ge' at: {supervisor_ge}"


# ---------------------------------------------------------------------------
# supervisor-log schema — optional-never-null fields have no null type
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("model_name", "field_name"),
    [
        pytest.param("FollowUpEvent", "reason", id="follow_up-reason"),
        pytest.param("FollowUpEvent", "text", id="follow_up-text"),
        pytest.param("LaunchEvent", "effort", id="launch-effort"),
        pytest.param("LaunchEvent", "max_usd", id="launch-max_usd"),
        pytest.param("LaunchEvent", "model", id="launch-model"),
        pytest.param("ModelPhaseEvent", "tool_name", id="model_phase-tool_name"),
        pytest.param("ModelPhaseEvent", "parent_tool_use_id", id="model_phase-parent_tool_use_id"),
        pytest.param(
            "ThinkingUpdateEvent", "parent_tool_use_id", id="thinking_update-parent_tool_use_id"
        ),
        pytest.param("ToolStartEvent", "parent_tool_use_id", id="tool_start-parent_tool_use_id"),
        pytest.param("ToolEndEvent", "parent_tool_use_id", id="tool_end-parent_tool_use_id"),
        pytest.param("TextDeltaEvent", "parent_tool_use_id", id="text_delta-parent_tool_use_id"),
    ],
)
def test_render_json_schemas_when_called_does_not_include_null_on_optional_never_null_field(
    model_name: str,
    field_name: str,
):
    _, supervisor_log = _schemas()

    model_schema = supervisor_log["$defs"][model_name]
    field_schema = model_schema["properties"][field_name]

    assert "anyOf" not in field_schema


def test_render_json_schemas_when_called_does_not_restrict_tool_start_input_away_from_null():
    _, supervisor_log = _schemas()

    tool_start = supervisor_log["$defs"]["ToolStartEvent"]
    properties = tool_start["properties"]

    assert "input" in properties
    input_prop = properties["input"]
    assert input_prop.get("type") is None


# ---------------------------------------------------------------------------
# session_event_adapter — public accessor and event_from_wire integration
# ---------------------------------------------------------------------------


def test_session_event_adapter_when_called_does_return_type_adapter():
    from gymrat.supervisor.events import session_event_adapter

    adapter = session_event_adapter()

    assert isinstance(adapter, TypeAdapter)


def test_session_event_adapter_when_called_does_validate_compaction_event():
    from gymrat.supervisor.events import CompactionEvent, session_event_adapter

    adapter = session_event_adapter()
    result = adapter.validate_python({"type": "compaction", "at": 1})

    assert isinstance(result, CompactionEvent)


# ---------------------------------------------------------------------------
# _OptNonNegativeInt — validation parity with _NonNegativeInt
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("field_name", "input_val", "expected"),
    [
        pytest.param("stdout_bytes", 5.0, 5, id="stdout-float-five"),
        pytest.param("stderr_bytes", 0.0, 0, id="stderr-float-zero"),
    ],
)
def test_keep_checks_when_opt_non_negative_int_given_integer_float_does_coerce(
    field_name: str,
    input_val: float,
    expected: int,
):
    from gymrat.session.records import KeepChecks

    checks = KeepChecks(configured=True, **{field_name: input_val})

    result = getattr(checks, field_name)
    assert result == expected
    assert isinstance(result, int)


@pytest.mark.parametrize(
    "field_name",
    [
        pytest.param("stdout_bytes", id="stdout"),
        pytest.param("stderr_bytes", id="stderr"),
    ],
)
def test_keep_checks_when_opt_non_negative_int_given_negative_does_reject(
    field_name: str,
):
    from pydantic import ValidationError

    from gymrat.session.records import KeepChecks

    with pytest.raises(ValidationError):
        KeepChecks(configured=True, **{field_name: -1})


# ---------------------------------------------------------------------------
# _OptNonNegativeInt — JSON Schema carries minimum: 0
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("model_name", "field_name"),
    [
        pytest.param("KeepChecks", "stdout_bytes", id="keep-checks-stdout-bytes"),
        pytest.param("KeepChecks", "stderr_bytes", id="keep-checks-stderr-bytes"),
        pytest.param("HookRecord", "stderr_bytes", id="hook-record-stderr-bytes"),
    ],
)
def test_render_json_schemas_when_called_does_use_minimum_zero_for_opt_non_negative_ints(
    model_name: str,
    field_name: str,
):
    session_log, _ = _schemas()

    model_schema = session_log["$defs"][model_name]
    field_prop = model_schema["properties"][field_name]

    assert field_prop.get("type") == "integer"
    assert field_prop.get("minimum") == 0


# ---------------------------------------------------------------------------
# import isolation — no unexpected modules pulled in
# ---------------------------------------------------------------------------


def test_importing_json_schema_when_loaded_does_not_import_unexpected_modules():
    unexpected = {"yaml", "jsonschema", "ruamel", "ruamel.yaml"}

    leaked = unexpected & modules_imported_by("gymrat.event_docs.json_schema")

    assert not leaked, f"importing json_schema pulled in unexpected modules: {leaked}"
