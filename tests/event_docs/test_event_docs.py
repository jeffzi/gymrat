"""Tests for the generated event docs: schemas, AsyncAPI, reference, writer, drift.

``render_json_schemas()`` builds two draft 2020-12 JSON Schema documents, one
for every ``SessionLogRecord`` member and one for every ``SessionEvent``
member, each discriminated on the ``type`` field. ``render_asyncapi(schemas)``
builds an AsyncAPI 3.0.0 document from them: two channels, their messages,
four receive operations, a shared envelope trait, and component messages that
reference the sibling JSON Schema files. ``render_reference(schemas)`` returns
the Markdown reference with a generated-file banner, per-log sections, field
tables, nested-object subsections, and a closing Readers section.

``render_all()`` maps repo-relative paths to the text of the four generated
artifacts, ``write_all(root)`` writes them, and ``python -m gymrat.event_docs``
invokes ``write_all`` against the repo root; ``main()`` renders a
``GymratError`` like the CLI. The drift test regenerates every artifact in
memory and asserts each matches its committed file byte for byte; a mismatch
tells the reader to run ``task schemas``.

The reader-agreement tests derive the expected type sets from the actual
reader functions (fold_session, status rendering, outcome_record_count, and
the supervisor event union), then compare them to the ``READERS`` constant so
the two cannot diverge.
"""

import importlib.metadata
import json
import subprocess
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any, NamedTuple, get_args
from unittest.mock import create_autospec

import pytest
from jsonschema import Draft7Validator, Draft202012Validator
from pydantic import BaseModel

from gymrat.errors import TOOL_FAILURE_EXIT_CODE
from gymrat.event_docs import (
    READERS,
    SESSION_LOG_ADDRESS,
    SUPERVISOR_LOG_ADDRESS,
    ReaderSpec,
    main,
    render_all,
    render_asyncapi,
    render_json_schemas,
    render_reference,
    write_all,
)
from gymrat.git import run_git
from gymrat.loop.status import status_session
from gymrat.session.paths import repository_lookup_error
from gymrat.session.records import SessionLogRecord
from gymrat.session.store import fold_session
from gymrat.supervisor.events import (
    FollowUpEvent,
    LaunchEvent,
    SessionEvent,
    ToolStartEvent,
    event_from_wire,
)
from gymrat.supervisor.turns import outcome_record_count
from tests._ansi import normalize
from tests._cli import run_module
from tests._config import benchless_config
from tests._imports import loaded_under, modules_imported_by
from tests._markdown import md_section
from tests.session.records._fixtures import (
    baseline_record,
    command_record,
    committed_keep,
    discard_record,
    finalize_record,
    hook_record,
    iteration_record,
    session_record,
    stop_record,
    worktrees_at,
    write_session_log,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]

_SESSION_LOG_SCHEMA = "schemas/session-log.schema.json"
_SUPERVISOR_LOG_SCHEMA = "schemas/supervisor-log.schema.json"
_ASYNCAPI_DOC = "schemas/asyncapi.yaml"
_REFERENCE_DOC = "docs/event-reference.md"

_EXPECTED_KEYS = frozenset({
    _SESSION_LOG_SCHEMA,
    _SUPERVISOR_LOG_SCHEMA,
    _ASYNCAPI_DOC,
    _REFERENCE_DOC,
})


class _Channel(NamedTuple):
    name: str
    schema_index: int
    members: tuple[type[BaseModel], ...]
    schema_path: str
    heading: str
    address: str
    title: str


#: The session-log channel: ``SessionLogRecord`` members, first schema of ``render_json_schemas()``.
_SESSION_LOG = _Channel(
    "session-log",
    0,
    get_args(SessionLogRecord.__value__),
    _SESSION_LOG_SCHEMA,
    "## Session Log",
    ".gymrat/session.jsonl",
    "gymrat session log record",
)
#: The supervisor-log channel: ``SessionEvent`` members, second schema of ``render_json_schemas()``.
_SUPERVISOR_LOG = _Channel(
    "supervisor-log",
    1,
    get_args(SessionEvent),
    _SUPERVISOR_LOG_SCHEMA,
    "## Supervisor Log",
    ".gymrat/supervisor-<ms>.jsonl",
    "gymrat supervisor log event",
)
_CHANNELS = (_SESSION_LOG, _SUPERVISOR_LOG)
_CHANNEL_PARAMS = [pytest.param(channel, id=channel.name) for channel in _CHANNELS]


def _wire_type(model: type[BaseModel]) -> str:
    return get_args(model.model_fields["type"].annotation)[0]


# ---------------------------------------------------------------------------
# write_all
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("root_type", [pytest.param(Path, id="path"), pytest.param(str, id="str")])
def test_write_all_when_called_does_write_four_nonempty_files_at_expected_paths(
    root_type: Callable[[Path], Path | str], tmp_path: Path
):
    paths = write_all(root_type(tmp_path))

    assert sorted(paths) == sorted(tmp_path / rel for rel in _EXPECTED_KEYS)
    assert [path for path in paths if path.stat().st_size == 0] == []


# ---------------------------------------------------------------------------
# __main__
# ---------------------------------------------------------------------------


def test_main_module_when_run_does_exit_zero_with_four_output_lines(
    create_scratch_repo: Callable[[], str],
):
    root = Path(create_scratch_repo())

    result = run_module("gymrat.event_docs", cwd=root)

    assert result.returncode == 0, f"stderr: {result.stderr}"
    written = [Path(line).resolve() for line in result.stdout.splitlines()]
    assert sorted(written) == sorted((root / rel).resolve() for rel in _EXPECTED_KEYS)
    assert all(path.is_file() for path in written)


# ---------------------------------------------------------------------------
# drift gate
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("rel_path", sorted(_EXPECTED_KEYS))
def test_render_all_when_compared_to_committed_artifacts_does_match_byte_for_byte(rel_path: str):
    committed = _REPO_ROOT / rel_path

    expected_content = render_all()[rel_path]

    assert committed.is_file(), (
        f"{rel_path} does not exist on disk. Run `task schemas` to generate it."
    )
    assert committed.read_text(encoding="utf-8") == expected_content, (
        f"{rel_path} is stale — committed file differs from render_all() output. "
        "Run `task schemas` to regenerate."
    )


# ---------------------------------------------------------------------------
# reader agreement — READERS matches the actual reader functions
# ---------------------------------------------------------------------------


#: One probe record per ``SessionLogRecord`` wire type, shared by every reader probe.
_RECORD_BY_TYPE: dict[str, SessionLogRecord] = {
    "session": session_record(),
    "baseline": baseline_record(),
    "iteration": iteration_record(),
    "keep": committed_keep(1),
    "discard": discard_record(1),
    "hook": hook_record(),
    "finalize": finalize_record(),
    "stop": stop_record(),
    "command": command_record(),
}

#: Records each probe needs before it in a log for ``fold_session`` to accept it.
_FOLD_CONTEXT: dict[str, tuple[SessionLogRecord, ...]] = {
    "session": (),
    "baseline": (session_record(),),
    "iteration": (session_record(),),
    "keep": (session_record(), iteration_record()),
    "discard": (session_record(), iteration_record()),
    "hook": (session_record(),),
    "finalize": (session_record(), iteration_record(), committed_keep(1)),
    "stop": (session_record(), iteration_record()),
    "command": (session_record(),),
}

#: Records each probe needs before it in a log for ``status_session`` to render it.
_STATUS_CONTEXT: dict[str, tuple[SessionLogRecord, ...]] = {
    "baseline": (),
    "iteration": (),
    "keep": (iteration_record(),),
    "discard": (iteration_record(),),
    "hook": (),
    "stop": (),
    "command": (),
}

#: Wire types the status-history probe skips; see ``_status_history_types``.
_STATUS_EXCLUDED = frozenset({"session", "finalize"})


def _fold_session_types() -> set[str]:
    """Derive the wire types that change ``SessionState`` under fold_session.

    Returns:
        Each type for which folding a log containing it (in valid context)
        produces a different state than folding the same log without it.
    """
    changes: set[str] = set()
    for wire_type, record in _RECORD_BY_TYPE.items():
        before = _FOLD_CONTEXT[wire_type]
        without = fold_session(list(before))
        with_record = fold_session([*before, record])
        if with_record != without:
            changes.add(wire_type)

    return changes


def _status_history_types() -> set[str]:
    """Derive the wire types that change ``status_session`` output.

    Each record type is probed by rendering a log containing it (in valid
    context) and the same log without it, from the same session header.

    ``finalize`` is excluded: its only effect on ``status_session`` is the
    trailing ``format_status_finalized`` line driven by ``state.finalized``,
    which belongs to the folded session state (already covered by the
    fold-session reader), not the ordered iteration history this reader
    builds. Probing it here would always register as a difference and produce
    a false positive against ``READERS["status-history"].types``. ``session``
    is excluded because every probe log already starts with the session header.

    Returns:
        Each probed type whose presence changes the rendered status output.
    """
    types: set[str] = set()
    with tempfile.TemporaryDirectory() as scratch:
        for wire_type, record in _RECORD_BY_TYPE.items():
            if wire_type in _STATUS_EXCLUDED:
                continue
            before = _STATUS_CONTEXT[wire_type]
            root_without = str(Path(scratch) / f"without-{wire_type}")
            root_with = str(Path(scratch) / f"with-{wire_type}")
            Path(root_without).mkdir(parents=True)
            Path(root_with).mkdir(parents=True)

            session = session_record(worktrees=worktrees_at(root_without))
            write_session_log(root_without, session, before)
            write_session_log(root_with, session, (*before, record))

            without_output = status_session(root_without, benchless_config())
            with_output = status_session(root_with, benchless_config())
            if with_output != without_output:
                types.add(wire_type)

    return types


def _supervisor_guard_types() -> set[str]:
    """Derive the wire types counted by ``outcome_record_count``.

    Returns:
        Each type whose singleton record list produces a count of 1 (not 0,
        which would mean the type is excluded).
    """
    types: set[str] = set()
    for wire_type, record in _RECORD_BY_TYPE.items():
        if outcome_record_count([record]) > 0:
            types.add(wire_type)
    return types


def _dashboard_types() -> set[str]:
    """Derive the dashboard type set from the supervisor event type union."""
    return {_wire_type(model) for model in _SUPERVISOR_LOG.members}


def test_reader_probes_when_compared_to_session_log_union_does_cover_every_wire_type():
    union_types = {_wire_type(model) for model in _SESSION_LOG.members}

    assert set(_RECORD_BY_TYPE) == union_types


@pytest.mark.parametrize(
    ("reader", "derive_types"),
    [
        pytest.param("fold-session", _fold_session_types, id="fold-session"),
        pytest.param("status-history", _status_history_types, id="status-history"),
        pytest.param("supervisor-guard", _supervisor_guard_types, id="supervisor-guard"),
        pytest.param("dashboard", _dashboard_types, id="dashboard"),
    ],
)
def test_readers_when_compared_to_the_reader_function_does_agree(
    reader: str, derive_types: Callable[[], set[str]]
):
    expected = derive_types()

    actual = set(READERS[reader].types)

    assert actual == expected


# ---------------------------------------------------------------------------
# render_json_schemas — envelope and structure
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("channel", _CHANNEL_PARAMS)
def test_render_json_schemas_when_called_does_map_every_type_in_a_draft_2020_12_envelope(
    channel: _Channel,
):
    expected_mapping = {
        _wire_type(member): f"#/$defs/{member.__name__}" for member in channel.members
    }

    schema = render_json_schemas()[channel.schema_index]

    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert schema["title"] == channel.title
    assert schema["$id"] == f"https://github.com/jeffzi/gymrat/{channel.schema_path}"
    assert schema["oneOf"] == [{"$ref": f"#/$defs/{member.__name__}"} for member in channel.members]
    assert schema["discriminator"] == {"propertyName": "type", "mapping": expected_mapping}


# ---------------------------------------------------------------------------
# member defs — const discriminator, additionalProperties per log, at as integer
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("channel", "model_name", "expected_const", "expected_additional"),
    [
        pytest.param(
            _SESSION_LOG, "IterationRecord", "iteration", False, id="session-log-iteration"
        ),
        # None stands for "key absent": events accept unknown fields.
        pytest.param(
            _SUPERVISOR_LOG, "ToolStartEvent", "tool_start", None, id="supervisor-log-tool-start"
        ),
    ],
)
def test_render_json_schemas_when_called_does_shape_member_def_per_log(
    channel: _Channel,
    model_name: str,
    expected_const: str,
    expected_additional: bool | None,
):
    model_schema = render_json_schemas()[channel.schema_index]["$defs"][model_name]

    properties = model_schema["properties"]
    assert properties["type"].get("const") == expected_const
    assert model_schema.get("additionalProperties") is expected_additional
    assert properties["at"].get("type") == "integer"


# ---------------------------------------------------------------------------
# schema required on header records, no default
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("channel", "model_name", "field_name"),
    [
        pytest.param(_SESSION_LOG, "SessionRecord", "schema", id="session-log-schema"),
        pytest.param(_SUPERVISOR_LOG, "LaunchEvent", "schema", id="supervisor-log-schema"),
        pytest.param(_SUPERVISOR_LOG, "LaunchEvent", "session_id", id="supervisor-log-session-id"),
    ],
)
def test_render_json_schemas_when_called_does_require_header_field_without_default(
    channel: _Channel,
    model_name: str,
    field_name: str,
):
    model_schema = render_json_schemas()[channel.schema_index]["$defs"][model_name]

    assert field_name in model_schema["required"]
    assert "default" not in model_schema["properties"][field_name]


# ---------------------------------------------------------------------------
# type required on every event — the schema rejects what event_from_wire rejects
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "member",
    [pytest.param(member, id=member.__name__) for member in get_args(SessionEvent)],
)
def test_render_json_schemas_when_called_does_require_type_on_every_event(member: type):
    _, supervisor_log = render_json_schemas()

    required = supervisor_log["$defs"][member.__name__].get("required", [])
    assert "type" in required


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param({"at": 1}, id="missing-type"),
        pytest.param({"type": "compaction", "at": 1}, id="compaction"),
    ],
)
def test_supervisor_log_schema_when_validating_payload_does_agree_with_event_from_wire(
    payload: dict[str, Any],
):
    _, supervisor_log = render_json_schemas()
    validator = Draft202012Validator(supervisor_log)

    schema_accepts = validator.is_valid(payload)

    assert schema_accepts == (event_from_wire(payload) is not None)


# ---------------------------------------------------------------------------
# minimum keywords — Field(ge=…) becomes {"minimum": N}
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("model_name", "field_name", "minimum"),
    [
        pytest.param("IterationRecord", "seq", 1, id="positive-int"),
        pytest.param("HookRecord", "stdout_bytes", 0, id="non-negative-int"),
        pytest.param("KeepChecks", "stdout_bytes", 0, id="optional-non-negative-int"),
    ],
)
def test_render_json_schemas_when_called_does_type_bounded_int_with_minimum(
    model_name: str, field_name: str, minimum: int
):
    session_log, _ = render_json_schemas()

    field_prop = session_log["$defs"][model_name]["properties"][field_name]
    assert (field_prop.get("type"), field_prop.get("minimum")) == ("integer", minimum)


# ---------------------------------------------------------------------------
# supervisor-log field keywords — descriptions carried, never-null fields reject
# null, free-form input stays untyped
# ---------------------------------------------------------------------------


def _supervisor_field_schema(model: type[BaseModel], field_name: str) -> dict[str, Any]:
    return render_json_schemas()[1]["$defs"][model.__name__]["properties"][field_name]


def test_render_json_schemas_when_supervisor_field_described_does_carry_its_description():
    field_schema = _supervisor_field_schema(LaunchEvent, "kickoff_summary")

    assert field_schema["description"] == LaunchEvent.model_fields["kickoff_summary"].description


@pytest.mark.parametrize(
    ("model", "field_name"),
    [
        pytest.param(FollowUpEvent, "reason", id="optional-str"),
        pytest.param(LaunchEvent, "max_usd", id="optional-float"),
        pytest.param(LaunchEvent, "effort", id="optional-effort"),
    ],
)
def test_render_json_schemas_when_supervisor_field_optional_does_reject_null(
    model: type[BaseModel], field_name: str
):
    field_schema = _supervisor_field_schema(model, field_name)

    assert not Draft202012Validator(field_schema).is_valid(None)


#: One value of every JSON type.
_ANY_JSON_VALUES: tuple[object, ...] = (None, 1, 1.5, "x", True, [], {})


def test_render_json_schemas_when_tool_start_input_free_form_does_accept_any_json_value():
    field_schema = _supervisor_field_schema(ToolStartEvent, "input")

    validator = Draft202012Validator(field_schema)
    assert [value for value in _ANY_JSON_VALUES if not validator.is_valid(value)] == []


# ---------------------------------------------------------------------------
# session-log field keywords — only delta_pct nullable, descriptions carried,
# nested models referenced without a null branch
# ---------------------------------------------------------------------------


def _session_log_fields_where(predicate: Callable[[dict[str, Any]], bool]) -> set[tuple[str, str]]:
    return {
        (model, field)
        for model, definition in render_json_schemas()[0]["$defs"].items()
        for field, prop in definition.get("properties", {}).items()
        if predicate(prop)
    }


def _is_nullable(prop: dict[str, Any]) -> bool:
    return prop.get("type") == "null" or {"type": "null"} in prop.get("anyOf", [])


def _lacks_description(prop: dict[str, Any]) -> bool:
    return "description" not in prop


def test_render_json_schemas_when_called_does_type_null_only_on_the_delta_pct_fields():
    nullable = _session_log_fields_where(_is_nullable)

    assert nullable == {("IterationPrimary", "delta_pct"), ("MetricVerdict", "delta_pct")}


def test_render_json_schemas_when_called_does_carry_descriptions_on_every_session_log_field():
    missing = _session_log_fields_where(_lacks_description)

    assert missing == set()


@pytest.mark.parametrize(
    ("model_name", "field", "definition"),
    [
        pytest.param("SessionConfig", "hooks", "SessionHooks", id="session-config-hooks"),
        pytest.param("IterationRecord", "confirm", "Confirm", id="iteration-record-confirm"),
    ],
)
def test_render_json_schemas_when_called_does_ref_the_nested_model_without_null(
    model_name: str, field: str, definition: str
):
    field_schema = render_json_schemas()[0]["$defs"][model_name]["properties"][field]

    assert (field_schema["$ref"], "anyOf" in field_schema) == (f"#/$defs/{definition}", False)


# ---------------------------------------------------------------------------
# render_asyncapi — helpers
# ---------------------------------------------------------------------------


def _render_with_schemas() -> tuple[tuple[dict[str, Any], dict[str, Any]], dict[str, Any]]:
    schemas = render_json_schemas()
    return schemas, render_asyncapi(schemas)


def _render_asyncapi_doc() -> dict[str, Any]:
    return _render_with_schemas()[1]


# ---------------------------------------------------------------------------
# render_asyncapi — envelope and top-level structure
# ---------------------------------------------------------------------------


def test_render_asyncapi_when_called_does_describe_the_logs_in_info():
    doc = _render_asyncapi_doc()

    assert doc["info"]["title"] == "gymrat logs"
    assert doc["info"]["version"] == importlib.metadata.version("gymrat")
    assert len(doc["info"]["description"]) > 0


def test_render_asyncapi_when_called_does_document_seq_in_envelope_trait():
    doc = _render_asyncapi_doc()

    envelope = doc["components"]["messageTraits"]["envelope"]["description"]
    assert "`seq` (non-negative integer)" in envelope


# ---------------------------------------------------------------------------
# channels — session-log and supervisor-log
# ---------------------------------------------------------------------------


def test_render_asyncapi_when_called_does_list_each_channel_in_union_order():
    doc = _render_asyncapi_doc()

    channels = {
        name: (spec["address"], list(spec["messages"].items()))
        for name, spec in doc["channels"].items()
    }
    assert channels == {
        channel.name: (
            channel.address,
            [
                (wt, {"$ref": f"#/components/messages/{wt}"})
                for wt in map(_wire_type, channel.members)
            ],
        )
        for channel in _CHANNELS
    }


def test_render_asyncapi_when_called_does_list_component_messages_in_union_order():
    expected = [_wire_type(model) for channel in _CHANNELS for model in channel.members]

    doc = _render_asyncapi_doc()

    assert list(doc["components"]["messages"]) == expected


def test_render_asyncapi_when_model_has_no_description_does_use_class_name_as_summary():
    schemas = render_json_schemas()
    del schemas[0]["$defs"]["KeepRecord"]["description"]

    doc: dict[str, Any] = render_asyncapi(schemas)

    assert doc["components"]["messages"]["keep"]["summary"] == "KeepRecord"


# ---------------------------------------------------------------------------
# components.messages — one per record/event type
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("channel", "model"),
    [
        pytest.param(channel, model, id=_wire_type(model))
        for channel in _CHANNELS
        for model in channel.members
    ],
)
def test_render_asyncapi_when_called_does_describe_each_message_payload(
    channel: _Channel,
    model: type[BaseModel],
):
    schemas, doc = _render_with_schemas()

    msg = doc["components"]["messages"][_wire_type(model)]

    description = schemas[channel.schema_index]["$defs"][model.__name__]["description"]
    assert msg["contentType"] == "application/json"
    assert msg["title"] == model.__name__
    assert msg["summary"] == description.split("\n")[0]
    assert msg["payload"] == {
        "schemaFormat": "application/schema+json;version=draft-2020-12",
        "schema": {"$ref": f"./{channel.name}.schema.json#/$defs/{model.__name__}"},
    }
    assert msg["traits"] == [{"$ref": "#/components/messageTraits/envelope"}]


# ---------------------------------------------------------------------------
# operations — one receive operation per reader, each referencing messages that exist
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("op_name", "reader"), [pytest.param(name, spec, id=name) for name, spec in READERS.items()]
)
def test_render_asyncapi_when_called_does_have_receive_operation_per_reader(
    op_name: str, reader: ReaderSpec
):
    doc = _render_asyncapi_doc()

    op = doc["operations"][op_name]
    assert op["action"] == "receive"
    assert op["channel"]["$ref"] == f"#/channels/{reader.channel}"
    assert op["messages"] == [
        {"$ref": f"#/channels/{reader.channel}/messages/{t}"} for t in reader.types
    ]
    channel_messages = doc["channels"][reader.channel]["messages"]
    missing = [
        msg_type
        for msg_type in reader.types
        if msg_type not in doc["components"]["messages"] or msg_type not in channel_messages
    ]
    assert missing == []


# ---------------------------------------------------------------------------
# meta-schema validation
# ---------------------------------------------------------------------------

_META_SCHEMA_PATH = Path(__file__).parent / "vendor" / "asyncapi-3.0.0.schema.json"


def test_render_asyncapi_when_called_does_validate_against_asyncapi_300_meta_schema():
    meta_schema = json.loads(_META_SCHEMA_PATH.read_text(encoding="utf-8"))
    validator = Draft7Validator(meta_schema)

    doc = _render_asyncapi_doc()

    errors = list(validator.iter_errors(doc))
    assert not errors, f"AsyncAPI meta-schema validation errors: {errors}"


# ---------------------------------------------------------------------------
# render_reference — helpers
# ---------------------------------------------------------------------------


def _render_reference_doc() -> str:
    schemas = render_json_schemas()
    return render_reference(schemas)


def _type_section(md: str, wire_type: str) -> str:
    # Stop at the next heading of any level so nested ``####`` tables stay out.
    return md_section(md, f"\n### `{wire_type}`\n", "\n#")


# ---------------------------------------------------------------------------
# document framing — banner, field-table header, single trailing newline
# ---------------------------------------------------------------------------


def test_render_reference_when_called_does_frame_the_document():
    md = _render_reference_doc()

    assert md.split("\n")[0] == "<!-- Generated by `task schemas`. Do not edit. -->"
    assert "\n| Name | Type | Status | Description |\n| --- | --- | --- | --- |\n" in md
    assert md.endswith("\n")
    assert not md.endswith("\n\n")


# ---------------------------------------------------------------------------
# introduction
# ---------------------------------------------------------------------------


def test_render_reference_when_called_does_introduce_log_conventions():
    md = _render_reference_doc()

    intro = normalize(md_section(md, None, "\n## "))
    assert [
        phrase
        for phrase in (
            f"a session log at `{SESSION_LOG_ADDRESS}`",
            f"a supervisor log at `{SUPERVISOR_LOG_ADDRESS}`",
            "the launch time in milliseconds since the Unix epoch",
            "one JSON object per line, with snake_case keys",
            "nanoseconds since the Unix epoch",
            "Optional fields are absent rather than `null`",
            "a field marked required may still be `null`",
        )
        if phrase not in intro
    ] == []


# ---------------------------------------------------------------------------
# per-record/event ### sections — one per union member, in declared order
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("channel", _CHANNEL_PARAMS)
def test_render_reference_when_called_does_order_sections_per_union_declaration(
    channel: _Channel,
):
    md = _render_reference_doc()

    positions = [md.find(f"\n### `{_wire_type(model)}`\n") for model in channel.members]
    assert -1 not in positions
    assert positions == sorted(positions)


# ---------------------------------------------------------------------------
# field tables — iteration and launch as representative types; columns: name,
# type, required/optional, description
# ---------------------------------------------------------------------------


def _field_statuses(section: str) -> dict[str, str]:
    """Map each field row's name to its required/optional column."""
    cells = (line.split(" | ") for line in section.split("\n") if line.startswith("| `"))
    return {name.removeprefix("| `").removesuffix("`"): status for name, _, status, *_ in cells}


@pytest.mark.parametrize(
    ("wire_type", "statuses"),
    [
        pytest.param(
            "iteration",
            {
                "seq": "required",
                "samples": "required",
                "metrics": "required",
                "confirm": "optional",
                "primary": "required",
                "outcome": "required",
                "target_reached": "required",
                "duration_ms": "optional",
                "measured_tree": "optional",
            },
            id="iteration",
        ),
        pytest.param(
            "launch",
            {
                "schema": "required",
                "session_id": "required",
                "head_sha": "required",
                "dirty": "required",
                "max_minutes": "required",
                "kickoff_summary": "required",
            },
            id="launch",
        ),
    ],
)
def test_render_reference_when_called_does_list_field_rows_with_their_status(
    wire_type: str, statuses: dict[str, str]
):
    md = _render_reference_doc()

    rows = _field_statuses(_type_section(md, wire_type))
    assert {field: rows.get(field) for field in statuses} == statuses


# ---------------------------------------------------------------------------
# nested object subsections
# ---------------------------------------------------------------------------


def test_render_reference_when_called_does_have_nested_object_subsections():
    nested_objects = [
        "SessionConfig",
        "SessionHooks",
        "PairedSamples",
        "Confirm",
        "IterationPrimary",
        "DirtyInfo",
    ]

    md = _render_reference_doc()

    headings = set(md.split("\n"))
    missing = [name for name in nested_objects if f"#### `{name}`" not in headings]
    assert missing == []


# ---------------------------------------------------------------------------
# readers section
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("op_name", [pytest.param(name, id=name) for name in READERS])
def test_render_reference_when_called_does_describe_each_reader_operation(op_name: str):
    spec = READERS[op_name]
    expected = normalize(f"- **{op_name}** (channel: `{spec.channel}`): {spec.description}")

    md = _render_reference_doc()

    assert expected in normalize(md_section(md, "\n## Readers\n", None))


# ---------------------------------------------------------------------------
# Entry point isolation
# ---------------------------------------------------------------------------


def test_import_event_docs_when_loaded_does_not_import_the_cli_package_or_doc_libraries():
    loaded = modules_imported_by("gymrat.event_docs")

    assert loaded_under(loaded, "gymrat.cli", "yaml", "jsonschema", "ruamel") == []


# ---------------------------------------------------------------------------
# main — repository lookup fails
# ---------------------------------------------------------------------------

#: What git writes for a directory outside every repository.
_NOT_A_REPOSITORY = "fatal: not a git repository (or any of the parent directories): .git\n"
#: What a git that declines to answer writes, as opposed to "not a git repository".
_DUBIOUS_OWNERSHIP = "fatal: detected dubious ownership in repository"


@pytest.mark.parametrize(
    "git_stderr",
    [
        pytest.param(_NOT_A_REPOSITORY, id="outside-repo-with-hint"),
        pytest.param(_DUBIOUS_OWNERSHIP, id="git-declines-without-hint"),
    ],
)
def test_main_when_repository_lookup_fails_does_report_error_to_stderr_without_writing(
    git_stderr: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    monkeypatch.chdir(tmp_path)
    cause = subprocess.CalledProcessError(128, ["git"], stderr=git_stderr)
    monkeypatch.setattr("gymrat.session.paths.run_git", create_autospec(run_git, side_effect=cause))
    error = repository_lookup_error(str(Path.cwd()), cause)
    expected_stderr = f"{error}\n" if error.hint is None else f"{error}\n{error.hint}\n"

    with pytest.raises(SystemExit) as exc_info:
        main()

    assert exc_info.value.code == TOOL_FAILURE_EXIT_CODE
    assert capsys.readouterr() == ("", expected_stderr)
    assert [rel for rel in sorted(_EXPECTED_KEYS) if (tmp_path / rel).exists()] == []
