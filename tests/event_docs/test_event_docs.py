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
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any, NamedTuple, get_args
from unittest.mock import patch

import pytest
import yaml
from jsonschema import Draft7Validator, Draft202012Validator
from pydantic import BaseModel

from gymrat.errors import TOOL_FAILURE_EXIT_CODE, GymratError
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
from gymrat.loop.status import status_session
from gymrat.session.records import SessionLogRecord
from gymrat.session.store import fold_session
from gymrat.supervisor.events import SessionEvent, event_from_wire
from gymrat.supervisor.turns import outcome_record_count
from tests._config import benchless_config
from tests._imports import loaded_under, modules_imported_by
from tests.event_docs._extended_unions import PROBE_WIRE_TYPE, ProbeModel
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


#: The session-log channel: ``SessionLogRecord`` members, first schema of ``render_json_schemas()``.
_SESSION_LOG = _Channel(
    "session-log",
    0,
    get_args(SessionLogRecord.__value__),
    _SESSION_LOG_SCHEMA,
    "## Session Log",
    ".gymrat/session.jsonl",
)
#: The supervisor-log channel: ``SessionEvent`` members, second schema of ``render_json_schemas()``.
_SUPERVISOR_LOG = _Channel(
    "supervisor-log",
    1,
    get_args(SessionEvent),
    _SUPERVISOR_LOG_SCHEMA,
    "## Supervisor Log",
    ".gymrat/supervisor-<ms>.jsonl",
)
_CHANNELS = (_SESSION_LOG, _SUPERVISOR_LOG)
_CHANNEL_PARAMS = [pytest.param(channel, id=channel.name) for channel in _CHANNELS]


def _wire_type(model: type[BaseModel]) -> str:
    return get_args(model.model_fields["type"].annotation)[0]


def _md_section(md: str, start_marker: str | None, stop_marker: str | None) -> str:
    """Slice ``md`` from ``start_marker`` (or the top) to the next ``stop_marker`` (or the end).

    A missing ``start_marker`` raises ``ValueError`` instead of returning a wrong slice.
    """
    start = md.index(start_marker) if start_marker is not None else 0
    end = md.find(stop_marker, start + 1) if stop_marker is not None else -1
    return md[start:end] if end != -1 else md[start:]


def _unwrap(text: str) -> str:
    return " ".join(text.split())


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

    result = subprocess.run(
        [sys.executable, "-m", "gymrat.event_docs"],
        capture_output=True,
        text=True,
        cwd=root,
        check=False,
    )

    assert result.returncode == 0, f"stderr: {result.stderr}"
    written = [Path(line).resolve() for line in result.stdout.splitlines()]
    assert sorted(written) == sorted((root / rel).resolve() for rel in _EXPECTED_KEYS)
    assert all(path.is_file() for path in written)


def test_main_module_when_run_outside_repo_does_fail_without_writing_artifacts(tmp_path: Path):
    result = subprocess.run(
        [sys.executable, "-m", "gymrat.event_docs"],
        capture_output=True,
        text=True,
        cwd=str(tmp_path),
        check=False,
    )

    assert result.returncode == TOOL_FAILURE_EXIT_CODE, f"stderr: {result.stderr}"
    assert "Traceback" not in result.stderr
    assert not result.stdout.strip()
    assert [rel for rel in sorted(_EXPECTED_KEYS) if (tmp_path / rel).exists()] == []


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
# union-driven docs — a model added to a union is documented with no generator edit
# ---------------------------------------------------------------------------


def _render_all_with_extended_union(channel: str) -> dict[str, str]:
    result = subprocess.run(  # noqa: S603 -- fixed interpreter and module; channel is a parametrize value
        [sys.executable, "-m", "tests.event_docs._extended_unions", channel],
        capture_output=True,
        text=True,
        cwd=str(_REPO_ROOT),
        check=True,
    )
    return json.loads(result.stdout)


@pytest.mark.parametrize("channel", _CHANNEL_PARAMS)
def test_render_all_when_union_gains_model_does_document_it_in_every_generated_doc(
    channel: _Channel,
):
    artifacts = _render_all_with_extended_union(channel.name)

    defs = json.loads(artifacts[channel.schema_path])["$defs"]
    summary = defs[ProbeModel.__name__]["description"].split("\n")[0]
    asyncapi = yaml.safe_load(artifacts[_ASYNCAPI_DOC])
    channels = asyncapi["channels"]
    channels_listing_probe = [
        name for name in channels if PROBE_WIRE_TYPE in channels[name]["messages"]
    ]
    last_message = list(channels[channel.name]["messages"].items())[-1]
    component_summary = asyncapi["components"]["messages"][PROBE_WIRE_TYPE]["summary"]
    reference = artifacts[_REFERENCE_DOC]
    probe_heading_count = reference.count(f"\n### `{PROBE_WIRE_TYPE}`\n")
    last_subsection = _md_section(reference, f"\n{channel.heading}\n", "\n## ").rsplit("\n### ", 1)[
        1
    ]
    assert channels_listing_probe == [channel.name]
    assert last_message == (
        PROBE_WIRE_TYPE,
        {"$ref": f"#/components/messages/{PROBE_WIRE_TYPE}"},
    )
    assert component_summary == summary
    assert probe_heading_count == 1
    assert last_subsection.startswith(f"`{PROBE_WIRE_TYPE}`\n")


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
    """Derive the set of wire types that change ``SessionState`` under fold_session.

    A type is in the set when folding a log containing it (in valid context)
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
    """Derive the set of wire types that change ``status_session`` output.

    For each record type, compare a log containing it (in valid context) against
    the same log without it, rendered from the same session header.  A type is
    in the set when its presence changes the rendered status output.

    ``finalize`` is excluded: its only effect on ``status_session`` is the
    trailing ``format_status_finalized`` line driven by ``state.finalized``,
    which belongs to the folded session state (already covered by the
    fold-session reader), not the ordered iteration history this reader
    builds. Probing it here would always register as a difference and produce
    a false positive against ``READERS["status-history"].types``. ``session``
    is excluded because every probe log already starts with the session header.
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
    """Derive the set of wire types counted by ``outcome_record_count``.

    A type is in the set when a singleton list of that record produces a count
    of 1 (not 0, which would mean the type is excluded).
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


def _render_schemas() -> tuple[dict[str, Any], dict[str, Any]]:
    return render_json_schemas()


@pytest.mark.parametrize(
    ("channel", "title"),
    [
        pytest.param(_SESSION_LOG, "gymrat session log record", id="session-log"),
        pytest.param(_SUPERVISOR_LOG, "gymrat supervisor log event", id="supervisor-log"),
    ],
)
def test_render_json_schemas_when_called_does_return_draft_2020_12_envelope(
    channel: _Channel, title: str
):
    schema = _render_schemas()[channel.schema_index]

    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert schema["title"] == title
    assert schema["$id"] == f"https://github.com/jeffzi/gymrat/{channel.schema_path}"


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
    model_schema = _render_schemas()[channel.schema_index]["$defs"][model_name]

    properties = model_schema["properties"]
    assert properties["type"].get("const") == expected_const
    assert model_schema.get("additionalProperties") is expected_additional
    assert properties["at"].get("type") == "integer"


@pytest.mark.parametrize("channel", _CHANNEL_PARAMS)
def test_render_json_schemas_when_called_does_map_every_type_to_its_member_def(channel: _Channel):
    expected_mapping = {
        _wire_type(member): f"#/$defs/{member.__name__}" for member in channel.members
    }

    schema = _render_schemas()[channel.schema_index]

    assert schema["oneOf"] == [{"$ref": f"#/$defs/{member.__name__}"} for member in channel.members]
    assert schema["discriminator"] == {"propertyName": "type", "mapping": expected_mapping}


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
    model_schema = _render_schemas()[channel.schema_index]["$defs"][model_name]

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
    _, supervisor_log = _render_schemas()

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
    _, supervisor_log = _render_schemas()
    validator = Draft202012Validator(supervisor_log)

    schema_accepts = validator.is_valid(payload)

    assert schema_accepts == (event_from_wire(payload) is not None)


# ---------------------------------------------------------------------------
# minimum keywords — Field(ge=…) becomes {"minimum": N}
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("model_name", "field_name", "minimum"),
    [
        pytest.param("IterationRecord", "seq", 1, id="iteration-seq"),
        pytest.param("SessionConfig", "samples", 1, id="session-config-samples"),
        pytest.param("HookRecord", "stdout_bytes", 0, id="hook-record-stdout-bytes"),
        pytest.param("HookRecord", "stderr_bytes", 0, id="hook-record-stderr-bytes"),
        pytest.param("CommandRecord", "duration_ms", 0, id="command-duration-ms"),
        pytest.param("KeepChecks", "stdout_bytes", 0, id="keep-checks-stdout-bytes"),
        pytest.param("KeepChecks", "stderr_bytes", 0, id="keep-checks-stderr-bytes"),
    ],
)
def test_render_json_schemas_when_called_does_type_bounded_int_with_minimum(
    model_name: str, field_name: str, minimum: int
):
    session_log, _ = _render_schemas()

    field_prop = session_log["$defs"][model_name]["properties"][field_name]

    assert (field_prop.get("type"), field_prop.get("minimum")) == ("integer", minimum)


# ---------------------------------------------------------------------------
# supervisor-log field keywords — descriptions carried, never-null fields have no
# null branch, free-form input stays untyped
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("model_name", "field_name", "keyword", "present"),
    [
        pytest.param(
            "LaunchEvent",
            "kickoff_summary",
            "description",
            True,
            id="launch-kickoff_summary-described",
        ),
        pytest.param("FollowUpEvent", "reason", "anyOf", False, id="follow_up-reason-never-null"),
        pytest.param("FollowUpEvent", "text", "anyOf", False, id="follow_up-text-never-null"),
        pytest.param("LaunchEvent", "effort", "anyOf", False, id="launch-effort-never-null"),
        pytest.param("LaunchEvent", "max_usd", "anyOf", False, id="launch-max_usd-never-null"),
        pytest.param("LaunchEvent", "model", "anyOf", False, id="launch-model-never-null"),
        pytest.param(
            "ModelPhaseEvent", "tool_name", "anyOf", False, id="model_phase-tool_name-never-null"
        ),
        pytest.param(
            "ModelPhaseEvent",
            "parent_tool_use_id",
            "anyOf",
            False,
            id="model_phase-parent_tool_use_id-never-null",
        ),
        pytest.param(
            "ThinkingUpdateEvent",
            "parent_tool_use_id",
            "anyOf",
            False,
            id="thinking_update-parent_tool_use_id-never-null",
        ),
        pytest.param(
            "ToolStartEvent",
            "parent_tool_use_id",
            "anyOf",
            False,
            id="tool_start-parent_tool_use_id-never-null",
        ),
        pytest.param(
            "ToolEndEvent",
            "parent_tool_use_id",
            "anyOf",
            False,
            id="tool_end-parent_tool_use_id-never-null",
        ),
        pytest.param(
            "TextDeltaEvent",
            "parent_tool_use_id",
            "anyOf",
            False,
            id="text_delta-parent_tool_use_id-never-null",
        ),
        pytest.param("ToolStartEvent", "input", "type", False, id="tool_start-input-untyped"),
    ],
)
def test_render_json_schemas_when_called_does_shape_supervisor_field_keyword(
    model_name: str,
    field_name: str,
    keyword: str,
    present: bool,
):
    _, supervisor_log = _render_schemas()

    field_schema = supervisor_log["$defs"][model_name]["properties"][field_name]

    assert (keyword in field_schema) is present


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------

_VENDOR_DIR = Path(__file__).parent / "vendor"
_META_SCHEMA_PATH = _VENDOR_DIR / "asyncapi-3.0.0.schema.json"


def _render_with_schemas() -> tuple[tuple[dict[str, Any], dict[str, Any]], dict[str, Any]]:
    schemas = render_json_schemas()
    return schemas, render_asyncapi(schemas)


def _render_asyncapi_doc() -> dict[str, Any]:
    return _render_with_schemas()[1]


# ---------------------------------------------------------------------------
# render_asyncapi — envelope and top-level structure
# ---------------------------------------------------------------------------


def test_render_asyncapi_when_called_does_return_valid_asyncapi_300_envelope():
    doc = _render_asyncapi_doc()

    assert doc["asyncapi"] == "3.0.0"
    assert doc["info"]["title"] == "gymrat logs"
    assert doc["info"]["version"] == importlib.metadata.version("gymrat")
    assert len(doc["info"]["description"]) > 0


# ---------------------------------------------------------------------------
# channels — session-log and supervisor-log
# ---------------------------------------------------------------------------


def test_render_asyncapi_when_called_does_list_each_channel_address_and_messages_in_union_order():
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
# components.messageTraits.envelope
# ---------------------------------------------------------------------------


def test_render_asyncapi_when_called_does_describe_envelope_seq_as_non_negative():
    doc = _render_asyncapi_doc()

    description = doc["components"]["messageTraits"]["envelope"]["description"]

    assert "`seq` (non-negative integer)" in description


# ---------------------------------------------------------------------------
# operations — four receive operations
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


# ---------------------------------------------------------------------------
# cross-check: every operation message exists in components and its channel
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("op_name", list(READERS))
def test_render_asyncapi_when_called_does_have_consistent_operation_message_refs(op_name: str):
    doc = _render_asyncapi_doc()

    op = doc["operations"][op_name]
    component_messages = doc["components"]["messages"]
    channel_messages = doc["channels"][op["channel"]["$ref"].split("/")[-1]]["messages"]
    referenced = [ref["$ref"].split("/")[-1] for ref in op["messages"]]
    missing = [
        msg_type
        for msg_type in referenced
        if msg_type not in component_messages or msg_type not in channel_messages
    ]
    assert missing == []


# ---------------------------------------------------------------------------
# meta-schema validation
# ---------------------------------------------------------------------------


def test_render_asyncapi_when_called_does_validate_against_asyncapi_300_meta_schema():
    doc = _render_asyncapi_doc()

    meta_schema = json.loads(_META_SCHEMA_PATH.read_text(encoding="utf-8"))
    validator = Draft7Validator(meta_schema)

    errors = list(validator.iter_errors(doc))

    assert not errors, f"AsyncAPI meta-schema validation errors: {errors}"


def _render_reference_doc() -> str:
    schemas = render_json_schemas()
    return render_reference(schemas)


def _type_section(md: str, wire_type: str) -> str:
    # Stop at the next heading of any level so nested ``####`` tables stay out.
    return _md_section(md, f"\n### `{wire_type}`\n", "\n#")


# ---------------------------------------------------------------------------
# banner
# ---------------------------------------------------------------------------


def test_render_reference_when_called_does_start_with_generated_file_banner():
    md = _render_reference_doc()

    assert md.split("\n")[0] == "<!-- Generated by `task schemas`. Do not edit. -->"


# ---------------------------------------------------------------------------
# introduction
# ---------------------------------------------------------------------------


def test_render_reference_when_called_does_introduce_log_conventions():
    md = _render_reference_doc()

    intro = _unwrap(_md_section(md, None, "\n## "))

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
# field tables — iteration and launch as representative types
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("wire_type", "fields"),
    [
        pytest.param(
            "iteration",
            [
                "seq",
                "samples",
                "metrics",
                "confirm",
                "primary",
                "outcome",
                "target_reached",
                "duration_ms",
                "measured_tree",
            ],
            id="iteration",
        ),
        pytest.param(
            "launch",
            ["schema", "session_id", "head_sha", "dirty", "max_minutes", "kickoff_summary"],
            id="launch",
        ),
    ],
)
def test_render_reference_when_called_does_have_field_rows(wire_type: str, fields: list[str]):
    md = _render_reference_doc()

    section = _type_section(md, wire_type)

    assert [field for field in fields if f"\n| `{field}` |" not in section] == []


# ---------------------------------------------------------------------------
# field table structure — columns: name, type, required/optional, description
# ---------------------------------------------------------------------------


def test_render_reference_when_called_does_have_table_header_with_four_columns():
    md = _render_reference_doc()

    assert "\n| Name | Type | Status | Description |\n| --- | --- | --- | --- |\n" in md


@pytest.mark.parametrize(
    ("wire_type", "field", "status"),
    [
        pytest.param("iteration", "seq", "required", id="required"),
        pytest.param("iteration", "duration_ms", "optional", id="optional"),
    ],
)
def test_render_reference_when_called_does_mark_field_status(
    wire_type: str, field: str, status: str
):
    md = _render_reference_doc()

    rows = [
        line
        for line in _type_section(md, wire_type).split("\n")
        if line.startswith(f"| `{field}` |")
    ]

    assert [row.split(" | ")[2] for row in rows] == [status]


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

    headings = set(_render_reference_doc().split("\n"))

    missing = [name for name in nested_objects if f"#### `{name}`" not in headings]

    assert missing == []


# ---------------------------------------------------------------------------
# readers section
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("op_name", [pytest.param(name, id=name) for name in READERS])
def test_render_reference_when_called_does_describe_each_reader_operation(op_name: str):
    spec = READERS[op_name]
    expected = _unwrap(f"- **{op_name}** (channel: `{spec.channel}`): {spec.description}")

    md = _render_reference_doc()

    assert expected in _unwrap(_md_section(md, "\n## Readers\n", None))


# ---------------------------------------------------------------------------
# formatting — trailing newline
# ---------------------------------------------------------------------------


def test_render_reference_when_called_does_end_with_single_trailing_newline():
    md = _render_reference_doc()

    assert md.endswith("\n")
    assert not md.endswith("\n\n")


# ---------------------------------------------------------------------------
# Entry point isolation
# ---------------------------------------------------------------------------


def test_import_event_docs_when_loaded_does_not_import_the_cli_package_or_doc_libraries():
    loaded = modules_imported_by("gymrat.event_docs")

    assert loaded_under(loaded, "gymrat.cli", "yaml", "jsonschema", "ruamel") == []


# ---------------------------------------------------------------------------
# main — GymratError from repo_root
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("error", "expected_stderr"),
    [
        pytest.param(
            GymratError("not a git repository", hint="run `git init` first"),
            "not a git repository\nrun `git init` first\n",
            id="with-hint",
        ),
        pytest.param(
            GymratError("not a git repository"), "not a git repository\n", id="without-hint"
        ),
    ],
)
def test_main_when_repo_root_raises_gymrat_error_does_report_error_to_stderr(
    error: GymratError, expected_stderr: str, capsys: pytest.CaptureFixture[str]
):
    with (
        patch("gymrat.event_docs.repo_root", side_effect=error, autospec=True),
        pytest.raises(SystemExit) as exc_info,
    ):
        main()

    assert exc_info.value.code == TOOL_FAILURE_EXIT_CODE
    assert capsys.readouterr() == ("", expected_stderr)
