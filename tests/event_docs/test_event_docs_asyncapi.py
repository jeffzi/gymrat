"""AsyncAPI 3.0.0 document generation for session-log and supervisor-log.

``render_asyncapi(schemas)`` builds an AsyncAPI 3.0.0 document from the two
JSON Schema dicts returned by ``render_json_schemas()``.  The document
describes two channels (session-log and supervisor-log), their messages,
four receive operations, a shared envelope trait, and component messages
that reference the sibling JSON Schema files.

``render_asyncapi_yaml(document)`` serializes a document dict to YAML
without importing ``yaml`` at module level.

``READERS`` is a module-level constant mapping operation names to their
channel and message type lists.
"""

import importlib.metadata
import json
from pathlib import Path
from typing import Any, get_args

import pytest
from jsonschema import Draft7Validator
from pydantic import BaseModel

from gymrat.event_docs import render_asyncapi, render_json_schemas
from gymrat.session.records import SessionLogRecord
from gymrat.supervisor.events import SessionEvent

# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------

_VENDOR_DIR = Path(__file__).parent / "vendor"
_META_SCHEMA_PATH = _VENDOR_DIR / "asyncapi-3.0.0.schema.json"

#: Record class names → wire type strings (session log).
SESSION_RECORD_CLASSES: dict[str, str] = {
    "SessionRecord": "session",
    "BaselineRecord": "baseline",
    "IterationRecord": "iteration",
    "KeepRecord": "keep",
    "DiscardRecord": "discard",
    "HookRecord": "hook",
    "FinalizeRecord": "finalize",
    "StopRecord": "stop",
    "CommandRecord": "command",
}

#: Event class names → wire type strings (supervisor log).
SUPERVISOR_EVENT_CLASSES: dict[str, str] = {
    "LaunchEvent": "launch",
    "ThinkingUpdateEvent": "thinking_update",
    "ToolStartEvent": "tool_start",
    "ToolEndEvent": "tool_end",
    "TextDeltaEvent": "text_delta",
    "UsageUpdateEvent": "usage_update",
    "CapEvent": "cap",
    "ModelPhaseEvent": "model_phase",
    "TurnEndEvent": "turn_end",
    "FollowUpEvent": "follow_up",
    "CompactionEvent": "compaction",
}


def _render_with_schemas() -> tuple[tuple[dict[str, Any], dict[str, Any]], dict[str, Any]]:
    from gymrat.event_docs import render_asyncapi

    schemas = render_json_schemas()
    return schemas, render_asyncapi(schemas)


def _render() -> dict[str, Any]:
    return _render_with_schemas()[1]


# ---------------------------------------------------------------------------
# render_asyncapi — envelope and top-level structure
# ---------------------------------------------------------------------------


def test_render_asyncapi_when_called_does_return_valid_asyncapi_300_envelope():
    doc = _render()

    assert doc["asyncapi"] == "3.0.0"
    assert doc["info"]["title"] == "gymrat logs"
    assert doc["info"]["version"] == importlib.metadata.version("gymrat")
    assert isinstance(doc["info"]["description"], str)
    assert len(doc["info"]["description"]) > 0


# ---------------------------------------------------------------------------
# channels — session-log and supervisor-log
# ---------------------------------------------------------------------------


def test_render_asyncapi_when_called_does_have_two_channels_with_correct_addresses():
    doc = _render()

    channels = doc["channels"]
    assert "session-log" in channels
    assert "supervisor-log" in channels
    assert channels["session-log"]["address"] == ".gymrat/session.jsonl"
    assert channels["supervisor-log"]["address"] == ".gymrat/supervisor-<ms>.jsonl"


#: Each channel with the members of the union it documents.
_CHANNEL_UNIONS = [
    pytest.param("session-log", get_args(SessionLogRecord.__value__), id="session-log"),
    pytest.param("supervisor-log", get_args(SessionEvent), id="supervisor-log"),
]

#: Each channel's position in the pair returned by ``render_json_schemas()``.
_SCHEMA_INDEX = {"session-log": 0, "supervisor-log": 1}


def _wire_type(model: type[BaseModel]) -> str:
    return get_args(model.model_fields["type"].annotation)[0]


@pytest.mark.parametrize(("channel", "members"), _CHANNEL_UNIONS)
def test_render_asyncapi_when_called_does_list_channel_messages_in_union_order(
    channel: str,
    members: tuple[type[BaseModel], ...],
):
    doc = _render()

    messages = doc["channels"][channel]["messages"]

    expected = [_wire_type(model) for model in members]
    assert list(messages) == expected
    assert messages == {wt: {"$ref": f"#/components/messages/{wt}"} for wt in expected}


def test_render_asyncapi_when_called_does_list_component_messages_in_union_order():
    doc = _render()

    messages = doc["components"]["messages"]

    members = (*get_args(SessionLogRecord.__value__), *get_args(SessionEvent))
    assert list(messages) == [_wire_type(model) for model in members]


def test_render_asyncapi_when_model_has_no_description_does_use_class_name_as_summary():
    schemas = render_json_schemas()
    del schemas[0]["$defs"]["KeepRecord"]["description"]

    doc: dict[str, Any] = render_asyncapi(schemas)

    assert doc["components"]["messages"]["keep"]["summary"] == "KeepRecord"


@pytest.mark.parametrize(("channel", "members"), _CHANNEL_UNIONS)
def test_render_asyncapi_when_called_does_take_message_summary_from_schema_description_first_line(
    channel: str,
    members: tuple[type[BaseModel], ...],
):
    schemas, doc = _render_with_schemas()
    defs = schemas[_SCHEMA_INDEX[channel]]["$defs"]

    summaries = {
        wire_type: doc["components"]["messages"][wire_type]["summary"]
        for wire_type in doc["channels"][channel]["messages"]
    }
    assert summaries == {
        _wire_type(model): defs[model.__name__]["description"].split("\n")[0] for model in members
    }


# ---------------------------------------------------------------------------
# components.messages — one per record/event type (20 total)
# ---------------------------------------------------------------------------


def _assert_message_payload(msg: dict[str, Any], class_name: str, schema_file: str) -> None:
    assert msg["contentType"] == "application/json"
    assert isinstance(msg["title"], str)
    assert isinstance(msg["summary"], str)
    assert len(msg["summary"]) > 0
    payload = msg["payload"]
    assert payload["schemaFormat"] == "application/schema+json;version=draft-2020-12"
    assert payload["schema"]["$ref"] == f"./{schema_file}#/$defs/{class_name}"


@pytest.mark.parametrize(
    ("class_name", "wire_type"),
    [pytest.param(cls, wire, id=wire) for cls, wire in SESSION_RECORD_CLASSES.items()],
)
def test_render_asyncapi_when_called_does_have_session_record_message_with_correct_payload(
    class_name: str,
    wire_type: str,
):
    doc = _render()

    msg = doc["components"]["messages"][wire_type]

    _assert_message_payload(msg, class_name, "session-log.schema.json")


@pytest.mark.parametrize(
    ("class_name", "wire_type"),
    [pytest.param(cls, wire, id=wire) for cls, wire in SUPERVISOR_EVENT_CLASSES.items()],
)
def test_render_asyncapi_when_called_does_have_supervisor_event_message_with_correct_payload(
    class_name: str,
    wire_type: str,
):
    doc = _render()

    msg = doc["components"]["messages"][wire_type]

    _assert_message_payload(msg, class_name, "supervisor-log.schema.json")


# ---------------------------------------------------------------------------
# components.messageTraits.envelope
# ---------------------------------------------------------------------------


def test_render_asyncapi_when_called_does_have_envelope_trait():
    doc = _render()

    envelope = doc["components"]["messageTraits"]["envelope"]

    assert "description" in envelope
    assert "type" in envelope["description"]
    assert "at" in envelope["description"]
    assert "seq" in envelope["description"]


def test_render_asyncapi_when_called_does_describe_envelope_seq_as_non_negative():
    doc = _render()

    description = doc["components"]["messageTraits"]["envelope"]["description"]

    assert "`seq` (non-negative integer)" in description


def test_render_asyncapi_when_called_does_apply_envelope_trait_to_every_message():
    doc = _render()

    for wire_type, msg in doc["components"]["messages"].items():
        traits = msg.get("traits", [])
        refs = [t.get("$ref") for t in traits]
        assert "#/components/messageTraits/envelope" in refs, (
            f"message {wire_type!r} missing envelope trait"
        )


# ---------------------------------------------------------------------------
# operations — four receive operations
# ---------------------------------------------------------------------------


def _assert_receive_operation(
    doc: dict[str, Any],
    op_name: str,
    channel: str,
    expected_types: set[str],
) -> dict[str, Any]:
    op = doc["operations"][op_name]

    assert op["action"] == "receive"
    assert op["channel"]["$ref"] == f"#/channels/{channel}"
    actual_refs = {m["$ref"] for m in op["messages"]}
    expected_refs = {f"#/channels/{channel}/messages/{t}" for t in expected_types}
    assert actual_refs == expected_refs

    return op


def test_render_asyncapi_when_called_does_have_fold_session_operation():
    doc = _render()

    _assert_receive_operation(
        doc,
        "fold-session",
        "session-log",
        {"session", "iteration", "keep", "discard", "finalize", "stop"},
    )


def test_render_asyncapi_when_called_does_have_status_history_operation():
    doc = _render()

    _assert_receive_operation(
        doc,
        "status-history",
        "session-log",
        {"baseline", "iteration", "keep", "discard", "stop"},
    )


def test_render_asyncapi_when_called_does_have_supervisor_guard_operation():
    doc = _render()

    op = _assert_receive_operation(
        doc,
        "supervisor-guard",
        "session-log",
        {"session", "baseline", "iteration", "keep", "discard", "hook", "finalize", "stop"},
    )
    assert "keep" in op.get("description", "")
    assert "discard" in op.get("description", "")
    assert "progress" in op.get("description", "").lower()


def test_render_asyncapi_when_called_does_have_dashboard_operation():
    doc = _render()

    _assert_receive_operation(
        doc,
        "dashboard",
        "supervisor-log",
        set(SUPERVISOR_EVENT_CLASSES.values()),
    )


# ---------------------------------------------------------------------------
# cross-check: every operation message exists in components and its channel
# ---------------------------------------------------------------------------


def test_render_asyncapi_when_called_does_have_consistent_operation_message_refs():
    doc = _render()

    component_messages = doc["components"]["messages"]
    for op_name, op in doc["operations"].items():
        channel_ref = op["channel"]["$ref"]
        channel_name = channel_ref.split("/")[-1]
        channel_messages = doc["channels"][channel_name]["messages"]

        for msg_ref_obj in op["messages"]:
            ref = msg_ref_obj["$ref"]
            msg_type = ref.split("/")[-1]
            assert msg_type in component_messages, (
                f"operation {op_name!r} references {msg_type!r} not in components.messages"
            )
            assert msg_type in channel_messages, (
                f"operation {op_name!r} references {msg_type!r} not in channel {channel_name!r}"
            )


# ---------------------------------------------------------------------------
# meta-schema validation
# ---------------------------------------------------------------------------


def test_render_asyncapi_when_called_does_validate_against_asyncapi_300_meta_schema():
    doc = _render()

    meta_schema = json.loads(_META_SCHEMA_PATH.read_text(encoding="utf-8"))
    validator = Draft7Validator(meta_schema)

    errors = list(validator.iter_errors(doc))

    assert not errors, f"AsyncAPI meta-schema validation errors: {errors}"


def test_asyncapi_meta_schema_when_document_has_misspelled_key_does_reject():
    doc = _render()
    bad_doc = dict(doc)
    bad_doc["asyncapii"] = bad_doc.pop("asyncapi")  # cspell:disable-line

    meta_schema = json.loads(_META_SCHEMA_PATH.read_text(encoding="utf-8"))
    validator = Draft7Validator(meta_schema)

    assert not validator.is_valid(bad_doc)


# ---------------------------------------------------------------------------
# render_asyncapi_yaml — serialization and import isolation
# ---------------------------------------------------------------------------


def test_render_asyncapi_yaml_when_called_does_return_yaml_string():
    doc = _render()

    from gymrat.event_docs import render_asyncapi_yaml

    result = render_asyncapi_yaml(doc)

    assert isinstance(result, str)
    assert "asyncapi:" in result
    assert "3.0.0" in result


# ---------------------------------------------------------------------------
# READERS module-level constant
# ---------------------------------------------------------------------------


def test_readers_when_imported_does_expose_module_level_constant():
    from gymrat.event_docs import READERS

    assert isinstance(READERS, dict)
    assert "fold-session" in READERS
    assert "status-history" in READERS
    assert "supervisor-guard" in READERS
    assert "dashboard" in READERS


def test_readers_when_imported_does_have_channel_and_types_per_entry():
    from gymrat.event_docs import READERS

    for op_name, entry in READERS.items():
        assert hasattr(entry, "channel"), f"{op_name!r} entry missing 'channel'"
        assert hasattr(entry, "types"), f"{op_name!r} entry missing 'types'"
        assert isinstance(entry.channel, str)
        assert len(entry.types) > 0
