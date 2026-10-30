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
"""

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, get_args

import pytest
from jsonschema import Draft7Validator, Draft202012Validator
from pydantic import BaseModel

from gymrat.errors import TOOL_FAILURE_EXIT_CODE
from gymrat.event_docs import (
    READERS,
    ReaderSpec,
    main,
    render_all,
    render_asyncapi,
    render_json_schemas,
    render_reference,
    write_all,
)
from gymrat.session.paths import repository_lookup_error
from gymrat.session.records import IterationRecord, SessionLogRecord
from gymrat.supervisor.events import (
    LaunchEvent,
    SessionEvent,
    ToolEndEvent,
    event_from_wire,
)
from tests._ansi import normalize
from tests._cli import run_module
from tests._git import refuse_git_lookups
from tests._imports import loaded_under, modules_imported_by
from tests._markdown import md_section

_REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True, slots=True)
class _Channel:
    schema_index: int
    members: tuple[type[BaseModel], ...]


#: The session-log channel: ``SessionLogRecord`` members, first schema of ``render_json_schemas()``.
_SESSION_LOG = _Channel(0, get_args(SessionLogRecord.__value__))
#: The supervisor-log channel: ``SessionEvent`` members, second schema of ``render_json_schemas()``.
_SUPERVISOR_LOG = _Channel(1, get_args(SessionEvent))
_CHANNELS = (_SESSION_LOG, _SUPERVISOR_LOG)


def _wire_type(model: type[BaseModel]) -> str:
    return get_args(model.model_fields["type"].annotation)[0]


# ---------------------------------------------------------------------------
# write_all
# ---------------------------------------------------------------------------


def test_write_all_when_called_does_write_each_artifact_nonempty_at_its_path(tmp_path: Path):
    paths = write_all(str(tmp_path))

    assert sorted(paths) == sorted(tmp_path / rel for rel in render_all())
    assert [path for path in paths if path.stat().st_size == 0] == []


# ---------------------------------------------------------------------------
# __main__
# ---------------------------------------------------------------------------


def test_main_module_when_run_does_print_each_written_path(
    create_scratch_repo: Callable[[], str],
):
    root = Path(create_scratch_repo())

    result = run_module("gymrat.event_docs", cwd=root)

    assert result.returncode == 0, f"stderr: {result.stderr}"
    written = [Path(line).resolve() for line in result.stdout.splitlines()]
    assert sorted(written) == sorted((root / rel).resolve() for rel in render_all())


# ---------------------------------------------------------------------------
# drift gate
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("rel_path", sorted(render_all()))
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
# type required on every event — the schema rejects what event_from_wire rejects
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param({"at": 1}, id="missing-type"),
        pytest.param({"type": "compaction", "at": 1}, id="compaction"),
    ],
)
def test_render_json_schemas_when_supervisor_payload_validated_does_agree_with_event_from_wire(
    payload: dict[str, Any],
):
    supervisor_log = render_json_schemas()[_SUPERVISOR_LOG.schema_index]
    validator = Draft202012Validator(supervisor_log)

    schema_accepts = validator.is_valid(payload)

    assert schema_accepts == (event_from_wire(payload) is not None)


# ---------------------------------------------------------------------------
# required fields kept — a field without a default stays required
# ---------------------------------------------------------------------------


def test_render_json_schemas_when_field_has_no_default_does_keep_it_required():
    tool_end = render_json_schemas()[_SUPERVISOR_LOG.schema_index]["$defs"][ToolEndEvent.__name__]

    assert "tool_name" in tool_end["required"]


# ---------------------------------------------------------------------------
# render_asyncapi — helpers
# ---------------------------------------------------------------------------


def _render_with_schemas() -> tuple[tuple[dict[str, Any], dict[str, Any]], dict[str, Any]]:
    schemas = render_json_schemas()
    return schemas, render_asyncapi(schemas)


def _render_asyncapi_doc() -> dict[str, Any]:
    return _render_with_schemas()[1]


# ---------------------------------------------------------------------------
# channels — session-log and supervisor-log
# ---------------------------------------------------------------------------


def test_render_asyncapi_when_called_does_list_each_channel_in_union_order():
    doc = _render_asyncapi_doc()

    channels = [list(spec["messages"].items()) for spec in doc["channels"].values()]
    assert channels == [
        [(wt, {"$ref": f"#/components/messages/{wt}"}) for wt in map(_wire_type, channel.members)]
        for channel in _CHANNELS
    ]


def test_render_asyncapi_when_called_does_list_component_messages_in_union_order():
    expected = [_wire_type(model) for channel in _CHANNELS for model in channel.members]

    doc = _render_asyncapi_doc()

    assert list(doc["components"]["messages"]) == expected


def test_render_asyncapi_when_model_has_no_description_does_use_class_name_as_summary():
    schemas = render_json_schemas()
    del schemas[_SESSION_LOG.schema_index]["$defs"]["KeepRecord"]["description"]

    doc: dict[str, Any] = render_asyncapi(schemas)

    assert doc["components"]["messages"]["keep"]["summary"] == "KeepRecord"


# ---------------------------------------------------------------------------
# components.messages — one per record/event type
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("channel", "model"),
    [
        pytest.param(_SESSION_LOG, IterationRecord, id="session-log"),
        pytest.param(_SUPERVISOR_LOG, LaunchEvent, id="supervisor-log"),
    ],
)
def test_render_asyncapi_when_called_does_describe_each_message_payload(
    channel: _Channel,
    model: type[BaseModel],
):
    schemas, doc = _render_with_schemas()

    msg = doc["components"]["messages"][_wire_type(model)]

    description = schemas[channel.schema_index]["$defs"][model.__name__]["description"]
    schema_file, fragment = msg["payload"]["schema"]["$ref"].removeprefix("./").split("#")
    assert msg["title"] == model.__name__
    assert msg["summary"] == description.split("\n")[0]
    assert fragment == f"/$defs/{model.__name__}"
    assert model.__name__ in json.loads(render_all()[f"schemas/{schema_file}"])["$defs"]


# ---------------------------------------------------------------------------
# operations — one receive operation per reader, each referencing messages that exist
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("op_name", "reader"),
    [
        pytest.param(name, READERS[name], id=name)
        for name in ("supervisor-guard", "dashboard")  # a reader with a note, one without
    ],
)
def test_render_asyncapi_when_called_does_have_receive_operation_per_reader(
    op_name: str, reader: ReaderSpec
):
    doc = _render_asyncapi_doc()

    op = doc["operations"][op_name]
    assert op["channel"]["$ref"] == f"#/channels/{reader.channel}"
    assert op["messages"] == [
        {"$ref": f"#/channels/{reader.channel}/messages/{t}"} for t in reader.types
    ]
    assert op.get("description") == reader.note


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
# document framing — single trailing newline
# ---------------------------------------------------------------------------


def test_render_reference_when_called_does_end_with_a_single_newline():
    md = _render_reference_doc()

    assert md.endswith("\n")
    assert not md.endswith("\n\n")


# ---------------------------------------------------------------------------
# per-record/event ### sections — one per union member, in declared order
# ---------------------------------------------------------------------------


def test_render_reference_when_called_does_order_sections_per_union_declaration():
    md = _render_reference_doc()

    positions = [md.find(f"\n### `{_wire_type(model)}`\n") for model in _SESSION_LOG.members]
    assert -1 not in positions
    assert positions == sorted(positions)


# ---------------------------------------------------------------------------
# field tables — iteration as the representative type; columns: name, type,
# required/optional, description
# ---------------------------------------------------------------------------


def _field_rows(section: str) -> dict[str, tuple[str, str]]:
    """Map each field row's name to its type and required/optional columns."""
    cells = (line.split(" | ") for line in section.split("\n") if line.startswith("| `"))
    return {
        name.removeprefix("| `").removesuffix("`"): (type_cell, status)
        for name, type_cell, status, *_ in cells
    }


def test_render_reference_when_called_does_render_each_field_row():
    statuses = {"seq": "required", "confirm": "optional"}
    verdict = "IterationMetricVerdict"
    types = {
        "confirm": "[Confirm](#confirm)",
        "metrics": f"object (values: [{verdict}](#{verdict.lower()}))",
        "outcome": '`"improved"` \\| `"regressed"` \\| `"no-signal"`',
    }

    md = _render_reference_doc()

    rows = _field_rows(_type_section(md, "iteration"))
    assert {field: rows[field][1] for field in statuses} == statuses
    assert {field: rows[field][0] for field in types} == types


# ---------------------------------------------------------------------------
# nested object subsections
# ---------------------------------------------------------------------------


def test_render_reference_when_called_does_have_nested_object_subsections():
    nested_objects = ["SessionHooks", "IterationMetricVerdict"]

    md = _render_reference_doc()

    headings = set(md.split("\n"))
    missing = [name for name in nested_objects if f"#### `{name}`" not in headings]
    assert missing == []


# ---------------------------------------------------------------------------
# closed objects — sections state that unlisted keys are rejected
# ---------------------------------------------------------------------------

#: The line a section carries when its schema sets ``additionalProperties: false``.
_CLOSED_NOTE = "Rejects keys not listed above."


def _carries_closed_note(md: str, heading: str) -> bool:
    # Stop at the next heading of any level so a nested section's note stays out.
    return _CLOSED_NOTE in md_section(md, f"\n{heading}\n", "\n#").split("\n")


@pytest.mark.parametrize(
    ("channel", "definition", "closed"),
    [
        pytest.param(_SESSION_LOG, "BaselineRef", False, id="opened-baseline-ref"),
        pytest.param(_SUPERVISOR_LOG, "DirtyInfo", True, id="closed-dirty-info"),
    ],
)
def test_render_reference_when_schema_closure_changes_does_follow_the_schema(
    channel: _Channel, definition: str, closed: bool
):
    schemas = render_json_schemas()
    object_schema = schemas[channel.schema_index]["$defs"][definition]
    object_schema.pop("additionalProperties", None)
    if closed:
        object_schema["additionalProperties"] = False

    md = render_reference(schemas)

    assert _carries_closed_note(md, f"#### `{definition}`") is closed


# ---------------------------------------------------------------------------
# readers section
# ---------------------------------------------------------------------------


def test_render_reference_when_called_does_describe_a_reader_operation():
    spec = READERS["dashboard"]

    md = _render_reference_doc()

    bullet = normalize(md_section(md, "\n- **dashboard**", r"\n- |\n#"))
    assert spec.channel in bullet
    assert [t for t in spec.types if f"`{t}`" not in bullet] == []


# ---------------------------------------------------------------------------
# Entry point isolation
# ---------------------------------------------------------------------------


def test_import_event_docs_when_loaded_does_not_import_the_cli_package_or_yaml():
    loaded = modules_imported_by("gymrat.event_docs")

    assert loaded_under(loaded, "gymrat.cli", "yaml") == []


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
    cause = refuse_git_lookups(monkeypatch, git_stderr)
    error = repository_lookup_error(str(Path.cwd()), cause)
    lines = (str(error), error.hint)
    expected_stderr = "".join(f"{line}\n" for line in lines if line is not None)

    with pytest.raises(SystemExit) as exc_info:
        main()

    assert exc_info.value.code == TOOL_FAILURE_EXIT_CODE
    assert capsys.readouterr() == ("", expected_stderr)
    assert [rel for rel in sorted(render_all()) if (tmp_path / rel).exists()] == []
