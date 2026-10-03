"""Event documentation artifacts: schemas, AsyncAPI spec, and Markdown reference.

``render_all()`` returns a dict mapping repo-relative paths to rendered text
for every generated artifact.  ``write_all(root)`` writes them to disk, and
``python -m gymrat.event_docs`` does so under the repo root, printing one line
per written file.

The two JSON Schema documents are built from the pydantic models that define
the session log records and supervisor events: ``render_json_schemas()``
returns both as draft 2020-12 documents, discriminated on the ``type`` field and
carrying ``description``, ``$id``, and ``title`` metadata suitable for publishing
as standalone schema files.

``render_asyncapi()`` builds an AsyncAPI 3.0.0 document describing the two
append-only JSONL log files gymrat writes during a session: the session log and
the supervisor log. Each channel's messages reference the sibling JSON Schema
files.

``render_reference()`` builds a human-readable Markdown document describing
every record and event type in both logs, with field tables, nested-object
subsections, and a closing Readers section.
"""

import importlib.metadata
import json
import sys
import textwrap
from pathlib import Path
from typing import Any, NamedTuple, get_args

from pydantic import BaseModel

from gymrat.errors import TOOL_FAILURE_EXIT_CODE, GymratError
from gymrat.session.paths import (
    SESSION_DIR_NAME,
    SESSION_LOG_NAME,
    repo_root,
    supervisor_log_name,
)
from gymrat.session.records import SESSION_LOG_ADAPTER, SESSION_LOG_MODELS, wire_type
from gymrat.supervisor.events import SESSION_EVENT_ADAPTER, SessionEvent

_SESSION_LOG_FILE = "./session-log.schema.json"
_SUPERVISOR_LOG_FILE = "./supervisor-log.schema.json"

SESSION_LOG_ADDRESS = f"{SESSION_DIR_NAME}/{SESSION_LOG_NAME}"
SUPERVISOR_LOG_ADDRESS = f"{SESSION_DIR_NAME}/{supervisor_log_name('<ms>')}"

_SCHEMA_FORMAT = "application/schema+json;version=draft-2020-12"
_YAML_WIDTH = 100

# ---------------------------------------------------------------------------
# Model registries and reader specs
# ---------------------------------------------------------------------------

#: Supervisor-log event models, in ``SessionEvent`` union order.
SUPERVISOR_LOG_MODELS: tuple[type[BaseModel], ...] = get_args(SessionEvent)


class ReaderSpec(NamedTuple):
    """A channel read: source channel, consumed wire-type strings, and doc sentence.

    Attributes:
        channel: The log channel the reader consumes.
        types: The wire types the reader consumes, in documentation order.
        description: What the reader does, for the Markdown reference.
        note: What the type selection means, for the AsyncAPI operation; absent
            when the selection needs no explanation.
    """

    channel: str
    types: tuple[str, ...]
    description: str
    note: str | None = None


READERS: dict[str, ReaderSpec] = {
    "fold-session": ReaderSpec(
        channel="session-log",
        types=("session", "iteration", "keep", "discard", "finalize", "stop"),
        description="Folds session records into cumulative session state.",
    ),
    "status-history": ReaderSpec(
        channel="session-log",
        types=("baseline", "iteration", "keep", "discard", "stop"),
        description="Builds an ordered history of iteration outcomes.",
    ),
    "supervisor-guard": ReaderSpec(
        channel="session-log",
        types=(
            "session",
            "baseline",
            "iteration",
            "keep",
            "discard",
            "hook",
            "finalize",
            "stop",
        ),
        description="Reads session log state for supervisor startup.",
        note=(
            "Every record type except command. "
            "keep and discard drive the discard streak "
            "and the rest count as progress."
        ),
    ),
    "dashboard": ReaderSpec(
        channel="supervisor-log",
        types=(
            "launch",
            "thinking_update",
            "tool_start",
            "tool_progress",
            "tool_end",
            "text_delta",
            "usage_update",
            "cap",
            "model_phase",
            "turn_end",
            "follow_up",
            "compaction",
        ),
        description="Streams supervisor events for live dashboard display.",
    ),
}


# ---------------------------------------------------------------------------
# Wire-type helpers
# ---------------------------------------------------------------------------


def wire_type_to_class_name(
    models: tuple[type[BaseModel], ...],
) -> dict[str, str]:
    """Map each model's wire type to its class name, in the models' order.

    Args:
        models: The union members documented by one log schema.

    Returns:
        The wire-type string of each model mapped to its class name.
    """
    return {wire_type(model): model.__name__ for model in models}


# ---------------------------------------------------------------------------
# AsyncAPI component builders
# ---------------------------------------------------------------------------


def _build_messages(
    wire_to_class: dict[str, str],
    defs: dict[str, dict[str, Any]],
    schema_file: str,
) -> dict[str, object]:
    """Build ``components.messages`` entries for one schema's message types."""
    messages: dict[str, object] = {}
    for wire, class_name in wire_to_class.items():
        messages[wire] = {
            "contentType": "application/json",
            "title": class_name,
            # A model with no docstring is summarized by its class name.
            "summary": defs[class_name].get("description", class_name).split("\n")[0],
            "payload": {
                "schemaFormat": _SCHEMA_FORMAT,
                "schema": {"$ref": f"{schema_file}#/$defs/{class_name}"},
            },
            "traits": [{"$ref": "#/components/messageTraits/envelope"}],
        }
    return messages


def _build_channel(
    wire_to_class: dict[str, str],
    address: str,
) -> dict[str, object]:
    return {
        "address": address,
        "messages": {wire: {"$ref": f"#/components/messages/{wire}"} for wire in wire_to_class},
    }


def _build_operation(reader: ReaderSpec) -> dict[str, object]:
    op: dict[str, object] = {
        "action": "receive",
        "channel": {"$ref": f"#/channels/{reader.channel}"},
        "messages": [{"$ref": f"#/channels/{reader.channel}/messages/{t}"} for t in reader.types],
    }
    if reader.note is not None:
        op["description"] = reader.note
    return op


# ---------------------------------------------------------------------------
# AsyncAPI document assembly
# ---------------------------------------------------------------------------


def render_asyncapi(
    schemas: tuple[dict[str, Any], dict[str, Any]],
) -> dict[str, object]:
    """Build an AsyncAPI 3.0.0 document from the two JSON Schema dicts.

    Args:
        schemas: The (session_log, supervisor_log) tuple returned by render_json_schemas().

    Returns:
        The AsyncAPI 3.0.0 document as a nested dict, ready for YAML
        serialization.
    """
    session_log_schema, supervisor_log_schema = schemas
    session_defs: dict[str, dict[str, Any]] = session_log_schema["$defs"]
    supervisor_defs: dict[str, dict[str, Any]] = supervisor_log_schema["$defs"]

    session_wire = wire_type_to_class_name(SESSION_LOG_MODELS)
    supervisor_wire = wire_type_to_class_name(SUPERVISOR_LOG_MODELS)

    session_messages = _build_messages(session_wire, session_defs, _SESSION_LOG_FILE)
    supervisor_messages = _build_messages(supervisor_wire, supervisor_defs, _SUPERVISOR_LOG_FILE)

    version = importlib.metadata.version("gymrat")

    return {
        "asyncapi": "3.0.0",
        "info": {
            "title": "gymrat logs",
            "version": version,
            "description": (
                "gymrat writes two append-only JSONL log files during a session: "
                f"the session log ({SESSION_LOG_ADDRESS}) records high-level "
                "session lifecycle events, and the supervisor log "
                f"({SUPERVISOR_LOG_ADDRESS}) captures fine-grained agent "
                "activity. Both files are strictly append-only; each line is a "
                "self-contained JSON object discriminated by a type field."
            ),
        },
        "channels": {
            "session-log": _build_channel(session_wire, SESSION_LOG_ADDRESS),
            "supervisor-log": _build_channel(supervisor_wire, SUPERVISOR_LOG_ADDRESS),
        },
        "operations": {name: _build_operation(reader) for name, reader in READERS.items()},
        "components": {
            "messages": {**session_messages, **supervisor_messages},
            "messageTraits": {
                "envelope": {
                    "description": (
                        "Every record and event carries `type` (string discriminator) "
                        "and `at` (integer nanoseconds since the Unix epoch). "
                        "Sequenced records also carry `seq` (positive integer)."
                    ),
                },
            },
        },
    }


def render_asyncapi_yaml(document: dict[str, object]) -> str:
    """Serialize an AsyncAPI document dict to YAML.

    Note:
        ``yaml`` is imported inside this function so that importing the
        module does not pull in PyYAML at load time.

    Args:
        document: The AsyncAPI document dict to serialize.

    Returns:
        The YAML string.
    """
    import yaml  # noqa: PLC0415

    return yaml.safe_dump(
        document,
        sort_keys=False,
        allow_unicode=True,
        width=_YAML_WIDTH,
    )


# ---------------------------------------------------------------------------
# Markdown reference
# ---------------------------------------------------------------------------

_BANNER = "<!-- Generated by `task schemas`. Do not edit. -->"

_INTRO = f"""\
gymrat writes two append-only JSONL log files during a session:
a session log at `{SESSION_LOG_ADDRESS}`
and a supervisor log at `{SUPERVISOR_LOG_ADDRESS}`,
where `<ms>` is the launch time in milliseconds since the Unix epoch.
Both files carry one JSON object per line, with snake_case keys.
The `at` field is an integer: nanoseconds since the Unix epoch.
Optional fields are absent rather than `null`;
a field marked required may still be `null` when the schema allows it.
The `schema` field appears on the first line only."""

_PROSE_WIDTH = 100

# ---------------------------------------------------------------------------
# Type and value rendering
# ---------------------------------------------------------------------------


def _type_summary(
    class_name: str,
    schema_def: dict[str, Any],
    wire_to_class: dict[str, str],
) -> str:
    description = schema_def.get("description")
    if not description:
        return class_name
    if class_name in wire_to_class.values():
        return str(description).split("\n")[0]
    return str(description)


def _ref_name(ref: str) -> str:
    return ref.rsplit("/", maxsplit=1)[-1]


def _ref_link(ref: str) -> str:
    ref_name = _ref_name(ref)
    return f"[{ref_name}](#{ref_name.lower()})"


def _render_typed_prop(type_val: str, prop: dict[str, Any]) -> str:
    """Render a property whose ``type`` key is already resolved."""
    if type_val == "array":
        items = prop.get("items", {})
        if isinstance(items, dict) and "type" in items:
            return f"array of {items['type']}"
        return "array"
    if type_val == "object":
        additional = prop.get("additionalProperties")
        if isinstance(additional, dict) and "$ref" in additional:
            return f"object (values: {_ref_link(str(additional['$ref']))})"
        return "object"
    return type_val


def _render_type(prop: dict[str, Any]) -> str:
    """Render a JSON Schema property to a human-readable type string."""
    if "$ref" in prop:
        return _ref_link(str(prop["$ref"]))

    if "const" in prop:
        return f"`{json.dumps(prop['const'])}`"

    if "enum" in prop:
        values = prop["enum"]
        return " \\| ".join(f"`{json.dumps(v)}`" for v in values)

    if "anyOf" in prop:
        return " \\| ".join(_render_type(variant) for variant in prop["anyOf"])

    type_val = prop.get("type")
    if type_val is not None:
        return _render_typed_prop(str(type_val), prop)

    return "object"


# ---------------------------------------------------------------------------
# Prose wrapping
# ---------------------------------------------------------------------------


def _wrap_line(line: str) -> str:
    """Wrap a single prose line, leaving Markdown list markers and structure lines intact."""
    if line.startswith("- "):
        return textwrap.fill(line, width=_PROSE_WIDTH, initial_indent="", subsequent_indent="  ")
    if line.startswith(("|", "#", "<!--")):
        return line
    return textwrap.fill(line, width=_PROSE_WIDTH)


def _wrap_prose(text: str) -> str:
    """Wrap prose paragraphs at the configured width."""
    paragraphs = text.split("\n\n")
    wrapped = [
        "\n".join(_wrap_line(line) for line in paragraph.split("\n")) for paragraph in paragraphs
    ]
    return "\n\n".join(wrapped)


# ---------------------------------------------------------------------------
# Field tables and nested subsections
# ---------------------------------------------------------------------------


def _build_field_table(
    schema_def: dict[str, Any],
) -> str:
    """Build a Markdown field table for one ``$defs`` entry."""
    props: dict[str, dict[str, Any]] = schema_def.get("properties", {})
    required_set: set[str] = set(schema_def.get("required", []))

    rows: list[str] = [
        "| Name | Type | Status | Description |",
        "| --- | --- | --- | --- |",
    ]

    for field_name, field_schema in props.items():
        type_str = _render_type(field_schema)
        is_const = "const" in field_schema
        status = "required" if field_name in required_set or is_const else "optional"
        description = str(field_schema.get("description", ""))
        rows.append(f"| `{field_name}` | {type_str} | {status} | {description} |")

    return "\n".join(rows)


def _nested_defs_for(
    schema_def: dict[str, Any],
    defs: dict[str, dict[str, Any]],
    wire_to_class: dict[str, str],
) -> list[str]:
    """Return ``$defs`` entries that are nested objects, not union members."""
    union_classes = set(wire_to_class.values())
    props: dict[str, dict[str, Any]] = schema_def.get("properties", {})
    candidate_refs = [
        ref_name for field_schema in props.values() for ref_name in _collect_refs(field_schema)
    ]

    nested: list[str] = []
    for ref_name in candidate_refs:
        if ref_name in defs and ref_name not in union_classes and ref_name not in nested:
            nested.append(ref_name)

    return nested


def _collect_refs(schema: dict[str, Any]) -> list[str]:
    """Collect all ``$ref`` target names from a property schema."""
    refs: list[str] = []
    if "$ref" in schema:
        refs.append(_ref_name(str(schema["$ref"])))
    if "anyOf" in schema:
        refs.extend(
            _ref_name(str(variant["$ref"])) for variant in schema["anyOf"] if "$ref" in variant
        )
    additional = schema.get("additionalProperties")
    if isinstance(additional, dict) and "$ref" in additional:
        refs.append(_ref_name(str(additional["$ref"])))
    return refs


def _render_type_block(
    heading: str,
    class_name: str,
    defs: dict[str, dict[str, Any]],
    wire_to_class: dict[str, str],
    rendered: set[str],
) -> list[str]:
    """Build a heading/doc/table block, with nested subsections appended.

    Each nested object gets one ``####`` subsection per log section, under the
    first type that refers to it, and recurses into its own nested refs.

    Args:
        heading: The Markdown heading line, including its `#` prefix.
        class_name: The schema class to render.
        defs: All schema definitions, keyed by class name.
        wire_to_class: Mapping from wire type to class name.
        rendered: Class names already rendered as nested subsections; mutated
            in place as nested subsections are appended.

    Returns:
        The heading, doc, and field-table lines, followed by any nested
        subsections.
    """
    schema_def = defs[class_name]
    doc = _type_summary(class_name, schema_def, wire_to_class)
    parts: list[str] = [heading, "", doc, "", _build_field_table(schema_def)]
    for nested in _nested_defs_for(schema_def, defs, wire_to_class):
        if nested not in rendered:
            rendered.add(nested)
            parts.append("")
            parts.extend(
                _render_type_block(f"#### `{nested}`", nested, defs, wire_to_class, rendered)
            )
    return parts


# ---------------------------------------------------------------------------
# Log-section rendering
# ---------------------------------------------------------------------------


def _render_log_section(
    heading: str,
    models: tuple[type[BaseModel], ...],
    defs: dict[str, dict[str, Any]],
) -> str:
    """Render a ## log section with ### subsections per wire type, in the models' order."""
    wire_to_class = wire_type_to_class_name(models)
    parts: list[str] = [f"## {heading}", ""]
    rendered_nested: set[str] = set()

    for wire, class_name in wire_to_class.items():
        block = _render_type_block(
            f"### `{wire}`", class_name, defs, wire_to_class, rendered_nested
        )
        parts.extend(block)
        parts.append("")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Readers section
# ---------------------------------------------------------------------------


def _render_readers_section() -> str:
    """Render the ## Readers section listing operations and their types."""
    parts: list[str] = [
        "## Readers",
        "",
        "Each reader operation consumes a subset of record or event types from one log file.",
        "",
    ]

    for op_name, spec in READERS.items():
        types_str = ", ".join(f"`{t}`" for t in spec.types)
        parts.append(
            f"- **{op_name}** (channel: `{spec.channel}`): {spec.description} Types: {types_str}"
        )

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------


def render_reference(schemas: tuple[dict[str, Any], dict[str, Any]]) -> str:
    """Build a Markdown reference document from JSON schemas and the reader specs.

    Args:
        schemas: The (session_log, supervisor_log) tuple returned by render_json_schemas().

    Returns:
        The rendered Markdown string with word-wrapped prose.
    """
    session_log, supervisor_log = schemas
    session_defs: dict[str, dict[str, Any]] = session_log.get("$defs", {})
    supervisor_defs: dict[str, dict[str, Any]] = supervisor_log.get("$defs", {})

    sections: list[str] = [
        _BANNER,
        "",
        "# Event Reference",
        "",
        _INTRO,
        "",
        _render_log_section("Session Log", SESSION_LOG_MODELS, session_defs),
        _render_log_section("Supervisor Log", SUPERVISOR_LOG_MODELS, supervisor_defs),
        _render_readers_section(),
    ]

    raw = "\n".join(sections)
    wrapped = _wrap_prose(raw)

    result = "\n".join(line.rstrip() for line in wrapped.split("\n"))
    return result.rstrip("\n") + "\n"


# ---------------------------------------------------------------------------
# JSON Schemas and the artifact set
# ---------------------------------------------------------------------------

_SESSION_LOG_TITLE = "gymrat session log record"
_SESSION_LOG_ID = "https://github.com/jeffzi/gymrat/schemas/session-log.schema.json"

_SUPERVISOR_LOG_TITLE = "gymrat supervisor log event"
_SUPERVISOR_LOG_ID = "https://github.com/jeffzi/gymrat/schemas/supervisor-log.schema.json"

_DRAFT_2020_12 = "https://json-schema.org/draft/2020-12/schema"


def _stamp(schema: dict[str, Any], title: str, schema_id: str) -> dict[str, Any]:
    """Add draft, title, and ``$id`` metadata to a generated JSON Schema dict, in place."""
    schema["$schema"] = _DRAFT_2020_12
    schema["title"] = title
    schema["$id"] = schema_id
    return schema


def render_json_schemas() -> tuple[dict[str, Any], dict[str, Any]]:
    """Render the session-log and supervisor-log JSON Schemas.

    Returns:
        A ``(session_log_schema, supervisor_log_schema)`` tuple: the session-log
        schema first, the supervisor-log schema second, both JSON-serializable
        draft 2020-12 JSON Schema documents.
    """
    session_log = _stamp(SESSION_LOG_ADAPTER.json_schema(), _SESSION_LOG_TITLE, _SESSION_LOG_ID)
    supervisor_log = _stamp(
        SESSION_EVENT_ADAPTER.json_schema(), _SUPERVISOR_LOG_TITLE, _SUPERVISOR_LOG_ID
    )
    return session_log, supervisor_log


def render_all() -> dict[str, str]:
    """Return a dict mapping repo-relative path to rendered text for each artifact."""
    schemas = render_json_schemas()
    session_log, supervisor_log = (
        json.dumps(schema, indent=2, sort_keys=True) + "\n" for schema in schemas
    )
    return {
        "schemas/session-log.schema.json": session_log,
        "schemas/supervisor-log.schema.json": supervisor_log,
        "schemas/asyncapi.yaml": render_asyncapi_yaml(render_asyncapi(schemas)),
        "docs/event-reference.md": render_reference(schemas),
    }


def write_all(root: str | Path) -> list[Path]:
    """Write every artifact under ``root``, creating directories as needed.

    Args:
        root: The repository root the artifact paths are relative to.

    Returns:
        The written paths, in artifact order.
    """
    root = Path(root)
    artifacts = render_all()
    paths: list[Path] = []
    for rel_path, content in artifacts.items():
        path = root / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        paths.append(path)
    return paths


def main() -> None:
    """Write artifacts to the repo root and print each path.

    Outside a git repository the error and its hint go to stderr and the
    process exits with ``TOOL_FAILURE_EXIT_CODE`` before anything is written.
    """
    try:
        root = repo_root()
    except GymratError as exc:
        print(exc, file=sys.stderr)  # noqa: T201 — CLI entry point
        if exc.hint is not None:
            print(exc.hint, file=sys.stderr)  # noqa: T201 — CLI entry point
        sys.exit(TOOL_FAILURE_EXIT_CODE)
    paths = write_all(root)
    for path in paths:
        print(path)  # noqa: T201 — CLI entry point


if __name__ == "__main__":
    main()
