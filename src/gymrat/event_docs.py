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

from pydantic import BaseModel, TypeAdapter

from gymrat.errors import TOOL_FAILURE_EXIT_CODE, GymratError
from gymrat.session.paths import (
    SESSION_DIR_NAME,
    SESSION_LOG_NAME,
    repo_root,
    supervisor_log_name,
)
from gymrat.session.records import SESSION_LOG_ADAPTER, SESSION_LOG_MODELS, wire_type
from gymrat.supervisor.events import SESSION_EVENT_ADAPTER, SessionEvent
from gymrat.utils import first_line

SESSION_LOG_ADDRESS = f"{SESSION_DIR_NAME}/{SESSION_LOG_NAME}"
SUPERVISOR_LOG_ADDRESS = f"{SESSION_DIR_NAME}/{supervisor_log_name('<ms>')}"

_SCHEMA_FORMAT = "application/schema+json;version=draft-2020-12"
_SCHEMA_BASE_URL = "https://github.com/jeffzi/gymrat/schemas/"
_DRAFT_2020_12 = "https://json-schema.org/draft/2020-12/schema"
_YAML_WIDTH = 100

# ---------------------------------------------------------------------------
# Model registries and reader specs
# ---------------------------------------------------------------------------


class _LogSpec(NamedTuple):
    """One JSONL log the artifacts document.

    Attributes:
        channel: The AsyncAPI channel name, also the stem of the schema file.
        heading: The log's section heading in the Markdown reference.
        title: The JSON Schema ``title``.
        address: Where the log lives, relative to the repository root.
        models: The union members, in documentation order.
        adapter: The adapter that renders the log's JSON Schema.
    """

    channel: str
    heading: str
    title: str
    address: str
    models: tuple[type[BaseModel], ...]
    adapter: TypeAdapter[Any]

    @property
    def schema_file(self) -> str:
        """The schema file name, relative to the ``schemas/`` directory."""
        return f"{self.channel}.schema.json"

    @property
    def wire_to_class(self) -> dict[str, str]:
        """Each model's wire type mapped to its class name, in the models' order."""
        return {wire_type(model): model.__name__ for model in self.models}


#: The documented logs, session log first; every artifact lists them in this order.
_LOGS: tuple[_LogSpec, _LogSpec] = (
    _LogSpec(
        channel="session-log",
        heading="Session Log",
        title="gymrat session log record",
        address=SESSION_LOG_ADDRESS,
        models=SESSION_LOG_MODELS,
        adapter=SESSION_LOG_ADAPTER,
    ),
    _LogSpec(
        channel="supervisor-log",
        heading="Supervisor Log",
        title="gymrat supervisor log event",
        address=SUPERVISOR_LOG_ADDRESS,
        # Supervisor-log event models, in ``SessionEvent`` union order.
        models=get_args(SessionEvent),
        adapter=SESSION_EVENT_ADAPTER,
    ),
)


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


def _summary(class_name: str, schema_def: dict[str, Any], *, first_line_only: bool) -> str:
    # A model with no docstring is summarized by its class name.
    description = schema_def.get("description")
    if not description:
        return class_name
    text = str(description)
    return first_line(text) if first_line_only else text


# ---------------------------------------------------------------------------
# AsyncAPI component builders
# ---------------------------------------------------------------------------


def _build_messages(
    wire_to_class: dict[str, str],
    defs: dict[str, dict[str, Any]],
    schema_file: str,
) -> dict[str, object]:
    """Build ``components.messages`` entries for one schema's message types."""
    return {
        wire: {
            "contentType": "application/json",
            "title": class_name,
            "summary": _summary(class_name, defs[class_name], first_line_only=True),
            "payload": {
                "schemaFormat": _SCHEMA_FORMAT,
                "schema": {"$ref": f"./{schema_file}#/$defs/{class_name}"},
            },
            "traits": [{"$ref": "#/components/messageTraits/envelope"}],
        }
        for wire, class_name in wire_to_class.items()
    }


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
    wire_maps = [log.wire_to_class for log in _LOGS]
    messages: dict[str, object] = {}
    for log, schema, wire_to_class in zip(_LOGS, schemas, wire_maps, strict=True):
        messages.update(_build_messages(wire_to_class, schema["$defs"], log.schema_file))

    return {
        "asyncapi": "3.0.0",
        "info": {
            "title": "gymrat logs",
            "version": importlib.metadata.version("gymrat"),
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
            log.channel: _build_channel(wire_to_class, log.address)
            for log, wire_to_class in zip(_LOGS, wire_maps, strict=True)
        },
        "operations": {name: _build_operation(reader) for name, reader in READERS.items()},
        "components": {
            "messages": messages,
            "messageTraits": {
                "envelope": {
                    "description": (
                        "Every record and event carries `type` (string discriminator) "
                        "and `at` (integer nanoseconds since the Unix epoch). "
                        "Sequenced records also carry `seq` (non-negative integer)."
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
        return " \\| ".join(f"`{json.dumps(v)}`" for v in prop["enum"])

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
        return textwrap.fill(line, width=_PROSE_WIDTH, subsequent_indent="  ")
    if line.startswith(("|", "#", "<!--")):
        return line
    return textwrap.fill(line, width=_PROSE_WIDTH)


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
    rendered: set[str],
    *,
    first_line_only: bool,
) -> list[str]:
    """Build a heading/doc/table block, with nested subsections appended.

    Each nested object gets one ``####`` subsection per log section, under the
    first type that refers to it, and recurses into its own nested refs.

    Args:
        heading: The Markdown heading line, including its `#` prefix.
        class_name: The schema class to render.
        defs: All schema definitions, keyed by class name.
        rendered: Class names that get no nested subsection: the log's union
            members and the nested objects already rendered. Mutated in place
            as nested subsections are appended.
        first_line_only: Whether the doc keeps only the first line of the
            class description, as a union member's does.

    Returns:
        The heading, doc, and field-table lines, followed by any nested
        subsections.
    """
    schema_def = defs[class_name]
    doc = _summary(class_name, schema_def, first_line_only=first_line_only)
    parts: list[str] = [heading, "", doc, "", _build_field_table(schema_def)]
    props: dict[str, dict[str, Any]] = schema_def.get("properties", {})
    for field_schema in props.values():
        for nested in _collect_refs(field_schema):
            if nested in defs and nested not in rendered:
                rendered.add(nested)
                parts += [
                    "",
                    *_render_type_block(
                        f"#### `{nested}`", nested, defs, rendered, first_line_only=False
                    ),
                ]
    return parts


# ---------------------------------------------------------------------------
# Log-section rendering
# ---------------------------------------------------------------------------


def _render_log_section(log: _LogSpec, defs: dict[str, dict[str, Any]]) -> str:
    """Render a ## log section with ### subsections per wire type, in the models' order."""
    wire_to_class = log.wire_to_class
    parts: list[str] = [f"## {log.heading}", ""]
    rendered: set[str] = set(wire_to_class.values())

    for wire, class_name in wire_to_class.items():
        parts += [
            *_render_type_block(f"### `{wire}`", class_name, defs, rendered, first_line_only=True),
            "",
        ]

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
    sections: list[str] = [
        _BANNER,
        "",
        "# Event Reference",
        "",
        _INTRO,
        "",
        *(
            _render_log_section(log, schema.get("$defs", {}))
            for log, schema in zip(_LOGS, schemas, strict=True)
        ),
        _render_readers_section(),
    ]

    raw = "\n".join(sections)
    return "\n".join(_wrap_line(line) for line in raw.split("\n")) + "\n"


# ---------------------------------------------------------------------------
# JSON Schemas and the artifact set
# ---------------------------------------------------------------------------


def _require_discriminator(schema_def: dict[str, Any]) -> None:
    # A defaulted ``type`` field is optional to pydantic, but every reader dispatches on it
    # and rejects an object without it, so the published schema must reject it too.
    required = {*schema_def.get("required", []), "type"}
    schema_def["required"] = [name for name in schema_def["properties"] if name in required]


def _json_schema(log: _LogSpec) -> dict[str, Any]:
    """Render one log's JSON Schema with its draft, title, and ``$id`` metadata."""
    schema = log.adapter.json_schema()
    for model in log.models:
        _require_discriminator(schema["$defs"][model.__name__])
    schema["$schema"] = _DRAFT_2020_12
    schema["title"] = log.title
    schema["$id"] = f"{_SCHEMA_BASE_URL}{log.schema_file}"
    return schema


def render_json_schemas() -> tuple[dict[str, Any], dict[str, Any]]:
    """Render the session-log and supervisor-log JSON Schemas.

    Returns:
        A ``(session_log_schema, supervisor_log_schema)`` tuple: the session-log
        schema first, the supervisor-log schema second, both JSON-serializable
        draft 2020-12 JSON Schema documents.
    """
    session_log, supervisor_log = _LOGS
    return _json_schema(session_log), _json_schema(supervisor_log)


def render_all() -> dict[str, str]:
    """Return a dict mapping repo-relative path to rendered text for each artifact."""
    schemas = render_json_schemas()
    return {
        **{
            f"schemas/{log.schema_file}": json.dumps(schema, indent=2, sort_keys=True) + "\n"
            for log, schema in zip(_LOGS, schemas, strict=True)
        },
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
