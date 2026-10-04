"""AsyncAPI 3.0.0 document generation for gymrat log files.

Builds an AsyncAPI document describing the two append-only JSONL log files
gymrat writes during a session: the session log and the supervisor log.
Each channel's messages reference the sibling JSON Schema files produced
by ``render_json_schemas()``.
"""

import importlib.metadata
from typing import Any, NamedTuple, get_args

from pydantic import BaseModel

from gymrat.session.paths import SESSION_DIR_NAME, SESSION_LOG_NAME, supervisor_log_name
from gymrat.session.records import SESSION_LOG_MODELS, wire_type
from gymrat.supervisor.events import SessionEvent

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
    """A channel read: source channel, consumed wire-type strings, and doc sentence."""

    channel: str
    types: tuple[str, ...]
    description: str


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
# Wire-type and summary helpers
# ---------------------------------------------------------------------------


def _wire_type_to_class_name(
    models: tuple[type[BaseModel], ...],
) -> dict[str, str]:
    """Map each model's wire type to its class name, in the models' order.

    Args:
        models: The union members documented by one log schema.

    Returns:
        The wire-type string of each model mapped to its class name.
    """
    return {wire_type(model): model.__name__ for model in models}


def _summary_line(defs: dict[str, dict[str, Any]], class_name: str) -> str:
    """First line of a top-level model's schema description, or its class name.

    Args:
        defs: The schema's ``$defs``, keyed by class name.
        class_name: The model class whose description to summarize.

    Returns:
        The first line of the description, or ``class_name`` when the schema
        carries none.
    """
    return defs[class_name].get("description", class_name).split("\n")[0]


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
            "summary": _summary_line(defs, class_name),
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
        "messages": {
            wire_type: {"$ref": f"#/components/messages/{wire_type}"} for wire_type in wire_to_class
        },
    }


def _build_operation(
    reader: ReaderSpec,
    *,
    description: str | None = None,
) -> dict[str, object]:
    op: dict[str, object] = {
        "action": "receive",
        "channel": {"$ref": f"#/channels/{reader.channel}"},
        "messages": [{"$ref": f"#/channels/{reader.channel}/messages/{t}"} for t in reader.types],
    }
    if description is not None:
        op["description"] = description
    return op


def _build_components(
    session_messages: dict[str, object],
    supervisor_messages: dict[str, object],
) -> dict[str, object]:
    return {
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
    }


# ---------------------------------------------------------------------------
# Top-level document assembly
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

    session_wire = _wire_type_to_class_name(SESSION_LOG_MODELS)
    supervisor_wire = _wire_type_to_class_name(SUPERVISOR_LOG_MODELS)

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
                "the session log (.gymrat/session.jsonl) records high-level "
                "session lifecycle events, and the supervisor log "
                "(.gymrat/supervisor-<ms>.jsonl) captures fine-grained agent "
                "activity. Both files are strictly append-only; each line is a "
                "self-contained JSON object discriminated by a type field."
            ),
        },
        "channels": {
            "session-log": _build_channel(session_wire, SESSION_LOG_ADDRESS),
            "supervisor-log": _build_channel(supervisor_wire, SUPERVISOR_LOG_ADDRESS),
        },
        "operations": {
            "fold-session": _build_operation(READERS["fold-session"]),
            "status-history": _build_operation(READERS["status-history"]),
            "supervisor-guard": _build_operation(
                READERS["supervisor-guard"],
                description=(
                    "Every record type except command. "
                    "keep and discard drive the discard streak "
                    "and the rest count as progress."
                ),
            ),
            "dashboard": _build_operation(READERS["dashboard"]),
        },
        "components": _build_components(session_messages, supervisor_messages),
    }


# ---------------------------------------------------------------------------
# YAML serialization
# ---------------------------------------------------------------------------


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
