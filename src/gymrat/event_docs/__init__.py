"""Event documentation artifacts: schemas, AsyncAPI spec, and Markdown reference.

``render_all()`` returns a dict mapping repo-relative paths to rendered text
for every generated artifact.  ``write_all(root)`` writes them to disk.

The two JSON Schema documents are built here, from the pydantic models that
define the session log records and supervisor events: ``render_json_schemas()``
returns both as draft 2020-12 documents, discriminated on the ``type`` field and
carrying ``description``, ``$id``, and ``title`` metadata suitable for publishing
as standalone schema files.
"""

import json
from pathlib import Path
from typing import Any

from gymrat.event_docs.asyncapi import render_asyncapi, render_asyncapi_yaml
from gymrat.event_docs.reference import render_reference
from gymrat.session.records import SESSION_LOG_ADAPTER
from gymrat.supervisor.events import SESSION_EVENT_ADAPTER

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
