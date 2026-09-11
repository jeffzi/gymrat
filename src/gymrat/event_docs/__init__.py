"""Event documentation artifacts: schemas, AsyncAPI spec, and Markdown reference.

``render_all()`` returns a dict mapping repo-relative paths to rendered text
for every generated artifact.  ``write_all(root)`` writes them to disk.
"""

import json
from pathlib import Path

from gymrat.event_docs.asyncapi import READERS, render_asyncapi, render_asyncapi_yaml
from gymrat.event_docs.json_schema import render_json_schemas
from gymrat.event_docs.reference import render_reference

ARTIFACT_PATHS: dict[str, str] = {
    "session_log_schema": "schemas/session-log.schema.json",
    "supervisor_log_schema": "schemas/supervisor-log.schema.json",
    "asyncapi": "schemas/asyncapi.yaml",
    "reference": "docs/event-reference.md",
}


def render_all() -> dict[str, str]:
    """Return a dict mapping repo-relative path to rendered text for each artifact."""
    schemas = render_json_schemas()
    session_log, supervisor_log = schemas
    asyncapi_doc = render_asyncapi(schemas)

    return {
        ARTIFACT_PATHS["session_log_schema"]: json.dumps(session_log, indent=2, sort_keys=True)
        + "\n",
        ARTIFACT_PATHS["supervisor_log_schema"]: json.dumps(
            supervisor_log, indent=2, sort_keys=True
        )
        + "\n",
        ARTIFACT_PATHS["asyncapi"]: render_asyncapi_yaml(asyncapi_doc),
        ARTIFACT_PATHS["reference"]: render_reference(schemas, READERS),
    }


def write_all(root: str | Path) -> list[Path]:
    """Write every artifact under *root*, creating directories as needed."""
    root = Path(root)
    artifacts = render_all()
    paths: list[Path] = []
    for rel_path, content in artifacts.items():
        path = root / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        paths.append(path)
    return paths
