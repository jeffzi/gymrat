"""JSON Schema generation for session-log and supervisor-log event documentation.

Builds two draft 2020-12 JSON Schema documents from the pydantic models that
define the session log records and supervisor events.  The schemas are
discriminated on the ``type`` field and carry ``description``, ``$id``, and
``title`` metadata suitable for publishing as standalone schema files.
"""

from typing import Annotated, Any

from pydantic import Field, TypeAdapter

from gymrat.session.records.models import SessionLogRecord
from gymrat.supervisor.events import session_event_adapter

_SESSION_LOG_TITLE = "gymrat session log record"
_SESSION_LOG_ID = "https://github.com/jeffzi/gymrat/schemas/session-log.schema.json"

_SUPERVISOR_LOG_TITLE = "gymrat supervisor log event"
_SUPERVISOR_LOG_ID = "https://github.com/jeffzi/gymrat/schemas/supervisor-log.schema.json"

_DRAFT_2020_12 = "https://json-schema.org/draft/2020-12/schema"

_SessionLogAdapter: TypeAdapter[SessionLogRecord] = TypeAdapter(
    Annotated[SessionLogRecord, Field(discriminator="type")]
)


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
    session_log = _stamp(_SessionLogAdapter.json_schema(), _SESSION_LOG_TITLE, _SESSION_LOG_ID)
    supervisor_log = _stamp(
        session_event_adapter().json_schema(), _SUPERVISOR_LOG_TITLE, _SUPERVISOR_LOG_ID
    )
    return session_log, supervisor_log
