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

_REF_TEMPLATE = "#/$defs/{model}"

_SessionLogAdapter: TypeAdapter[SessionLogRecord] = TypeAdapter(
    Annotated[SessionLogRecord, Field(discriminator="type")]
)


def render_json_schemas() -> tuple[dict[str, Any], dict[str, Any]]:
    """Return (session_log_schema, supervisor_log_schema) as JSON-serializable dicts."""
    session_log = _SessionLogAdapter.json_schema(mode="validation", ref_template=_REF_TEMPLATE)
    session_log["$schema"] = _DRAFT_2020_12
    session_log["title"] = _SESSION_LOG_TITLE
    session_log["$id"] = _SESSION_LOG_ID

    event_adapter = session_event_adapter()
    supervisor_log = event_adapter.json_schema(mode="validation", ref_template=_REF_TEMPLATE)
    supervisor_log["$schema"] = _DRAFT_2020_12
    supervisor_log["title"] = _SUPERVISOR_LOG_TITLE
    supervisor_log["$id"] = _SUPERVISOR_LOG_ID

    return session_log, supervisor_log
