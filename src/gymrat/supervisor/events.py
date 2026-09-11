"""Session event vocabulary and the helpers that render and fan them out.

A session emits a fixed set of twelve events to any number of
:data:`SessionObserver` callbacks. Each event is a frozen pydantic model that
inherits ``_EventModel`` — carrying a per-event ``Literal`` ``type``
discriminator and ``at: int`` (nanoseconds since epoch) — together forming
the :data:`SessionEvent` union.

:func:`to_json_line` renders an event to a single compact JSON line with
snake_case keys and ``exclude_none=True`` — the shared wire form the event log
and the stdio driver both write. :func:`summarize` and :func:`summarize_input`
produce the compact, single-line summaries carried on tool events.
:func:`combine_observers` fans one event out to several observers in order.
"""

import json
import os
import re
import warnings
from collections.abc import Callable
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError
from pydantic.json_schema import SkipJsonSchema

from gymrat.config.types import Effort
from gymrat.finite_json import null_non_finite
from gymrat.paths import abbreviate_home

# ---------------------------------------------------------------------------
# Event vocabulary
# ---------------------------------------------------------------------------

# Maximum code-point length for a session-event summary before it is truncated.
SUMMARY_MAX_CHARS = 200

# json.dumps separators for the wire's no-padding compact form: `{"key":"value"}`.
_COMPACT_JSON_SEPARATORS = (",", ":")

# Repeated optional-field unions, shared across event fields below. Each field
# still builds its own `Field(...)` so pydantic's per-model `FieldInfo` is unaffected.
_OptStr = str | SkipJsonSchema[None]
_OptFloat = float | SkipJsonSchema[None]
_OptEffort = Effort | SkipJsonSchema[None]

# Description shared by every `parent_tool_use_id` field below.
_PARENT_TOOL_USE_ID_DESCRIPTION = "Tool use ID of the enclosing tool call, if any."


class _EventModel(BaseModel):
    """Shared config for the event vocabulary: frozen, snake_case wire."""

    model_config = ConfigDict(frozen=True, validate_by_name=True, serialize_by_alias=True)


class DirtyInfo(_EventModel):
    """The dirty-worktree provenance carried on a launch event."""

    file_count: int = Field(description="Number of dirty files in the worktree.")


class ThinkingUpdateEvent(_EventModel):
    """Emitted as the model's extended-thinking token estimate changes mid-turn."""

    type: Literal["thinking_update"] = Field(
        "thinking_update", description="Event type discriminator."
    )
    at: int = Field(description="Nanoseconds since the Unix epoch when the event was created.")
    estimated_tokens: int = Field(description="Cumulative estimated thinking tokens so far.")
    delta: int = Field(description="Token count change since the last thinking update.")
    parent_tool_use_id: _OptStr = Field(default=None, description=_PARENT_TOOL_USE_ID_DESCRIPTION)


class ToolStartEvent(_EventModel):
    """Emitted when the model invokes a tool."""

    type: Literal["tool_start"] = Field("tool_start", description="Event type discriminator.")
    at: int = Field(description="Nanoseconds since the Unix epoch when the event was created.")
    tool_use_id: str = Field(description="Unique identifier for this tool invocation.")
    tool_name: str = Field(description="Name of the tool being invoked.")
    input: object = Field(description="Raw input passed to the tool.")
    input_summary: str = Field(description="Human-readable summary of the tool input.")
    parent_tool_use_id: _OptStr = Field(default=None, description=_PARENT_TOOL_USE_ID_DESCRIPTION)


class ToolProgressEvent(_EventModel):
    """Emitted periodically while a long-running tool call is still in flight."""

    type: Literal["tool_progress"] = Field("tool_progress", description="Event type discriminator.")
    at: int = Field(description="Nanoseconds since the Unix epoch when the event was created.")
    tool_use_id: str = Field(description="Unique identifier of the in-flight tool invocation.")
    elapsed_ms: int = Field(description="Milliseconds elapsed since the tool call started.")


class ToolEndEvent(_EventModel):
    """Emitted when a tool call completes and its result is available."""

    type: Literal["tool_end"] = Field("tool_end", description="Event type discriminator.")
    at: int = Field(description="Nanoseconds since the Unix epoch when the event was created.")
    tool_use_id: str = Field(description="Unique identifier of the completed tool invocation.")
    tool_name: str = Field(description="Name of the tool that completed.")
    duration_ms: int = Field(description="Wall-clock milliseconds the tool call took.")
    result: str = Field(description="Raw result returned by the tool.")
    result_summary: str = Field(description="Human-readable summary of the tool result.")
    parent_tool_use_id: _OptStr = Field(default=None, description=_PARENT_TOOL_USE_ID_DESCRIPTION)


class TextDeltaEvent(_EventModel):
    """Emitted for each chunk of assistant text as it streams in."""

    type: Literal["text_delta"] = Field("text_delta", description="Event type discriminator.")
    at: int = Field(description="Nanoseconds since the Unix epoch when the event was created.")
    chunk: str = Field(description="Text chunk streamed from the assistant.")
    parent_tool_use_id: _OptStr = Field(default=None, description=_PARENT_TOOL_USE_ID_DESCRIPTION)


class UsageUpdateEvent(_EventModel):
    """Emitted when the driver observes updated cumulative cost.

    ``settled`` marks a usage update carried by a result message that has
    already settled the session on its own; a spend-cap observer must not
    treat it as a live crossing of the cap, since the session is ending
    regardless. Always written to the wire (``"settled": false`` on an
    unsettled update).
    """

    type: Literal["usage_update"] = Field("usage_update", description="Event type discriminator.")
    at: int = Field(description="Nanoseconds since the Unix epoch when the event was created.")
    cost_usd: float = Field(description="Cumulative session cost in US dollars.")
    settled: bool = Field(default=False, description="Whether the session has already settled.")


class CapEvent(_EventModel):
    """Emitted when a supervision cap (wall-clock or spend) fires."""

    type: Literal["cap"] = Field("cap", description="Event type discriminator.")
    at: int = Field(description="Nanoseconds since the Unix epoch when the event was created.")
    cap: Literal["wall-clock", "spend-cap"] = Field(description="Which supervision cap fired.")


class ModelPhaseEvent(_EventModel):
    """Emitted when the model transitions between processing phases within a turn."""

    type: Literal["model_phase"] = Field("model_phase", description="Event type discriminator.")
    at: int = Field(description="Nanoseconds since the Unix epoch when the event was created.")
    phase: Literal["thinking", "responding", "tool_input", "turn_end"] = Field(
        description="Model processing phase the turn entered."
    )
    tool_name: _OptStr = Field(default=None, description="Tool name when the phase is tool_input.")
    parent_tool_use_id: _OptStr = Field(default=None, description=_PARENT_TOOL_USE_ID_DESCRIPTION)


class LaunchEvent(_EventModel):
    """Written by the supervisor as the log's first line: launch provenance.

    ``max_usd``, ``model``, and ``effort`` are ``None``-optional and omitted
    from the wire form via the global ``exclude_none=True`` in
    :func:`to_json_line`.
    """

    type: Literal["launch"] = Field("launch", description="Event type discriminator.")
    at: int = Field(description="Nanoseconds since the Unix epoch when the event was created.")
    schema_version: Literal[1] = Field(alias="schema", description="Supervisor log format version.")
    session_id: str = Field(description="Unique identifier for this session.")

    head_sha: str = Field(description="Git HEAD commit SHA at launch time.")
    dirty: Literal[False] | DirtyInfo = Field(
        description="False when the worktree is clean, or dirty-file details."
    )
    max_minutes: float = Field(description="Wall-clock timeout in minutes.")
    max_usd: _OptFloat = Field(default=None, description="Spend cap in US dollars, if configured.")
    model: _OptStr = Field(default=None, description="Model identifier, if specified.")
    effort: _OptEffort = Field(default=None, description="Effort level, if specified.")
    runbook_path: str = Field(description="Path to the runbook file.")
    kickoff_summary: str = Field(description="One-line summary of the kickoff prompt.")


class TurnEndEvent(_EventModel):
    """Emitted when the agent finishes a conversational turn."""

    type: Literal["turn_end"] = Field("turn_end", description="Event type discriminator.")
    at: int = Field(description="Nanoseconds since the Unix epoch when the event was created.")
    text: str = Field(description="Full text the agent produced in this turn.")
    cost_usd: float = Field(description="Cost of this turn in US dollars.")
    origin: Literal["agent", "injected"] = Field(
        description="Whether the turn was agent-generated or injected by the supervisor."
    )
    budget_exhausted: bool = Field(description="Whether the turn exhausted the remaining budget.")


class FollowUpEvent(_EventModel):
    """Emitted when the supervisor acts on a completed turn.

    ``reason`` and ``text`` are ``None``-optional and omitted from the wire
    form via the global ``exclude_none=True`` in :func:`to_json_line`.
    """

    type: Literal["follow_up"] = Field("follow_up", description="Event type discriminator.")
    at: int = Field(description="Nanoseconds since the Unix epoch when the event was created.")
    action: Literal["replied", "waiting", "ended"] = Field(
        description="Supervisor action taken after the turn."
    )
    reason: _OptStr = Field(default=None, description="Reason for the action, if applicable.")
    text: _OptStr = Field(default=None, description="Follow-up text sent to the agent, if any.")


class CompactionEvent(_EventModel):
    """Emitted when the agent SDK reports a context compaction boundary."""

    type: Literal["compaction"] = Field("compaction", description="Event type discriminator.")
    at: int = Field(description="Nanoseconds since the Unix epoch when the event was created.")


SessionEvent = (
    ThinkingUpdateEvent
    | ToolStartEvent
    | ToolProgressEvent
    | ToolEndEvent
    | TextDeltaEvent
    | UsageUpdateEvent
    | CapEvent
    | ModelPhaseEvent
    | LaunchEvent
    | TurnEndEvent
    | FollowUpEvent
    | CompactionEvent
)
"""The union of every event a session can emit to a :data:`SessionObserver`."""

SessionObserver = Callable[[SessionEvent], None]
"""Receives :data:`SessionEvent`s as a session streams them."""

_SessionEventAdapter: TypeAdapter[SessionEvent] = TypeAdapter(
    Annotated[SessionEvent, Field(discriminator="type")]
)


def session_event_adapter() -> TypeAdapter[SessionEvent]:
    """Expose the module-private ``TypeAdapter`` without making it public."""
    return _SessionEventAdapter


# ---------------------------------------------------------------------------
# Wire serialization and dispatch
# ---------------------------------------------------------------------------

# The wire key for LaunchEvent.schema_version, derived from its alias so the
# non-launch-event guard below can never drift from the field it mirrors.
_SCHEMA_KEY = LaunchEvent.model_fields["schema_version"].alias


def to_json_line(event: SessionEvent) -> str:
    """Serialize an event to a single compact JSON line with snake_case keys.

    Args:
        event: The session event to serialize.

    Returns:
        A single JSON line with no trailing newline.
    """
    wire = event.model_dump(mode="python", exclude_none=True)
    # exclude_none drops every None, but required fields whose domain includes
    # None (e.g. ToolStartEvent.input) must stay on the wire as null.
    for name, info in type(event).model_fields.items():
        if getattr(event, name) is None and info.default is not None:
            wire[info.serialization_alias or info.alias or name] = None
    return json.dumps(null_non_finite(wire), separators=_COMPACT_JSON_SEPARATORS, default=str)


def event_from_wire(obj: object) -> SessionEvent | None:
    """Reconstruct a session event from its snake_case wire object.

    The inverse of :func:`to_json_line`'s rendering: given a decoded JSON object,
    return the matching event model. Returns ``None`` when ``obj`` is not a
    dict, carries no recognized ``type``, is missing a required field, has one
    with the wrong type, or carries the schema key (:data:`_SCHEMA_KEY`) on an
    event other than ``launch``.

    Args:
        obj: The decoded JSON object to reconstruct into an event.

    Returns:
        The deserialized event, or ``None`` when the object is unrecognized or
        invalid.
    """
    if not isinstance(obj, dict):
        return None
    if obj.get("type") != "launch" and _SCHEMA_KEY in obj:
        return None
    try:
        return session_event_adapter().validate_python(obj)
    except ValidationError:
        return None


def combine_observers(*observers: SessionObserver) -> SessionObserver:
    """Fan one event out to each observer in order with the identical object.

    With no observers the result is a no-op. If an observer raises, a
    :class:`RuntimeWarning` is emitted and later observers still run.

    Args:
        *observers: The observers to fan each event out to, in call order.

    Returns:
        A combined observer that dispatches to all given observers.
    """

    def combined(event: SessionEvent) -> None:
        for observer in observers:
            try:
                observer(event)
            except Exception as error:  # noqa: BLE001 - observer failure must not break the chain
                warnings.warn(str(error), RuntimeWarning, stacklevel=2)

    return combined


# ---------------------------------------------------------------------------
# summarize
# ---------------------------------------------------------------------------

_WHITESPACE_RUN = re.compile(r"\s+")


def summarize(text: str, max_chars: int = SUMMARY_MAX_CHARS) -> str:
    """Produce a compact, single-line summary of ``text``.

    Whitespace runs collapse to single spaces and the ends are trimmed. When the
    collapsed text fits within ``max_chars`` code points it is returned as-is;
    otherwise it is cut on a code-point boundary and suffixed with a bare ``…``.

    Args:
        text: The text to collapse and possibly truncate.
        max_chars: The code-point budget before truncation kicks in.

    Returns:
        The collapsed and possibly truncated single-line summary.
    """
    collapsed = _WHITESPACE_RUN.sub(" ", text).strip()

    if len(collapsed) <= max_chars:
        return collapsed

    return f"{collapsed[:max_chars]}…"


# ---------------------------------------------------------------------------
# summarize_input
# ---------------------------------------------------------------------------


# Tool names whose input carries a file path as the primary summary value.
_FILE_PATH_TOOLS: dict[str, str] = {
    "Read": "file_path",
    "Edit": "file_path",
    "Write": "file_path",
    "NotebookEdit": "notebook_path",
}


def _render_path(path: str, supervised_root: str | None) -> str:
    """Render a file path for display.

    Paths under the supervised root render relative to it; paths under the
    user's home directory render ``~``-prefixed; anything else renders verbatim.
    The supervised root takes priority when a path falls under both.  All
    returned paths use forward slashes for consistent cross-platform display.

    Args:
        path: The file path to render.
        supervised_root: The supervised root directory, or ``None`` when no
            root is configured.

    Returns:
        The display-ready path string.
    """
    if supervised_root is not None:
        try:
            rel = os.path.relpath(path, supervised_root)
        except ValueError:
            pass
        else:
            if not rel.startswith(".."):
                return rel.replace(os.sep, "/")

    return abbreviate_home(path)


def summarize_input(
    value: object,
    max_chars: int = SUMMARY_MAX_CHARS,
    *,
    tool_name: str | None = None,
    supervised_root: str | None = None,
) -> str:
    """Summarize a tool-call input for display.

    When ``tool_name`` identifies a known tool, the summary extracts the
    human-relevant field (file path, command, or subagent type + description)
    rather than dumping the entire input as JSON.

    Falls back to compact JSON when the tool is unknown or the expected field is
    absent. A value ``json.dumps`` cannot encode falls back to ``str(value)``.

    Args:
        value: The raw tool-call input to summarize.
        max_chars: The code-point budget before truncation kicks in.
        tool_name: The tool identifier, used to pick a field-specific extractor.
        supervised_root: The root a file-path summary should render relative to.

    Returns:
        A concise, single-line summary of the tool input.
    """
    if isinstance(value, dict) and tool_name is not None:
        extracted = _extract_tool_summary(value, tool_name, supervised_root)
        if extracted is not None:
            return summarize(extracted, max_chars)

    try:
        encoded = json.dumps(value, separators=_COMPACT_JSON_SEPARATORS)
    except (TypeError, ValueError):
        return summarize(str(value), max_chars)
    return summarize(encoded, max_chars)


def _extract_tool_summary(
    input_dict: dict[str, object],
    tool_name: str,
    supervised_root: str | None,
) -> str | None:
    """Extract a human-readable summary from a tool input dict, or ``None``."""
    path_key = _FILE_PATH_TOOLS.get(tool_name)
    if path_key is not None:
        path = input_dict.get(path_key)
        return _render_path(path, supervised_root) if isinstance(path, str) else None

    if tool_name == "Bash":
        command = input_dict.get("command")
        return command if isinstance(command, str) else None

    if tool_name in ("Agent", "Task"):
        description = input_dict.get("description")
        if not isinstance(description, str):
            return None
        agent_type = input_dict.get("subagent_type") or input_dict.get("type")
        prefix = f"{agent_type}: " if isinstance(agent_type, str) else ""
        return f"{prefix}{description}"

    if tool_name == "Skill":
        skill = input_dict.get("skill")
        if isinstance(skill, str):
            args = input_dict.get("args")
            return f"{skill} {args}" if isinstance(args, str) else skill

    return None
