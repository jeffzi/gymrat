"""Session event vocabulary and the helpers that render and fan them out.

A session emits a fixed set of nine events to any number of
:data:`SessionObserver` callbacks. Each event is a frozen pydantic model tagged
by a ``type`` literal and stamped with an epoch-millisecond ``timestamp``;
together they form the :data:`SessionEvent` union.

:func:`to_json_line` renders an event to a single compact JSON line with
camelCase keys — the shared wire form the event log and the stdio driver both
write. :func:`summarize` and :func:`summarize_input` produce the compact,
single-line summaries carried on tool events. :func:`combine_observers` fans one
event out to several observers in order.
"""

import json
import os
import re
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SerializerFunctionWrapHandler,
    TypeAdapter,
    ValidationError,
    model_serializer,
)
from pydantic.alias_generators import to_camel

from gymrat.finite_json import null_non_finite

# ---------------------------------------------------------------------------
# Event vocabulary
# ---------------------------------------------------------------------------

# Maximum code-point length for a session-event summary before it is truncated.
SUMMARY_MAX_CHARS = 200

# json.dumps separators for the wire's no-padding compact form: `{"key":"value"}`.
_COMPACT_JSON_SEPARATORS = (",", ":")


class _EventModel(BaseModel):
    """Shared config for the event vocabulary: frozen, camelCase wire aliases."""

    model_config = ConfigDict(
        frozen=True,
        populate_by_name=True,
        alias_generator=to_camel,
    )


class DirtyInfo(_EventModel):
    """The dirty-worktree provenance carried on a launch event."""

    file_count: int


class ThinkingUpdateEvent(_EventModel):
    """Emitted as the model's extended-thinking token estimate changes mid-turn."""

    type: Literal["thinking_update"] = "thinking_update"
    timestamp: int
    estimated_tokens: int
    delta: int
    parent_tool_use_id: str | None = None


class ToolStartEvent(_EventModel):
    """Emitted when the model invokes a tool."""

    type: Literal["tool_start"] = "tool_start"
    timestamp: int
    tool_use_id: str
    tool_name: str
    input: object
    input_summary: str
    parent_tool_use_id: str | None = None


class ToolProgressEvent(_EventModel):
    """Emitted periodically while a long-running tool call is still in flight."""

    type: Literal["tool_progress"] = "tool_progress"
    timestamp: int
    tool_use_id: str
    elapsed_ms: int


class ToolEndEvent(_EventModel):
    """Emitted when a tool call completes and its result is available."""

    type: Literal["tool_end"] = "tool_end"
    timestamp: int
    tool_use_id: str
    tool_name: str
    duration_ms: int
    result: str
    result_summary: str
    parent_tool_use_id: str | None = None


class TextDeltaEvent(_EventModel):
    """Emitted for each chunk of assistant text as it streams in."""

    type: Literal["text_delta"] = "text_delta"
    timestamp: int
    chunk: str
    parent_tool_use_id: str | None = None


class UsageUpdateEvent(_EventModel):
    """Emitted when the driver observes updated cumulative cost.

    ``settled`` marks a usage update carried by a result message that has
    already settled the session on its own; a spend-cap observer must not
    treat it as a live crossing of the cap, since the session is ending
    regardless. Omitted from the wire form when ``False`` (the common case),
    matching the shipped log format.
    """

    type: Literal["usage_update"] = "usage_update"
    timestamp: int
    cost_usd: float
    settled: bool = False

    @model_serializer(mode="wrap")
    def _omit_unsettled(self, handler: SerializerFunctionWrapHandler) -> dict[str, object]:
        data: dict[str, object] = handler(self)
        if not self.settled:
            data.pop("settled", None)
        return data


class CapEvent(_EventModel):
    """Emitted when a supervision cap (wall-clock or spend) fires."""

    type: Literal["cap"] = "cap"
    timestamp: int
    cap: Literal["wall-clock", "spend-cap"]


class ModelPhaseEvent(_EventModel):
    """Emitted when the model transitions between processing phases within a turn."""

    type: Literal["model_phase"] = "model_phase"
    timestamp: int
    phase: Literal["thinking", "responding", "tool_input", "turn_end"]
    tool_name: str | None = None
    parent_tool_use_id: str | None = None


class LaunchEvent(_EventModel):
    """Written by the supervisor as the log's first line: launch provenance.

    ``max_usd`` and ``model`` are omitted from the wire form when ``None``
    (not serialized as ``null`` — absent from the dict), matching the shipped
    log format.
    """

    type: Literal["launch"] = "launch"
    timestamp: int
    head_sha: str
    dirty: Literal[False] | DirtyInfo
    max_minutes: float
    max_usd: float | None = None
    model: str | None = None
    runbook_path: str
    kickoff_summary: str

    @model_serializer(mode="wrap")
    def _omit_none_optionals(
        self,
        handler: SerializerFunctionWrapHandler,
    ) -> dict[str, object]:
        data: dict[str, object] = handler(self)
        if self.max_usd is None:
            # Pop both spellings: the handler's own by_alias setting decides which
            # one is present in `data`, and callers dump both ways (see tests).
            data.pop(to_camel("max_usd"), None)
            data.pop("max_usd", None)
        if self.model is None:
            data.pop("model", None)
        return data


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
)
"""The union of every event a session can emit to a :data:`SessionObserver`."""

SessionObserver = Callable[[SessionEvent], None]
"""Receives :data:`SessionEvent`s as a session streams them."""

_SessionEventAdapter: TypeAdapter[SessionEvent] = TypeAdapter(
    Annotated[SessionEvent, Field(discriminator="type")]
)


# ---------------------------------------------------------------------------
# Wire serialization and dispatch
# ---------------------------------------------------------------------------


def to_json_line(event: SessionEvent) -> str:
    """Serialize an event to a single compact JSON line with camelCase keys."""
    wire = event.model_dump(mode="python", by_alias=True)
    return json.dumps(null_non_finite(wire), separators=_COMPACT_JSON_SEPARATORS, default=str)


def event_from_wire(obj: object) -> SessionEvent | None:
    """Reconstruct a session event from its camelCase wire object.

    The inverse of :func:`to_json_line`'s rendering: given a decoded JSON object,
    return the matching event model. Returns ``None`` when ``obj`` is not a
    dict, carries no recognized ``type``, or is missing a required field or has
    one with the wrong type.
    """
    if not isinstance(obj, dict):
        return None
    try:
        return _SessionEventAdapter.validate_python(obj)
    except ValidationError:
        return None


def combine_observers(*observers: SessionObserver) -> SessionObserver:
    """Fan one event out to each observer in order with the identical object.

    With no observers the result is a no-op. If an observer raises, a
    :class:`RuntimeWarning` is emitted and later observers still run.
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
    """
    if supervised_root is not None:
        try:
            rel = os.path.relpath(path, supervised_root)
        except ValueError:
            pass
        else:
            if not rel.startswith(".."):
                return rel.replace(os.sep, "/")

    try:
        return "~/" + Path(path).relative_to(Path.home()).as_posix()
    except (ValueError, RuntimeError):
        pass

    return path


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
