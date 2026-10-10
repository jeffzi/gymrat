"""Session event vocabulary and the helpers that render and fan them out.

A session emits a fixed set of eleven events to any number of
:data:`SessionObserver` callbacks. Each event is a frozen pydantic model that
inherits ``_EventModel`` — carrying a per-event ``Literal`` ``type``
discriminator and ``at: int`` (nanoseconds since epoch) — together forming
the :data:`SessionEvent` union.

:func:`to_json_line` renders an event to a single compact JSON line with
snake_case keys, leaving unset optional fields off — the wire form the event
log writes. :func:`summarize` and
:func:`summarize_input` produce the compact, single-line summaries carried on
tool events.
:func:`combine_observers` fans one event out to several observers in order.
:func:`create_event_log_writer` returns the observer that appends each event to
the event log in that wire form, and :func:`probe_event_log_path` checks the log
is writable before a session starts.
"""

import json
import math
import os
import re
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, TypeAdapter, ValidationError
from pydantic.json_schema import SkipJsonSchema
from pydantic_core import PydanticSerializationError

from gymrat.config import Effort
from gymrat.errors import GymratError
from gymrat.supervisor.tool_names import ITERATE_TOOL, PROBE_TOOL
from gymrat.utils import UNICODE_LINE_BREAKS, abbreviate_home, fan_out

# ---------------------------------------------------------------------------
# Event vocabulary
# ---------------------------------------------------------------------------

# Maximum code-point length for a session-event summary before it is truncated.
SUMMARY_MAX_CHARS = 200

# Input summary for an iterate call, shared with the dashboard frame's match
# against a Bash-invoked `gymrat iterate`.
ITERATE_SUMMARY = "gymrat iterate"

# json.dumps separators for the wire's no-padding compact form: `{"key":"value"}`.
_COMPACT_JSON_SEPARATORS = (",", ":")


def _is_none(value: object) -> bool:
    return value is None


# Repeated optional-field unions, shared across event fields below. An unset
# (None) value is left off the wire; a required field whose domain includes None
# (ToolStartEvent.input) does not use these and stays on the wire as null. Each
# field still builds its own `Field(...)`, which pydantic merges with this one.
_OMIT_NONE = Field(exclude_if=_is_none)
_OptStr = Annotated[str | SkipJsonSchema[None], _OMIT_NONE]
_OptFloat = Annotated[FiniteFloat | SkipJsonSchema[None], _OMIT_NONE]
_OptEffort = Annotated[Effort | SkipJsonSchema[None], _OMIT_NONE]

_At = Annotated[
    int, Field(description="Nanoseconds since the Unix epoch when the event was created.")
]

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
    at: _At
    estimated_tokens: int = Field(description="Cumulative estimated thinking tokens so far.")
    delta: int = Field(description="Token count change since the last thinking update.")
    parent_tool_use_id: _OptStr = Field(default=None, description=_PARENT_TOOL_USE_ID_DESCRIPTION)


class ToolStartEvent(_EventModel):
    """Emitted when the model invokes a tool."""

    type: Literal["tool_start"] = Field("tool_start", description="Event type discriminator.")
    at: _At
    tool_use_id: str = Field(description="Unique identifier for this tool invocation.")
    tool_name: str = Field(description="Name of the tool being invoked.")
    input: object = Field(description="Raw input passed to the tool.")
    input_summary: str = Field(description="Human-readable summary of the tool input.")
    parent_tool_use_id: _OptStr = Field(default=None, description=_PARENT_TOOL_USE_ID_DESCRIPTION)


class ToolEndEvent(_EventModel):
    """Emitted when a tool call completes and its result is available."""

    type: Literal["tool_end"] = Field("tool_end", description="Event type discriminator.")
    at: _At
    tool_use_id: str = Field(description="Unique identifier of the completed tool invocation.")
    tool_name: str = Field(description="Name of the tool that completed.")
    duration_ms: int = Field(description="Wall-clock milliseconds the tool call took.")
    result: str = Field(description="Raw result returned by the tool.")
    result_summary: str = Field(description="Human-readable summary of the tool result.")
    parent_tool_use_id: _OptStr = Field(default=None, description=_PARENT_TOOL_USE_ID_DESCRIPTION)


class TextDeltaEvent(_EventModel):
    """Emitted for each chunk of assistant text as it streams in."""

    type: Literal["text_delta"] = Field("text_delta", description="Event type discriminator.")
    at: _At
    chunk: str = Field(description="Text chunk streamed from the assistant.")
    parent_tool_use_id: _OptStr = Field(default=None, description=_PARENT_TOOL_USE_ID_DESCRIPTION)


class UsageUpdateEvent(_EventModel):
    """Emitted when the driver observes updated cumulative cost."""

    type: Literal["usage_update"] = Field("usage_update", description="Event type discriminator.")
    at: _At
    cost_usd: FiniteFloat = Field(description="Cumulative session cost in US dollars.")


CapType = Literal["wall-clock", "spend-cap"]
"""The cap variety that ended or is ending a supervised run."""

CapAction = Literal["ending", "interrupting"]
"""Whether the cap is ending the idle session or interrupting an in-flight turn."""


class CapEvent(_EventModel):
    """Emitted when a supervision cap (wall-clock or spend) fires."""

    type: Literal["cap"] = Field("cap", description="Event type discriminator.")
    at: _At
    cap: CapType = Field(description="Which supervision cap fired.")
    action: CapAction = Field(
        description=(
            "Whether the supervisor is ending the idle session or interrupting an in-flight turn."
        ),
    )


ModelPhase = Literal["thinking", "responding", "tool_input", "turn_end"]
"""The processing phases a turn moves through."""


class ModelPhaseEvent(_EventModel):
    """Emitted when the model transitions between processing phases within a turn."""

    type: Literal["model_phase"] = Field("model_phase", description="Event type discriminator.")
    at: _At
    phase: ModelPhase = Field(description="Model processing phase the turn entered.")
    tool_name: _OptStr = Field(default=None, description="Tool name when the phase is tool_input.")
    parent_tool_use_id: _OptStr = Field(default=None, description=_PARENT_TOOL_USE_ID_DESCRIPTION)


class LaunchEvent(_EventModel):
    """Written by the supervisor as the log's first line: launch provenance.

    ``max_usd``, ``model``, and ``effort`` are ``None``-optional and omitted
    from the wire form when unset.
    """

    type: Literal["launch"] = Field("launch", description="Event type discriminator.")
    at: _At
    schema_version: Literal[1] = Field(alias="schema", description="Supervisor log format version.")
    session_id: str = Field(description="Unique identifier for this session.")

    head_sha: str = Field(description="Git HEAD commit SHA at launch time.")
    dirty: Literal[False] | DirtyInfo = Field(
        description="False when the worktree is clean, or dirty-file details."
    )
    max_minutes: FiniteFloat = Field(description="Wall-clock timeout in minutes.")
    max_usd: _OptFloat = Field(default=None, description="Spend cap in US dollars, if configured.")
    model: _OptStr = Field(default=None, description="Model identifier, if specified.")
    effort: _OptEffort = Field(default=None, description="Effort level, if specified.")
    runbook_path: str = Field(description="Path to the runbook file.")
    kickoff_summary: str = Field(description="One-line summary of the kickoff prompt.")


class TurnEndEvent(_EventModel):
    """Emitted when the agent finishes a conversational turn."""

    type: Literal["turn_end"] = Field("turn_end", description="Event type discriminator.")
    at: _At
    text: str = Field(description="Full text the agent produced in this turn.")
    cost_usd: FiniteFloat = Field(
        description="Cumulative session cost in US dollars when the turn ended."
    )
    origin: Literal["agent", "injected"] = Field(
        description="Whether the turn was agent-generated or injected by the supervisor."
    )
    budget_exhausted: bool = Field(description="Whether the turn exhausted the remaining budget.")


class FollowUpEvent(_EventModel):
    """Emitted when the supervisor acts on a completed turn.

    ``reason`` and ``text`` are ``None``-optional and omitted from the wire
    form when unset.
    """

    type: Literal["follow_up"] = Field("follow_up", description="Event type discriminator.")
    at: _At
    action: Literal["replied", "waiting", "ended"] = Field(
        description="Supervisor action taken after the turn."
    )
    reason: _OptStr = Field(default=None, description="Reason for the action, if applicable.")
    text: _OptStr = Field(default=None, description="Follow-up text sent to the agent, if any.")


class CompactionEvent(_EventModel):
    """Emitted when the agent SDK reports a context compaction boundary."""

    type: Literal["compaction"] = Field("compaction", description="Event type discriminator.")
    at: _At


SessionEvent = (
    ThinkingUpdateEvent
    | ToolStartEvent
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

SESSION_EVENT_ADAPTER: TypeAdapter[SessionEvent] = TypeAdapter(
    Annotated[SessionEvent, Field(discriminator="type")]
)
"""Validates a wire object into its :data:`SessionEvent` and renders the union's JSON Schema."""


# ---------------------------------------------------------------------------
# Wire serialization and dispatch
# ---------------------------------------------------------------------------

# The wire key for LaunchEvent.schema_version, derived from its alias so the
# non-launch-event guard below can never drift from the field it mirrors.
_SCHEMA_KEY = LaunchEvent.model_fields["schema_version"].alias


def to_json_line(event: SessionEvent) -> str:
    """Serialize an event to a single compact JSON line with snake_case keys.

    Non-ASCII text is written raw except U+0085, U+2028 and U+2029, which are
    written as JSON unicode escapes so the event never spans several lines. A NaN
    or infinite float nested in a free-form payload such as
    ``ToolStartEvent.input`` is written as ``null`` (the typed float fields
    refuse non-finite values at construction), and a value pydantic cannot
    serialize is written as its ``str()``. An event holding a lone surrogate,
    which cannot be encoded as UTF-8, is written with every non-ASCII character
    escaped.

    Args:
        event: The session event to serialize.

    Returns:
        A single JSON line with no trailing newline.
    """
    try:
        return event.model_dump_json(fallback=str).translate(UNICODE_LINE_BREAKS)
    except PydanticSerializationError:
        return json.dumps(
            _null_non_finite(event.model_dump(mode="json", fallback=str)),
            separators=_COMPACT_JSON_SEPARATORS,
            ensure_ascii=True,
            allow_nan=False,
        )


def _null_non_finite(value: object) -> object:
    # Mirrors pydantic's JSON mode, which writes NaN and infinities as null: the
    # strict log decoder rejects the bare NaN/Infinity literals json.dumps emits.
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _null_non_finite(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_null_non_finite(item) for item in value]
    return value


def event_from_wire(obj: object) -> SessionEvent | None:
    """Reconstruct a session event from its snake_case wire object.

    The inverse of :func:`to_json_line`'s rendering.

    Args:
        obj: The decoded JSON object to reconstruct into an event.

    Returns:
        The matching event model, or ``None`` when ``obj`` is not a dict,
        carries no recognized ``type``, is missing a required field, has one
        with the wrong type, or carries the schema key (:data:`_SCHEMA_KEY`) on
        an event other than ``launch``.
    """
    if not isinstance(obj, dict):
        return None
    if obj.get("type") != "launch" and _SCHEMA_KEY in obj:
        return None
    try:
        return SESSION_EVENT_ADAPTER.validate_python(obj)
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
    return fan_out(observers, _warn_observer_failure)


def _warn_observer_failure(error: Exception) -> None:
    # stacklevel=3 skips this sink and fan_out's dispatch loop, attributing the
    # warning to whoever called the combined observer.
    warnings.warn(str(error), RuntimeWarning, stacklevel=3)


# ---------------------------------------------------------------------------
# Event log
# ---------------------------------------------------------------------------


def probe_event_log_path(log_path: str | Path) -> None:
    """Verify ``log_path`` is writable before a session starts.

    Attempts to create the parent directory and open the file for appending, so
    the command can fail up front rather than after the session.

    Args:
        log_path: The event log path to verify.

    Raises:
        GymratError: Naming the path, when the path or its parent directory is
            not writable.
    """
    path = Path(log_path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8"):
            pass
    except OSError as error:
        message = f"Event log path is not writable: {path}"
        raise GymratError(message) from error


def create_event_log_writer(log_path: str | Path) -> SessionObserver:
    """Return a :data:`SessionObserver` that appends each event to ``log_path``.

    Each event is one JSON line ending in a bare line feed on every platform,
    as in the session log; Windows never gets a carriage return added. The
    parent directory is created (recursively) on the first write if it does not
    already exist. A write failure surfaces as a :class:`GymratError` naming the
    log path, chaining the underlying OS error as its cause.

    Args:
        log_path: The event log path to append to.

    Returns:
        An observer callback that appends each event as a JSON line.
    """
    path = Path(log_path)

    def write(event: SessionEvent) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8", newline="\n") as log:
                log.write(to_json_line(event) + "\n")
        except OSError as error:
            message = f"Failed to write event log: {path}"
            raise GymratError(message) from error

    return write


# ---------------------------------------------------------------------------
# Tool-event summaries
# ---------------------------------------------------------------------------

_WHITESPACE_RUN = re.compile(r"\s+")


def summarize(text: str) -> str:
    """Produce a compact, single-line summary of ``text``.

    Whitespace runs collapse to single spaces and the ends are trimmed. When the
    collapsed text fits within ``SUMMARY_MAX_CHARS`` code points it is returned
    as-is; otherwise it is cut on a code-point boundary and suffixed with a bare
    ``…``.

    Args:
        text: The text to collapse and possibly truncate.

    Returns:
        The collapsed and possibly truncated single-line summary.
    """
    collapsed = _WHITESPACE_RUN.sub(" ", text).strip()

    if len(collapsed) <= SUMMARY_MAX_CHARS:
        return collapsed

    return f"{collapsed[:SUMMARY_MAX_CHARS]}…"


# Tool names whose input carries a file path as the primary summary value.
FILE_PATH_TOOLS: dict[str, str] = {
    "Read": "file_path",
    "Edit": "file_path",
    "Write": "file_path",
    "MultiEdit": "file_path",
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
            if rel != os.pardir and not rel.startswith(os.pardir + os.sep):
                return rel.replace(os.sep, "/")

    return abbreviate_home(path)


def summarize_input(
    value: object,
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
        tool_name: The tool identifier, used to pick a field-specific extractor.
        supervised_root: The root a file-path summary should render relative to.

    Returns:
        A concise, single-line summary of the tool input.
    """
    if isinstance(value, dict) and tool_name is not None:
        extracted = _extract_tool_summary(value, tool_name, supervised_root)
        if extracted is not None:
            return summarize(extracted)

    try:
        encoded = json.dumps(value, separators=_COMPACT_JSON_SEPARATORS)
    except (TypeError, ValueError):
        return summarize(str(value))
    return summarize(encoded)


def _extract_tool_summary(
    input_dict: dict[str, object],
    tool_name: str,
    supervised_root: str | None,
) -> str | None:
    """Extract a human-readable summary from a tool input dict, or ``None``."""
    path_key = FILE_PATH_TOOLS.get(tool_name)
    if path_key is not None:
        path = input_dict.get(path_key)
        return _render_path(path, supervised_root) if isinstance(path, str) else None
    return _extract_non_path_summary(input_dict, tool_name)


def _extract_non_path_summary(
    input_dict: dict[str, object],
    tool_name: str,
) -> str | None:
    """Summarize tools that don't use a file-path key."""
    if tool_name == "Bash":
        command = input_dict.get("command")
        return command if isinstance(command, str) else None
    if tool_name in ("Agent", "Task"):
        description = input_dict.get("description")
        if isinstance(description, str):
            agent_type = input_dict.get("subagent_type") or input_dict.get("type")
            prefix = f"{agent_type}: " if isinstance(agent_type, str) else ""
            return f"{prefix}{description}"
    elif tool_name == "Skill":
        skill = input_dict.get("skill")
        if isinstance(skill, str):
            args = input_dict.get("args")
            return f"{skill} {args}" if isinstance(args, str) else skill
    elif tool_name == ITERATE_TOOL:
        return ITERATE_SUMMARY
    elif tool_name == PROBE_TOOL:
        return _summarize_probe(input_dict)
    return None


def _summarize_probe(input_dict: dict[str, object]) -> str:
    """Build a CLI-style summary for a gymrat probe tool call.

    Malformed fields (``names`` not a list of strings, ``samples`` not an int)
    are silently dropped rather than raised on.

    Args:
        input_dict: The raw tool input dict.

    Returns:
        A ``gymrat probe [names...] [--samples N]`` summary string.
    """
    parts: list[str] = ["gymrat", "probe"]

    names = input_dict.get("names")
    if isinstance(names, list) and all(isinstance(n, str) for n in names):
        parts.extend(names)

    samples = input_dict.get("samples")
    if isinstance(samples, int) and not isinstance(samples, bool):
        parts.append(f"--samples {samples}")

    return " ".join(parts)
