"""Session event vocabulary and the helpers that render and fan them out.

A session emits a fixed set of eight events to any number of
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
import re
import warnings
from collections.abc import Callable
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


class ToolStartEvent(_EventModel):
    """Emitted when the model invokes a tool."""

    type: Literal["tool_start"] = "tool_start"
    timestamp: int
    tool_use_id: str
    tool_name: str
    input: object
    input_summary: str


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


class TextDeltaEvent(_EventModel):
    """Emitted for each chunk of assistant text as it streams in."""

    type: Literal["text_delta"] = "text_delta"
    timestamp: int
    chunk: str


class UsageUpdateEvent(_EventModel):
    """Emitted when the driver observes updated cumulative cost."""

    type: Literal["usage_update"] = "usage_update"
    timestamp: int
    cost_usd: float


class CapEvent(_EventModel):
    """Emitted when a supervision cap (wall-clock or spend) fires."""

    type: Literal["cap"] = "cap"
    timestamp: int
    cap: Literal["wall-clock", "spend-cap"]


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
            data.pop("maxUsd", None)
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
    | LaunchEvent
)
"""The union of every event a session can emit to a :data:`SessionObserver`."""

SessionObserver = Callable[[SessionEvent], None]
"""Receives :data:`SessionEvent`s as a session streams them."""

_SessionEventAdapter: TypeAdapter[SessionEvent] = TypeAdapter(
    Annotated[SessionEvent, Field(discriminator="type")]
)


# ---------------------------------------------------------------------------
# to_json_line
# ---------------------------------------------------------------------------


def to_json_line(event: SessionEvent) -> str:
    """Serialize an event to a single compact JSON line with camelCase keys."""
    wire = event.model_dump(mode="python", by_alias=True)
    return json.dumps(null_non_finite(wire), separators=_COMPACT_JSON_SEPARATORS, default=str)


# ---------------------------------------------------------------------------
# event_from_wire
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# combine_observers
# ---------------------------------------------------------------------------


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
    otherwise it is cut on a code-point boundary and a suffix reports the number
    of dropped code points and the line count of the *original* text.
    """
    collapsed = _WHITESPACE_RUN.sub(" ", text).strip()

    if len(collapsed) <= max_chars:
        return collapsed

    remaining = len(collapsed) - max_chars
    line_count = text.count("\n") + 1
    return f"{collapsed[:max_chars]}… ({remaining} more chars, {line_count} lines)"


# ---------------------------------------------------------------------------
# summarize_input
# ---------------------------------------------------------------------------


def summarize_input(value: object, max_chars: int = SUMMARY_MAX_CHARS) -> str:
    """Summarize a value by JSON-encoding it, then passing it through :func:`summarize`.

    The JSON uses no separators padding so ``{"key": "value"}`` renders as
    ``{"key":"value"}``. A value ``json.dumps`` cannot encode falls back to
    ``summarize(str(value))``.
    """
    try:
        encoded = json.dumps(value, separators=_COMPACT_JSON_SEPARATORS)
    except (TypeError, ValueError):
        return summarize(str(value), max_chars)
    return summarize(encoded, max_chars)
