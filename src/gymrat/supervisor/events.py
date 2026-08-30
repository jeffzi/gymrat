"""Session event vocabulary and the helpers that render and fan them out.

A session emits a fixed set of eight events to any number of
:data:`SessionObserver` callbacks. Each event is a frozen dataclass tagged by a
``type`` literal and stamped with an epoch-millisecond ``timestamp``; together
they form the :data:`SessionEvent` union.

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
from dataclasses import dataclass
from typing import Any, Literal, assert_never, cast

from gymrat.finite_json import null_non_finite

# ---------------------------------------------------------------------------
# Event vocabulary
# ---------------------------------------------------------------------------

# Maximum code-point length for a session-event summary before it is truncated.
SUMMARY_MAX_CHARS = 200


@dataclass(frozen=True, slots=True)
class DirtyInfo:
    """The dirty-worktree provenance carried on a launch event."""

    file_count: int


@dataclass(frozen=True, slots=True, kw_only=True)
class ThinkingUpdateEvent:
    """Emitted as the model's extended-thinking token estimate changes mid-turn."""

    type: Literal["thinking_update"] = "thinking_update"
    timestamp: int
    estimated_tokens: int
    delta: int


@dataclass(frozen=True, slots=True, kw_only=True)
class ToolStartEvent:
    """Emitted when the model invokes a tool."""

    type: Literal["tool_start"] = "tool_start"
    timestamp: int
    tool_use_id: str
    tool_name: str
    input: object
    input_summary: str


@dataclass(frozen=True, slots=True, kw_only=True)
class ToolProgressEvent:
    """Emitted periodically while a long-running tool call is still in flight."""

    type: Literal["tool_progress"] = "tool_progress"
    timestamp: int
    tool_use_id: str
    elapsed_ms: int


@dataclass(frozen=True, slots=True, kw_only=True)
class ToolEndEvent:
    """Emitted when a tool call completes and its result is available."""

    type: Literal["tool_end"] = "tool_end"
    timestamp: int
    tool_use_id: str
    tool_name: str
    duration_ms: int
    result: str
    result_summary: str


@dataclass(frozen=True, slots=True, kw_only=True)
class TextDeltaEvent:
    """Emitted for each chunk of assistant text as it streams in."""

    type: Literal["text_delta"] = "text_delta"
    timestamp: int
    chunk: str


@dataclass(frozen=True, slots=True, kw_only=True)
class UsageUpdateEvent:
    """Emitted when the driver observes updated cumulative cost."""

    type: Literal["usage_update"] = "usage_update"
    timestamp: int
    cost_usd: float


@dataclass(frozen=True, slots=True, kw_only=True)
class CapEvent:
    """Emitted when a supervision cap (wall-clock or spend) fires."""

    type: Literal["cap"] = "cap"
    timestamp: int
    cap: Literal["wall-clock", "spend-cap"]


@dataclass(frozen=True, slots=True, kw_only=True)
class LaunchEvent:
    """Written by the supervisor as the log's first line: launch provenance."""

    type: Literal["launch"] = "launch"
    timestamp: int
    head_sha: str
    dirty: Literal[False] | DirtyInfo
    max_minutes: float
    max_usd: float | None
    model: str | None
    runbook_path: str
    kickoff_summary: str


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


# ---------------------------------------------------------------------------
# to_json_line
# ---------------------------------------------------------------------------


def _dirty_to_wire(dirty: Literal[False] | DirtyInfo) -> object:
    if dirty is False:
        return False
    return {"fileCount": dirty.file_count}


def _launch_to_wire(event: LaunchEvent) -> dict[str, object]:
    """Render a launch event, omitting ``maxUsd``/``model`` when they are ``None``.

    Omission (rather than a ``null``) matches the shipped log format.
    """
    wire: dict[str, object] = {
        "type": event.type,
        "timestamp": event.timestamp,
        "headSha": event.head_sha,
        "dirty": _dirty_to_wire(event.dirty),
        "maxMinutes": event.max_minutes,
    }
    if event.max_usd is not None:
        wire["maxUsd"] = event.max_usd
    if event.model is not None:
        wire["model"] = event.model
    wire["runbookPath"] = event.runbook_path
    wire["kickoffSummary"] = event.kickoff_summary
    return wire


def _to_wire(event: SessionEvent) -> dict[str, object]:
    """Render an event to its camelCase wire dict in declared field order."""
    wire: dict[str, object]
    match event:
        case ThinkingUpdateEvent():
            wire = {
                "type": event.type,
                "timestamp": event.timestamp,
                "estimatedTokens": event.estimated_tokens,
                "delta": event.delta,
            }
        case ToolStartEvent():
            wire = {
                "type": event.type,
                "timestamp": event.timestamp,
                "toolUseId": event.tool_use_id,
                "toolName": event.tool_name,
                "input": event.input,
                "inputSummary": event.input_summary,
            }
        case ToolProgressEvent():
            wire = {
                "type": event.type,
                "timestamp": event.timestamp,
                "toolUseId": event.tool_use_id,
                "elapsedMs": event.elapsed_ms,
            }
        case ToolEndEvent():
            wire = {
                "type": event.type,
                "timestamp": event.timestamp,
                "toolUseId": event.tool_use_id,
                "toolName": event.tool_name,
                "durationMs": event.duration_ms,
                "result": event.result,
                "resultSummary": event.result_summary,
            }
        case TextDeltaEvent():
            wire = {"type": event.type, "timestamp": event.timestamp, "chunk": event.chunk}
        case UsageUpdateEvent():
            wire = {"type": event.type, "timestamp": event.timestamp, "costUsd": event.cost_usd}
        case CapEvent():
            wire = {"type": event.type, "timestamp": event.timestamp, "cap": event.cap}
        case LaunchEvent():
            wire = _launch_to_wire(event)
        case _ as unreachable:  # pragma: no cover — exhaustive match over the event union
            assert_never(unreachable)
    return wire


def to_json_line(event: SessionEvent) -> str:
    """Serialize an event to a single compact JSON line with camelCase keys."""
    wire = _to_wire(event)
    return json.dumps(null_non_finite(wire), separators=(",", ":"), default=str)


# ---------------------------------------------------------------------------
# event_from_wire
# ---------------------------------------------------------------------------


def _dirty_from_wire(value: object) -> Literal[False] | DirtyInfo:
    if isinstance(value, dict):
        file_count = value["fileCount"]
        if not isinstance(file_count, int):
            msg = f"dirty.fileCount must be an int, got {type(file_count).__name__}"
            raise TypeError(msg)
        return DirtyInfo(file_count=file_count)
    return False


def _launch_from_wire(wire: dict[str, Any]) -> SessionEvent:
    return LaunchEvent(
        timestamp=wire["timestamp"],
        head_sha=wire["headSha"],
        dirty=_dirty_from_wire(wire["dirty"]),
        max_minutes=wire["maxMinutes"],
        max_usd=wire.get("maxUsd"),
        model=wire.get("model"),
        runbook_path=wire["runbookPath"],
        kickoff_summary=wire["kickoffSummary"],
    )


# Each builder maps a camelCase wire dict back to its event, raising KeyError on
# a missing required field or TypeError on a malformed one so :func:`event_from_wire`
# can report it as unparsed.
_WIRE_BUILDERS: dict[str, Callable[[dict[str, Any]], SessionEvent]] = {
    "thinking_update": lambda w: ThinkingUpdateEvent(
        timestamp=w["timestamp"], estimated_tokens=w["estimatedTokens"], delta=w["delta"]
    ),
    "tool_start": lambda w: ToolStartEvent(
        timestamp=w["timestamp"],
        tool_use_id=w["toolUseId"],
        tool_name=w["toolName"],
        input=w["input"],
        input_summary=w["inputSummary"],
    ),
    "tool_progress": lambda w: ToolProgressEvent(
        timestamp=w["timestamp"], tool_use_id=w["toolUseId"], elapsed_ms=w["elapsedMs"]
    ),
    "tool_end": lambda w: ToolEndEvent(
        timestamp=w["timestamp"],
        tool_use_id=w["toolUseId"],
        tool_name=w["toolName"],
        duration_ms=w["durationMs"],
        result=w["result"],
        result_summary=w["resultSummary"],
    ),
    "text_delta": lambda w: TextDeltaEvent(timestamp=w["timestamp"], chunk=w["chunk"]),
    "usage_update": lambda w: UsageUpdateEvent(timestamp=w["timestamp"], cost_usd=w["costUsd"]),
    "cap": lambda w: CapEvent(timestamp=w["timestamp"], cap=w["cap"]),
    "launch": _launch_from_wire,
}


def event_from_wire(obj: object) -> SessionEvent | None:
    """Reconstruct a session event from its camelCase wire object.

    The inverse of :func:`to_json_line`'s rendering: given a decoded JSON object,
    return the matching event dataclass. Returns ``None`` when ``obj`` is not a
    dict, carries no recognized ``type``, or is missing a required field or has
    one with the wrong type.
    """
    if not isinstance(obj, dict):
        return None
    wire = cast("dict[str, Any]", obj)
    type_name = wire.get("type")
    builder = _WIRE_BUILDERS.get(type_name) if isinstance(type_name, str) else None
    if builder is None:
        return None
    try:
        return builder(wire)
    except (KeyError, TypeError):
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
        encoded = json.dumps(value, separators=(",", ":"))
    except (TypeError, ValueError):
        return summarize(str(value), max_chars)
    return summarize(encoded, max_chars)
