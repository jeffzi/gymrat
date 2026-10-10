"""Types for the supervise progress display.

The liveness states and tool records the reducer tracks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from gymrat.supervisor.events import CapAction, CapType


@dataclass(frozen=True, slots=True)
class Starting:
    """No tool has run yet."""


@dataclass(frozen=True, slots=True)
class InFlight:
    """A tool is currently running, started at ``since``."""

    tool_use_id: str
    tool_name: str
    since: int
    input_summary: str = ""


@dataclass(frozen=True, slots=True)
class Thinking:
    """The model is in extended thinking, started at ``since``."""

    since: int
    estimated_tokens: int


@dataclass(frozen=True, slots=True)
class Responding:
    """The model is emitting a response, started at ``since``."""

    since: int


@dataclass(frozen=True, slots=True)
class Composing:
    """The model is composing a tool call for ``tool_name``, started at ``since``."""

    tool_name: str
    since: int


@dataclass(frozen=True, slots=True)
class Waiting:
    """No tool is running.

    Attributes:
        since: The timestamp of the last observed activity.
        last_tool: The last finished top-level tool, or ``None`` when no tool
            has finished yet.
    """

    since: int
    last_tool: FinishedTool | None = None


@dataclass(frozen=True, slots=True)
class Capped:
    """A cap fired; liveness is frozen and the action comes from the event."""

    cap_type: CapType
    action: CapAction


@dataclass(frozen=True, slots=True)
class Exiting:
    """The run-end exit sequence entered a phase.

    Attributes:
        kind: The phase the sequence entered.
        since: When it entered the phase.
        pid: The process holding the repository lock while the sequence waits
            on it, or ``None`` when its holder record cannot be read or while
            settling.
    """

    kind: Literal["waiting-lock", "settling"]
    since: int
    pid: int | None


type Liveness = Starting | InFlight | Thinking | Responding | Composing | Waiting | Capped | Exiting
"""The reporter's current view of what the model or tool is doing."""


@dataclass(frozen=True, slots=True)
class RunningTool:
    """A top-level or nested subagent tool seen to start at ``since`` and not yet end."""

    tool_name: str
    input_summary: str
    since: int


@dataclass(frozen=True, slots=True)
class FinishedTool:
    """A tool that has finished, kept for the last-three log."""

    tool_name: str
    input_summary: str
    duration_ms: int
    result: str
    ended_at: int


type NestedModelPhase = Literal["thinking", "responding", "tool_input"]
"""A model phase a nested subagent can be in; a nested turn end clears it instead."""


@dataclass(frozen=True, slots=True)
class NestedPhase:
    """A nested subagent model phase (thinking, responding, or tool_input)."""

    phase: NestedModelPhase
    since: int
    tool_name: str | None = None


type NestedActivity = RunningTool | NestedPhase
"""What a nested subagent is currently doing, keyed by its parent tool-use id."""
