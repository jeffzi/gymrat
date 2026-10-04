"""Types for the supervise progress display.

The session read result and the liveness states and tool records the reducer
tracks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from gymrat.session.store import SessionState
    from gymrat.supervisor.events import CapAction, CapType


@dataclass(frozen=True, slots=True)
class BestIteration:
    """The committed-keep iteration with the best primary delta.

    Attributes:
        delta_pct: Its primary delta, in percent.
        seq: Its sequence number.
        label: Its primary: the metric name for a named-metric primary, else
            the kind (``"geomean"``).
    """

    delta_pct: float
    seq: int
    label: str


@dataclass(frozen=True, slots=True)
class ReadSessionResult:
    """The folded session state plus whether a baseline has been recorded.

    Attributes:
        state: The folded session state as of the last read.
        has_baseline: Whether a baseline record has been recorded for the session.
        best: The best committed-keep iteration. ``None`` when no keep has been
            committed. ``read_live_session`` computes it from the session
            records; injected test readers set it directly.
        baseline_sha: The commit the session started from, taken from the
            session record by ``read_live_session``. ``None`` before the session
            record has been written.
        stop_message: The newest stop record's message. Holds a value only
            while the folded log ends on a stop; ``None`` once any iteration,
            keep, discard, or finalize record supersedes it.
    """

    state: SessionState
    has_baseline: bool
    best: BestIteration | None = None
    baseline_sha: str | None = None
    stop_message: str | None = None


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
    """No tool is running; ``since`` is the timestamp of the last observed activity.

    ``last_tool`` is the last finished top-level tool, ``None`` when no tool has
    finished yet.
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
    """The run-end exit sequence entered phase ``kind`` at ``since``.

    ``pid`` is the process holding the repository lock while the sequence waits
    on it; ``None`` when its holder record cannot be read or while settling.
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


@dataclass(frozen=True, slots=True)
class NestedPhase:
    """A nested subagent model phase (thinking, responding, or tool_input)."""

    phase: str
    since: int
    tool_name: str | None = None


type NestedActivity = RunningTool | NestedPhase
"""What a nested subagent is currently doing, keyed by its parent tool-use id."""
