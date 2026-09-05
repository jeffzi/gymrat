"""Liveness states and reporter context for the supervise progress display."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from collections import deque
    from collections.abc import Callable
    from datetime import tzinfo

    from rich.console import RenderableType
    from rich.live import Live

    from gymrat.config import Effort
    from gymrat.session.progress_file import ProgressSnapshot
    from gymrat.session.store import SessionState
    from gymrat.supervisor.events import SessionObserver

IDLE_WARN_MS = 30_000
"""After 30 seconds of no tool activity, the liveness line escalates to alert styling."""

type CapType = Literal["wall-clock", "spend-cap"]
"""The cap variety that ended or is ending a supervised run."""


@dataclass(frozen=True, slots=True)
class ReadSessionResult:
    """The folded session state plus whether a baseline has been recorded.

    The ``best_*`` fields track the committed-keep iteration with the best
    primary delta.  ``make_default_read`` computes them from the session
    records; injected test readers set them directly.
    """

    state: SessionState
    has_baseline: bool
    best_delta_pct: float | None = None
    best_seq: int | None = None
    primary_label: str | None = None
    baseline_sha: str | None = None


@dataclass(frozen=True, slots=True)
class SuperviseReporter:
    """The observer/stop/frame/warn surface that drives the supervise progress display.

    ``session_result`` hands back the session state as of the last re-read, which
    is what the closing summary reports once the display has stopped.
    """

    observer: SessionObserver
    stop: Callable[[], None]
    frame: Callable[[], RenderableType]
    warn: Callable[[str], None]
    session_result: Callable[[], ReadSessionResult | None]
    final_text: Callable[[], str | None]


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

    The ``tool_*`` and ``result`` fields describe the last finished top-level tool.
    All three are ``None`` when no tool has finished yet.
    """

    since: int
    tool_name: str | None = None
    tool_ended_at: int | None = None
    result: str | None = None


@dataclass(frozen=True, slots=True)
class Capped:
    """A cap fired; the run is interrupting and liveness is frozen."""

    cap_type: CapType


type Liveness = Starting | InFlight | Thinking | Responding | Composing | Waiting | Capped
"""The reporter's current view of what the model or tool is doing."""


@dataclass(frozen=True, slots=True)
class TrackedTool:
    """A tool the reporter has seen start but not yet end."""

    tool_name: str
    started_at: int
    input_summary: str


@dataclass(frozen=True, slots=True)
class FinishedTool:
    """A tool that has finished, kept for the last-three log."""

    tool_name: str
    input_summary: str
    duration_ms: int
    result: str
    ended_at: int


@dataclass(frozen=True, slots=True)
class NestedTool:
    """A nested subagent tool that is currently running."""

    tool_name: str
    input_summary: str
    since: int


@dataclass(frozen=True, slots=True)
class NestedPhase:
    """A nested subagent model phase (thinking, responding, or tool_input)."""

    phase: str
    since: int
    tool_name: str | None = None


type NestedActivity = NestedTool | NestedPhase
"""What a nested subagent is currently doing, keyed by its parent tool-use id."""


@dataclass(slots=True)
class ReporterCtx:
    """Mutable state shared by the event handlers."""

    now: Callable[[], int]
    read_session_fn: Callable[[], ReadSessionResult]
    read_progress_fn: Callable[[str], ProgressSnapshot | None]
    root: str
    max_minutes: float
    max_usd: float | None
    max_iterations: int | None
    is_plain: bool
    label: str
    session_id: str
    branch: str
    in_flight_tools: dict[str, TrackedTool]
    finished_tools: deque[FinishedTool]
    launch_timestamp: int | None
    cost_usd: float | None
    session_result: ReadSessionResult | None
    liveness: Liveness
    last_loop_text: str
    plain_write_fn: Callable[[str], None]
    tz: tzinfo | None
    warn_fn: Callable[[str], None]
    live: Live | None
    nested: dict[str, NestedActivity]
    nested_tool_ids: dict[str, str]
    no_color: bool
    log_path: str
    last_top_level_text: str | None
    model: str | None
    effort: Effort | None
