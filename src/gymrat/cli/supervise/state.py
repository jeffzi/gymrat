"""Liveness states and reporter context for the supervise progress display."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import asyncio
    from collections.abc import Callable
    from datetime import tzinfo

    from rich.console import RenderableType
    from rich.live import Live

    from gymrat.cli.supervise.reducer import ReporterState
    from gymrat.session.progress_file import ProgressSnapshot
    from gymrat.session.store import SessionState
    from gymrat.supervisor.events import CapAction, CapType, SessionObserver

IDLE_WARN_MS = 30_000
"""After 30 seconds of no tool activity, the liveness line escalates to alert styling."""


@dataclass(frozen=True, slots=True)
class ReadSessionResult:
    """The folded session state plus whether a baseline has been recorded.

    Attributes:
        state: The folded session state as of the last read.
        has_baseline: Whether a baseline record has been recorded for the session.
        best_delta_pct: The best primary delta, in percent, among committed-keep
            iterations. ``None`` when no keep has been committed.
            ``make_default_read`` computes it from the session records;
            injected test readers set it directly.
        best_seq: The sequence number of the committed-keep iteration with the
            best primary delta. ``None`` under the same condition as
            ``best_delta_pct``, and set alongside it.
        primary_label: The kind or name of the primary metric for the best
            committed-keep iteration. ``None`` under the same condition as
            ``best_delta_pct``, and set alongside it.
        baseline_sha: The commit sha of the session's baseline. ``None`` when
            no baseline record has been recorded; set from the session log by
            ``make_default_read``.
        stop_message: The newest stop record's message. Holds a value only
            while the folded log ends on a stop; ``None`` once any iteration,
            keep, discard, or finalize record supersedes it.
    """

    state: SessionState
    has_baseline: bool
    best_delta_pct: float | None = None
    best_seq: int | None = None
    primary_label: str | None = None
    baseline_sha: str | None = None
    stop_message: str | None = None


@dataclass(frozen=True, slots=True)
class SuperviseReporter:
    """The observer/stop/frame/warn surface that drives the supervise progress display.

    ``session_result`` hands back the session state as of the last re-read, which
    is what the closing summary reports once the display has stopped.

    ``start`` must be called from within a running event loop — it schedules the
    tick task via ``asyncio.create_task``. In plain mode it is a no-op.
    """

    observer: SessionObserver
    start: Callable[[], None]
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
    """A cap fired; liveness is frozen and the action comes from the event."""

    cap_type: CapType
    action: CapAction


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
    """The terminal, clock, and I/O handles the reporter shell owns.

    Everything the dashboard renders lives in ``state``, which the shell
    replaces after each event.  Only the shell writes to this object; the
    reducer never sees it.
    """

    state: ReporterState
    now: Callable[[], int]
    read_session_fn: Callable[[], ReadSessionResult]
    read_progress_fn: Callable[[str], ProgressSnapshot | None]
    plain_write_fn: Callable[[str], None]
    warn_fn: Callable[[str], None]
    live: Live | None
    tz: tzinfo | None
    no_color: bool
    is_plain: bool
    idle_warn_ms: int
    refresh_ms: int
    tick_task: asyncio.Task[None] | None = None
