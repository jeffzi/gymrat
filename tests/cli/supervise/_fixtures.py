"""Shared test doubles, builders, and reporter factories for the supervise progress tests.

The module is name-prefixed with ``_`` so pytest never collects it: it is a
helper imported as ``tests.cli.supervise._fixtures``.
"""

from __future__ import annotations

from datetime import UTC, datetime, tzinfo
from io import StringIO
from typing import TYPE_CHECKING, Literal, NamedTuple

if TYPE_CHECKING:
    from collections.abc import Callable

    from gymrat.config import Effort
    from gymrat.session.progress_file import ProgressSnapshot
    from gymrat.session.store import SessionState
    from gymrat.supervisor.supervise import EndedBy

from rich.console import Console, RenderableType

from gymrat.cli.style import CLI_THEME
from gymrat.cli.supervise.progress import (
    REFRESH_MS,
    SuperviseReporter,
    create_supervise_reporter,
)
from gymrat.cli.supervise.types import IDLE_WARN_MS, ReadSessionResult
from gymrat.eta import NS_PER_MS
from gymrat.loop.start import start_session
from gymrat.session.records import IterationPrimary, IterationRecord
from gymrat.supervisor.driver import SessionOutcome
from gymrat.supervisor.events import (
    CapAction,
    CapEvent,
    CapType,
    FollowUpEvent,
    LaunchEvent,
    ModelPhaseEvent,
    SessionObserver,
    ThinkingUpdateEvent,
    ToolEndEvent,
    ToolStartEvent,
    TurnEndEvent,
    UsageUpdateEvent,
)
from gymrat.supervisor.supervise import SupervisionResult
from tests._rich import frame_text
from tests.loop.iterate._fixtures import resolved_config
from tests.session.records._fixtures import (
    empty_session_state,
    iteration_record,
    session_state,
)

# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class Clock:
    """A mutable millisecond clock a test advances by assigning ``now``."""

    def __init__(self, start: int):
        self.now = start

    def __call__(self) -> int:
        return self.now


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def start_open_session(repo: str) -> None:
    """Start a gymrat session so the experiment worktree and session log exist."""
    start_session(repo, "main", resolved_config())


def _epoch_ms_to_local_hms(epoch_ms: int) -> str:
    """Epoch milliseconds to local ``HH:MM:SS``.

    Uses the same epoch-to-local conversion the implementation should use, so
    tests are timezone-independent — they compute the expected string rather
    than hard-coding a clock time.
    """
    return datetime.fromtimestamp(epoch_ms / 1000, tz=UTC).astimezone().strftime("%H:%M:%S")


def make_iteration(delta_pct: float | None, outcome: str, seq: int = 1) -> IterationRecord:
    """An iteration whose only reporter-visible fields are its delta and outcome."""
    return iteration_record(
        seq=seq,
        primary=IterationPrimary(kind="geomean", delta_pct=delta_pct),
        outcome=outcome,
    )


def session_state_three_iterations(delta_pct: float, outcome: str, *, seq: int = 1) -> SessionState:
    """Build a three-iteration session state for loop-row tests.

    The shared "loop row has content" arrangement used by summary, frame, and
    reporter tests that only vary the last iteration's delta and outcome.
    """
    return session_state(
        iteration_count=3,
        keep_count=2,
        discard_count=1,
        last_iteration=make_iteration(delta_pct, outcome, seq=seq),
    )


def make_read_session(
    state: SessionState,
    *,
    has_baseline: bool,
    best_delta_pct: float | None = None,
    best_seq: int | None = None,
    primary_label: str | None = None,
    baseline_sha: str | None = None,
    stop_message: str | None = None,
) -> Callable[[], ReadSessionResult]:
    """A ``read_session`` that always returns ``state`` and ``has_baseline``.

    ``best_*`` / ``baseline_sha`` / ``stop_message`` default to ``None`` on
    ``ReadSessionResult`` itself, so callers that omit them still get a valid
    result.
    """
    result = ReadSessionResult(
        state=state,
        has_baseline=has_baseline,
        best_delta_pct=best_delta_pct,
        best_seq=best_seq,
        primary_label=primary_label,
        baseline_sha=baseline_sha,
        stop_message=stop_message,
    )
    return lambda: result


def make_supervision_result(
    *,
    reason: Literal["completed", "error", "interrupted"] = "completed",
    ended_by: EndedBy = "session",
    duration_ms: int = 60_000,
    cost_usd: float = 0.05,
    message: str | None = None,
    end_reason: str | None = None,
) -> SupervisionResult:
    """Build a default-completed supervision result.

    Args:
        reason: Outcome reason recorded on the ``SessionOutcome``.
        ended_by: Value recorded on ``SupervisionResult.ended_by``.
        duration_ms: Value recorded on ``SupervisionResult.duration_ms``.
        cost_usd: Duplicated onto both the ``outcome`` and the top-level
            result, mirroring the shape ``supervise`` returns.
        message: Value recorded on ``SessionOutcome.message``.
        end_reason: Value recorded on ``SupervisionResult.end_reason``.

    Returns:
        A ``SupervisionResult`` built from the given arguments.
    """
    outcome = SessionOutcome(reason=reason, cost_usd=cost_usd, message=message)
    return SupervisionResult(
        outcome=outcome,
        ended_by=ended_by,
        duration_ms=duration_ms,
        cost_usd=cost_usd,
        end_reason=end_reason,
    )


def _throwing_read() -> ReadSessionResult:
    message = "no session file"
    raise RuntimeError(message)


# ---------------------------------------------------------------------------
# Event firers
# ---------------------------------------------------------------------------

# Default timestamp of a tool start; a tool end's default duration is measured
# from it, so the two stay in sync.
_DEFAULT_TOOL_START_TS = 2000


def launch_event(
    at_ms: int = 1000,
    *,
    max_minutes: float = 60,
    max_usd: float | None = None,
) -> LaunchEvent:
    """A ``LaunchEvent`` stamped at *at_ms* milliseconds, with sensible cap and model defaults.

    ``at_ms`` is in the test's millisecond vocabulary; the event is stamped
    ``at=at_ms * NS_PER_MS`` (nanoseconds) so the dashboard's ingestion
    boundary (``event.at // 1_000_000``) recovers the same millisecond value.
    """
    return LaunchEvent(
        at=at_ms * NS_PER_MS,
        schema_version=1,
        head_sha="abc123",
        dirty=False,
        max_minutes=max_minutes,
        max_usd=max_usd,
        model=None,
        runbook_path="/path/to/runbook.md",
        kickoff_summary="test kickoff",
        session_id="20260813-125044-34ec",
    )


def tool_start_event(
    tool_name: str,
    tool_use_id: str,
    at_ms: int = _DEFAULT_TOOL_START_TS,
    *,
    input_summary: str = "...",
    parent_tool_use_id: str | None = None,
) -> ToolStartEvent:
    """A ``ToolStartEvent`` for *tool_name* stamped at *at_ms* milliseconds."""
    return ToolStartEvent(
        at=at_ms * NS_PER_MS,
        tool_use_id=tool_use_id,
        tool_name=tool_name,
        input={},
        input_summary=input_summary,
        parent_tool_use_id=parent_tool_use_id,
    )


def tool_end_event(
    tool_name: str,
    tool_use_id: str,
    at_ms: int = 3000,
    *,
    result: str = "ok",
    result_summary: str = "ok",
    started_at_ms: int = _DEFAULT_TOOL_START_TS,
    parent_tool_use_id: str | None = None,
) -> ToolEndEvent:
    """A ``ToolEndEvent`` whose duration is measured from *started_at_ms*."""
    return ToolEndEvent(
        at=at_ms * NS_PER_MS,
        tool_use_id=tool_use_id,
        tool_name=tool_name,
        duration_ms=at_ms - started_at_ms,
        result=result,
        result_summary=result_summary,
        parent_tool_use_id=parent_tool_use_id,
    )


def usage_event(cost_usd: float, at_ms: int = 4000) -> UsageUpdateEvent:
    """A ``UsageUpdateEvent`` carrying the given cumulative cost."""
    return UsageUpdateEvent(at=at_ms * NS_PER_MS, cost_usd=cost_usd)


def cap_event(cap: CapType, at_ms: int = 5000, *, action: CapAction = "interrupting") -> CapEvent:
    """A ``CapEvent`` signaling that *cap* has fired."""
    return CapEvent(at=at_ms * NS_PER_MS, cap=cap, action=action)


def model_phase_event(
    at_ms: int,
    phase: str,
    *,
    tool_name: str | None = None,
    parent_tool_use_id: str | None = None,
) -> ModelPhaseEvent:
    """A ``ModelPhaseEvent``; *parent_tool_use_id* scopes it to a nested tool."""
    return ModelPhaseEvent(
        at=at_ms * NS_PER_MS,
        phase=phase,  # type: ignore[arg-type]
        tool_name=tool_name,
        parent_tool_use_id=parent_tool_use_id,
    )


def thinking_event(
    at_ms: int,
    *,
    estimated_tokens: int = 100,
    delta: int = 10,
    parent_tool_use_id: str | None = None,
) -> ThinkingUpdateEvent:
    """A ``ThinkingUpdateEvent`` carrying the given cumulative token estimate."""
    return ThinkingUpdateEvent(
        at=at_ms * NS_PER_MS,
        estimated_tokens=estimated_tokens,
        delta=delta,
        parent_tool_use_id=parent_tool_use_id,
    )


def turn_end_event(
    at_ms: int = 5000,
    *,
    text: str = "Turn summary.",
    cost_usd: float = 0.01,
    origin: Literal["agent", "injected"] = "agent",
    budget_exhausted: bool = False,
) -> TurnEndEvent:
    """A ``TurnEndEvent`` attributed to *origin*."""
    return TurnEndEvent(
        at=at_ms * NS_PER_MS,
        text=text,
        cost_usd=cost_usd,
        origin=origin,
        budget_exhausted=budget_exhausted,
    )


def follow_up_event(
    at_ms: int = 6000,
    *,
    action: Literal["replied", "waiting", "ended"] = "replied",
    reason: str | None = None,
    text: str | None = None,
) -> FollowUpEvent:
    """A ``FollowUpEvent`` carrying the supervisor's decision for the turn."""
    return FollowUpEvent(at=at_ms * NS_PER_MS, action=action, reason=reason, text=text)


def fire_launch_and_bash_cycle(observer: SessionObserver) -> None:
    """Minimum event sequence that gets session state into the loop/best rows.

    The Bash end triggers the reporter's session re-read.
    """
    observer(launch_event(1000))
    observer(tool_start_event("Bash", "bash-1", 2000))
    observer(tool_end_event("Bash", "bash-1", 3000))


def fire_launch_and_bash_start(observer: SessionObserver) -> None:
    """Fire launch and Bash start, leaving the tool in flight.

    Used by the in-flight-guard and nested-event tests, which fire a second
    event on top of the still-running Bash call and assert whether it takes
    effect or is ignored.
    """
    observer(launch_event(1000))
    observer(tool_start_event("Bash", "bash-1", 1500))


# ---------------------------------------------------------------------------
# Reporter setup
# ---------------------------------------------------------------------------

# Fixed width for all golden-snapshot tests so frames are stable.
FRAME_WIDTH = 100

#: Patch target for the ``ErasableLive`` the live-mode reporter drives.
LIVE_CLASS_PATH = "gymrat.cli.supervise.progress.ErasableLive"


# Every reporter make_reporter builds, so teardown can stop the live refresh
# thread each one starts.
_built_reporters: list[SuperviseReporter] = []


class ReporterKit(NamedTuple):
    """A live-mode reporter paired with the injectable clock that drives it."""

    reporter: SuperviseReporter
    clock: Clock


def make_reporter(
    *,
    mode: Literal["live", "plain"] = "live",
    max_minutes: float = 480,
    max_usd: float | None = None,
    max_iterations: int | None = None,
    read_session: Callable[[], ReadSessionResult] | None = None,
    clock_start: int = 1000,
    root: str = "/tmp/repo",
    read_progress: Callable[[str], ProgressSnapshot | None] | None = None,
    plain_write: Callable[[str], None] | None = None,
    session_id: str = "20260813-125044-34ec",
    branch: str = "gymrat/20260813-125044-34ec",
    color: bool | None = None,
    tz: tzinfo | None = UTC,
    model: str | None = None,
    effort: Effort | None = None,
    idle_warn_ms: int = IDLE_WARN_MS,
    refresh_ms: int = REFRESH_MS,
    clock: Clock | None = None,
) -> ReporterKit:
    """Build a reporter with injectable dependencies for deterministic testing.

    The reporter is recorded so ``stop_built_reporters`` can stop it at
    teardown.

    Args:
        mode: ``"live"`` for a Rich Live dashboard, ``"plain"`` for line-by-line output.
        max_minutes: Wall-clock cap in minutes.
        max_usd: Spend cap in USD, or ``None`` for uncapped.
        max_iterations: Iteration cap, or ``None`` for uncapped.
        read_session: Reads the current session state; defaults to an empty
            session with no baseline.
        clock_start: Start value of the ``Clock`` built when ``clock`` is omitted.
        root: Project root whose session directory is monitored.
        read_progress: Reads the iterate progress sidecar, or ``None`` for the
            standard reader.
        plain_write: Line writer for plain mode, or ``None`` for the standard writer.
        session_id: Session identifier propagated to the frame.
        branch: Git branch name shown in the frame header.
        color: Tri-state color override: ``True`` forces color, ``False``
            disables it, ``None`` auto-detects.
        tz: Timezone for wall-clock timestamps. Defaults to ``UTC`` so snapshot
            tests produce stable timestamps regardless of host timezone; pass
            ``None`` to exercise the system-local fallback path.
        model: Model name shown as a labelled row when set.
        effort: Effort level shown as a labelled row when set.
        idle_warn_ms: Milliseconds of inactivity before the liveness line
            escalates to alert styling.
        refresh_ms: Live dashboard refresh interval in milliseconds.
        clock: Drives the reporter, e.g. a ``Clock`` subclass; replaces the one
            built from ``clock_start``.

    Returns:
        The reporter paired with the clock that drives it.
    """
    if clock is None:
        clock = Clock(clock_start)
    if read_session is None:
        read_session = make_read_session(empty_session_state(), has_baseline=False)
    reporter = create_supervise_reporter(
        root=root,
        max_minutes=max_minutes,
        mode=mode,
        now=clock,
        read_session=read_session,
        session_id=session_id,
        branch=branch,
        tz=tz,
        max_iterations=max_iterations,
        max_usd=max_usd,
        read_progress=read_progress,
        plain_write=plain_write,
        color=color,
        model=model,
        effort=effort,
        idle_warn_ms=idle_warn_ms,
        refresh_ms=refresh_ms,
    )
    _built_reporters.append(reporter)
    return ReporterKit(reporter, clock)


def stop_built_reporters() -> None:
    """Stop every reporter ``make_reporter`` built, ending its live refresh thread."""
    while _built_reporters:
        _built_reporters.pop().stop()


def render_frame(reporter: SuperviseReporter, *, width: int = FRAME_WIDTH) -> str:
    """Render the reporter's current frame through a non-terminal console."""
    return frame_text(reporter.frame(), width=width)


def line_after(frame: str, needle: str) -> str:
    """Return the line immediately following the first line containing *needle*."""
    lines = frame.splitlines()
    idx = next(i for i, line in enumerate(lines) if needle in line)
    return lines[idx + 1]


def _render_sealed(
    renderable: RenderableType,
    *,
    width: int,
    no_color: bool,
    color_system: Literal["standard"] | None,
) -> str:
    """Render *renderable* through a sealed terminal console with the given color settings."""
    buf = StringIO()
    console = Console(
        file=buf,
        width=width,
        force_terminal=True,
        no_color=no_color,
        color_system=color_system,
        legacy_windows=False,
        _environ={},
        theme=CLI_THEME,
    )
    console.print(renderable)
    return buf.getvalue()


def render_colored(renderable: RenderableType, *, width: int = FRAME_WIDTH) -> str:
    """Render ``renderable`` through a sealed console with standard color."""
    return _render_sealed(renderable, width=width, no_color=False, color_system="standard")


def render_colorless(renderable: RenderableType, *, width: int = FRAME_WIDTH) -> str:
    """Render ``renderable`` through a sealed terminal console with colour off, as ``--no-color`` does."""
    return _render_sealed(renderable, width=width, no_color=True, color_system=None)
