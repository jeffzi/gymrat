"""Shared test doubles, builders, and reporter factories for the supervise progress tests.

The module is name-prefixed with ``_`` so pytest never collects it: it is a
helper imported as ``tests.cli.supervise._fixtures``.
"""

from __future__ import annotations

from datetime import UTC, tzinfo
from typing import TYPE_CHECKING, Any, Literal, NamedTuple

if TYPE_CHECKING:
    from collections.abc import Callable

    import pytest
    from rich.console import Console, RenderableType

    from gymrat.config import Effort
    from gymrat.session.progress_file import ProgressSnapshot
    from gymrat.session.store import SessionState
    from gymrat.supervisor.supervise import EndedBy

from gymrat.cli.supervise.progress import (
    IDLE_WARN_MS,
    REFRESH_MS,
    SuperviseReporter,
    create_supervise_reporter,
)
from gymrat.cli.supervise.types import BestIteration, ReadSessionResult
from gymrat.loop.start import start_session
from gymrat.supervisor.driver import SessionOutcome
from gymrat.supervisor.events import (
    CapAction,
    CapEvent,
    CapType,
    FollowUpEvent,
    LaunchEvent,
    ModelPhase,
    ModelPhaseEvent,
    SessionObserver,
    ThinkingUpdateEvent,
    ToolEndEvent,
    ToolStartEvent,
    TurnEndEvent,
    UsageUpdateEvent,
)
from gymrat.supervisor.supervise import SupervisionResult
from gymrat.utils import NS_PER_MS
from tests._ansi import (
    strip_sgr,
)
from tests._config import resolved_config
from tests._rich import (
    Clock,
    console_output,
    frame_text,
    sealed_console,
    track,
)
from tests.report._measurements import create_measurement_result
from tests.session.records._fixtures import (
    SUPERVISED_SESSION_ID,
    baseline_record,
    empty_session_state,
    make_iteration,
    session_state,
)
from tests.supervisor._fixtures import make_launch, make_turn_end

# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def start_open_session(repo: str) -> None:
    """Start a gymrat session so the experiment worktree and session log exist."""
    start_session(repo, "main", resolved_config())


def install_baseline_seam(
    monkeypatch: pytest.MonkeyPatch, *, on_call: Callable[[], None] | None = None
) -> list[dict[str, Any]]:
    """Replace the baseline measurement path so no real bench runs.

    Args:
        monkeypatch: The fixture that installs the stand-in measurement.
        on_call: Run at the start of each measurement, to observe state while it runs.

    Returns:
        A list that records each call's keyword arguments.
    """
    calls: list[dict[str, Any]] = []

    async def fake_measure(target: object, run_options: object) -> Any:
        calls.append({"target": target, "run_options": run_options})
        if on_call is not None:
            on_call()
        record = baseline_record(duration_ms=5000)
        result = create_measurement_result(label=record.label, samples=1, rounds=record.samples)
        return result, record

    monkeypatch.setattr("gymrat.cli.supervise.preflight.measure_baseline", fake_measure)
    return calls


def session_state_three_iterations(delta_pct: float, outcome: str, *, seq: int = 1) -> SessionState:
    """Build a three-iteration session state for loop-row tests.

    The shared "loop row has content" arrangement used by summary, frame, and
    reporter tests that only vary the last iteration's delta and outcome.

    Args:
        delta_pct: The last iteration's delta.
        outcome: The last iteration's outcome.
        seq: The last iteration's sequence number.

    Returns:
        A state with two kept iterations and one discarded.
    """
    return session_state(
        iteration_count=3,
        keep_count=2,
        discard_count=1,
        last_iteration=make_iteration(delta_pct, outcome, seq=seq),
    )


def read_result(
    state: SessionState | None = None,
    *,
    has_baseline: bool = False,
    best: BestIteration | None = None,
    stop_message: str | None = None,
) -> ReadSessionResult:
    """A session read result.

    Args:
        state: The session state the result carries; the empty session when omitted.
        has_baseline: Whether the session has recorded a baseline.
        best: The best iteration, if any.
        stop_message: The stop-condition message, if any.

    Returns:
        The read result.
    """
    return ReadSessionResult(
        state=state if state is not None else empty_session_state(),
        has_baseline=has_baseline,
        best=best,
        stop_message=stop_message,
    )


def make_read_session(
    state: SessionState,
    *,
    has_baseline: bool,
    best: BestIteration | None = None,
    stop_message: str | None = None,
) -> Callable[[], ReadSessionResult]:
    """A ``read_session`` that always returns the same result.

    Args:
        state: The session state the result carries.
        has_baseline: Whether the session has recorded a baseline.
        best: The best iteration, if any.
        stop_message: The stop-condition message, if any.

    Returns:
        A callable returning one fixed ``ReadSessionResult``.
    """
    result = read_result(state, has_baseline=has_baseline, best=best, stop_message=stop_message)
    return lambda: result


#: A session read of a session with a baseline and no iteration yet.
EMPTY_READ = read_result(has_baseline=True)

#: A session read of a session whose one iteration was kept as an improvement.
KEPT_READ = read_result(
    session_state(iteration_count=1, keep_count=1, last_iteration=make_iteration(-2.0, "improved")),
    has_baseline=True,
)


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
        end_reason=end_reason,
    )


def _throwing_read() -> ReadSessionResult:
    message = "no session file"
    raise RuntimeError(message)


# ---------------------------------------------------------------------------
# Event firers
# ---------------------------------------------------------------------------

#: Default timestamp of a tool start, in milliseconds; a tool end's default
#: duration is measured from it, so the two stay in sync.
TOOL_START_MS = 2000


def launch_event(
    at_ms: int = 1000,
    *,
    max_minutes: float = 60,
    max_usd: float | None = None,
) -> LaunchEvent:
    """A ``LaunchEvent`` with sensible cap and model defaults.

    The event is stamped in nanoseconds so the dashboard's ingestion boundary
    (``event.at // 1_000_000``) recovers the same millisecond value.

    Args:
        at_ms: The launch time, in milliseconds.
        max_minutes: The wall-clock cap the launch carries.
        max_usd: The spend cap the launch carries, if any.

    Returns:
        The launch event.
    """
    return make_launch(
        at=at_ms * NS_PER_MS, head_sha="abc123", max_minutes=max_minutes, max_usd=max_usd
    )


def tool_start_event(
    tool_name: str,
    tool_use_id: str,
    at_ms: int = TOOL_START_MS,
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
    started_at_ms: int = TOOL_START_MS,
    parent_tool_use_id: str | None = None,
) -> ToolEndEvent:
    """A ``ToolEndEvent`` whose duration is measured from *started_at_ms*."""
    return ToolEndEvent(
        at=at_ms * NS_PER_MS,
        tool_use_id=tool_use_id,
        tool_name=tool_name,
        duration_ms=at_ms - started_at_ms,
        result=result,
        result_summary="ok",
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
    phase: ModelPhase,
    *,
    tool_name: str | None = None,
    parent_tool_use_id: str | None = None,
) -> ModelPhaseEvent:
    """A ``ModelPhaseEvent``; *parent_tool_use_id* scopes it to a nested tool."""
    return ModelPhaseEvent(
        at=at_ms * NS_PER_MS,
        phase=phase,
        tool_name=tool_name,
        parent_tool_use_id=parent_tool_use_id,
    )


def thinking_event(
    at_ms: int,
    *,
    estimated_tokens: int = 100,
    parent_tool_use_id: str | None = None,
) -> ThinkingUpdateEvent:
    """A ``ThinkingUpdateEvent`` carrying the given cumulative token estimate."""
    return ThinkingUpdateEvent(
        at=at_ms * NS_PER_MS,
        estimated_tokens=estimated_tokens,
        delta=10,
        parent_tool_use_id=parent_tool_use_id,
    )


def turn_end_event(
    at_ms: int = 5000,
    *,
    text: str = "Turn summary.",
    origin: Literal["agent", "injected"] = "agent",
) -> TurnEndEvent:
    """A ``TurnEndEvent`` attributed to *origin*, costing one cent with budget left."""
    return make_turn_end(at=at_ms * NS_PER_MS, text=text, origin=origin)


def follow_up_event(
    at_ms: int = 6000,
    *,
    action: Literal["replied", "waiting", "ended"] = "replied",
    reason: str | None = None,
) -> FollowUpEvent:
    """A ``FollowUpEvent`` carrying the supervisor's decision for the turn."""
    return FollowUpEvent(at=at_ms * NS_PER_MS, action=action, reason=reason)


#: When :func:`fire_launch_and_bash_cycle` stamps its Bash end, in milliseconds.
BASH_CYCLE_END_MS = 3000


def fire_launch_and_bash_cycle(
    observer: SessionObserver, *, clock: Clock[int] | None = None, result: str = "ok"
) -> None:
    """Minimum event sequence that gets session state into the loop/best rows.

    Fires a launch at 1000 ms, then a Bash start at 2000 ms and its end at
    :data:`BASH_CYCLE_END_MS`. The Bash end triggers the reporter's session
    re-read.

    Args:
        observer: The reporter observer the events are fired at.
        clock: The reporter's clock, advanced to each event's time before it
            fires so idle timing starts from the Bash end; left alone when
            omitted.
        result: The Bash call's result, ``"error"`` for a failed call.
    """
    for at_ms, event in (
        (1000, launch_event(1000)),
        (2000, tool_start_event("Bash", "bash-1", 2000)),
        (
            BASH_CYCLE_END_MS,
            tool_end_event("Bash", "bash-1", BASH_CYCLE_END_MS, result=result),
        ),
    ):
        if clock is not None:
            clock.now = at_ms
        observer(event)


def fire_launch_and_bash_start(observer: SessionObserver) -> None:
    """Fire launch and Bash start, leaving the tool in flight.

    Used by the in-flight-guard and nested-event tests, which fire a second
    event on top of the still-running Bash call and assert whether it takes
    effect or is ignored.

    Args:
        observer: The reporter observer the events are fired at.
    """
    observer(launch_event(1000))
    observer(tool_start_event("Bash", "bash-1", 1500))


def fire_launch_and_iterate_start(kit: ReporterKit, *, tool_name: str = "Bash") -> None:
    """Fire a launch at 1000 ms, then an iterate call at 2000 ms, leaving it in flight.

    Args:
        kit: The reporter the events are fired at; its clock is moved to 2000 ms.
        tool_name: The tool that runs the iterate call.
    """
    kit.reporter.observer(launch_event(1000))
    kit.clock.now = 2000
    kit.reporter.observer(
        tool_start_event(tool_name, "bash-1", 2000, input_summary="gymrat iterate")
    )


def fire_launch_and_edit_cycle(kit: ReporterKit, *, result: str = "ok") -> None:
    """Fire a launch at 1000 ms, then an Edit of ``src/archetype.ts`` from 2000 to 3000 ms.

    Args:
        kit: The reporter the events are fired at; its clock is moved to each event's time.
        result: The Edit call's result, ``"error"`` for a failed call.
    """
    kit.reporter.observer(launch_event(1000))
    kit.clock.now = 2000
    kit.reporter.observer(
        tool_start_event("Edit", "edit-1", 2000, input_summary="src/archetype.ts")
    )
    kit.clock.now = 3000
    kit.reporter.observer(tool_end_event("Edit", "edit-1", 3000, result=result))


# ---------------------------------------------------------------------------
# Reporter setup
# ---------------------------------------------------------------------------

# Fixed width for all golden-snapshot tests so frames are stable.
FRAME_WIDTH = 100

#: Patch target for the ``ErasableLive`` the live-mode reporter drives.
LIVE_CLASS_PATH = "gymrat.cli.supervise.progress.ErasableLive"


#: The root every built reporter carries. Both session readers are stubbed, so
#: nothing ever reads below it.
_UNREAD_ROOT = "repo"


def _no_progress(_root: str) -> ProgressSnapshot | None:
    return None


class ReporterKit(NamedTuple):
    """A live-mode reporter paired with the injectable clock that drives it."""

    reporter: SuperviseReporter
    clock: Clock[int]


def make_reporter(
    *,
    mode: Literal["live", "plain"] = "live",
    max_minutes: float = 480,
    max_usd: float | None = None,
    max_iterations: int | None = None,
    read_session: Callable[[], ReadSessionResult] | None = None,
    clock_start: int = 1000,
    read_progress: Callable[[str], ProgressSnapshot | None] | None = None,
    plain_write: Callable[[str], None] | None = None,
    session_id: str = SUPERVISED_SESSION_ID,
    branch: str = f"gymrat/{SUPERVISED_SESSION_ID}",
    color: bool | None = None,
    tz: tzinfo | None = UTC,
    model: str | None = None,
    effort: Effort | None = None,
    idle_warn_ms: int = IDLE_WARN_MS,
    refresh_ms: int = REFRESH_MS,
    clock: Clock[int] | None = None,
) -> ReporterKit:
    """Build a reporter with injectable dependencies for deterministic testing.

    The reporter is tracked, so the CLI tests' autouse teardown stops it.

    Args:
        mode: ``"live"`` for a Rich Live dashboard, ``"plain"`` for line-by-line output.
        max_minutes: Wall-clock cap in minutes.
        max_usd: Spend cap in USD, or ``None`` for uncapped.
        max_iterations: Iteration cap, or ``None`` for uncapped.
        read_session: Reads the current session state; defaults to an empty
            session with no baseline.
        clock_start: Start value of the ``Clock`` built when ``clock`` is omitted.
        read_progress: Reads the iterate progress sidecar; defaults to a reader
            that finds none, so no test reads a sidecar off the disk.
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
    if read_progress is None:
        read_progress = _no_progress
    reporter = create_supervise_reporter(
        root=_UNREAD_ROOT,
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
    return ReporterKit(track(reporter), clock)


def reporter_with_nested_read() -> ReporterKit:
    """A reporter whose in-flight Bash call runs a nested Read of ``src/config.ts``.

    The Bash call starts at 1500 ms, the nested Read at 2000 ms, and the clock
    stands at 5000 ms.
    """
    kit = make_reporter()
    fire_launch_and_bash_start(kit.reporter.observer)
    kit.clock.now = 2000
    kit.reporter.observer(
        tool_start_event(
            "Read",
            "nested-read-1",
            2000,
            parent_tool_use_id="bash-1",
            input_summary="src/config.ts",
        )
    )
    kit.clock.now = 5000
    return kit


def render_frame(reporter: SuperviseReporter, *, width: int = FRAME_WIDTH) -> str:
    """Render the reporter's current frame through a non-terminal console."""
    return frame_text(reporter.frame(), width=width)


def line_after(frame: str, needle: str) -> str:
    """Return the line immediately following the first line containing *needle*."""
    lines = frame.splitlines()
    idx = next(i for i, line in enumerate(lines) if needle in line)
    return lines[idx + 1]


def color_console(*, width: int = FRAME_WIDTH) -> Console:
    """A sealed console with standard color, *width* columns wide."""
    return sealed_console(width=width, no_color=False, color_system="standard")


def render_colored(renderable: RenderableType, *, width: int = FRAME_WIDTH) -> str:
    """Render ``renderable`` through a sealed console with standard color."""
    console = color_console(width=width)
    console.print(renderable)
    return console_output(console)


def lines_containing(frame: str, needle: str) -> list[str]:
    """Return the raw lines of *frame* whose text, color codes stripped, contains *needle*."""
    return [line for line in frame.splitlines() if needle in strip_sgr(line)]


def row_content(line: str) -> str:
    """Strip the panel's side borders and padding from one frame row."""
    return line.strip("│").strip()


def content_line(frame: str, needle: str) -> str:
    """The sole frame row containing *needle*, with panel border and padding stripped."""
    lines = lines_containing(frame, needle)
    assert len(lines) == 1, f"expected exactly one line containing {needle!r}, got {lines}"
    return row_content(lines[0])
