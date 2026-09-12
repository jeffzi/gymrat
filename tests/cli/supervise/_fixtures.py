"""Shared test doubles, builders, and reporter factories for the supervise progress tests.

The module is name-prefixed with ``_`` so pytest never collects it: it is a
helper imported as ``tests.cli.supervise._fixtures``.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, tzinfo
from typing import TYPE_CHECKING, Any, Literal, NamedTuple

if TYPE_CHECKING:
    from collections.abc import Callable

    from gymrat.config import Effort
    from gymrat.session.progress_file import ProgressSnapshot

from gymrat.cli.supervise.progress import (
    REFRESH_MS,
    ReadSessionResult,
    SuperviseReporter,
    create_supervise_reporter,
)
from gymrat.cli.supervise.state import IDLE_WARN_MS
from gymrat.loop.start import start_session
from gymrat.session import (
    BaselineRecord,
    IterationPrimary,
    IterationRecord,
    append_record,
    session_jsonl_path,
)
from gymrat.session.store import SessionState
from gymrat.supervisor import SessionOutcome, SupervisionResult
from gymrat.supervisor.events import (
    CapAction,
    CapEvent,
    CapType,
    CompactionEvent,
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
from tests._rich import frame_text
from tests.loop.iterate._fixtures import resolved_config
from tests.session.records._fixtures import AT, finalize_record, iteration_record

__all__ = [
    "FRAME_WIDTH",
    "LIVE_CLASS_PATH",
    "Clock",
    "PlainCapture",
    "ReporterKit",
    "baseline_record",
    "cap_event",
    "empty_session_state",
    "finalize_record",
    "fire_cap",
    "fire_compaction",
    "fire_follow_up",
    "fire_launch",
    "fire_launch_and_bash_cycle",
    "fire_launch_and_bash_start",
    "fire_model_phase",
    "fire_thinking_update",
    "fire_tool_end",
    "fire_tool_start",
    "fire_turn_end",
    "fire_usage_update",
    "follow_up_event",
    "launch_event",
    "make_iteration",
    "make_plain_reporter",
    "make_read_session",
    "make_reporter",
    "make_supervision_result",
    "model_phase_event",
    "render_frame",
    "seed_session_with_baseline",
    "seed_session_with_iteration",
    "session_state",
    "session_state_three_iterations",
    "start_open_session",
    "thinking_event",
    "tool_end_event",
    "tool_start_event",
    "turn_end_event",
    "usage_event",
]


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


def empty_session_state() -> SessionState:
    """A session that has opened but measured nothing yet."""
    return SessionState(
        session=None,
        iteration_count=0,
        last_iteration=None,
        unsettled=False,
        keep_count=0,
        discard_count=0,
        target_reached_and_kept=False,
        last_seq=0,
        last_kept_commit=None,
        ends_on_gating_block=False,
        ends_on_stop=False,
        finalized=None,
    )


def session_state(**changes: Any) -> SessionState:
    """The empty session state with the named fields overridden."""
    return replace(empty_session_state(), **changes)


def baseline_record(
    *,
    label: str = ".gymrat/worktrees/baseline",
    duration_ms: float | None = None,
    at: int = AT,
) -> BaselineRecord:
    """A baseline record with an optional wall-clock duration."""
    return BaselineRecord(
        type="baseline",
        at=at,
        label=label,
        samples=({"total_ms": 15200},),
        duration_ms=duration_ms,
    )


def start_open_session(repo: str) -> None:
    """Start a gymrat session so the experiment worktree and session log exist."""
    start_session(repo, "main", resolved_config())


def seed_session_with_baseline(
    repo: str, *, baseline_duration_ms: float, label: str = ".gymrat/worktrees/baseline"
) -> None:
    """Open a session and append a single baseline record with the given duration."""
    start_open_session(repo)
    log = session_jsonl_path(repo)
    append_record(log, baseline_record(label=label, duration_ms=baseline_duration_ms))


def seed_session_with_iteration(
    repo: str,
    *,
    iteration_duration_ms: float,
    include_baseline: bool = True,
    label: str = ".gymrat/worktrees/baseline",
) -> None:
    """Seed a session whose iteration carries the given duration.

    The seeded baseline (when included) gets no ``duration_ms`` of its own —
    there is no parameter to set one — so any feasibility math a test exercises
    is driven entirely by ``iteration_duration_ms``.
    """
    start_open_session(repo)
    log = session_jsonl_path(repo)
    if include_baseline:
        append_record(log, baseline_record(label=label))
    append_record(log, iteration_record(duration_ms=iteration_duration_ms))


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
    ended_by: Literal["session", "spend-cap", "wall-clock", "guard"] = "session",
    duration_ms: int = 60_000,
    cost_usd: float = 0.05,
    message: str | None = None,
    end_reason: str | None = None,
) -> SupervisionResult:
    """Build a default-completed supervision result.

    ``cost_usd`` is duplicated onto both the ``outcome`` and the top-level
    result, mirroring the shape ``supervise`` returns.
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

# Default timestamp for fire_tool_start; fire_tool_end's default duration_ms
# is computed against it so the two stay in sync.
_DEFAULT_TOOL_START_TS = 2000

# Converts an event firer's `at_ms` (milliseconds) to the `at` field's
# nanosecond-since-epoch unit.
_NS_PER_MS = 1_000_000


def launch_event(
    at_ms: int = 1000,
    *,
    max_minutes: float = 60,
    max_usd: float | None = None,
) -> LaunchEvent:
    """A ``LaunchEvent`` stamped at *at_ms* milliseconds, with sensible cap and model defaults."""
    return LaunchEvent(
        at=at_ms * _NS_PER_MS,
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


def fire_launch(
    observer: SessionObserver,
    at_ms: int = 1000,
    *,
    max_minutes: float = 60,
    max_usd: float | None = None,
) -> None:
    """Publish a ``LaunchEvent`` with sensible defaults for cap and model fields.

    ``at_ms`` is in the test's millisecond vocabulary; the event is stamped
    ``at=at_ms * _NS_PER_MS`` (nanoseconds) so the dashboard's ingestion
    boundary (``event.at // 1_000_000``) recovers the same millisecond value.
    """
    observer(launch_event(at_ms, max_minutes=max_minutes, max_usd=max_usd))


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
        at=at_ms * _NS_PER_MS,
        tool_use_id=tool_use_id,
        tool_name=tool_name,
        input={},
        input_summary=input_summary,
        parent_tool_use_id=parent_tool_use_id,
    )


def fire_tool_start(
    observer: SessionObserver,
    tool_name: str,
    tool_use_id: str,
    at_ms: int = _DEFAULT_TOOL_START_TS,
    *,
    input_summary: str = "...",
    parent_tool_use_id: str | None = None,
) -> None:
    """Publish a ``ToolStartEvent`` at the default start timestamp used by ``fire_tool_end``."""
    observer(
        tool_start_event(
            tool_name,
            tool_use_id,
            at_ms,
            input_summary=input_summary,
            parent_tool_use_id=parent_tool_use_id,
        )
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
        at=at_ms * _NS_PER_MS,
        tool_use_id=tool_use_id,
        tool_name=tool_name,
        duration_ms=at_ms - started_at_ms,
        result=result,
        result_summary=result_summary,
        parent_tool_use_id=parent_tool_use_id,
    )


def fire_tool_end(
    observer: SessionObserver,
    tool_name: str,
    tool_use_id: str,
    at_ms: int = 3000,
    *,
    result: str = "ok",
    result_summary: str = "ok",
    parent_tool_use_id: str | None = None,
) -> None:
    """Publish a ``ToolEndEvent`` with duration measured from the default start timestamp."""
    observer(
        tool_end_event(
            tool_name,
            tool_use_id,
            at_ms,
            result=result,
            result_summary=result_summary,
            parent_tool_use_id=parent_tool_use_id,
        )
    )


def usage_event(cost_usd: float, at_ms: int = 4000) -> UsageUpdateEvent:
    """A ``UsageUpdateEvent`` carrying the given cumulative cost."""
    return UsageUpdateEvent(at=at_ms * _NS_PER_MS, cost_usd=cost_usd)


def fire_usage_update(observer: SessionObserver, cost_usd: float, at_ms: int = 4000) -> None:
    """Publish a ``UsageUpdateEvent`` carrying the given cumulative cost."""
    observer(usage_event(cost_usd, at_ms))


def cap_event(cap: CapType, at_ms: int = 5000, *, action: CapAction = "interrupting") -> CapEvent:
    """A ``CapEvent`` signaling that *cap* has fired."""
    return CapEvent(at=at_ms * _NS_PER_MS, cap=cap, action=action)


def fire_cap(
    observer: SessionObserver,
    cap: CapType,
    at_ms: int = 5000,
    *,
    action: CapAction = "interrupting",
) -> None:
    """Publish a ``CapEvent`` signaling that the given cap has fired."""
    observer(cap_event(cap, at_ms, action=action))


def fire_compaction(observer: SessionObserver, at_ms: int = 5000) -> None:
    """Publish a ``CompactionEvent`` marking a context-window compaction."""
    observer(CompactionEvent(at=at_ms * _NS_PER_MS))


def model_phase_event(
    at_ms: int,
    phase: str,
    *,
    tool_name: str | None = None,
    parent_tool_use_id: str | None = None,
) -> ModelPhaseEvent:
    """A ``ModelPhaseEvent``; *parent_tool_use_id* scopes it to a nested tool."""
    return ModelPhaseEvent(
        at=at_ms * _NS_PER_MS,
        phase=phase,  # type: ignore[arg-type]
        tool_name=tool_name,
        parent_tool_use_id=parent_tool_use_id,
    )


def fire_model_phase(
    observer: SessionObserver,
    at_ms: int,
    phase: str,
    *,
    tool_name: str | None = None,
    parent_tool_use_id: str | None = None,
) -> None:
    """Publish a ``ModelPhaseEvent``; ``tool_name``/``parent_tool_use_id`` scope it to a nested tool."""
    observer(
        model_phase_event(at_ms, phase, tool_name=tool_name, parent_tool_use_id=parent_tool_use_id)
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
        at=at_ms * _NS_PER_MS,
        estimated_tokens=estimated_tokens,
        delta=delta,
        parent_tool_use_id=parent_tool_use_id,
    )


def fire_thinking_update(
    observer: SessionObserver,
    at_ms: int,
    *,
    estimated_tokens: int = 100,
    delta: int = 10,
    parent_tool_use_id: str | None = None,
) -> None:
    """Publish a ``ThinkingUpdateEvent`` with the given token estimate and delta."""
    observer(
        thinking_event(
            at_ms,
            estimated_tokens=estimated_tokens,
            delta=delta,
            parent_tool_use_id=parent_tool_use_id,
        )
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
        at=at_ms * _NS_PER_MS,
        text=text,
        cost_usd=cost_usd,
        origin=origin,
        budget_exhausted=budget_exhausted,
    )


def fire_turn_end(
    observer: SessionObserver,
    at_ms: int = 5000,
    *,
    text: str = "Turn summary.",
    cost_usd: float = 0.01,
    origin: Literal["agent", "injected"] = "agent",
    budget_exhausted: bool = False,
) -> None:
    """Publish a ``TurnEndEvent``; ``budget_exhausted`` gates the cap-triggered path."""
    observer(
        turn_end_event(
            at_ms, text=text, cost_usd=cost_usd, origin=origin, budget_exhausted=budget_exhausted
        )
    )


def follow_up_event(
    at_ms: int = 6000,
    *,
    action: Literal["replied", "waiting", "ended"] = "replied",
    reason: str | None = None,
    text: str | None = None,
) -> FollowUpEvent:
    """A ``FollowUpEvent`` carrying the supervisor's decision for the turn."""
    return FollowUpEvent(at=at_ms * _NS_PER_MS, action=action, reason=reason, text=text)


def fire_follow_up(
    observer: SessionObserver,
    at_ms: int = 6000,
    *,
    action: Literal["replied", "waiting", "ended"] = "replied",
    reason: str | None = None,
    text: str | None = None,
) -> None:
    """Publish a ``FollowUpEvent`` with the given follow-up action."""
    observer(follow_up_event(at_ms, action=action, reason=reason, text=text))


def fire_launch_and_bash_cycle(observer: SessionObserver) -> None:
    """Minimum event sequence that gets session state into the loop/best rows.

    The Bash end triggers the reporter's session re-read.
    """
    fire_launch(observer, 1000)
    fire_tool_start(observer, "Bash", "bash-1", 2000)
    fire_tool_end(observer, "Bash", "bash-1", 3000)


def fire_launch_and_bash_start(observer: SessionObserver) -> None:
    """Fire launch and Bash start, leaving the tool in flight.

    Used by the in-flight-guard and nested-event tests, which fire a second
    event on top of the still-running Bash call and assert whether it takes
    effect or is ignored.
    """
    fire_launch(observer, 1000)
    fire_tool_start(observer, "Bash", "bash-1", 1500)


# ---------------------------------------------------------------------------
# Reporter setup
# ---------------------------------------------------------------------------

# Fixed width for all golden-snapshot tests so frames are stable.
FRAME_WIDTH = 100

#: Patch target for the ``Live`` the live-mode reporter drives.
LIVE_CLASS_PATH = "gymrat.cli.supervise.progress.Live"


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
    label: str = "ecstatic-ts",
    session_id: str = "20260813-125044-34ec",
    branch: str = "gymrat/20260813-125044-34ec",
    color: bool | None = None,
    tz: tzinfo | None = UTC,
    model: str | None = None,
    effort: Effort | None = None,
    idle_warn_ms: int = IDLE_WARN_MS,
    refresh_ms: int = REFRESH_MS,
) -> ReporterKit:
    """Build a reporter with injectable dependencies for deterministic testing.

    The ``tz`` parameter defaults to ``UTC`` so snapshot tests produce stable
    timestamps regardless of host timezone.  Pass ``tz=None`` to exercise the
    system-local fallback path.
    """
    clock = Clock(clock_start)
    if read_session is None:
        read_session = make_read_session(empty_session_state(), has_baseline=False)
    reporter = create_supervise_reporter(
        root=root,
        max_minutes=max_minutes,
        mode=mode,
        now=clock,
        read_session=read_session,
        label=label,
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
    return ReporterKit(reporter, clock)


def render_frame(reporter: SuperviseReporter, *, width: int = FRAME_WIDTH) -> str:
    """Render the reporter's current frame through a non-terminal console."""
    return frame_text(reporter.frame(), width=width)


# ---------------------------------------------------------------------------
# Plain mode helpers
# ---------------------------------------------------------------------------


class PlainCapture(NamedTuple):
    """A plain-mode reporter paired with a write recorder."""

    kit: ReporterKit
    writes: list[str]

    @property
    def reporter(self) -> SuperviseReporter:
        return self.kit.reporter

    @property
    def observer(self) -> SessionObserver:
        return self.kit.reporter.observer


def make_plain_reporter(
    *,
    max_minutes: float = 60,
    max_usd: float | None = None,
    max_iterations: int | None = None,
    read_session: Callable[[], ReadSessionResult] | None = None,
    clock_start: int = 1000,
    tz: tzinfo | None = UTC,
) -> PlainCapture:
    """Build a plain-mode reporter with a write-capturing callback.

    Each milestone line the reporter emits is appended to the ``writes`` list.
    The ``tz`` parameter defaults to ``UTC`` for stable assertions.
    """
    writes: list[str] = []
    kit = make_reporter(
        mode="plain",
        max_minutes=max_minutes,
        max_usd=max_usd,
        max_iterations=max_iterations,
        read_session=read_session,
        clock_start=clock_start,
        plain_write=writes.append,
        tz=tz,
    )
    return PlainCapture(kit, writes)
