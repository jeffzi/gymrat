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

    from gymrat.session.progress_file import ProgressSnapshot

from gymrat.cli.supervise.progress import (
    IDLE_WARN_MS,
    CapType,
    ReadSessionResult,
    SuperviseReporter,
    create_supervise_reporter,
)
from gymrat.session import IterationPrimary, IterationRecord
from gymrat.session.store import SessionState
from gymrat.supervisor.events import (
    CapEvent,
    LaunchEvent,
    SessionObserver,
    ToolEndEvent,
    ToolStartEvent,
    UsageUpdateEvent,
)
from tests._rich import frame_text
from tests.session.records._fixtures import finalize_record, iteration_record

__all__ = [
    "FRAME_WIDTH",
    "IDLE_WARN_MS",
    "Clock",
    "PlainCapture",
    "ReporterKit",
    "empty_session_state",
    "finalize_record",
    "fire_cap",
    "fire_launch",
    "fire_launch_and_bash_cycle",
    "fire_tool_end",
    "fire_tool_start",
    "fire_usage_update",
    "make_iteration",
    "make_plain_reporter",
    "make_read_session",
    "make_reporter",
    "render_frame",
    "session_state",
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
        finalized=None,
    )


def session_state(**changes: Any) -> SessionState:
    """The empty session state with the named fields overridden."""
    return replace(empty_session_state(), **changes)


def _epoch_ms_to_local_hms(epoch_ms: int) -> str:
    """Convert epoch milliseconds to local-time ``HH:MM:SS`` for test assertions.

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


def make_read_session(
    state: SessionState,
    *,
    has_baseline: bool,
    best_delta_pct: float | None = None,
    best_seq: int | None = None,
    primary_label: str | None = None,
    baseline_sha: str | None = None,
) -> Callable[[], ReadSessionResult]:
    """A ``read_session`` that always returns ``state`` and ``has_baseline``.

    Only forward a ``best_*`` / ``baseline_sha`` kwarg when the caller supplies
    a value, so callers that omit best-tracking still get a valid two-field
    result.
    """
    best_kwargs: dict[str, Any] = {}
    if best_delta_pct is not None:
        best_kwargs["best_delta_pct"] = best_delta_pct
    if best_seq is not None:
        best_kwargs["best_seq"] = best_seq
    if primary_label is not None:
        best_kwargs["primary_label"] = primary_label
    if baseline_sha is not None:
        best_kwargs["baseline_sha"] = baseline_sha
    result = ReadSessionResult(state=state, has_baseline=has_baseline, **best_kwargs)
    return lambda: result


def _throwing_read() -> ReadSessionResult:
    message = "no session file"
    raise RuntimeError(message)


# ---------------------------------------------------------------------------
# Event firers
# ---------------------------------------------------------------------------


def fire_launch(
    observer: SessionObserver,
    timestamp: int = 1000,
    *,
    max_minutes: float = 60,
    max_usd: float | None = None,
) -> None:
    observer(
        LaunchEvent(
            timestamp=timestamp,
            head_sha="abc123",
            dirty=False,
            max_minutes=max_minutes,
            max_usd=max_usd,
            model=None,
            runbook_path="/path/to/runbook.md",
            kickoff_summary="test kickoff",
        )
    )


def fire_tool_start(
    observer: SessionObserver,
    tool_name: str,
    tool_use_id: str,
    timestamp: int = 2000,
    *,
    input_summary: str = "...",
) -> None:
    observer(
        ToolStartEvent(
            timestamp=timestamp,
            tool_use_id=tool_use_id,
            tool_name=tool_name,
            input={},
            input_summary=input_summary,
        )
    )


def fire_tool_end(
    observer: SessionObserver,
    tool_name: str,
    tool_use_id: str,
    timestamp: int = 3000,
    *,
    result: str = "ok",
    result_summary: str = "ok",
) -> None:
    observer(
        ToolEndEvent(
            timestamp=timestamp,
            tool_use_id=tool_use_id,
            tool_name=tool_name,
            duration_ms=timestamp - 2000,
            result=result,
            result_summary=result_summary,
        )
    )


def fire_usage_update(observer: SessionObserver, cost_usd: float, timestamp: int = 4000) -> None:
    observer(UsageUpdateEvent(timestamp=timestamp, cost_usd=cost_usd))


def fire_cap(observer: SessionObserver, cap: CapType, timestamp: int = 5000) -> None:
    observer(CapEvent(timestamp=timestamp, cap=cap))


def fire_launch_and_bash_cycle(observer: SessionObserver) -> None:
    """Launch, then run a Bash tool start/end cycle at the default timestamps.

    The Bash end is what triggers the reporter's session re-read (see the
    "session re-read" tests below), so this is the minimum event sequence
    that gets session state into the loop/best rows.
    """
    fire_launch(observer, 1000)
    fire_tool_start(observer, "Bash", "bash-1", 2000)
    fire_tool_end(observer, "Bash", "bash-1", 3000)


# ---------------------------------------------------------------------------
# Reporter setup
# ---------------------------------------------------------------------------

# Fixed width for all golden-snapshot tests so frames are stable.
FRAME_WIDTH = 100


class ReporterKit(NamedTuple):
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
) -> ReporterKit:
    """Build a reporter with injectable dependencies for deterministic testing.

    The ``tz`` parameter defaults to ``UTC`` so snapshot tests produce stable
    timestamps regardless of host timezone.  Pass ``tz=None`` to exercise the
    system-local fallback path.
    """
    clock = Clock(clock_start)
    if read_session is None:
        read_session = make_read_session(empty_session_state(), has_baseline=False)
    kwargs: dict[str, Any] = {}
    if max_iterations is not None:
        kwargs["max_iterations"] = max_iterations
    if max_usd is not None:
        kwargs["max_usd"] = max_usd
    if read_progress is not None:
        kwargs["read_progress"] = read_progress
    if plain_write is not None:
        kwargs["plain_write"] = plain_write
    if color is not None:
        kwargs["color"] = color
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
        **kwargs,
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
