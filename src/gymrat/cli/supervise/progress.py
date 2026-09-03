"""Progress reporter for a supervised optimization run.

Event handlers, factory, and session-reading helpers for the supervise
dashboard. Frame builders live in :mod:`.frame` and state types
in :mod:`.state`.
"""

from __future__ import annotations

import contextlib
import logging
import operator
import sys
from collections import deque
from typing import TYPE_CHECKING, Literal, assert_never

from rich.live import Live

from gymrat.cli.console import stderr_console
from gymrat.cli.supervise.frame import (
    build_frame,
    build_loop_text,
    format_caps,
    format_cost,
)
from gymrat.cli.supervise.state import (
    IDLE_WARN_MS,
    Capped,
    CapType,
    Composing,
    FinishedTool,
    InFlight,
    NestedPhase,
    NestedTool,
    ReadSessionResult,
    ReporterCtx,
    Responding,
    Starting,
    SuperviseReporter,
    Thinking,
    TrackedTool,
    Waiting,
)
from gymrat.session.clock import now_ms
from gymrat.session.paths import session_jsonl_path
from gymrat.session.progress_file import read_progress as _default_read_progress
from gymrat.session.records import BaselineRecord, IterationRecord, KeepRecord, SessionRecord
from gymrat.session.store import fold_session, read_records
from gymrat.supervisor.events import (
    CapEvent,
    LaunchEvent,
    ModelPhaseEvent,
    SessionEvent,
    TextDeltaEvent,
    ThinkingUpdateEvent,
    ToolEndEvent,
    ToolProgressEvent,
    ToolStartEvent,
    UsageUpdateEvent,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from datetime import tzinfo

    from gymrat.session.progress_file import ProgressSnapshot
    from gymrat.session.records import SessionLogRecord

logger = logging.getLogger(__name__)

__all__ = [
    "IDLE_WARN_MS",
    "CapType",
    "ReadSessionResult",
    "SuperviseReporter",
    "create_supervise_reporter",
]


# ---------------------------------------------------------------------------
# Emit / refresh
# ---------------------------------------------------------------------------


def _try_read_session(ctx: ReporterCtx) -> None:
    try:
        ctx.session_result = ctx.read_session_fn()
    except Exception as exc:
        logger.exception("session read failed")
        ctx.warn_fn(f"session read failed: {exc}")
        ctx.session_result = None


def _emit_live(ctx: ReporterCtx) -> None:
    if ctx.is_plain or ctx.live is None:
        return
    ctx.live.update(build_frame(ctx))


def _emit(ctx: ReporterCtx, plain_text: str) -> None:
    if ctx.is_plain:
        ctx.plain_write_fn(plain_text)
    else:
        _emit_live(ctx)


def _plain_loop_update(ctx: ReporterCtx) -> None:
    if not ctx.is_plain:
        return
    plain = build_loop_text(ctx.session_result, ctx.max_iterations).plain
    if plain not in {ctx.last_loop_text, "no session yet"}:
        ctx.last_loop_text = plain
        ctx.plain_write_fn(plain)


def _refresh_session(ctx: ReporterCtx) -> None:
    _try_read_session(ctx)
    _plain_loop_update(ctx)
    _emit_live(ctx)


def _next_liveness_after_tool_end(
    ctx: ReporterCtx, event: ToolEndEvent
) -> Capped | InFlight | Waiting:
    if isinstance(ctx.liveness, Capped):
        return ctx.liveness
    last_entry = next(reversed(ctx.in_flight_tools.items()), None)
    if last_entry is not None:
        tool_id, tracked = last_entry
        return InFlight(
            tool_use_id=tool_id,
            tool_name=tracked.tool_name,
            since=tracked.started_at,
            input_summary=tracked.input_summary,
        )
    return Waiting(
        since=event.timestamp,
        tool_name=event.tool_name,
        tool_ended_at=event.timestamp,
        result=event.result,
    )


# ---------------------------------------------------------------------------
# Event handlers
# ---------------------------------------------------------------------------


def _handle_cap_event(ctx: ReporterCtx, cap: CapEvent) -> None:
    ctx.liveness = Capped(cap_type=cap.cap)
    _emit(ctx, f"cap {cap.cap} — interrupting")


def _handle_launch(ctx: ReporterCtx, event: LaunchEvent) -> None:
    ctx.launch_timestamp = event.timestamp
    _try_read_session(ctx)
    _emit(ctx, format_caps(ctx.max_minutes, ctx.max_usd))


def _handle_usage_update(ctx: ReporterCtx, event: UsageUpdateEvent) -> None:
    ctx.cost_usd = event.cost_usd
    _emit(ctx, f"cost {format_cost(event.cost_usd)}")


def _handle_tool_start(ctx: ReporterCtx, event: ToolStartEvent) -> None:
    if event.parent_tool_use_id is not None:
        if event.parent_tool_use_id not in ctx.in_flight_tools:
            return
        ctx.nested[event.parent_tool_use_id] = NestedTool(
            tool_name=event.tool_name,
            input_summary=event.input_summary,
            since=event.timestamp,
        )
        ctx.nested_tool_ids[event.tool_use_id] = event.parent_tool_use_id
        return

    ctx.in_flight_tools[event.tool_use_id] = TrackedTool(
        tool_name=event.tool_name, started_at=event.timestamp, input_summary=event.input_summary
    )
    if not isinstance(ctx.liveness, Capped):
        ctx.liveness = InFlight(
            tool_use_id=event.tool_use_id,
            tool_name=event.tool_name,
            since=event.timestamp,
            input_summary=event.input_summary,
        )
    _emit_live(ctx)


def _handle_tool_end(ctx: ReporterCtx, event: ToolEndEvent) -> None:
    if event.parent_tool_use_id is not None:
        parent_id = ctx.nested_tool_ids.pop(event.tool_use_id, None)
        if parent_id is not None and isinstance(ctx.nested.get(parent_id), NestedTool):
            del ctx.nested[parent_id]
        return

    tracked = ctx.in_flight_tools.get(event.tool_use_id)
    tool_name = tracked.tool_name if tracked is not None else event.tool_name
    input_summary = tracked.input_summary if tracked is not None else ""
    should_refresh_session = tracked is None or tool_name == "Bash"

    ctx.in_flight_tools.pop(event.tool_use_id, None)
    ctx.nested.pop(event.tool_use_id, None)
    ctx.nested_tool_ids = {k: v for k, v in ctx.nested_tool_ids.items() if v != event.tool_use_id}

    ctx.finished_tools.append(
        FinishedTool(
            tool_name=tool_name,
            input_summary=input_summary,
            duration_ms=event.duration_ms,
            result=event.result,
            ended_at=event.timestamp,
        )
    )

    if tracked is not None:
        ctx.liveness = _next_liveness_after_tool_end(ctx, event)

    if should_refresh_session:
        _refresh_session(ctx)
    else:
        _emit_live(ctx)


def _handle_thinking_update(ctx: ReporterCtx, event: ThinkingUpdateEvent) -> None:
    if event.parent_tool_use_id is not None:
        return
    if isinstance(ctx.liveness, (Capped, InFlight)):
        return
    if isinstance(ctx.liveness, Thinking):
        ctx.liveness = Thinking(since=ctx.liveness.since, estimated_tokens=event.estimated_tokens)
    else:
        ctx.liveness = Thinking(since=event.timestamp, estimated_tokens=event.estimated_tokens)
    _emit_live(ctx)


def _handle_model_phase(ctx: ReporterCtx, event: ModelPhaseEvent) -> None:
    if event.parent_tool_use_id is not None:
        parent_id = event.parent_tool_use_id
        if parent_id not in ctx.in_flight_tools:
            return
        if event.phase == "turn_end":
            ctx.nested.pop(parent_id, None)
        elif not isinstance(ctx.nested.get(parent_id), NestedTool):
            tool_name = event.tool_name if event.phase == "tool_input" else None
            ctx.nested[parent_id] = NestedPhase(
                phase=event.phase, since=event.timestamp, tool_name=tool_name
            )
        return

    if isinstance(ctx.liveness, (Capped, InFlight)):
        return

    match event.phase:
        case "thinking":
            tokens = ctx.liveness.estimated_tokens if isinstance(ctx.liveness, Thinking) else 0
            ctx.liveness = Thinking(since=event.timestamp, estimated_tokens=tokens)
        case "responding":
            ctx.liveness = Responding(since=event.timestamp)
        case "tool_input":
            tool_name = event.tool_name if event.tool_name is not None else "unknown"
            ctx.liveness = Composing(tool_name=tool_name, since=event.timestamp)
        case "turn_end":
            last = ctx.finished_tools[-1] if ctx.finished_tools else None
            ctx.liveness = Waiting(
                since=event.timestamp,
                tool_name=last.tool_name if last is not None else None,
                tool_ended_at=last.ended_at if last is not None else None,
                result=last.result if last is not None else None,
            )
    _emit_live(ctx)


def _handle_text_delta(ctx: ReporterCtx, event: TextDeltaEvent) -> None:
    if event.parent_tool_use_id is None:
        ctx.last_top_level_text = event.chunk


def _handle_event(ctx: ReporterCtx, event: SessionEvent) -> None:
    match event:
        case CapEvent():
            _handle_cap_event(ctx, event)
        case LaunchEvent():
            _handle_launch(ctx, event)
        case UsageUpdateEvent():
            _handle_usage_update(ctx, event)
        case ToolStartEvent():
            _handle_tool_start(ctx, event)
        case ToolEndEvent():
            _handle_tool_end(ctx, event)
        case ToolProgressEvent():
            _emit_live(ctx)
        case ThinkingUpdateEvent():
            _handle_thinking_update(ctx, event)
        case ModelPhaseEvent():
            _handle_model_phase(ctx, event)
        case TextDeltaEvent():
            _handle_text_delta(ctx, event)
        case _:  # pragma: no cover - exhaustive over the event union
            assert_never(event)


# ---------------------------------------------------------------------------
# Reporter factory
# ---------------------------------------------------------------------------


def _stderr_write(text: str) -> None:
    sys.stderr.write(f"{text}\n")


def _stop_live(live: Live | None) -> None:
    if live is not None:
        with contextlib.suppress(Exception):
            live.stop()


def _new_ctx(  # noqa: PLR0913 - one field per reporter knob
    *,
    root: str,
    max_minutes: float,
    max_usd: float | None,
    max_iterations: int | None,
    is_plain: bool,
    now: Callable[[], int],
    read_session: Callable[[], ReadSessionResult],
    read_progress: Callable[[str], ProgressSnapshot | None],
    plain_write: Callable[[str], None],
    label: str,
    session_id: str,
    branch: str,
    tz: tzinfo | None,
    no_color: bool,
    log_path: str,
) -> ReporterCtx:
    return ReporterCtx(
        now=now,
        read_session_fn=read_session,
        read_progress_fn=read_progress,
        root=root,
        max_minutes=max_minutes,
        max_usd=max_usd,
        max_iterations=max_iterations,
        is_plain=is_plain,
        label=label,
        session_id=session_id,
        branch=branch,
        in_flight_tools={},
        finished_tools=deque(maxlen=3),
        launch_timestamp=None,
        cost_usd=None,
        session_result=None,
        liveness=Starting(),
        last_loop_text="",
        tz=tz,
        plain_write_fn=plain_write,
        warn_fn=plain_write,
        live=None,
        nested={},
        nested_tool_ids={},
        no_color=no_color,
        log_path=log_path,
        last_top_level_text=None,
    )


def create_supervise_reporter(  # noqa: PLR0913 - one parameter per reporter knob
    *,
    root: str,
    max_minutes: float,
    max_usd: float | None = None,
    max_iterations: int | None = None,
    mode: Literal["live", "plain"],
    log_path: str = "",
    now: Callable[[], int] | None = None,
    read_session: Callable[[], ReadSessionResult] | None = None,
    label: str = "",
    session_id: str = "",
    branch: str = "",
    plain_write: Callable[[str], None] | None = None,
    read_progress: Callable[[str], ProgressSnapshot | None] | None = None,
    color: bool | None = None,
    tz: tzinfo | None = None,
) -> SuperviseReporter:
    """Build the observer/stop/frame/warn surface for the supervise dashboard."""
    ctx = _new_ctx(
        root=root,
        max_minutes=max_minutes,
        max_usd=max_usd,
        max_iterations=max_iterations,
        is_plain=mode == "plain",
        now=now if now is not None else now_ms,
        read_session=read_session if read_session is not None else make_default_read(root),
        read_progress=read_progress if read_progress is not None else _default_read_progress,
        plain_write=plain_write if plain_write is not None else _stderr_write,
        label=label,
        session_id=session_id,
        branch=branch,
        tz=tz,
        no_color=color is False,
        log_path=log_path,
    )

    # Live is created after ctx — Rich's Live.__init__ eagerly calls
    # get_renderable() to size the initial layout, so ctx must be populated.
    if not ctx.is_plain:
        ctx.live = Live(
            console=stderr_console(color_flag=color),
            refresh_per_second=1,
            transient=True,
            get_renderable=lambda: build_frame(ctx),
        )
        ctx.live.start()

    def warn(message: str) -> None:
        if ctx.is_plain:
            ctx.plain_write_fn(message)
        elif ctx.live is not None:
            ctx.live.console.print(message)

    ctx.warn_fn = warn
    return SuperviseReporter(
        observer=lambda event: _handle_event(ctx, event),
        stop=lambda: _stop_live(ctx.live),
        frame=lambda: build_frame(ctx),
        warn=warn,
        session_result=lambda: ctx.session_result,
        final_text=lambda: ctx.last_top_level_text,
    )


# ---------------------------------------------------------------------------
# Session reading
# ---------------------------------------------------------------------------


def _find_best_kept_iteration(
    records: list[SessionLogRecord], committed_seqs: set[int]
) -> tuple[float | None, int | None, str | None]:
    """Return ``(delta_pct, seq, primary_label)`` for the best committed keep."""
    candidates = [
        (r, delta)
        for r in records
        if isinstance(r, IterationRecord)
        and r.seq in committed_seqs
        and (delta := r.primary.delta_pct) is not None
    ]
    if not candidates:
        return None, None, None

    best, best_delta = min(candidates, key=operator.itemgetter(1))
    label = best.primary.kind if best.primary.kind != "single" else best.primary.name
    return best_delta, best.seq, label


def _find_baseline_sha(records: list[SessionLogRecord]) -> str | None:
    return next((r.baseline.sha for r in records if isinstance(r, SessionRecord)), None)


def make_default_read(root: str) -> Callable[[], ReadSessionResult]:
    """Build a session-reader closure that folds the live session log at ``root``."""

    def _read() -> ReadSessionResult:
        records = read_records(session_jsonl_path(root))
        state = fold_session(records)
        has_baseline = any(isinstance(r, BaselineRecord) for r in records)

        committed_seqs = {
            r.seq for r in records if isinstance(r, KeepRecord) and r.status == "committed"
        }
        best_delta_pct, best_seq, primary_label = _find_best_kept_iteration(records, committed_seqs)

        return ReadSessionResult(
            state=state,
            has_baseline=has_baseline,
            best_delta_pct=best_delta_pct,
            best_seq=best_seq,
            primary_label=primary_label,
            baseline_sha=_find_baseline_sha(records),
        )

    return _read
