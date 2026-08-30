"""Progress reporter for a supervised optimization run.

Event handlers, factory, and session-reading helpers for the supervise
dashboard. Frame builders live in :mod:`.frame` and state types
in :mod:`.state`.
"""

from __future__ import annotations

import contextlib
import logging
import sys
from collections import deque
from typing import TYPE_CHECKING, Literal, assert_never

from rich.live import Live

from gymrat.cli.console import stderr_console
from gymrat.cli.supervise.frame import (
    _build_loop_text,
    _format_cost,
    build_frame,
    format_caps,
)
from gymrat.cli.supervise.state import (
    IDLE_WARN_MS,
    Capped,
    CapType,
    Ended,
    FinishedTool,
    InFlight,
    ReadSessionResult,
    ReporterCtx,
    Starting,
    SuperviseReporter,
    TrackedTool,
)
from gymrat.session.clock import now_ms
from gymrat.session.paths import session_jsonl_path
from gymrat.session.progress_file import read_progress as _default_read_progress
from gymrat.session.records import IterationRecord, KeepRecord, SessionRecord
from gymrat.session.store import fold_session, read_records
from gymrat.supervisor.events import (
    CapEvent,
    LaunchEvent,
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


def _plain_emit(ctx: ReporterCtx, text: str) -> None:
    ctx.plain_write_fn(text)


def _emit(ctx: ReporterCtx, plain_text: str) -> None:
    if ctx.is_plain:
        _plain_emit(ctx, plain_text)
    else:
        _emit_live(ctx)


def _plain_loop_update(ctx: ReporterCtx) -> None:
    if not ctx.is_plain:
        return
    loop_text = _build_loop_text(ctx.session_result, ctx.max_iterations)
    if loop_text not in {ctx.last_loop_text, "no session yet"}:
        ctx.last_loop_text = loop_text
        _plain_emit(ctx, loop_text)


def _refresh_session(ctx: ReporterCtx) -> None:
    _try_read_session(ctx)
    _plain_loop_update(ctx)
    _emit_live(ctx)


def _next_liveness_after_tool_end(
    ctx: ReporterCtx, event: ToolEndEvent
) -> Capped | InFlight | Ended:
    if isinstance(ctx.liveness, Capped):
        return ctx.liveness
    last = next(reversed(ctx.in_flight_tools.values()), None)
    if last is not None:
        return InFlight(
            tool_name=last.tool_name, since=last.started_at, input_summary=last.input_summary
        )
    return Ended(tool_name=event.tool_name, since=event.timestamp, result=event.result)


def _handle_cap_event(ctx: ReporterCtx, cap: CapEvent) -> None:
    ctx.liveness = Capped(cap_type=cap.cap)
    _emit(ctx, f"cap {cap.cap} — interrupting")


def _handle_launch(ctx: ReporterCtx, event: LaunchEvent) -> None:
    ctx.launch_timestamp = event.timestamp
    _try_read_session(ctx)
    _emit(ctx, format_caps(ctx.max_minutes, ctx.max_usd))


def _handle_usage_update(ctx: ReporterCtx, event: UsageUpdateEvent) -> None:
    ctx.cost_usd = event.cost_usd
    _emit(ctx, f"cost {_format_cost(event.cost_usd)}")


def _handle_tool_start(ctx: ReporterCtx, event: ToolStartEvent) -> None:
    ctx.in_flight_tools[event.tool_use_id] = TrackedTool(
        tool_name=event.tool_name, started_at=event.timestamp, input_summary=event.input_summary
    )
    if not isinstance(ctx.liveness, Capped):
        ctx.liveness = InFlight(
            tool_name=event.tool_name, since=event.timestamp, input_summary=event.input_summary
        )
    _emit_live(ctx)


def _handle_tool_end(ctx: ReporterCtx, event: ToolEndEvent) -> None:
    tracked = ctx.in_flight_tools.get(event.tool_use_id)
    tool_name = tracked.tool_name if tracked is not None else event.tool_name
    input_summary = tracked.input_summary if tracked is not None else ""
    is_bash_end = tracked is None or tool_name == "Bash"

    ctx.in_flight_tools.pop(event.tool_use_id, None)

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

    if is_bash_end:
        _refresh_session(ctx)
    else:
        _emit_live(ctx)


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
        case TextDeltaEvent() | ThinkingUpdateEvent():
            pass
        case _:  # pragma: no cover - exhaustive over the event union
            assert_never(event)


def _stderr_write(text: str) -> None:
    sys.stderr.write(f"{text}\n")


def _build_ctx(  # noqa: PLR0913 - mirrors create_supervise_reporter parameters
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
        plain_write_fn=plain_write,
        warn_fn=plain_write,
        live=None,
    )


def create_supervise_reporter(  # noqa: PLR0913 - one parameter per reporter knob
    *,
    root: str,
    max_minutes: float,
    max_usd: float | None = None,
    max_iterations: int | None = None,
    mode: Literal["live", "plain"],
    now: Callable[[], int] | None = None,
    read_session: Callable[[], ReadSessionResult] | None = None,
    label: str = "",
    session_id: str = "",
    branch: str = "",
    plain_write: Callable[[str], None] | None = None,
    read_progress: Callable[[str], ProgressSnapshot | None] | None = None,
    color: bool = True,
) -> SuperviseReporter:
    """Build the observer/stop/frame/warn surface for the supervise dashboard."""
    is_plain = mode == "plain"
    ctx = _build_ctx(
        root=root,
        max_minutes=max_minutes,
        max_usd=max_usd,
        max_iterations=max_iterations,
        is_plain=is_plain,
        now=now if now is not None else now_ms,
        read_session=read_session if read_session is not None else make_default_read(root),
        read_progress=read_progress if read_progress is not None else _default_read_progress,
        plain_write=plain_write if plain_write is not None else _stderr_write,
        label=label,
        session_id=session_id,
        branch=branch,
    )

    # Live is created after ctx — Rich's Live.__init__ eagerly calls
    # get_renderable() to size the initial layout, so ctx must be populated.
    if not is_plain:
        ctx.live = Live(
            console=stderr_console(color_flag=color),
            refresh_per_second=1,
            transient=True,
            get_renderable=lambda: build_frame(ctx),
        )
        ctx.live.start()

    def stop() -> None:
        if ctx.live is not None:
            with contextlib.suppress(Exception):
                ctx.live.stop()

    def warn(message: str) -> None:
        if ctx.is_plain:
            _plain_emit(ctx, message)
        elif ctx.live is not None:
            ctx.live.console.print(message)

    ctx.warn_fn = warn

    return SuperviseReporter(
        observer=lambda event: _handle_event(ctx, event),
        stop=stop,
        frame=lambda: build_frame(ctx),
        warn=warn,
    )


def _find_best_kept_iteration(
    records: list[SessionLogRecord], committed_seqs: set[int]
) -> tuple[float | None, int | None, str | None]:
    """Return ``(delta_pct, seq, primary_label)`` for the best committed keep."""
    best_delta_pct: float | None = None
    best_seq: int | None = None
    primary_label: str | None = None

    for r in records:
        if (
            isinstance(r, IterationRecord)
            and r.seq in committed_seqs
            and r.primary.delta_pct is not None
            and (best_delta_pct is None or r.primary.delta_pct < best_delta_pct)
        ):
            best_delta_pct = float(r.primary.delta_pct)
            best_seq = r.seq
            primary_label = r.primary.kind if r.primary.kind != "single" else r.primary.name

    return best_delta_pct, best_seq, primary_label


def _find_baseline_sha(records: list[SessionLogRecord]) -> str | None:
    return next((r.baseline.sha for r in records if isinstance(r, SessionRecord)), None)


def make_default_read(root: str) -> Callable[[], ReadSessionResult]:
    """Build a session-reader closure that folds the live session log at ``root``."""

    def _read() -> ReadSessionResult:
        records = read_records(session_jsonl_path(root))
        state = fold_session(records)
        has_baseline = any(getattr(r, "type", None) == "baseline" for r in records)

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
