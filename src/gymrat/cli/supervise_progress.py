"""Progress reporter for a supervised optimization run.

Turns a stream of :class:`~gymrat.supervisor.events.SessionEvent` values into
a Rich ``Live`` dashboard (live mode) or discrete milestone lines (plain mode).
The dashboard shows a bordered panel with time/cost/loop summary rows and a
liveness section tracking tool activity. The reporter re-reads session state on
launch and after each Bash tool ends (or when a tool ends whose start it never
saw), and otherwise only re-renders.
"""

from __future__ import annotations

import contextlib
import logging
import sys
from collections import deque
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Literal, assert_never

if TYPE_CHECKING:
    from collections.abc import Callable

from rich.console import Console, RenderableType
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TextColumn
from rich.table import Table
from rich.text import Text

from gymrat.eta import format_duration, format_eta
from gymrat.model import Effect
from gymrat.report.format import format_delta
from gymrat.session.clock import now_ms
from gymrat.session.paths import session_jsonl_path
from gymrat.session.progress_file import ProgressSnapshot
from gymrat.session.progress_file import read_progress as _default_read_progress
from gymrat.session.records import (
    IterationRecord,
    KeepRecord,
    SessionLogRecord,
    SessionRecord,
)
from gymrat.session.store import SessionState, fold_session, read_records
from gymrat.supervisor.events import (
    CapEvent,
    LaunchEvent,
    SessionEvent,
    SessionObserver,
    TextDeltaEvent,
    ThinkingUpdateEvent,
    ToolEndEvent,
    ToolProgressEvent,
    ToolStartEvent,
    UsageUpdateEvent,
)

logger = logging.getLogger(__name__)

# After 3 minutes of no tool activity, the liveness segment turns to "idle".
IDLE_WARN_MS = 180_000

_MS_PER_MINUTE = 60_000

type CapType = Literal["wall-clock", "spend-cap"]


@dataclass(frozen=True, slots=True)
class ReadSessionResult:
    """The folded session state plus whether a baseline has been recorded.

    The ``best_*`` fields track the committed-keep iteration with the best
    primary delta.  ``_make_default_read`` computes them from the session
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
    """The observer/stop/frame/warn surface that drives the supervise progress display."""

    observer: SessionObserver
    stop: Callable[[], None]
    frame: Callable[[], RenderableType]
    warn: Callable[[str], None]


# ---------------------------------------------------------------------------
# Liveness states
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Starting:
    """No tool has run yet."""


@dataclass(frozen=True, slots=True)
class _InFlight:
    """A tool is currently running, started at ``since``."""

    tool_name: str
    since: int
    input_summary: str = ""


@dataclass(frozen=True, slots=True)
class _Ended:
    """The last tool finished at ``since`` and nothing is running."""

    tool_name: str
    since: int
    result: str


@dataclass(frozen=True, slots=True)
class _Capped:
    """A cap fired; the run is interrupting and liveness is frozen."""

    cap_type: CapType


_Liveness = _Starting | _InFlight | _Ended | _Capped


@dataclass(frozen=True, slots=True)
class _TrackedTool:
    """A tool the reporter has seen start but not yet end."""

    tool_name: str
    started_at: int
    input_summary: str


@dataclass(frozen=True, slots=True)
class _FinishedTool:
    """A tool that has finished, kept for the last-three log."""

    tool_name: str
    input_summary: str
    duration_ms: int
    result: str
    ended_at: int


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def _format_cost(usd: float) -> str:
    return f"${usd:.2f}"


def _format_minutes(minutes: float) -> str:
    # Render the wall-clock cap at its actual value: whole numbers drop the
    # decimal point (10.0 -> "10"), fractional caps keep minimal digits
    # (5.5 -> "5.5"). ``:g`` stays in fixed notation for every cap in range
    # (the flag ceiling is 35791), so no scientific notation can leak out.
    return f"{minutes:g}"


def _format_iter_label(count: int, max_iterations: int | None) -> str:
    return f"iter {count}/{max_iterations}" if max_iterations is not None else f"iter {count}"


def _format_time_label(elapsed_ms: int, max_minutes: float) -> str:
    """Render elapsed time in the format used by the dashboard header: ``2h 41m / 8h 00m``."""
    max_ms = max_minutes * _MS_PER_MINUTE
    return f"{format_duration(elapsed_ms)} / {format_duration(max_ms)}"


def _is_iterate_tool(tool: _TrackedTool | _InFlight) -> bool:
    """Whether the tool looks like a ``gymrat iterate`` invocation."""
    return tool.tool_name == "Bash" and "gymrat iterate" in tool.input_summary


def _format_wall_clock(epoch_ms: int) -> str:
    return datetime.fromtimestamp(epoch_ms / 1000, tz=UTC).astimezone().strftime("%H:%M:%S")


def _format_result_mark(result: str) -> str:
    return "✔" if result != "error" else "✗"


# ---------------------------------------------------------------------------
# Renderable builders
# ---------------------------------------------------------------------------


def _build_time_bar(elapsed_ms: int, max_minutes: float) -> RenderableType:
    max_ms = max_minutes * _MS_PER_MINUTE
    progress = Progress(
        TextColumn("time"),
        BarColumn(bar_width=None),
        TextColumn(_format_time_label(elapsed_ms, max_minutes)),
        auto_refresh=False,
        expand=True,
    )
    task = progress.add_task("time", total=max_ms, completed=int(min(elapsed_ms, max_ms)))
    progress.update(task)
    return progress


def _build_cost_text(cost_usd: float | None, max_usd: float | None) -> str:
    cost_str = "$—" if cost_usd is None else _format_cost(cost_usd)
    if max_usd is not None:
        return f"cost {cost_str} / {_format_cost(max_usd)}"
    return f"cost {cost_str}"


def _build_loop_text(session_result: ReadSessionResult | None, max_iterations: int | None) -> str:
    """Shared by the live dashboard loop row and the plain-mode loop milestone."""
    if session_result is None:
        return "no session yet"

    state = session_result.state

    if state.finalized is not None:
        return f"{_format_iter_label(state.iteration_count, max_iterations)} · finalized"

    if session_result.has_baseline and state.iteration_count == 0:
        return f"{_format_iter_label(0, max_iterations)} · baseline recorded"

    if state.iteration_count == 0:
        return "no session yet"

    parts = [
        _format_iter_label(state.iteration_count, max_iterations),
        f"{state.keep_count} kept",
        f"{state.discard_count} discarded",
    ]

    last = state.last_iteration
    if last is not None:
        delta_pct = last.primary.delta_pct
        delta = "—" if delta_pct is None else format_delta(Effect(value=delta_pct, unit="percent"))
        last_part = f"last {delta} {last.outcome}"
        if state.unsettled:
            last_part += ", unsettled"
        parts.append(last_part)

    return " · ".join(parts)


def _build_best_text(session_result: ReadSessionResult | None) -> str | None:
    """Render the best-kept row, or ``None`` when no committed keep exists."""
    if session_result is None:
        return None
    if session_result.best_delta_pct is None or session_result.best_seq is None:
        return None
    delta = format_delta(Effect(value=session_result.best_delta_pct, unit="percent"))
    parts = [f"best {delta}"]
    if session_result.primary_label is not None:
        parts.append(session_result.primary_label)
    if session_result.baseline_sha is not None:
        parts.append(f"vs baseline {session_result.baseline_sha[:7]}")
    parts.append(f"(seq {session_result.best_seq})")
    return " ".join(parts)


def _build_liveness_text(liveness: _Liveness, now: int) -> str:
    """Render the liveness line for the current state.

    Plain elapsed text for starting/in-flight/recently-ended states, or an
    idle warning with wall-clock and last-tool context once the ended state's
    elapsed time exceeds ``IDLE_WARN_MS``.
    """
    match liveness:
        case _Starting():
            return "starting"
        case _InFlight(tool_name=name, since=since, input_summary=summary):
            elapsed = format_duration(now - since)
            label = f"{name} {elapsed}"
            if summary:
                label += f"  {summary}"
            return label
        case _Ended(tool_name=name, since=since, result=result):
            ago = now - since
            if ago >= IDLE_WARN_MS:
                return (
                    f"idle {format_duration(ago)}"
                    f" — no tool call since {_format_wall_clock(since)}"
                    f" (last: {name} {_format_result_mark(result)})"
                )
            return f"{name} {format_duration(ago)} ago"
        case _Capped(cap_type=cap_type):
            return f"interrupting ({cap_type})"
        case _:  # pragma: no cover - exhaustive over the liveness union
            assert_never(liveness)


def _build_finished_tool_line(tool: _FinishedTool) -> str:
    """Render one finished tool line for the liveness log."""
    wall_clock = _format_wall_clock(tool.ended_at)
    mark = _format_result_mark(tool.result)
    duration = format_duration(tool.duration_ms)
    return f"  {wall_clock}  {tool.tool_name}   {tool.input_summary}  {mark} {duration}"


def _build_iterate_nest(
    sidecar: ProgressSnapshot | None, now_ms_val: int, tool_started_at: int
) -> str | None:
    """Render the nested iterate passes bar text when a sidecar is available."""
    if sidecar is None:
        return None
    remaining = sidecar.passes_total - sidecar.passes_completed
    if remaining > 0 and sidecar.last_pass_duration_ms > 0:
        eta_ms = remaining * sidecar.last_pass_duration_ms
        eta_text = format_eta(eta_ms)
    else:
        eta_text = ""
    elapsed = format_duration(now_ms_val - tool_started_at)
    text = f"passes {sidecar.passes_completed}/{sidecar.passes_total} · {elapsed}"
    if eta_text:
        text += f" · {eta_text}"
    return text


def _build_frame(ctx: _ReporterCtx) -> RenderableType:
    now = ctx.now()
    elapsed = now - ctx.launch_timestamp if ctx.launch_timestamp is not None else 0

    summary = Table.grid(padding=(0, 1))
    summary.add_column()

    summary.add_row(_build_time_bar(elapsed, ctx.max_minutes))

    summary.add_row(Text(_build_cost_text(ctx.cost_usd, ctx.max_usd)))

    loop_text = _build_loop_text(ctx.session_result, ctx.max_iterations)
    summary.add_row(Text(f"loop   {loop_text}"))

    best_text = _build_best_text(ctx.session_result)
    if best_text is not None:
        summary.add_row(Text(best_text))

    liveness_table = Table.grid(padding=(0, 1))
    liveness_table.add_column()

    liveness_text = _build_liveness_text(ctx.liveness, now)
    liveness_table.add_row(Text(liveness_text))

    if isinstance(ctx.liveness, _InFlight) and _is_iterate_tool(ctx.liveness):
        sidecar = ctx.read_progress_fn(ctx.root)
        nest_text = _build_iterate_nest(sidecar, now, ctx.liveness.since)
        if nest_text is not None:
            liveness_table.add_row(Text(f"  {nest_text}"))

    for tool in ctx.finished_tools:
        liveness_table.add_row(Text(_build_finished_tool_line(tool)))

    body = Table.grid(padding=(0, 0))
    body.add_column()
    body.add_row(summary)
    body.add_row(liveness_table)

    title = f"supervise {ctx.label} · session {ctx.session_id} · branch {ctx.branch}"
    return Panel(body, title=title, title_align="left")


# ---------------------------------------------------------------------------
# Reporter context — mutable state shared by the event handlers
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _ReporterCtx:
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
    in_flight_tools: dict[str, _TrackedTool]
    finished_tools: deque[_FinishedTool]
    launch_timestamp: int | None
    cost_usd: float | None
    session_result: ReadSessionResult | None
    liveness: _Liveness
    last_loop_text: str
    plain_write_fn: Callable[[str], None]
    live: Live | None


# ---------------------------------------------------------------------------
# Context-based helpers
# ---------------------------------------------------------------------------


def _try_read_session(ctx: _ReporterCtx) -> None:
    try:
        ctx.session_result = ctx.read_session_fn()
    except Exception:
        logger.exception("session read failed")
        ctx.session_result = None


def _emit_live(ctx: _ReporterCtx) -> None:
    """Update the live display if we're in live mode."""
    if ctx.is_plain or ctx.live is None:
        return
    ctx.live.update(_build_frame(ctx))


def _plain_emit(ctx: _ReporterCtx, text: str) -> None:
    """Emit a milestone line in plain mode."""
    ctx.plain_write_fn(text)


def _emit(ctx: _ReporterCtx, plain_text: str) -> None:
    """Emit ``plain_text`` as a milestone line in plain mode, or refresh the live display."""
    if ctx.is_plain:
        _plain_emit(ctx, plain_text)
    else:
        _emit_live(ctx)


def _plain_loop_update(ctx: _ReporterCtx) -> None:
    if not ctx.is_plain:
        return
    loop_text = _build_loop_text(ctx.session_result, ctx.max_iterations)
    if loop_text not in {ctx.last_loop_text, "no session yet"}:
        ctx.last_loop_text = loop_text
        _plain_emit(ctx, loop_text)


def _refresh_session(ctx: _ReporterCtx) -> None:
    _try_read_session(ctx)
    _plain_loop_update(ctx)
    _emit_live(ctx)


def _next_liveness_after_tool_end(ctx: _ReporterCtx, event: ToolEndEvent) -> _Liveness:
    if isinstance(ctx.liveness, _Capped):
        return ctx.liveness
    last = next(reversed(ctx.in_flight_tools.values()), None)
    if last is not None:
        return _InFlight(
            tool_name=last.tool_name, since=last.started_at, input_summary=last.input_summary
        )
    return _Ended(tool_name=event.tool_name, since=event.timestamp, result=event.result)


# ---------------------------------------------------------------------------
# Per-event handlers
# ---------------------------------------------------------------------------


def _handle_cap_event(ctx: _ReporterCtx, cap: CapEvent) -> None:
    ctx.liveness = _Capped(cap_type=cap.cap)
    _emit(ctx, f"cap {cap.cap} — interrupting")


def _handle_launch(ctx: _ReporterCtx, event: LaunchEvent) -> None:
    ctx.launch_timestamp = event.timestamp
    _try_read_session(ctx)
    caps_parts = [f"{_format_minutes(ctx.max_minutes)}m"]
    if ctx.max_usd is not None:
        caps_parts.append(_format_cost(ctx.max_usd))
    _emit(ctx, f"caps {', '.join(caps_parts)}")


def _handle_usage_update(ctx: _ReporterCtx, event: UsageUpdateEvent) -> None:
    ctx.cost_usd = event.cost_usd
    _emit(ctx, f"cost {_format_cost(event.cost_usd)}")


def _handle_tool_start(ctx: _ReporterCtx, event: ToolStartEvent) -> None:
    ctx.in_flight_tools[event.tool_use_id] = _TrackedTool(
        tool_name=event.tool_name, started_at=event.timestamp, input_summary=event.input_summary
    )
    if not isinstance(ctx.liveness, _Capped):
        ctx.liveness = _InFlight(
            tool_name=event.tool_name, since=event.timestamp, input_summary=event.input_summary
        )
    _emit_live(ctx)


def _handle_tool_end(ctx: _ReporterCtx, event: ToolEndEvent) -> None:
    tracked = ctx.in_flight_tools.get(event.tool_use_id)
    tool_name = tracked.tool_name if tracked is not None else event.tool_name
    input_summary = tracked.input_summary if tracked is not None else ""
    # A tool_end whose start we never saw is treated as a Bash end so a
    # background Bash edit still refreshes the session view.
    is_bash_end = tracked is None or tool_name == "Bash"

    ctx.in_flight_tools.pop(event.tool_use_id, None)

    ctx.finished_tools.append(
        _FinishedTool(
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


def _handle_event(ctx: _ReporterCtx, event: SessionEvent) -> None:
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


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def _find_best_kept_iteration(
    records: list[SessionLogRecord], committed_seqs: set[int]
) -> tuple[float | None, int | None, str | None]:
    """Return ``(delta_pct, seq, primary_label)`` for the best committed keep.

    Scans for the committed keep with the lowest primary delta.  Returns
    all-``None`` when no committed keep has a delta.
    """
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


def _make_default_read(root: str) -> Callable[[], ReadSessionResult]:
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


def _stderr_write(text: str) -> None:
    """Default plain_write: write to stderr with a newline."""
    sys.stderr.write(f"{text}\n")


def create_supervise_reporter(  # noqa: PLR0913 - one parameter per reporter knob, mirroring the option surface
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
) -> SuperviseReporter:
    """Build the observer/stop/frame/warn surface for the supervise dashboard.

    The returned ``observer`` consumes session events, re-reading session state
    on launch and after each Bash tool ends; a failed ``read_session`` call is
    swallowed and treated as "no session data" rather than propagated. ``stop``
    tears the live display down. ``frame()`` exposes the current renderable for
    testing. ``warn(message)`` prints above the live block (live mode) or as a
    plain line (plain mode).
    """
    resolved_now = now if now is not None else now_ms
    resolved_read = read_session if read_session is not None else _make_default_read(root)
    resolved_read_progress = read_progress if read_progress is not None else _default_read_progress
    resolved_plain_write = plain_write if plain_write is not None else _stderr_write
    is_plain = mode == "plain"

    ctx = _ReporterCtx(
        now=resolved_now,
        read_session_fn=resolved_read,
        read_progress_fn=resolved_read_progress,
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
        liveness=_Starting(),
        last_loop_text="",
        plain_write_fn=resolved_plain_write,
        live=None,
    )

    # Live is created after ctx so that ``get_renderable`` can close over the
    # fully-initialized context — Rich's ``Live.__init__`` eagerly calls
    # ``get_renderable()`` to size the initial layout.
    if not is_plain:
        ctx.live = Live(
            console=Console(stderr=True),
            refresh_per_second=1,
            transient=True,
            get_renderable=lambda: _build_frame(ctx),
        )
        ctx.live.start()

    def get_frame() -> RenderableType:
        return _build_frame(ctx)

    def observer(event: SessionEvent) -> None:
        _handle_event(ctx, event)

    def stop() -> None:
        if ctx.live is not None:
            with contextlib.suppress(Exception):
                ctx.live.stop()

    def warn(message: str) -> None:
        if ctx.is_plain:
            _plain_emit(ctx, message)
        elif ctx.live is not None:
            ctx.live.console.print(message)

    return SuperviseReporter(
        observer=observer,
        stop=stop,
        frame=get_frame,
        warn=warn,
    )
