"""Progress reporter for a supervised optimization run.

Turns a stream of :class:`~gymrat_py.supervisor.events.SessionEvent` values into
a single ``budget · loop · liveness`` status line. The reporter re-reads session
state on launch and after each Bash tool ends (or when a tool ends whose start it
never saw), and otherwise only re-renders. In the TTY modes the whole line is
rewritten on every handled event and on the periodic tick; in plain mode only
discrete milestones are printed (caps, cost, cap interruption, and a changed loop
segment).
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, assert_never

from gymrat_py.cli.status_line import RenderMode, StatusLine, create_status_line
from gymrat_py.eta import format_duration
from gymrat_py.model import Effect
from gymrat_py.report.format import format_delta
from gymrat_py.session.clock import now_ms
from gymrat_py.session.paths import session_jsonl_path
from gymrat_py.session.store import SessionState, fold_session, read_records
from gymrat_py.supervisor.events import (
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

# After 3 minutes of no tool activity, the liveness segment turns to "idle".
IDLE_WARN_MS = 180_000

type CapType = Literal["wall-clock", "spend-cap"]


@dataclass(frozen=True, slots=True)
class ReadSessionResult:
    """The folded session state plus whether a baseline has been recorded."""

    state: SessionState
    has_baseline: bool


@dataclass(frozen=True, slots=True)
class SuperviseReporter:
    """The observer/stop pair that drives the supervise progress display."""

    observer: SessionObserver
    stop: Callable[[], None]


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


@dataclass(frozen=True, slots=True)
class _Ended:
    """The last tool finished at ``since`` and nothing is running."""

    tool_name: str
    since: int


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


# ---------------------------------------------------------------------------
# Segment builders
# ---------------------------------------------------------------------------


def _build_budget_segment(
    elapsed: int, cost_usd: float | None, max_minutes: float, max_usd: float | None
) -> str:
    budget = f"{format_duration(elapsed)} / {_format_minutes(max_minutes)}m"
    if max_usd is not None:
        cost_str = _format_cost(cost_usd) if cost_usd is not None else "$—"
        budget += f" · {cost_str} / {_format_cost(max_usd)}"
    elif cost_usd is not None:
        budget += f" · {_format_cost(cost_usd)}"
    return budget


def _build_loop_segment(
    session_result: ReadSessionResult | None, max_iterations: int | None
) -> str:
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


def _build_liveness_segment(liveness: _Liveness, now: int) -> str:
    match liveness:
        case _Starting():
            return "starting…"
        case _InFlight(tool_name=name, since=since):
            return f"{name} {format_duration(now - since)}"
        case _Ended(tool_name=name, since=since):
            ago = now - since
            if ago >= IDLE_WARN_MS:
                return f"idle {format_duration(ago)}"
            return f"{name} {format_duration(ago)} ago"
        case _Capped(cap_type=cap_type):
            return f"interrupting ({cap_type})…"
        case _:  # pragma: no cover - exhaustive over the liveness union
            assert_never(liveness)


# ---------------------------------------------------------------------------
# Reporter context — mutable state shared by the event handlers
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _ReporterCtx:
    now: Callable[[], int]
    read_session_fn: Callable[[], ReadSessionResult]
    max_minutes: float
    max_usd: float | None
    max_iterations: int | None
    is_plain: bool
    status_line: StatusLine
    in_flight_tools: dict[str, _TrackedTool]
    launch_timestamp: int | None
    cost_usd: float | None
    session_result: ReadSessionResult | None
    liveness: _Liveness
    last_loop_text: str


# ---------------------------------------------------------------------------
# Context-based helpers
# ---------------------------------------------------------------------------


def _try_read_session(ctx: _ReporterCtx) -> None:
    try:
        ctx.session_result = ctx.read_session_fn()
    except Exception:  # noqa: BLE001 - a failed read must never break the display; treat as no data
        ctx.session_result = None


def _build_line(ctx: _ReporterCtx) -> str:
    now = ctx.now()
    elapsed = now - ctx.launch_timestamp if ctx.launch_timestamp is not None else 0
    return " · ".join(
        [
            _build_budget_segment(elapsed, ctx.cost_usd, ctx.max_minutes, ctx.max_usd),
            _build_loop_segment(ctx.session_result, ctx.max_iterations),
            _build_liveness_segment(ctx.liveness, now),
        ]
    )


def _emit_line(ctx: _ReporterCtx) -> None:
    if ctx.is_plain:
        return
    ctx.status_line.write(_build_line(ctx))


def _plain_loop_update(ctx: _ReporterCtx) -> None:
    if not ctx.is_plain:
        return
    loop_text = _build_loop_segment(ctx.session_result, ctx.max_iterations)
    if loop_text not in {ctx.last_loop_text, "no session yet"}:
        ctx.last_loop_text = loop_text
        ctx.status_line.write(loop_text)


def _refresh_session(ctx: _ReporterCtx) -> None:
    _try_read_session(ctx)
    _plain_loop_update(ctx)
    _emit_line(ctx)


def _next_liveness_after_tool_end(ctx: _ReporterCtx, event: ToolEndEvent) -> _Liveness:
    if isinstance(ctx.liveness, _Capped):
        return ctx.liveness
    last = next(reversed(ctx.in_flight_tools.values()), None)
    if last is not None:
        return _InFlight(tool_name=last.tool_name, since=last.started_at)
    return _Ended(tool_name=event.tool_name, since=event.timestamp)


# ---------------------------------------------------------------------------
# Per-event handlers
# ---------------------------------------------------------------------------


def _handle_cap_event(ctx: _ReporterCtx, cap: CapEvent) -> None:
    ctx.liveness = _Capped(cap_type=cap.cap)
    if ctx.is_plain:
        ctx.status_line.write(f"cap {cap.cap} — interrupting")
    else:
        _emit_line(ctx)


def _handle_launch(ctx: _ReporterCtx, event: LaunchEvent) -> None:
    ctx.launch_timestamp = event.timestamp
    _try_read_session(ctx)
    if ctx.is_plain:
        caps_parts = [f"{_format_minutes(ctx.max_minutes)}m"]
        if ctx.max_usd is not None:
            caps_parts.append(_format_cost(ctx.max_usd))
        ctx.status_line.write(f"caps {', '.join(caps_parts)}")
    else:
        _emit_line(ctx)


def _handle_usage_update(ctx: _ReporterCtx, event: UsageUpdateEvent) -> None:
    ctx.cost_usd = event.cost_usd
    if ctx.is_plain:
        ctx.status_line.write(f"cost {_format_cost(event.cost_usd)}")
    else:
        _emit_line(ctx)


def _handle_tool_start(ctx: _ReporterCtx, event: ToolStartEvent) -> None:
    ctx.in_flight_tools[event.tool_use_id] = _TrackedTool(
        tool_name=event.tool_name, started_at=event.timestamp
    )
    if not isinstance(ctx.liveness, _Capped):
        ctx.liveness = _InFlight(tool_name=event.tool_name, since=event.timestamp)
    _emit_line(ctx)


def _handle_tool_end(ctx: _ReporterCtx, event: ToolEndEvent) -> None:
    tracked = ctx.in_flight_tools.get(event.tool_use_id)
    tool_name = tracked.tool_name if tracked is not None else event.tool_name
    # A tool_end whose start we never saw is treated as a Bash end so a
    # background Bash edit still refreshes the session view.
    is_bash_end = tracked is None or tool_name == "Bash"

    ctx.in_flight_tools.pop(event.tool_use_id, None)
    if tracked is not None:
        ctx.liveness = _next_liveness_after_tool_end(ctx, event)

    if is_bash_end:
        _refresh_session(ctx)
    else:
        _emit_line(ctx)


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
            _emit_line(ctx)
        case TextDeltaEvent() | ThinkingUpdateEvent():
            pass
        case _:  # pragma: no cover - exhaustive over the event union
            assert_never(event)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def _make_default_read(root: str) -> Callable[[], ReadSessionResult]:
    def _read() -> ReadSessionResult:
        records = read_records(session_jsonl_path(root))
        return ReadSessionResult(
            state=fold_session(records),
            has_baseline=any(getattr(r, "type", None) == "baseline" for r in records),
        )

    return _read


def create_supervise_reporter(  # noqa: PLR0913 - one parameter per reporter knob, mirroring the option surface
    *,
    root: str,
    max_minutes: float,
    max_usd: float | None = None,
    max_iterations: int | None = None,
    mode: RenderMode,
    now: Callable[[], int] | None = None,
    read_session: Callable[[], ReadSessionResult] | None = None,
) -> SuperviseReporter:
    """Build the observer/stop pair that drives the supervise progress display.

    The returned ``observer`` consumes session events, re-reading session state
    on launch and after each Bash tool ends; a failed ``read_session`` call is
    swallowed and treated as "no session data" rather than propagated. ``stop``
    only tears the status line down — it neither flushes nor persists anything.
    """
    resolved_now = now if now is not None else now_ms
    resolved_read = read_session if read_session is not None else _make_default_read(root)

    def on_tick() -> str:
        return _build_line(ctx)

    ctx = _ReporterCtx(
        now=resolved_now,
        read_session_fn=resolved_read,
        max_minutes=max_minutes,
        max_usd=max_usd,
        max_iterations=max_iterations,
        is_plain=mode == "plain",
        status_line=create_status_line(mode, on_tick),
        in_flight_tools={},
        launch_timestamp=None,
        cost_usd=None,
        session_result=None,
        liveness=_Starting(),
        last_loop_text="",
    )

    def observer(event: SessionEvent) -> None:
        _handle_event(ctx, event)

    def stop() -> None:
        ctx.status_line.stop()

    return SuperviseReporter(observer=observer, stop=stop)
