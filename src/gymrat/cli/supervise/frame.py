"""Formatting helpers and frame builders for the supervise dashboard."""

from __future__ import annotations

from datetime import UTC, datetime, tzinfo
from typing import TYPE_CHECKING, assert_never

from rich.panel import Panel
from rich.progress import BarColumn, Progress, TextColumn
from rich.table import Table
from rich.text import Text

from gymrat.cli.style import (
    STYLE_ALERT,
    STYLE_COUNT,
    STYLE_DONE,
    STYLE_META,
    STYLE_PENDING,
    STYLE_REGRESSED,
    STYLE_RUNNING,
)
from gymrat.cli.supervise.state import (
    IDLE_WARN_MS,
    Capped,
    Ended,
    FinishedTool,
    InFlight,
    ReadSessionResult,
    ReporterCtx,
    Starting,
    TrackedTool,
)
from gymrat.eta import MS_PER_SECOND, format_duration, format_eta
from gymrat.model import Effect
from gymrat.report.format import format_delta
from gymrat.report.loop import SHORT_SHA_LENGTH

if TYPE_CHECKING:
    from rich.console import RenderableType

    from gymrat.cli.supervise.state import Liveness
    from gymrat.session.progress_file import ProgressSnapshot

_MS_PER_MINUTE = 60 * MS_PER_SECOND


def format_cost(usd: float) -> str:
    """Format a USD amount as a two-decimal dollar string."""
    return f"${usd:.2f}"


def _format_minutes(minutes: float) -> str:
    return f"{minutes:g}"


def _format_iter_label(count: int, max_iterations: int | None) -> str:
    return f"iter {count}/{max_iterations}" if max_iterations is not None else f"iter {count}"


def _minutes_to_ms(minutes: float) -> float:
    return minutes * _MS_PER_MINUTE


def _format_time_label(elapsed_ms: int, max_minutes: float) -> str:
    max_ms = _minutes_to_ms(max_minutes)
    return f"{format_duration(elapsed_ms)} / {format_duration(max_ms)}"


def _is_iterate_tool(tool: TrackedTool | InFlight) -> bool:
    return tool.tool_name == "Bash" and "gymrat iterate" in tool.input_summary


def _format_wall_clock(epoch_ms: int, tz: tzinfo | None) -> str:
    dt = datetime.fromtimestamp(epoch_ms / 1000, tz=UTC)
    return dt.astimezone(tz).strftime("%H:%M:%S")


def _format_result_mark(result: str) -> str:
    return "✔" if result != "error" else "✗"


def _build_time_bar(elapsed_ms: int, max_minutes: float) -> RenderableType:
    max_ms = _minutes_to_ms(max_minutes)
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


def _build_cost_text(cost_usd: float | None, max_usd: float | None) -> Text:
    cost_str = "$—" if cost_usd is None else format_cost(cost_usd)
    text = Text()
    text.append("cost ", style=STYLE_META)
    text.append(cost_str)
    if max_usd is not None:
        text.append(f" / {format_cost(max_usd)}")
    return text


def _outcome_style(outcome: str) -> str:
    """Map an iteration outcome to its display style."""
    if outcome in ("improved", "kept"):
        return STYLE_DONE
    if outcome == "regressed":
        return STYLE_REGRESSED
    return STYLE_META


def _iter_label_text(count: int, max_iterations: int | None) -> Text:
    text = Text()
    text.append(_format_iter_label(count, max_iterations), style=STYLE_COUNT)
    return text


def build_loop_text(session_result: ReadSessionResult | None, max_iterations: int | None) -> Text:
    """Build the iteration-progress summary shown in the supervise frame."""
    if session_result is None:
        return Text("no session yet", style=STYLE_PENDING)

    state = session_result.state

    if state.finalized is not None:
        text = _iter_label_text(state.iteration_count, max_iterations)
        text.append(" · finalized", style=STYLE_DONE)
        return text

    if session_result.has_baseline and state.iteration_count == 0:
        text = _iter_label_text(0, max_iterations)
        text.append(" · baseline recorded")
        return text

    if state.iteration_count == 0:
        return Text("no session yet", style=STYLE_PENDING)

    text = _iter_label_text(state.iteration_count, max_iterations)
    text.append(" · ")
    text.append(f"{state.keep_count} kept", style=STYLE_DONE)
    text.append(" · ")
    text.append(f"{state.discard_count} discarded", style=STYLE_META)

    last = state.last_iteration
    if last is not None:
        delta_pct = last.primary.delta_pct
        delta = "—" if delta_pct is None else format_delta(Effect(value=delta_pct, unit="percent"))
        style = _outcome_style(last.outcome)
        text.append(" · last ")
        text.append(delta, style=style)
        text.append(" ")
        text.append(last.outcome, style=style)
        if state.unsettled:
            text.append(", unsettled", style=STYLE_ALERT)

    return text


def _build_best_text(session_result: ReadSessionResult | None) -> Text | None:
    if session_result is None:
        return None
    if session_result.best_delta_pct is None or session_result.best_seq is None:
        return None
    delta = format_delta(Effect(value=session_result.best_delta_pct, unit="percent"))
    delta_style = STYLE_DONE if session_result.best_delta_pct < 0 else STYLE_REGRESSED
    text = Text()
    text.append("best ", style=STYLE_META)
    text.append(delta, style=delta_style)
    if session_result.primary_label is not None:
        text.append(f" {session_result.primary_label}")
    if session_result.baseline_sha is not None:
        text.append(
            f" vs baseline {session_result.baseline_sha[:SHORT_SHA_LENGTH]}", style=STYLE_META
        )
    text.append(f" (seq {session_result.best_seq})", style=STYLE_META)
    return text


def _build_liveness_text(liveness: Liveness, now: int, tz: tzinfo | None) -> Text:
    match liveness:
        case Starting():
            return Text("starting", style=STYLE_PENDING)
        case InFlight(tool_name=name, since=since, input_summary=summary):
            elapsed = format_duration(now - since)
            text = Text()
            text.append(f"{name} {elapsed}", style=STYLE_RUNNING)
            if summary:
                text.append(f"  {summary}", style=STYLE_RUNNING)
            return text
        case Ended(tool_name=name, since=since, result=result):
            ago = now - since
            if ago >= IDLE_WARN_MS:
                return Text(
                    f"idle {format_duration(ago)}"
                    f" — no tool call since {_format_wall_clock(since, tz)}"
                    f" (last: {name} {_format_result_mark(result)})",
                    style=STYLE_ALERT,
                )
            return Text(f"{name} {format_duration(ago)} ago", style=STYLE_META)
        case Capped(cap_type=cap_type):
            return Text(f"interrupting ({cap_type})", style=STYLE_ALERT)
        case _:  # pragma: no cover - exhaustive over the liveness union
            assert_never(liveness)


def _build_finished_tool_line(tool: FinishedTool, tz: tzinfo | None) -> Text:
    wall_clock = _format_wall_clock(tool.ended_at, tz)
    mark = _format_result_mark(tool.result)
    duration = format_duration(tool.duration_ms)
    return Text(
        f"  {wall_clock}  {tool.tool_name}   {tool.input_summary}  {mark} {duration}",
        style=STYLE_META,
    )


def _build_iterate_nest(
    sidecar: ProgressSnapshot | None, now_ms: int, tool_started_at: int
) -> str | None:
    if sidecar is None:
        return None
    remaining = sidecar.passes_total - sidecar.passes_completed
    if remaining > 0 and sidecar.last_pass_duration_ms > 0:
        eta_ms = remaining * sidecar.last_pass_duration_ms
        eta_text = format_eta(eta_ms)
    else:
        eta_text = ""
    elapsed = format_duration(now_ms - tool_started_at)
    text = f"passes {sidecar.passes_completed}/{sidecar.passes_total} · {elapsed}"
    if eta_text:
        text += f" · {eta_text}"
    return text


def build_frame(ctx: ReporterCtx) -> RenderableType:
    """Assemble the supervise dashboard panel from the current reporter context."""
    now = ctx.now()
    elapsed = now - ctx.launch_timestamp if ctx.launch_timestamp is not None else 0

    summary = Table.grid(padding=(0, 1))
    summary.add_column()

    summary.add_row(_build_time_bar(elapsed, ctx.max_minutes))
    summary.add_row(_build_cost_text(ctx.cost_usd, ctx.max_usd))

    loop_row = Text("loop   ", style=STYLE_META)
    loop_row.append_text(build_loop_text(ctx.session_result, ctx.max_iterations))
    summary.add_row(loop_row)

    best_text = _build_best_text(ctx.session_result)
    if best_text is not None:
        summary.add_row(best_text)

    liveness_table = Table.grid(padding=(0, 1))
    liveness_table.add_column()
    liveness_table.add_row(_build_liveness_text(ctx.liveness, now, ctx.tz))

    if isinstance(ctx.liveness, InFlight) and _is_iterate_tool(ctx.liveness):
        sidecar = ctx.read_progress_fn(ctx.root)
        nest_text = _build_iterate_nest(sidecar, now, ctx.liveness.since)
        if nest_text is not None:
            liveness_table.add_row(Text(f"  {nest_text}", style=STYLE_META))

    for tool in ctx.finished_tools:
        liveness_table.add_row(_build_finished_tool_line(tool, ctx.tz))

    body = Table.grid(padding=(0, 0))
    body.add_column()
    body.add_row(summary)
    body.add_row(liveness_table)

    title = f"supervise {ctx.label} · session {ctx.session_id} · branch {ctx.branch}"
    return Panel(body, title=title, title_align="left", border_style=STYLE_META)


def format_caps(max_minutes: float, max_usd: float | None) -> str:
    """Format the cap summary line for the launch event."""
    caps_parts = [f"{_format_minutes(max_minutes)}m"]
    if max_usd is not None:
        caps_parts.append(format_cost(max_usd))
    return f"caps {', '.join(caps_parts)}"
