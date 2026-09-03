"""Formatting helpers and frame builders for the supervise dashboard."""

from __future__ import annotations

from datetime import UTC, datetime, tzinfo
from pathlib import Path
from typing import TYPE_CHECKING, assert_never

from rich.console import Group
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TextColumn
from rich.table import Table
from rich.text import Text

from gymrat.cli.style import (
    GLYPH_ALERT,
    GLYPH_DONE,
    GLYPH_ERROR,
    STYLE_ALERT,
    STYLE_COUNT,
    STYLE_DONE,
    STYLE_LABEL,
    STYLE_META,
    STYLE_PENDING,
    STYLE_REGRESSED,
    STYLE_RUNNING,
)
from gymrat.cli.supervise.state import (
    IDLE_WARN_MS,
    Capped,
    Composing,
    FinishedTool,
    InFlight,
    NestedPhase,
    NestedTool,
    ReadSessionResult,
    ReporterCtx,
    Responding,
    Starting,
    Thinking,
    TrackedTool,
    Waiting,
)
from gymrat.eta import MS_PER_SECOND, format_duration, format_eta
from gymrat.model import Effect
from gymrat.report.format import format_delta
from gymrat.report.loop import SHORT_SHA_LENGTH

if TYPE_CHECKING:
    from rich.console import RenderableType

    from gymrat.cli.supervise.state import Liveness
    from gymrat.session.progress_file import ProgressSnapshot
    from gymrat.supervisor import SupervisionResult

_MS_PER_MINUTE = 60 * MS_PER_SECOND

# Bounds for the tool-name column: floor prevents jitter across short names
# (Read, Edit, Bash); ceiling prevents a long name from pushing the layout.
_MIN_TOOL_NAME_WIDTH = 5
_MAX_TOOL_NAME_WIDTH = 8


# ---------------------------------------------------------------------------
# Time and cost formatting
# ---------------------------------------------------------------------------


def format_cost(usd: float) -> str:
    """Format a USD amount as a two-decimal dollar string."""
    return f"${usd:.2f}"


def _format_minutes(minutes: float) -> str:
    return f"{minutes:g}"


def _format_iter_label(count: int, max_iterations: int | None) -> str:
    if max_iterations is not None:
        # The noun agrees with the cap, so the capped form stays plural at any count.
        return f"{count}/{max_iterations} iterations"
    return f"{count} iteration" if count == 1 else f"{count} iterations"


def _minutes_to_ms(minutes: float) -> float:
    return minutes * _MS_PER_MINUTE


def _format_time_label(elapsed_ms: int, max_minutes: float) -> str:
    max_ms = _minutes_to_ms(max_minutes)
    remaining_ms = max(0, max_ms - elapsed_ms)
    elapsed = format_duration(elapsed_ms)
    remaining = format_duration(remaining_ms)
    return (
        f"[{STYLE_RUNNING}]{elapsed}[/{STYLE_RUNNING}]"
        f"  [{STYLE_META}]cap in {remaining}[/{STYLE_META}]"
    )


def _is_iterate_tool(tool: TrackedTool | InFlight) -> bool:
    return tool.tool_name == "Bash" and "gymrat iterate" in tool.input_summary


def _format_wall_clock(epoch_ms: int, tz: tzinfo | None) -> str:
    dt = datetime.fromtimestamp(epoch_ms / 1000, tz=UTC)
    return dt.astimezone(tz).strftime("%H:%M:%S")


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
    cost_str = format_cost(cost_usd if cost_usd is not None else 0.0)
    text = Text()
    text.append("cost ", style=STYLE_META)
    text.append(cost_str)
    if max_usd is not None:
        text.append(f" / {format_cost(max_usd)}")
    return text


# ---------------------------------------------------------------------------
# Loop and best summary
# ---------------------------------------------------------------------------


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
        return Text("baseline recorded · no iterations yet")

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
    """The best-iteration content, without its ``best`` label.

    The label is styled by each caller — dim inside the dashboard frame, plain in
    the closing summary — so it is not baked into the shared content.
    """
    if session_result is None:
        return None
    if session_result.best_delta_pct is None or session_result.best_seq is None:
        return None
    delta = format_delta(Effect(value=session_result.best_delta_pct, unit="percent"))
    delta_style = STYLE_DONE if session_result.best_delta_pct < 0 else STYLE_REGRESSED
    text = Text()
    text.append(delta, style=delta_style)
    if session_result.primary_label is not None:
        text.append(f" {session_result.primary_label}")
    if session_result.baseline_sha is not None:
        text.append(
            f" vs baseline {session_result.baseline_sha[:SHORT_SHA_LENGTH]}", style=STYLE_META
        )
    text.append(f" (iteration {session_result.best_seq})", style=STYLE_META)
    return text


# ---------------------------------------------------------------------------
# Dashboard frame
# ---------------------------------------------------------------------------


def _style_unless_no_color(style: str, *, no_color: bool) -> str:
    """Rich style string, or empty when the frame is rendered without color."""
    return "" if no_color else style


def _build_waiting_text(waiting: Waiting, now: int, tz: tzinfo | None, *, no_color: bool) -> Text:
    ago = now - waiting.since
    if ago < IDLE_WARN_MS:
        style = _style_unless_no_color(STYLE_PENDING, no_color=no_color)
        return Text(f"  waiting  {format_duration(ago)}", style=style)
    style = _style_unless_no_color(STYLE_ALERT, no_color=no_color)
    if waiting.tool_name is None:
        return Text(f"no output for {format_duration(ago)}", style=style)
    at = waiting.tool_ended_at if waiting.tool_ended_at is not None else waiting.since
    clock = _format_wall_clock(at, tz)
    mark = f" {GLYPH_ERROR}" if waiting.result == "error" else ""
    label = f"(last tool: {waiting.tool_name}{mark} at {clock})"
    return Text(f"no output for {format_duration(ago)} {label}", style=style)


def _build_liveness_text(
    liveness: Liveness, now: int, tz: tzinfo | None, *, tool_col: int, no_color: bool
) -> Text | None:
    match liveness:
        case Starting():
            return Text("starting", style=STYLE_PENDING)
        case InFlight(tool_name=name, since=since, input_summary=summary):
            elapsed = format_duration(now - since)
            wall = _format_wall_clock(since, tz)
            text = Text(no_wrap=True, overflow="ellipsis")
            text.append(f"  {wall}  {name:<{tool_col}}  {summary}  {elapsed}")
            return text
        case Thinking(since=since, estimated_tokens=tokens):
            elapsed = format_duration(now - since)
            text = Text(no_wrap=True, overflow="ellipsis")
            text.append(f"  thinking  ~{tokens:,} tokens  {elapsed}", style=STYLE_PENDING)
            return text
        case Responding(since=since) | Composing(since=since):
            elapsed = format_duration(now - since)
            if isinstance(liveness, Composing):
                label = f"preparing {liveness.tool_name}"
            else:
                label = "responding"
            style = _style_unless_no_color(STYLE_PENDING, no_color=no_color)
            return Text(f"  {label}  {elapsed}", style=style)
        case Waiting():
            return _build_waiting_text(liveness, now, tz, no_color=no_color)
        case Capped(cap_type=cap_type):
            return Text(f"interrupting ({cap_type})", style=STYLE_ALERT)
        case _:  # pragma: no cover - exhaustive over the liveness union
            assert_never(liveness)


def _build_finished_tool_line(tool: FinishedTool, tz: tzinfo | None, *, tool_col: int) -> Text:
    wall_clock = _format_wall_clock(tool.ended_at, tz)
    duration = "<1s" if tool.duration_ms < MS_PER_SECOND else format_duration(tool.duration_ms)
    is_error = tool.result == "error"

    glyph = f"{GLYPH_ERROR} " if is_error else ""
    line = f"  {wall_clock}  {tool.tool_name:<{tool_col}}  {tool.input_summary}  {glyph}{duration}"
    style = f"dim {STYLE_REGRESSED}" if is_error else STYLE_META

    return Text(line, style=style, no_wrap=True, overflow="ellipsis")


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


def _build_summary_table(ctx: ReporterCtx, elapsed_ms: int) -> Table:
    summary = Table.grid(padding=(0, 1))
    summary.add_column()

    summary.add_row(_build_time_bar(elapsed_ms, ctx.max_minutes))
    summary.add_row(_build_cost_text(ctx.cost_usd, ctx.max_usd))

    loop_row = Text("loop   ", style=STYLE_META)
    loop_row.append_text(build_loop_text(ctx.session_result, ctx.max_iterations))
    summary.add_row(loop_row)

    best_text = _build_best_text(ctx.session_result)
    if best_text is not None:
        best_row = Text("best ", style=STYLE_META)
        best_row.append_text(best_text)
        summary.add_row(best_row)

    return summary


def _tool_name_column_width(ctx: ReporterCtx) -> int:
    """Widest tool name among finished tools and the in-flight one, clamped to bounds."""
    tool_names = [t.tool_name for t in ctx.finished_tools]
    if isinstance(ctx.liveness, InFlight):
        tool_names.append(ctx.liveness.tool_name)
    raw = max((len(n) for n in tool_names), default=_MIN_TOOL_NAME_WIDTH)
    return max(_MIN_TOOL_NAME_WIDTH, min(raw, _MAX_TOOL_NAME_WIDTH))


def _build_nested_activity_line(activity: NestedTool | NestedPhase, now: int) -> Text:
    """Build a dim ``↳`` line for nested subagent activity."""
    elapsed = format_duration(now - activity.since)
    if isinstance(activity, NestedTool):
        content = f"    ↳ {activity.tool_name} {activity.input_summary}  {elapsed}"
    else:
        match activity.phase:
            case "tool_input":
                tool = activity.tool_name or "unknown"
                label = f"preparing {tool}"
            case "responding":
                label = "responding"
            case _:
                label = "thinking"
        content = f"    ↳ {label}  {elapsed}"
    return Text(content, style=STYLE_META, no_wrap=True, overflow="ellipsis")


def _build_liveness_table(ctx: ReporterCtx, now: int, *, tool_col: int) -> Table:
    liveness_table = Table.grid(padding=(0, 1))
    liveness_table.add_column()
    liveness_text = _build_liveness_text(
        ctx.liveness, now, ctx.tz, tool_col=tool_col, no_color=ctx.no_color
    )
    if liveness_text is not None:
        liveness_table.add_row(liveness_text)

    if isinstance(ctx.liveness, InFlight):
        if _is_iterate_tool(ctx.liveness):
            sidecar = ctx.read_progress_fn(ctx.root)
            nest_text = _build_iterate_nest(sidecar, now, ctx.liveness.since)
            if nest_text is not None:
                liveness_table.add_row(Text(f"  {nest_text}", style=STYLE_META))

        nested = ctx.nested.get(ctx.liveness.tool_use_id)
        if nested is not None:
            liveness_table.add_row(_build_nested_activity_line(nested, now))

    for tool in reversed(ctx.finished_tools):
        liveness_table.add_row(_build_finished_tool_line(tool, ctx.tz, tool_col=tool_col))

    return liveness_table


def _append_meta_field(text: Text, label: str, value: str) -> None:
    text.append(" · ", style=STYLE_META)
    text.append(label, style=STYLE_META)
    text.append(f" {value}")


def _build_title(ctx: ReporterCtx) -> Text:
    title = Text()
    title.append("supervise", style=STYLE_LABEL)
    if ctx.label:
        title.append(f" {ctx.label}", style=STYLE_LABEL)
    if ctx.session_id:
        _append_meta_field(title, "session", ctx.session_id)
    if ctx.branch:
        _append_meta_field(title, "branch", ctx.branch)
    return title


def build_frame(ctx: ReporterCtx) -> RenderableType:
    """Assemble the supervise dashboard panel from the current reporter context."""
    now = ctx.now()
    elapsed = now - ctx.launch_timestamp if ctx.launch_timestamp is not None else 0
    tool_col = _tool_name_column_width(ctx)

    body = Table.grid(padding=(0, 0))
    body.add_column()
    body.add_row(_build_summary_table(ctx, elapsed))
    body.add_row(_build_liveness_table(ctx, now, tool_col=tool_col))

    panel = Panel(body, title=_build_title(ctx), title_align="left", border_style=STYLE_META)
    if not ctx.log_path:
        return panel
    log_line = Text("log: ")
    log_line.append_text(_log_path_text(ctx.log_path))
    return Group(log_line, panel)


# ---------------------------------------------------------------------------
# Closing summary
# ---------------------------------------------------------------------------

# The label column of the summary rows, wide enough for "agent"; the two
# spaces on either side indent the block and separate label from content.
_SUMMARY_LABEL_WIDTH = 5

#: How a cap that stopped the session reads in the closing headline.
_CAP_LABELS: dict[str, str] = {"wall-clock": "wall-clock cap", "spend-cap": "spend cap"}


def _build_outcome_text(result: SupervisionResult) -> Text:
    """The glyph-led headline: how the run ended, then its duration and cost."""
    text = Text()
    if _completed_on_its_own(result):
        text.append(f"{GLYPH_DONE} completed", style=STYLE_DONE)
    elif result.outcome.reason == "error":
        text.append(f"{GLYPH_ERROR} error", style=STYLE_REGRESSED)
    else:
        cap = _CAP_LABELS[result.ended_by]
        text.append(f"{GLYPH_ALERT} interrupted by {cap}", style=STYLE_ALERT)
    text.append(" · ", style=STYLE_META)
    text.append(format_duration(result.duration_ms))
    text.append(" · ", style=STYLE_META)
    text.append(format_cost(result.cost_usd))
    return text


def _row_prefix(label: str) -> str:
    """The two-space-indented, padded label lead-in shared by every summary row."""
    return f"  {label:<{_SUMMARY_LABEL_WIDTH}}  "


def _summary_row(label: str, content: Text) -> Text:
    row = Text(_row_prefix(label))
    row.append_text(content)
    return row


def _abbreviate_home(path: str) -> str:
    """Shorten a path under the user's home directory to a ``~`` prefix."""
    try:
        return str(Path("~") / Path(path).relative_to(Path.home()))
    except (ValueError, RuntimeError):
        return path


def _log_path_text(log_path: str) -> Text:
    """Build a ``Text`` for a log path with an OSC 8 file hyperlink."""
    display = _abbreviate_home(log_path)
    uri = Path(log_path).resolve().as_uri()
    return Text(display, style=f"link {uri}")


def _build_agent_row(final_text: str) -> Text:
    """Build the agent summary row with continuation-line indentation.

    Each line after the first is indented to align under the content column so
    paragraph breaks in the agent's final message are preserved visually.
    """
    label = "agent"
    indent = " " * len(_row_prefix(label))
    indented = final_text.replace("\n", f"\n{indent}")
    return _summary_row(label, Text(indented))


def _completed_on_its_own(result: SupervisionResult) -> bool:
    """Whether the session ended by itself, not by a cap trip or an error."""
    return result.ended_by == "session" and result.outcome.reason != "error"


def build_summary(
    result: SupervisionResult,
    *,
    log_path: str,
    session_result: ReadSessionResult | None,
    final_text: str | None = None,
) -> Text:
    """Build the closing summary ``gymrat supervise`` prints when a run ends.

    The headline states how the run ended; the rows below it reuse the
    dashboard's best and loop renderables, so the last thing printed reads like
    the frame it replaces, and end with where the event log landed.

    When the session ended on its own (not by a cap or error) and the agent
    produced text, an ``agent`` row appears after the headline showing the
    agent's last text block with paragraph breaks preserved.
    """
    rows = [_build_outcome_text(result)]
    if _completed_on_its_own(result) and final_text is not None:
        rows.append(_build_agent_row(final_text))
    best_text = _build_best_text(session_result)
    if best_text is not None:
        rows.append(_summary_row("best", best_text))
    rows.append(_summary_row("loop", build_loop_text(session_result, None)))
    rows.append(_summary_row("log", _log_path_text(log_path)))
    return Text("\n").join(rows)


def format_caps(max_minutes: float, max_usd: float | None) -> str:
    """Format the cap summary line for the launch event."""
    caps_parts = [f"{_format_minutes(max_minutes)}m"]
    if max_usd is not None:
        caps_parts.append(format_cost(max_usd))
    return f"caps {', '.join(caps_parts)}"
