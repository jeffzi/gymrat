"""Rich-based progress renderer for the ``gymrat iterate`` command.

Live mode shows a flat checklist of the iteration's phases; plain mode prints
timestamped milestone lines. The renderer owns the terminal — the ``Live``, the
spinners, and the progress bars — while the checklist's data lives in
:mod:`.state`, which every event is folded through first.

Each row of the checklist is a :class:`~gymrat.cli.iterate.state.NodeState`, a
frozen value carrying the phase's three verb forms, its timing, and its completion
status. The row functions here turn one such row into a Rich renderable, using the
spinner and progress bar the renderer owns for that row.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from rich.console import Console, Group, RenderableType
from rich.spinner import Spinner
from rich.text import Text

from gymrat.cli.iterate.state import (
    MISSING_DELTA,
    REGRESSED_NAME_CAP,
    JudgeDetail,
    advance,
    format_primary_delta,
    initial_state,
    plain_line,
)
from gymrat.cli.progress import compact_progress, passes_progress
from gymrat.cli.style import (
    COMPACT_HEIGHT_THRESHOLD,
    GLYPH_ALERT,
    GLYPH_DONE,
    GLYPH_PENDING,
    SPINNER_NAME,
    STYLE_ALERT,
    STYLE_DONE,
    STYLE_LABEL,
    STYLE_META,
    STYLE_PENDING,
    STYLE_RUNNING,
    STYLE_TIMER_DONE,
    STYLE_TIMER_RUNNING,
    STYLE_VERB,
    ErasableLive,
    LiveDisplayMixin,
)
from gymrat.eta import MS_PER_SECOND, format_clock, format_duration, format_timestamp
from gymrat.metric_name import format_inline, parse
from gymrat.progress_events import (
    ConfirmStarted,
    PassFinished,
    PassStarted,
    ProgressEvent,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from rich.progress import Progress, TaskID

    from gymrat.cli.iterate.state import NodeState, PhaseCounters
    from gymrat.cli.progress import _ClockColumn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Checklist rows
# ---------------------------------------------------------------------------


def render_row(
    node: NodeState,
    *,
    spinner: Spinner,
    bar: Progress | None,
    running_ms: float | None,
) -> RenderableType:
    """Render one checklist row in whichever state it is in.

    Args:
        node: The row to render.
        spinner: The renderer's persistent spinner for this row, updated in
            place so its animation stays continuous across frames.
        bar: The renderer's progress bar for this row, or ``None`` for a row
            that has none. A running row with a bar renders as that bar.
        running_ms: Elapsed milliseconds to show on a running row, or ``None``
            to show no timer.

    Returns:
        The renderable for the row.
    """
    match node.status:
        case "running" if bar is not None:
            return bar
        case "running":
            return render_running_row(node, spinner, running_ms)
        case "done":
            return render_done_row(node)
        case _:
            return render_idle_row(node)


def render_running_row(
    node: NodeState, spinner: Spinner, running_ms: float | None
) -> RenderableType:
    """Render a running checklist row: verb, note, target, and live timer.

    ``spinner`` is updated in place so its animation carries across frames.

    Args:
        node: The row's state.
        spinner: The row's spinner, reused from frame to frame.
        running_ms: How long the row has been running, or ``None`` to show no
            timer.

    Returns:
        A static alert-glyph line when the row is in alert state, otherwise the
        updated ``spinner``.
    """
    style = STYLE_ALERT if node.alert else STYLE_RUNNING
    text = Text()
    text.append(node.gerund, style=STYLE_VERB)
    if node.note:
        text.append(f" {node.note}", style=STYLE_META)
    if node.target:
        text.append(" · ", style=STYLE_META)
        text.append(node.target, style=STYLE_LABEL)
    if running_ms is not None:
        text.append(f" {format_duration(running_ms)}", style=STYLE_TIMER_RUNNING)
    if node.alert:
        return Text.assemble((f"{GLYPH_ALERT} ", style), text)
    spinner.update(text=text, style=style)
    return spinner


def render_done_row(node: NodeState) -> Text:
    """Render a completed phase: its glyph, past-tense label, detail, and elapsed time.

    A :class:`JudgeDetail` is styled by :func:`build_judge_detail`; a plain
    string detail gets ``STYLE_META``.

    Args:
        node: The row's state.

    Returns:
        The styled row.
    """
    text = Text()
    text.append(f"{_glyph(node)} ", style=STYLE_ALERT if node.alert else STYLE_DONE)
    text.append(node.past)
    if node.detail:
        if isinstance(node.detail, JudgeDetail):
            text.append(" ")
            text.append_text(build_judge_detail(node.detail))
        else:
            text.append(f" {node.detail}", style=STYLE_META)
    if node.elapsed_ms > 0:
        text.append(f" {format_duration(node.elapsed_ms)}", style=STYLE_TIMER_DONE)
    return text


def render_idle_row(node: NodeState) -> Text:
    """Render a not-yet-started phase: its glyph, noun, and optional hint."""
    text = Text()
    text.append(f"{_glyph(node)} {node.noun}", style=STYLE_PENDING)
    if node.hint:
        text.append(f" ({node.hint})", style=STYLE_PENDING)
    return text


def _glyph(node: NodeState) -> str:
    if node.alert:
        return GLYPH_ALERT
    match node.status:
        case "done":
            return GLYPH_DONE
        case _:
            return GLYPH_PENDING


# ---------------------------------------------------------------------------
# Judge detail builder
# ---------------------------------------------------------------------------


def build_judge_detail(detail: JudgeDetail) -> Text:
    """Build the rich Text detail for the judge's done row.

    Args:
        detail: The judge's verdict. At most :data:`REGRESSED_NAME_CAP`
            regressed names are spelled out; the rest are collapsed to ``"…"``.
            The delta renders through :func:`format_primary_delta`; the
            primary metric's name is shown only beside a printable delta.

    Returns:
        A styled ``Text`` for the judge row's detail.
    """
    delta_str = format_primary_delta(detail.primary_delta_pct)
    primary = delta_str if delta_str == MISSING_DELTA else f"{delta_str} on {detail.primary_metric}"
    regressed = detail.regressed_names

    text = Text()
    text.append(primary, style=STYLE_META)
    text.append(" · ", style=STYLE_META)
    if regressed:
        text.append(f"{len(regressed)} regressed: ", style=STYLE_META)
        names = [
            Text.from_markup(format_inline(parse(name))) for name in regressed[:REGRESSED_NAME_CAP]
        ]
        if len(regressed) > REGRESSED_NAME_CAP:
            names.append(Text.styled("…", STYLE_META))
        # Text.styled, not Text(style=...): join copies the separator's base
        # style onto the whole result, which would dim the names too.
        text.append_text(Text.styled(", ", STYLE_META).join(names))
    else:
        text.append("no gating regression", style=STYLE_META)
    return text


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------


@dataclass
class _PhaseView:
    """The bar, clock column, and task id backing one sampling phase's row."""

    bar: Progress | None = None
    clock_col: _ClockColumn | None = None
    task_id: TaskID | None = None


class IterateRenderer(LiveDisplayMixin):
    """Single-use progress renderer for ``gymrat iterate``.

    Args:
        mode: Display mode — ``"live"`` for a rich interactive checklist,
            ``"plain"`` for timestamped milestone lines.
        console: The Rich console to render into.
        seq: The 1-based iteration sequence number, shown in the header.
        session_id: Displayed in the header for identification.
        sample_count: Total passes per target — used to size progress bars and
            compute ETAs.
        metric_count: Number of metrics the judge evaluates, shown in the judge
            hint.
        primary_metric: Name of the primary metric, shown in the judge hint and
            in the judge detail line.
        verbose: When ``True`` the live display is not transient — frames
            persist after the renderer stops.
        clock: Monotonic clock returning seconds, injected for testing. When
            ``None`` the renderer skips elapsed-time and ETA display.
        checks_cmd: The shell command the ``record`` phase mentions will run at
            ``gymrat keep``. ``None`` omits the note.
        has_before_hook: Whether a before-hook node is included in the
            checklist.
        has_after_hook: Whether the record node's hint mentions a subsequent
            after hook.
    """

    def __init__(  # noqa: PLR0913, PLR0917 -- one parameter per renderer concern
        self,
        mode: Literal["live", "plain"],
        console: Console,
        seq: int,
        session_id: str,
        sample_count: int,
        metric_count: int,
        primary_metric: str,
        *,
        verbose: bool = False,
        clock: Callable[[], float] | None = None,
        checks_cmd: str | None = None,
        has_before_hook: bool = False,
        has_after_hook: bool = False,
    ) -> None:
        self._console = console
        self._seq = seq
        self._session_id = session_id
        self._verbose = verbose
        self._clock = clock
        self._start_clock_time: float | None = None

        self._state = initial_state(
            sample_count=sample_count,
            metric_count=metric_count,
            primary_metric=primary_metric,
            checks_cmd=checks_cmd,
            has_before_hook=has_before_hook,
            has_after_hook=has_after_hook,
        )

        self._is_live = mode == "live" and console.width > 0
        self._compact = False
        self._stopped = False
        self._live: ErasableLive | None = None
        self._uninstall_cleanup: Callable[[], None] = lambda: None

        self._spinners: dict[str, Spinner] = {}
        self._pass_view = _PhaseView()
        self._confirm_view = _PhaseView()

        self._compact_progress: Progress | None = None
        self._compact_clock_col: _ClockColumn | None = None
        self._compact_task_id: TaskID | None = None

        if self._is_live:
            self._init_live()

    def _init_live(self) -> None:
        self._compact = self._console.height < COMPACT_HEIGHT_THRESHOLD

        if self._compact:
            self._compact_progress, self._compact_clock_col = compact_progress(
                self._console, clock=self._clock
            )
        else:
            self._pass_view.bar, self._pass_view.clock_col = passes_progress(
                self._console, clock=self._clock
            )
            self._confirm_view.bar, self._confirm_view.clock_col = passes_progress(
                self._console, clock=self._clock
            )

        self._mount_live(transient=not self._verbose, get_renderable=self.frame)

    # -----------------------------------------------------------------------
    # Rendering
    # -----------------------------------------------------------------------

    def frame(self) -> RenderableType:
        """Return the renderable the live display paints from."""
        if self._compact and self._compact_progress is not None:
            return self._compact_progress

        rows: list[RenderableType] = [self._header_text()]
        for node in self._state.nodes.all_nodes:
            if node.status == "skipped":
                continue
            rows.append(
                render_row(
                    node,
                    spinner=self._spinner_for(node),
                    bar=self._bar_for(node),
                    running_ms=self._running_elapsed_ms(node),
                )
            )
        return Group(*rows)

    def _header_text(self) -> Text:
        header = Text()
        header.append(f"iterate #{self._seq}", style=STYLE_LABEL)

        header.append(" · ", style=STYLE_META)
        header.append(f"session {self._session_id}", style=STYLE_META)

        elapsed_ms = self._clock_elapsed_ms(self._start_clock_time)
        if elapsed_ms is None:
            return header

        header.append(" · ", style=STYLE_META)
        eta_ms = self._state.pass_phase.eta.eta_ms
        if eta_ms is None:
            header.append(f"{format_duration(elapsed_ms)} elapsed", style=STYLE_META)
        else:
            header.append(format_clock(elapsed_ms), style=STYLE_TIMER_RUNNING)
            header.append(f"/{format_clock(elapsed_ms + eta_ms)}", style=STYLE_META)
        return header

    def _spinner_for(self, node: NodeState) -> Spinner:
        """Return the row's spinner, created once so its animation stays continuous."""
        spinner = self._spinners.get(node.noun)
        if spinner is None:
            spinner = Spinner(SPINNER_NAME)
            self._spinners[node.noun] = spinner
        return spinner

    def _bar_for(self, node: NodeState) -> Progress | None:
        view = self._view_for(node)
        return view.bar if view is not None else None

    def _view_for(self, node: NodeState) -> _PhaseView | None:
        nodes = self._state.nodes
        if node is nodes.passes:
            return self._pass_view
        if node is nodes.confirm:
            return self._confirm_view
        return None

    def _running_elapsed_ms(self, node: NodeState) -> float | None:
        if node is not self._state.nodes.judge or node.status != "running":
            return None
        if self._clock is None or node.start_ms <= 0:
            return None
        return self._clock() * MS_PER_SECOND - node.start_ms

    def _clock_elapsed_ms(self, start_clock: float | None) -> float | None:
        if start_clock is None or self._clock is None:
            return None
        return (self._clock() - start_clock) * MS_PER_SECOND

    def _print_plain(self, at_ms: float, message: str) -> None:
        ts = format_timestamp(at_ms, self._state.run_start_ms)
        self._console.print(f"{ts} {message}", highlight=False, markup=False)

    # -----------------------------------------------------------------------
    # Event handling
    # -----------------------------------------------------------------------

    def report(self, event: ProgressEvent) -> None:
        """Fold the event into the checklist state, then paint or print the result."""
        before = self._state
        self._state = advance(before, event)
        if before.run_start_ms is None and self._clock is not None:
            self._start_clock_time = self._clock()

        if not self._is_live:
            line = plain_line(before, self._state, event)
            if line is not None:
                self._print_plain(event.at_ms, line)
            return

        self._sync_live(event)
        self._refresh_live()

    def _sync_live(self, event: ProgressEvent) -> None:
        """Mirror the new state onto the bars the reducer knows nothing about."""
        match event:
            case PassStarted():
                self._sync_pass_started(event)
            case PassFinished():
                self._sync_pass_finished(event)
            case ConfirmStarted():
                self._start_confirm_task()
            case _:
                pass

    def _sync_pass_started(self, event: PassStarted) -> None:
        is_confirm = event.phase == "confirm"
        completed = self._counters(is_confirm=is_confirm).eta.completed

        if self._compact and self._compact_progress is not None:
            if self._compact_task_id is None:
                self._compact_task_id = self._compact_progress.add_task(
                    "sampling", total=self._state.total, target=event.label
                )
            elif is_confirm:
                self._compact_progress.update(self._compact_task_id, target=event.label)
            else:
                self._compact_progress.update(
                    self._compact_task_id, target=event.label, completed=completed
                )
            return

        view = self._confirm_view if is_confirm else self._pass_view
        if view.bar is None:
            return
        if view.task_id is not None:
            view.bar.update(view.task_id, target=event.label)
            return

        view.task_id = view.bar.add_task(
            "confirming" if is_confirm else "sampling",
            total=self._state.total,
            target=event.label,
        )

    def _sync_pass_finished(self, event: PassFinished) -> None:
        is_confirm = event.phase == "confirm"
        counters = self._counters(is_confirm=is_confirm)

        eta_ms = counters.eta.eta_ms
        if eta_ms is not None:
            view = self._confirm_view if is_confirm else self._pass_view
            for column in (view.clock_col, self._compact_clock_col):
                if column is not None:
                    column.set_eta(eta_ms)

        self._advance_bar(is_confirm=is_confirm, completed=counters.eta.completed)

    def _advance_bar(self, *, is_confirm: bool, completed: int) -> None:
        if self._compact:
            if self._compact_progress is not None and self._compact_task_id is not None:
                self._compact_progress.update(self._compact_task_id, completed=completed)
            return
        view = self._confirm_view if is_confirm else self._pass_view
        if view.bar is not None and view.task_id is not None:
            view.bar.update(view.task_id, completed=completed)

    def _start_confirm_task(self) -> None:
        """Swap the compact bar over to the confirm run, or open the confirm row's bar."""
        if self._compact and self._compact_progress is not None:
            if self._compact_task_id is not None:
                self._compact_progress.remove_task(self._compact_task_id)
            self._compact_task_id = self._compact_progress.add_task(
                "confirming", total=self._state.total
            )
            if self._compact_clock_col is not None:
                self._compact_clock_col.set_eta(0)
            return

        if self._confirm_view.bar is not None and self._confirm_view.task_id is None:
            self._confirm_view.task_id = self._confirm_view.bar.add_task(
                "confirming", total=self._state.total, note=self._state.nodes.confirm.note
            )

    def _counters(self, *, is_confirm: bool) -> PhaseCounters:
        return self._state.confirm_phase if is_confirm else self._state.pass_phase

    def stop(self) -> None:
        """Stop the renderer and clean up any live display."""
        if not self._claim_stop():
            return
        self._uninstall_cleanup()
        if self._live is not None:
            self._live.stop()
            self._live = None
