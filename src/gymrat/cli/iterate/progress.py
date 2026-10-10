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

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from rich.console import Console, Group, RenderableType
from rich.spinner import Spinner
from rich.text import Text

from gymrat.cli.iterate.state import (
    JudgeDetail,
    advance,
    initial_state,
    judge_segments,
    plain_line,
)
from gymrat.cli.live_display import LiveDisplayMixin
from gymrat.cli.progress import (
    COMPACT_HEIGHT_THRESHOLD,
    SPINNER_NAME,
    clock_text,
    compact_progress,
    passes_progress,
    phase_text,
)
from gymrat.cli.style import (
    GLYPH_ALERT,
    GLYPH_DONE,
    GLYPH_PENDING,
    STYLE_ALERT,
    STYLE_DONE,
    STYLE_LABEL,
    STYLE_META,
    STYLE_PENDING,
    STYLE_RUNNING,
    STYLE_TIMER_DONE,
    STYLE_TIMER_RUNNING,
)
from gymrat.metric_name import format_inline
from gymrat.metric_name import parse as parse_metric_name
from gymrat.progress_events import (
    ConfirmStarted,
    PassFinished,
    PassStarted,
    ProgressEvent,
)
from gymrat.utils import MS_PER_SECOND, format_duration

if TYPE_CHECKING:
    from collections.abc import Callable

    from rich.progress import Progress, TaskID

    from gymrat.cli.iterate.state import NodeState, PhaseCounters
    from gymrat.cli.progress import _ClockColumn


GLYPH_SKIPPED = "\N{EN DASH}"
"""Glyph of a checklist row the iteration decided not to run."""


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
        case "skipped":
            return render_skipped_row(node)
        case _:
            return render_idle_row(node)


def render_running_row(
    node: NodeState, spinner: Spinner, running_ms: float | None
) -> RenderableType:
    """Render a running checklist row: verb, note, target, and live timer.

    Args:
        node: The row's state.
        spinner: The row's spinner, updated in place so its animation carries
            across frames.
        running_ms: How long the row has been running, or ``None`` to show no
            timer.

    Returns:
        A static alert-glyph line when the row is in alert state, otherwise the
        updated ``spinner``.
    """
    style = STYLE_ALERT if node.alert else STYLE_RUNNING
    text = phase_text(node.gerund, node.note, node.target)
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
    if node.alert:
        text.append(f"{GLYPH_ALERT} ", style=STYLE_ALERT)
    else:
        text.append(f"{GLYPH_DONE} ", style=STYLE_DONE)
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
    glyph = GLYPH_ALERT if node.alert else GLYPH_PENDING
    text.append(f"{glyph} {node.noun}", style=STYLE_PENDING)
    if node.hint:
        text.append(f" ({node.hint})", style=STYLE_PENDING)
    return text


def render_skipped_row(node: NodeState) -> Text:
    """Render a phase the iteration decided not to run: its glyph, noun, and optional hint."""
    text = Text(f"{GLYPH_SKIPPED} {node.noun} skipped", style=STYLE_PENDING)
    if node.hint:
        text.append(f" ({node.hint})", style=STYLE_PENDING)
    return text


# ---------------------------------------------------------------------------
# Judge detail builder
# ---------------------------------------------------------------------------


def build_judge_detail(detail: JudgeDetail) -> Text:
    """Build the rich Text detail for the judge's done row.

    Args:
        detail: The judge's verdict, split into words by
            :func:`~gymrat.cli.iterate.state.judge_segments`. Regressed names
            are highlighted inline; the rest of the wording is dimmed.

    Returns:
        A styled ``Text`` for the judge row's detail.
    """
    text = Text()
    for segment in judge_segments(detail):
        if segment.role == "name":
            text.append_text(
                Text.from_markup(format_inline(parse_metric_name(segment.text)), emoji=False)
            )
        else:
            text.append(segment.text, style=STYLE_META)
    return text


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------


@dataclass(slots=True)
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
        clock: Monotonic clock returning seconds, behind the elapsed-time and
            ETA display.
        verbose: When ``True`` the live display is not transient — frames
            persist after the renderer stops.
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
        clock: Callable[[], float],
        verbose: bool = False,
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

        self._spinners: dict[str, Spinner] = {}
        self._pass_view = _PhaseView()
        self._confirm_view = _PhaseView()
        # The single bar a short terminal shows in place of the checklist.
        self._compact_view = _PhaseView()

        if self._resolve_live(mode):
            self._init_live()

    def _init_live(self) -> None:
        if self._console.height < COMPACT_HEIGHT_THRESHOLD:
            self._compact_view.bar, self._compact_view.clock_col = compact_progress(
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
        if self._compact_view.bar is not None:
            return self._compact_view.bar

        rows: list[RenderableType] = [self._header_text()]
        for node in self._state.nodes.all_nodes:
            if node.status == "skipped" and not self._skip_follows_regression(node):
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

    def _skip_follows_regression(self, node: NodeState) -> bool:
        nodes = self._state.nodes
        verdict = nodes.judge.detail
        return (
            node is nodes.confirm
            and isinstance(verdict, JudgeDetail)
            and bool(verdict.regressed_names)
        )

    def _header_text(self) -> Text:
        header = Text()
        header.append(f"iterate #{self._seq}", style=STYLE_LABEL)

        header.append(" · ", style=STYLE_META)
        header.append(f"session {self._session_id}", style=STYLE_META)

        if self._start_clock_time is None:
            return header
        elapsed_ms = (self._clock() - self._start_clock_time) * MS_PER_SECOND

        header.append(" · ", style=STYLE_META)
        eta_ms = self._state.pass_phase.eta.eta_ms
        if eta_ms is None:
            header.append(f"{format_duration(elapsed_ms)} elapsed", style=STYLE_META)
        else:
            header.append_text(clock_text(elapsed_ms, eta_ms))
        return header

    def _spinner_for(self, node: NodeState) -> Spinner:
        """Return the row's spinner, created once so its animation stays continuous."""
        spinner = self._spinners.get(node.noun)
        if spinner is None:
            spinner = Spinner(SPINNER_NAME)
            self._spinners[node.noun] = spinner
        return spinner

    def _bar_for(self, node: NodeState) -> Progress | None:
        nodes = self._state.nodes
        if node is nodes.passes:
            return self._pass_view.bar
        if node is nodes.confirm:
            return self._confirm_view.bar
        return None

    def _running_elapsed_ms(self, node: NodeState) -> float | None:
        if node is not self._state.nodes.judge or node.status != "running":
            return None
        if node.start_ms <= 0:
            return None
        return self._clock() * MS_PER_SECOND - node.start_ms

    # -----------------------------------------------------------------------
    # Event handling
    # -----------------------------------------------------------------------

    def report(self, event: ProgressEvent) -> None:
        """Fold the event into the checklist state, then paint or print the result."""
        before = self._state
        self._state = advance(before, event)
        if before.run_start_ms is None:
            self._start_clock_time = self._clock()

        if not self._is_live:
            line = plain_line(before, self._state, event)
            if line is not None:
                self._print_milestone(line, event.at_ms, self._state.run_start_ms)
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

        compact = self._compact_view
        if compact.bar is not None:
            if compact.task_id is None:
                compact.task_id = compact.bar.add_task(
                    self._state.nodes.passes.gerund, total=self._state.total, target=event.label
                )
            elif is_confirm:
                compact.bar.update(compact.task_id, target=event.label)
            else:
                compact.bar.update(compact.task_id, target=event.label, completed=completed)
            return

        view = self._confirm_view if is_confirm else self._pass_view
        if view.bar is None:
            return
        if view.task_id is not None:
            view.bar.update(view.task_id, target=event.label)
            return

        node = self._state.nodes.confirm if is_confirm else self._state.nodes.passes
        view.task_id = view.bar.add_task(
            node.gerund,
            total=self._state.total,
            target=event.label,
        )

    def _sync_pass_finished(self, event: PassFinished) -> None:
        is_confirm = event.phase == "confirm"
        counters = self._counters(is_confirm=is_confirm)

        eta_ms = counters.eta.eta_ms
        if eta_ms is not None:
            view = self._confirm_view if is_confirm else self._pass_view
            for column in (view.clock_col, self._compact_view.clock_col):
                if column is not None:
                    column.set_eta(eta_ms)

        self._advance_bar(is_confirm=is_confirm, completed=counters.eta.completed)

    def _advance_bar(self, *, is_confirm: bool, completed: int) -> None:
        compact = self._compact_view
        if compact.bar is not None:
            if compact.task_id is not None:
                compact.bar.update(compact.task_id, completed=completed)
            return
        view = self._confirm_view if is_confirm else self._pass_view
        if view.bar is not None and view.task_id is not None:
            view.bar.update(view.task_id, completed=completed)

    def _start_confirm_task(self) -> None:
        """Swap the compact bar over to the confirm run, or open the confirm row's bar."""
        compact = self._compact_view
        if compact.bar is not None:
            if compact.task_id is not None:
                compact.bar.remove_task(compact.task_id)
            compact.task_id = compact.bar.add_task(
                self._state.nodes.confirm.gerund, total=self._state.total
            )
            if compact.clock_col is not None:
                compact.clock_col.set_eta(0)
            return

        if self._confirm_view.bar is not None and self._confirm_view.task_id is None:
            confirm = self._state.nodes.confirm
            self._confirm_view.task_id = self._confirm_view.bar.add_task(
                confirm.gerund, total=self._state.total, note=confirm.note
            )

    def _counters(self, *, is_confirm: bool) -> PhaseCounters:
        return self._state.confirm_phase if is_confirm else self._state.pass_phase
