"""Rich-based progress renderer for the ``gymrat iterate`` command.

Live mode shows a flat checklist of the iteration's phases; plain mode prints
timestamped milestone lines. The renderer owns the terminal — the ``Live``, the
spinners, and the progress bars — while the checklist's data lives in
:mod:`.state`, which every event is folded through first. Row rendering helpers
live in :mod:`.rows`.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal

from rich.console import Console, Group, RenderableType
from rich.live import Live
from rich.spinner import Spinner
from rich.text import Text

from gymrat.cli.iterate.rows import render_row
from gymrat.cli.iterate.state import advance, initial_state, plain_line
from gymrat.cli.progress import compact_progress, passes_progress
from gymrat.cli.style import (
    COMPACT_HEIGHT_THRESHOLD,
    LIVE_REFRESH_PER_SECOND,
    SPINNER_NAME,
    STYLE_LABEL,
    STYLE_META,
    STYLE_TIMER_RUNNING,
    LiveDisplayMixin,
)
from gymrat.eta import MS_PER_SECOND, format_clock, format_duration, format_timestamp
from gymrat.progress_events import (
    ConfirmStarted,
    PassFinished,
    PassStarted,
    ProgressEvent,
)
from gymrat.signals import install_termination_cleanup

if TYPE_CHECKING:
    from collections.abc import Callable

    from rich.progress import Progress, TaskID

    from gymrat.cli.iterate.state import NodeState
    from gymrat.cli.progress import _ClockColumn

logger = logging.getLogger(__name__)


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
        self._live: Live | None = None
        self._uninstall_cleanup: Callable[[], None] = lambda: None

        self._spinners: dict[str, Spinner] = {}
        self._pass_bar: Progress | None = None
        self._confirm_bar: Progress | None = None
        self._pass_clock_col: _ClockColumn | None = None
        self._confirm_clock_col: _ClockColumn | None = None
        self._pass_task_id: TaskID | None = None
        self._confirm_task_id: TaskID | None = None

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
            self._pass_bar, self._pass_clock_col = passes_progress(self._console, clock=self._clock)
            self._confirm_bar, self._confirm_clock_col = passes_progress(
                self._console, clock=self._clock
            )

        self._live = Live(
            console=self._console,
            auto_refresh=True,
            refresh_per_second=LIVE_REFRESH_PER_SECOND,
            transient=not self._verbose,
            redirect_stderr=False,
            get_renderable=self.frame,
        )
        self._live.start()

        self._uninstall_cleanup = install_termination_cleanup(self.clear_on_signal)

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
        nodes = self._state.nodes
        if node is nodes.passes:
            return self._pass_bar
        if node is nodes.confirm:
            return self._confirm_bar
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
        completed = self._phase_completed(is_confirm=is_confirm)

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

        bar = self._confirm_bar if is_confirm else self._pass_bar
        if bar is None:
            return
        task_id = self._confirm_task_id if is_confirm else self._pass_task_id
        if task_id is not None:
            bar.update(task_id, target=event.label)
            return

        task_id = bar.add_task(
            "confirming" if is_confirm else "sampling",
            total=self._state.total,
            target=event.label,
        )
        if is_confirm:
            self._confirm_task_id = task_id
        else:
            self._pass_task_id = task_id

    def _sync_pass_finished(self, event: PassFinished) -> None:
        is_confirm = event.phase == "confirm"
        counters = self._state.confirm_phase if is_confirm else self._state.pass_phase

        eta_ms = counters.eta.eta_ms
        if eta_ms is not None:
            phase_col = self._confirm_clock_col if is_confirm else self._pass_clock_col
            for column in (phase_col, self._compact_clock_col):
                if column is not None:
                    column.set_eta(eta_ms)

        self._advance_bar(is_confirm=is_confirm, completed=counters.eta.completed)

    def _advance_bar(self, *, is_confirm: bool, completed: int) -> None:
        if self._compact:
            if self._compact_progress is not None and self._compact_task_id is not None:
                self._compact_progress.update(self._compact_task_id, completed=completed)
            return
        bar = self._confirm_bar if is_confirm else self._pass_bar
        task_id = self._confirm_task_id if is_confirm else self._pass_task_id
        if bar is not None and task_id is not None:
            bar.update(task_id, completed=completed)

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

        if self._confirm_bar is not None and self._confirm_task_id is None:
            self._confirm_task_id = self._confirm_bar.add_task(
                "confirming", total=self._state.total, note=self._state.nodes.confirm.note
            )

    def _phase_completed(self, *, is_confirm: bool) -> int:
        counters = self._state.confirm_phase if is_confirm else self._state.pass_phase
        return counters.eta.completed

    def stop(self) -> None:
        """Stop the renderer and clean up any live display."""
        if self._stopped:
            return
        self._stopped = True
        self._uninstall_cleanup()
        if self._live is not None:
            self._live.stop()
            self._live = None
