"""Rich-based progress renderer for measure/compare commands.

Live mode (TTY) shows a header, a prepare row while the prepare command runs,
and a sampling bar with an elapsed-over-total clock. The prepare row is removed
once prepare finishes, so the display never grows past those two rows. Plain
mode (non-TTY) prints timestamped milestone lines without ANSI escape codes.

This module is the shell: it owns the terminal, the ``rich`` objects, and the
live/plain branch. Every decision about what to show lives in the pure reducer
:mod:`gymrat.cli.progress_state`.

Glyphs, verb forms, and timer colors follow the conventions in
:mod:`gymrat.cli.style`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, override

if TYPE_CHECKING:
    from collections.abc import Callable

    from gymrat.progress_events import ProgressEvent

from rich.console import Console, Group, RenderableType
from rich.live import Live
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    ProgressColumn,
    SpinnerColumn,
    Task,
    TaskID,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Column
from rich.text import Text

from gymrat.cli.progress_state import ProgressState, advance, plain_line
from gymrat.cli.style import (
    COMPACT_HEIGHT_THRESHOLD,
    LIVE_REFRESH_PER_SECOND,
    SPINNER_NAME,
    STYLE_LABEL,
    STYLE_META,
    STYLE_TIMER_DONE,
    STYLE_TIMER_RUNNING,
    STYLE_VERB,
    LiveDisplayMixin,
)
from gymrat.eta import MS_PER_SECOND, format_clock, format_duration, format_timestamp
from gymrat.signals import install_termination_cleanup


class _ClockColumn(ProgressColumn):
    """Media-player clock: elapsed over the projected total run time.

    The total is the row's elapsed time plus the remaining estimate last handed
    to :meth:`set_eta`. Until an estimate exists the total reads ``--:--``.
    """

    def __init__(self) -> None:
        super().__init__()
        self._remaining_ms: float | None = None

    def set_eta(self, ms: float) -> None:
        """Record the milliseconds left, which the total is projected from."""
        self._remaining_ms = ms

    @override
    def render(self, task: Task) -> Text:
        elapsed_ms = (task.elapsed or 0.0) * MS_PER_SECOND
        total = (
            "--:--" if self._remaining_ms is None else format_clock(elapsed_ms + self._remaining_ms)
        )
        text = Text()
        text.append(format_clock(elapsed_ms), style=STYLE_TIMER_RUNNING)
        text.append(f"/{total}", style=STYLE_META)
        return text


class _TargetColumn(ProgressColumn):
    """Renders ``· <label> ·`` between the percentage and elapsed columns."""

    @override
    def render(self, task: Task) -> Text:
        label = task.fields.get("target", "")
        if not label:
            return Text("")
        text = Text()
        text.append("· ", style=STYLE_META)
        text.append(str(label), style=STYLE_LABEL)
        text.append(" ·", style=STYLE_META)
        return text


class _PhaseColumn(ProgressColumn):
    """Renders the running verb plus the optional context the row carries.

    The task description holds the gerund (``"sampling"``); the optional
    ``note`` field adds dim context after it, and the optional ``target`` field
    adds the in-flight target label behind a dim separator.

    The column never wraps: a narrow terminal shrinks the bar rather than
    spilling the verb onto a second line and breaking the checklist alignment.
    """

    def __init__(self) -> None:
        super().__init__(table_column=Column(no_wrap=True))

    @override
    def render(self, task: Task) -> Text:
        fields = task.fields
        text = Text()
        text.append(task.description, style=STYLE_VERB)
        note = fields.get("note", "")
        if note:
            text.append(f" {note}", style=STYLE_META)
        target = fields.get("target", "")
        if target:
            text.append(" · ", style=STYLE_META)
            text.append(str(target), style=STYLE_LABEL)
        return text


def _clocked_progress(
    console: Console,
    clock: Callable[[], float] | None,
    *columns: ProgressColumn,
) -> tuple[Progress, _ClockColumn]:
    """Build a ``Progress`` from ``columns`` plus a trailing clock column.

    Args:
        console: The console the progress bar renders to.
        clock: Optional time source injected for testing; ``None`` defaults to
            ``time.monotonic``.
        columns: The columns to show before the clock column.

    Returns:
        The ``Progress`` and its ``_ClockColumn``.
    """
    clock_col = _ClockColumn()
    progress = Progress(
        *columns,
        clock_col,
        console=console,
        auto_refresh=False,
        get_time=clock,
    )
    return progress, clock_col


def compact_progress(
    console: Console, *, clock: Callable[[], float] | None = None
) -> tuple[Progress, _ClockColumn]:
    """Build a single-row compact progress bar for narrow terminals.

    Callers can update the remaining estimate as passes complete via the
    returned clock column.

    Args:
        console: The console the progress bar renders to.
        clock: Optional time source injected for testing; defaults to
            ``time.monotonic``.

    Returns:
        The ``Progress`` and its ``_ClockColumn``.
    """
    return _clocked_progress(
        console,
        clock,
        SpinnerColumn(SPINNER_NAME),
        TextColumn("{task.description}", style=STYLE_VERB),
        BarColumn(),
        TaskProgressColumn(),
        _TargetColumn(),
    )


def passes_progress(
    console: Console, *, clock: Callable[[], float] | None = None
) -> tuple[Progress, _ClockColumn]:
    """Build the sampling bar row shared by the measure/compare and iterate views.

    Callers can update the remaining estimate as passes complete via the
    returned clock column.

    Args:
        console: The console the progress bar renders to.
        clock: Optional time source injected for testing; defaults to
            ``time.monotonic``.

    Returns:
        The ``Progress`` and its ``_ClockColumn``.
    """
    return _clocked_progress(
        console,
        clock,
        SpinnerColumn(SPINNER_NAME),
        _PhaseColumn(),
        BarColumn(),
        MofNCompleteColumn(),
    )


class ProgressReporter(LiveDisplayMixin):
    """Single-use progress reporter for measure/compare commands.

    Call ``stop`` exactly once after the run ends. The reporter renders to the
    given ``console`` using either a rich ``Live`` block (live mode) or plain
    timestamped lines (plain mode).

    Args:
        mode: ``"live"`` for a rich live display or ``"plain"`` for timestamped
            milestone lines.
        console: The console to render progress to.
        target_count: How many targets (baseline + candidates) the run covers.
        sample_count: The number of samples per target, or ``None`` when the
            total is discovered at runtime from ``PassStarted.total_rounds``.
        clock: Optional time source injected for testing; defaults to
            ``time.monotonic``.
        command: A label printed in the header row of the live display.
        target_labels: Labels for each target, shown when ``target_count > 1``.
    """

    def __init__(  # noqa: PLR0913 -- mirrors the factory below
        self,
        mode: Literal["live", "plain"],
        console: Console,
        target_count: int,
        sample_count: int | None = None,
        *,
        clock: Callable[[], float] | None = None,
        command: str | None = None,
        target_labels: list[str] | None = None,
    ) -> None:
        self._console = console
        self._command = command
        self._target_labels = target_labels or []
        self._state = ProgressState.start(target_count=target_count, sample_count=sample_count)

        # Read only by ``report``; ``__init__`` branches on the local so the
        # live/plain split stays in exactly one place.
        is_live = mode == "live" and console.width > 0
        self._is_live = is_live
        self._live: Live | None = None
        self._clock_column: _ClockColumn | None = None
        self._prepare_progress: Progress | None = None
        self._pass_progress: Progress | None = None
        self._prepare_task_id: TaskID | None = None
        self._pass_task_id: TaskID | None = None
        self._compact = False
        self._stopped = False
        self._uninstall_cleanup: Callable[[], None] = lambda: None

        if is_live:
            self._init_live(console, clock)

    def _init_live(self, console: Console, clock: Callable[[], float] | None) -> None:
        self._compact = console.height < COMPACT_HEIGHT_THRESHOLD

        if self._compact:
            self._pass_progress, self._clock_column = compact_progress(console, clock=clock)
        else:
            self._prepare_progress = Progress(
                SpinnerColumn(SPINNER_NAME),
                _PhaseColumn(),
                TimeElapsedColumn(),
                console=console,
                auto_refresh=False,
                get_time=clock,
            )
            self._pass_progress, self._clock_column = passes_progress(console, clock=clock)

        self._live = Live(
            console=console,
            auto_refresh=True,
            refresh_per_second=LIVE_REFRESH_PER_SECOND,
            transient=True,
            redirect_stderr=False,
            get_renderable=self.frame,
        )
        self._live.start()

        # A termination signal exits via os._exit without unwinding the run's
        # finally block, so the live display would strand its last frame on the
        # terminal. Clearing it here keeps the terminal clean.
        self._uninstall_cleanup = install_termination_cleanup(self.clear_on_signal)

    def _header_text(self) -> Text | None:
        if not self._command:
            return None
        header = Text()
        header.append(self._command, style=STYLE_LABEL)
        sample_count = self._state.sample_count
        label_str = ", ".join(self._target_labels) if self._target_labels else ""
        sample_str = f"{sample_count} samples" if sample_count is not None else ""
        dim_parts = [p for p in (label_str, sample_str) if p]
        if dim_parts:
            header.append(" ")
            header.append(" · ".join(dim_parts), style=STYLE_META)
        return header

    def _target_field(self, label: str) -> str:
        """The label a row shows for ``label``, empty when there is only one target."""
        return label if self._state.target_count > 1 else ""

    def frame(self) -> Group:
        """Return the renderable the live display paints from."""
        parts: list[RenderableType] = []
        header = self._header_text()
        if header is not None:
            parts.append(header)
        if self._prepare_progress is not None and self._prepare_task_id is not None:
            parts.append(self._prepare_progress)
        if self._pass_progress is not None and self._pass_task_id is not None:
            parts.append(self._pass_progress)
        if not parts:
            parts.append(Text(""))
        return Group(*parts)

    def report(self, event: ProgressEvent) -> None:
        """Fold ``event`` into the run state and paint the result; ignore unrelated types."""
        before = self._state
        after = advance(before, event)
        # The reducer hands back the identical state for events the display has
        # nothing to say about.
        if after is before:
            return
        self._state = after

        if self._is_live:
            self._sync_live(after)
            return

        line = plain_line(before, after, event)
        if line is not None:
            ts = format_timestamp(event.at_ms, after.run_start_ms)
            self._console.print(f"{ts} {line}", highlight=False, markup=False)

    def _sync_live(self, state: ProgressState) -> None:
        self._sync_prepare_row(state)
        self._sync_pass_row(state)
        eta_ms = state.eta.eta_ms
        if eta_ms is not None and self._clock_column is not None:
            self._clock_column.set_eta(eta_ms)
        self._refresh_live()

    def _sync_prepare_row(self, state: ProgressState) -> None:
        if self._prepare_progress is None:
            return
        if state.prepare_visible:
            if self._prepare_task_id is None:
                self._prepare_task_id = self._prepare_progress.add_task(
                    "preparing", target=self._target_field(state.current_target)
                )
        elif self._prepare_task_id is not None:
            # The prepare row has nothing left to say once sampling starts, so it
            # leaves the display rather than lingering as a completed row.
            self._prepare_progress.remove_task(self._prepare_task_id)
            self._prepare_task_id = None

    def _sync_pass_row(self, state: ProgressState) -> None:
        if self._pass_progress is None or not state.pass_visible:
            return
        target = self._target_field(state.current_target)
        if self._pass_task_id is None:
            self._pass_task_id = self._pass_progress.add_task(
                "sampling", total=state.total, target=target
            )
        self._pass_progress.update(
            self._pass_task_id,
            total=state.total,
            target=target,
            completed=state.eta.completed,
        )

    def warn(self, message: str) -> None:
        """Surface a warning without disturbing any active live display."""
        self._console.print(message, highlight=False, markup=False)

    def stop(self) -> None:
        """Stop the reporter and clean up any live display."""
        if self._stopped:
            return
        self._stopped = True
        self._uninstall_cleanup()
        if self._live is not None:
            self._live.stop()
            self._print_summary()
            self._live = None

    def _print_summary(self) -> None:
        """Print the run's timing; the report right below carries everything else."""
        state = self._state
        elapsed_ms = (
            0.0
            if state.run_start_ms is None or state.run_end_ms is None
            else state.run_end_ms - state.run_start_ms
        )
        verb = "compared" if state.target_count > 1 else "measured"

        summary = Text()
        summary.append(f"{verb} in ", style=STYLE_META)
        summary.append(format_duration(elapsed_ms), style=STYLE_TIMER_DONE)
        self._console.print(summary, highlight=False)
