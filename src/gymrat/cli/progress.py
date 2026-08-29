"""Rich-based progress renderer for measure/compare commands.

Live mode (TTY) shows a header, a prepare row while the prepare command runs,
and a sampling bar with an elapsed-over-total clock. The prepare row is removed
once prepare finishes, so the display never grows past those two rows. Plain
mode (non-TTY) prints timestamped milestone lines without ANSI escape codes.

Glyphs, verb forms, and timer colors follow the conventions in
:mod:`gymrat.cli.style`.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Literal, override

if TYPE_CHECKING:
    from collections.abc import Callable

from rich.console import Console, Group, RenderableType
from rich.live import Live
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    ProgressColumn,
    SpinnerColumn,
    TaskID,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Column
from rich.text import Text

from gymrat.cli.style import (
    LIVE_REFRESH_PER_SECOND,
    SPINNER_NAME,
    STYLE_LABEL,
    STYLE_META,
    STYLE_TIMER_DONE,
    STYLE_TIMER_RUNNING,
    STYLE_VERB,
)
from gymrat.eta import format_clock, format_duration
from gymrat.progress_events import (
    PassFinished,
    PassStarted,
    PrepareFinished,
    PrepareStarted,
    ProgressEvent,
)
from gymrat.signals import install_termination_cleanup

_CLEAR_LINE = "\r\x1b[K"

# Below this terminal height, the header plus the prepare and sampling rows
# can't fit, so the reporter switches to a single-row compact bar.
_COMPACT_HEIGHT_THRESHOLD = 12

_SECONDS_PER_MINUTE = 60
_SECONDS_PER_HOUR = 3600
_MS_PER_SECOND = 1000


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
    def render(self, task: object) -> Text:
        elapsed_ms = (getattr(task, "elapsed", None) or 0.0) * _MS_PER_SECOND
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
    def render(self, task: object) -> Text:
        label = getattr(task, "fields", {}).get("target", "")
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
    def render(self, task: object) -> Text:
        fields = getattr(task, "fields", {})
        text = Text()
        text.append(str(getattr(task, "description", "")), style=STYLE_VERB)
        note = fields.get("note", "")
        if note:
            text.append(f" {note}", style=STYLE_META)
        target = fields.get("target", "")
        if target:
            text.append(" · ", style=STYLE_META)
            text.append(str(target), style=STYLE_LABEL)
        return text


def compact_progress(
    console: Console, *, clock: Callable[[], float] | None = None
) -> tuple[Progress, _ClockColumn]:
    """Build a single-row compact progress bar for narrow terminals.

    Returns the ``Progress`` and its ``_ClockColumn`` so callers can update the
    remaining estimate as passes complete.
    """
    clock_col = _ClockColumn()
    progress = Progress(
        SpinnerColumn(SPINNER_NAME),
        TextColumn("{task.description}", style=STYLE_VERB),
        BarColumn(),
        TaskProgressColumn(),
        _TargetColumn(),
        clock_col,
        console=console,
        auto_refresh=False,
        get_time=clock,
    )
    return progress, clock_col


def passes_progress(
    console: Console, *, clock: Callable[[], float] | None = None
) -> tuple[Progress, _ClockColumn]:
    """Build the sampling bar row shared by the measure/compare and iterate views.

    Returns the ``Progress`` and its ``_ClockColumn`` so callers can update the
    remaining estimate as passes complete.
    """
    clock_col = _ClockColumn()
    progress = Progress(
        SpinnerColumn(SPINNER_NAME),
        _PhaseColumn(),
        BarColumn(),
        MofNCompleteColumn(),
        clock_col,
        console=console,
        auto_refresh=False,
        get_time=clock,
    )
    return progress, clock_col


class ProgressReporter:
    """Single-use progress reporter for measure/compare commands.

    Call ``stop`` exactly once after the run ends. The reporter renders to the
    given ``console`` using either a rich ``Live`` block (live mode) or plain
    timestamped lines (plain mode).
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
        self._target_count = target_count
        self._sample_count = sample_count
        self._total = (sample_count or 0) * target_count
        self._clock = clock
        self._command = command
        self._target_labels = target_labels or []

        self._completed = 0
        self._prepare_start_ms = 0.0
        self._pass_start_ms = 0.0
        self._total_pass_time_ms = 0.0
        self._pass_finish_count = 0
        self._run_start_ms: float | None = None
        self._run_end_ms: float | None = None

        self._is_live = mode == "live" and console.width > 0
        self._live: Live | None = None
        self._clock_column: _ClockColumn | None = None
        self._prepare_progress: Progress | None = None
        self._pass_progress: Progress | None = None
        self._prepare_task_id: TaskID | None = None
        self._pass_task_id: TaskID | None = None
        self._compact = False
        self._stopped = False
        self._uninstall_cleanup: Callable[[], None] = lambda: None

        if self._is_live:
            self._init_live(console, clock)

    def _init_live(self, console: Console, clock: Callable[[], float] | None) -> None:
        self._compact = console.height < _COMPACT_HEIGHT_THRESHOLD

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
        self._uninstall_cleanup = install_termination_cleanup(self._clear_on_signal)

    def _header_text(self) -> Text | None:
        if not self._command:
            return None
        header = Text()
        header.append(self._command, style=STYLE_LABEL)
        label_str = ", ".join(self._target_labels) if self._target_labels else ""
        sample_str = f"{self._sample_count} samples" if self._sample_count is not None else ""
        dim_parts = [p for p in (label_str, sample_str) if p]
        if dim_parts:
            header.append(" ")
            header.append(" · ".join(dim_parts), style=STYLE_META)
        return header

    def _target_field(self, label: str) -> str:
        """The label a row shows for ``label``, empty when there is only one target."""
        return label if self._target_count > 1 else ""

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

    def _refresh_live(self) -> None:
        if self._live is not None:
            self._live.refresh()

    def report(self, event: ProgressEvent) -> None:
        """Dispatch ``event`` to the matching handler; ignore unrelated types."""
        if not isinstance(event, PrepareStarted | PrepareFinished | PassStarted | PassFinished):
            return

        self._track_timestamp(event.at_ms)

        match event:
            case PrepareStarted():
                self._on_prepare_started(event)
            case PrepareFinished():
                self._on_prepare_finished(event)
            case PassStarted():
                self._on_pass_started(event)
            case PassFinished():
                self._on_pass_finished(event)

    def _track_timestamp(self, at_ms: float) -> None:
        if self._run_start_ms is None:
            self._run_start_ms = at_ms
        self._run_end_ms = at_ms

    def _on_prepare_started(self, event: PrepareStarted) -> None:
        self._prepare_start_ms = event.at_ms
        if self._is_live and self._prepare_progress is not None:
            self._prepare_task_id = self._prepare_progress.add_task(
                "preparing", target=self._target_field(event.label)
            )
            self._refresh_live()

    def _on_prepare_finished(self, event: PrepareFinished) -> None:
        elapsed_ms = event.at_ms - self._prepare_start_ms

        if not self._is_live:
            elapsed = format_duration(elapsed_ms)
            self._print_plain(event.at_ms, f"prepared {event.label} ({elapsed})")
            return

        # The prepare row has nothing left to say once sampling starts, so it
        # leaves the display rather than lingering as a completed row.
        if self._prepare_progress is not None and self._prepare_task_id is not None:
            self._prepare_progress.remove_task(self._prepare_task_id)
            self._prepare_task_id = None
        self._refresh_live()

    def _on_pass_started(self, event: PassStarted) -> None:
        self._pass_start_ms = event.at_ms

        if self._total == 0 and self._sample_count is None:
            self._total = event.total_rounds * self._target_count

        if not self._is_live or self._pass_progress is None:
            return

        target = self._target_field(event.label)
        if self._pass_task_id is None:
            self._pass_task_id = self._pass_progress.add_task(
                "sampling", total=self._total, target=target
            )
        self._pass_progress.update(self._pass_task_id, target=target, completed=self._completed)

        self._refresh_live()

    def _on_pass_finished(self, event: PassFinished) -> None:
        duration_ms = event.at_ms - self._pass_start_ms
        self._total_pass_time_ms += duration_ms
        self._completed += 1
        self._pass_finish_count += 1

        remaining = self._total - self._completed
        if remaining > 0:
            avg_ms = self._total_pass_time_ms / self._pass_finish_count
            eta_ms = avg_ms * remaining
            if self._clock_column is not None:
                self._clock_column.set_eta(eta_ms)

        if self._is_live and self._pass_progress is not None and self._pass_task_id is not None:
            self._pass_progress.update(self._pass_task_id, completed=self._completed)
            self._refresh_live()
        elif not self._is_live:
            self._print_plain(
                event.at_ms,
                (
                    f"pass {event.round}/{event.total_rounds}"
                    f" · {event.label}"
                    f" ({format_duration(duration_ms)})"
                ),
            )

    def _print_plain(self, at_ms: float, message: str) -> None:
        ts = self._format_timestamp(at_ms)
        self._console.print(f"{ts} {message}", highlight=False, markup=False)

    def _format_timestamp(self, at_ms: float) -> str:
        run_start_ms = at_ms if self._run_start_ms is None else self._run_start_ms
        elapsed_ms = at_ms - run_start_ms
        total_seconds = int(elapsed_ms / 1000)
        hours, remainder = divmod(total_seconds, _SECONDS_PER_HOUR)
        minutes, seconds = divmod(remainder, _SECONDS_PER_MINUTE)
        return f"[{hours:02d}:{minutes:02d}:{seconds:02d}]"

    def _clear_on_signal(self) -> None:
        # os._exit skips buffer flushing, so the clear must be flushed explicitly
        # or it never reaches the terminal.
        if self._stopped:
            return
        self._stopped = True
        sys.stderr.write(_CLEAR_LINE)
        sys.stderr.flush()

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
        elapsed_ms = (
            0.0
            if self._run_start_ms is None or self._run_end_ms is None
            else self._run_end_ms - self._run_start_ms
        )
        verb = "compared" if self._target_count > 1 else "measured"

        summary = Text()
        summary.append(f"{verb} in ", style=STYLE_META)
        summary.append(format_duration(elapsed_ms), style=STYLE_TIMER_DONE)
        self._console.print(summary, highlight=False)


def create_progress_reporter(  # noqa: PLR0913 -- mirrors the constructor above
    mode: Literal["live", "plain"],
    console: Console,
    target_count: int,
    sample_count: int | None = None,
    *,
    clock: Callable[[], float] | None = None,
    command: str | None = None,
    target_labels: list[str] | None = None,
) -> ProgressReporter:
    """Build the reporter a run streams its progress through.

    Args:
        mode: ``"live"`` for rich animated output, ``"plain"`` for timestamped
            milestone lines.
        console: The ``Console`` to render to.
        target_count: Number of targets (1 for measure, N for compare).
        sample_count: Samples per target. When ``None``, inferred from the first
            ``PassStarted`` event.
        clock: Optional deterministic clock for ``Progress(get_time=...)``.
        command: Command name (e.g. ``"measure"``, ``"compare"``) for the header line.
        target_labels: Target display labels for the header line.
    """
    return ProgressReporter(
        mode,
        console,
        target_count,
        sample_count,
        clock=clock,
        command=command,
        target_labels=target_labels,
    )
