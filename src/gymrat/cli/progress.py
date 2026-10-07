"""Rich-based progress renderer for measure/compare commands.

Live mode (TTY) shows a header, a prepare row while the prepare command runs,
and a sampling bar with an elapsed-over-total clock. The prepare row is removed
once prepare finishes, so the display never grows past those two rows. Plain
mode (non-TTY) prints timestamped milestone lines without ANSI escape codes.

:class:`ProgressReporter` is the shell: it owns the terminal, the ``rich``
objects, and the live/plain branch. Every decision about what to show lives in
the pure reducer made of :class:`ProgressState`, :func:`advance` and
:func:`plain_line` -- which rows are visible, how many passes are done, what
the remaining estimate is, and which milestone line plain mode prints. The
reducer touches no ``rich`` object and reads no clock: ``now`` always comes
from the event's own ``at_ms``, so a transition is fully determined by
``(state, event)``.

Glyphs, verb forms, and timer colors follow the conventions in
:mod:`gymrat.cli.style`.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Literal, Self, override

if TYPE_CHECKING:
    from collections.abc import Callable

    from gymrat.progress_events import ProgressEvent

from rich.console import Console, Group, RenderableType
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

from gymrat.cli.live_display import LiveDisplayMixin
from gymrat.cli.style import (
    COMPACT_HEIGHT_THRESHOLD,
    SPINNER_NAME,
    STYLE_LABEL,
    STYLE_META,
    STYLE_TIMER_DONE,
    STYLE_TIMER_RUNNING,
    STYLE_VERB,
)
from gymrat.progress_events import (
    PassFinished,
    PassStarted,
    PrepareFinished,
    PrepareStarted,
)
from gymrat.utils import (
    MS_PER_SECOND,
    SamplingEta,
    format_clock,
    format_duration,
)


@dataclass(frozen=True, slots=True)
class ProgressState:
    """Everything the progress display needs to paint a frame.

    Attributes:
        target_count: How many targets (baseline plus candidates) the run covers.
        sample_count: Samples per target, or ``None`` when the total is only
            discovered at runtime from ``PassStarted.total_rounds``.
        prepare_start_ms: Timestamp the current prepare step began.
        pass_start_ms: Timestamp the current pass began.
        run_start_ms: Timestamp of the first event seen, or ``None`` until the
            first event arrives.
        run_end_ms: Timestamp of the most recent event, or ``None`` until the
            first event arrives.
        eta: Finished-pass samples and the remaining-time estimate they make.
            Its ``total`` is ``0`` while the pass count is still unknown, which
            happens when no ``--samples`` flag pinned it up front; the first
            ``PassStarted`` fills it in.
        prepare_visible: Whether the prepare row is shown. Cleared on
            ``PrepareFinished`` even though the run is still in progress; it
            tracks row visibility, not whether the prepare phase is done.
        pass_visible: Whether the pass row is shown.
        current_target: Label of the in-flight target shown on whichever row
            is currently visible, or ``""`` before one is set.
    """

    target_count: int
    sample_count: int | None = None
    prepare_start_ms: float = 0.0
    pass_start_ms: float = 0.0
    run_start_ms: float | None = None
    run_end_ms: float | None = None
    eta: SamplingEta = field(default_factory=lambda: SamplingEta(total=0))
    prepare_visible: bool = False
    pass_visible: bool = False
    current_target: str = ""

    @classmethod
    def start(cls, *, target_count: int, sample_count: int | None) -> Self:
        """Build the state a run begins in.

        Args:
            target_count: How many targets (baseline plus candidates) the run covers.
            sample_count: Samples per target, or ``None`` when the total is only
                discovered at runtime from ``PassStarted.total_rounds``.

        Returns:
            A state with nothing visible and the total pre-sized when it is known.
        """
        total = (sample_count or 0) * target_count
        return cls(
            target_count=target_count,
            sample_count=sample_count,
            eta=SamplingEta(total=total),
        )

    @property
    def total(self) -> int:
        """Passes the run expects in all, as the estimate counts them."""
        return self.eta.total


def _pass_started(state: ProgressState, event: PassStarted) -> ProgressState:
    total = state.total or event.total_rounds * event.target_count
    return replace(
        state,
        eta=replace(state.eta, total=total),
        pass_start_ms=event.at_ms,
        pass_visible=True,
        current_target=event.label,
    )


def advance(state: ProgressState, event: ProgressEvent) -> ProgressState:
    """Fold ``event`` into ``state``.

    Args:
        state: The state the run is in before the event.
        event: The event to apply; anything outside the four prepare/pass
            milestones is not a display change.

    Returns:
        The state the display should paint next, or ``state`` itself when the
        event carries nothing the display shows.
    """
    match event:
        case PrepareStarted():
            updated = replace(
                state,
                prepare_start_ms=event.at_ms,
                prepare_visible=True,
                current_target=event.label,
            )
        case PrepareFinished():
            # The prepare row has nothing left to say once sampling starts, so it
            # leaves the display rather than lingering as a completed row.
            updated = replace(state, prepare_visible=False)
        case PassStarted():
            updated = _pass_started(state, event)
        case PassFinished():
            updated = replace(state, eta=state.eta.advanced(event.at_ms - state.pass_start_ms))
        case _:
            return state

    return replace(
        updated,
        run_start_ms=event.at_ms if state.run_start_ms is None else state.run_start_ms,
        run_end_ms=event.at_ms,
    )


def plain_line(before: ProgressState, after: ProgressState, event: ProgressEvent) -> str | None:
    """Render the milestone line plain mode prints for ``event``, without its timestamp.

    Args:
        before: The state before ``event`` was applied.
        after: The state ``advance`` returned for ``event``.
        event: The event being reported.

    Returns:
        The line to print, or ``None`` when the event is not a milestone plain
        mode announces.
    """
    match event:
        case PrepareFinished():
            elapsed = format_duration(event.at_ms - before.prepare_start_ms)
            return f"prepared {event.label} ({elapsed})"
        case PassFinished():
            # Taking the duration from the ETA delta rather than recomputing it
            # keeps the printed number and the bar's estimate from ever disagreeing.
            elapsed = format_duration(after.eta.total_time_ms - before.eta.total_time_ms)
            return f"pass {event.round}/{event.total_rounds} · {event.label} ({elapsed})"
        case _:
            return None


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


def phase_text(verb: str, note: str, target: str) -> Text:
    """Build a running row: the verb, then the context the row carries.

    Args:
        verb: The gerund the row leads with (``"sampling"``).
        note: Dim context shown after the verb; empty for none.
        target: The in-flight target label, shown behind a dim separator;
            empty for none.

    Returns:
        The styled row text.
    """
    text = Text()
    text.append(verb, style=STYLE_VERB)
    if note:
        text.append(f" {note}", style=STYLE_META)
    if target:
        text.append(" · ", style=STYLE_META)
        text.append(target, style=STYLE_LABEL)
    return text


class _PhaseColumn(ProgressColumn):
    """Renders the running verb plus the optional context the row carries.

    The task description holds the gerund (``"sampling"``); the optional
    ``note`` and ``target`` fields are the context :func:`phase_text` adds.

    The column never wraps: a narrow terminal shrinks the bar rather than
    spilling the verb onto a second line and breaking the checklist alignment.
    """

    def __init__(self) -> None:
        super().__init__(table_column=Column(no_wrap=True))

    @override
    def render(self, task: Task) -> Text:
        fields = task.fields
        return phase_text(
            task.description, str(fields.get("note", "")), str(fields.get("target", ""))
        )


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
        *columns: The columns to show before the clock column.

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

        self._clock_column: _ClockColumn | None = None
        self._prepare_progress: Progress | None = None
        self._pass_progress: Progress | None = None
        self._prepare_task_id: TaskID | None = None
        self._pass_task_id: TaskID | None = None

        if self._resolve_live(mode):
            self._init_live(console, clock)

    def _init_live(self, console: Console, clock: Callable[[], float] | None) -> None:
        if console.height < COMPACT_HEIGHT_THRESHOLD:
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

        # A termination signal exits via os._exit without unwinding the run's
        # finally block, so the live display would strand its last frame on the
        # terminal; the mount erases it instead.
        self._mount_live(transient=True, get_renderable=self.frame)

    def _header_text(self) -> Text | None:
        if not self._command:
            return None
        header = Text()
        header.append(self._command, style=STYLE_LABEL)
        sample_count = self._state.sample_count
        label_str = ", ".join(self._target_labels)
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
            self._print_milestone(line, event.at_ms, after.run_start_ms)

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

    @override
    def _after_live_stopped(self) -> None:
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
