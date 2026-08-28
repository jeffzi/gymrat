"""Rich-based progress renderer for measure/compare commands.

Live mode (TTY) shows an animated progress bar with custom ETA, prepare
spinners, and a detail line. Plain mode (non-TTY) prints timestamped milestone
lines without ANSI escape codes.
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
from rich.text import Text

from gymrat_py.eta import format_duration, format_eta
from gymrat_py.progress_events import (
    PassFinished,
    PassStarted,
    PrepareFinished,
    PrepareStarted,
    ProgressEvent,
)
from gymrat_py.signals import install_termination_cleanup

_CLEAR_LINE = "\r\x1b[K"

# Below this terminal height, the two-block layout (prepare spinner + pass bar +
# detail line) can't fit, so the reporter switches to a single-row compact bar.
_COMPACT_HEIGHT_THRESHOLD = 12

_SECONDS_PER_MINUTE = 60
_SECONDS_PER_HOUR = 3600


class _EtaColumn(ProgressColumn):
    """Custom ETA that shows a pending label until the first pass finishes."""

    def __init__(self) -> None:
        super().__init__()
        self._text = "estimating time left…"

    def set_eta(self, ms: float) -> None:
        """Replace the pending label with a computed ETA."""
        self._text = format_eta(ms)

    @override
    def render(self, task: object) -> Text:
        return Text(self._text, style="dim")


class _TargetColumn(ProgressColumn):
    """Renders ``· <label> ·`` between the percentage and elapsed columns."""

    @override
    def render(self, task: object) -> Text:
        label = getattr(task, "fields", {}).get("target", "")
        if not label:
            return Text("")
        text = Text()
        text.append("· ", style="dim")
        text.append(str(label), style="bold blue")
        text.append(" ·", style="dim")
        return text


def compact_progress(
    console: Console, *, clock: Callable[[], float] | None = None
) -> tuple[Progress, _EtaColumn]:
    """Build a single-row compact progress bar for narrow terminals.

    Returns the ``Progress`` and its ``_EtaColumn`` so callers can update the
    ETA as passes complete.
    """
    eta_col = _EtaColumn()
    progress = Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        _TargetColumn(),
        TimeElapsedColumn(),
        eta_col,
        console=console,
        auto_refresh=False,
        get_time=clock,
    )
    return progress, eta_col


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
        self._prepare_elapsed_ms = 0.0
        self._pass_start_ms = 0.0
        self._total_pass_time_ms = 0.0
        self._pass_finish_count = 0
        self._last_pass_duration_ms = 0.0
        self._run_start_ms: float | None = None
        self._run_end_ms: float | None = None
        self._labels: set[str] = set()

        self._is_live = mode == "live" and console.width > 0
        self._live: Live | None = None
        self._eta_column: _EtaColumn | None = None
        self._prepare_progress: Progress | None = None
        self._pass_progress: Progress | None = None
        self._prepare_task_id: TaskID | None = None
        self._pass_task_id: TaskID | None = None
        self._current_pass_event: PassStarted | None = None
        self._compact = False
        self._stopped = False
        self._metric_count: int | None = None
        self._uninstall_cleanup: Callable[[], None] = lambda: None

        if self._is_live:
            self._init_live(console, clock)

    def _init_live(self, console: Console, clock: Callable[[], float] | None) -> None:
        self._compact = console.height < _COMPACT_HEIGHT_THRESHOLD

        if self._compact:
            self._pass_progress, self._eta_column = compact_progress(console, clock=clock)
        else:
            self._prepare_progress = Progress(
                SpinnerColumn(),
                TextColumn("{task.description}"),
                TimeElapsedColumn(),
                console=console,
                auto_refresh=False,
                get_time=clock,
            )

            self._eta_column = _EtaColumn()
            self._pass_progress = Progress(
                TextColumn("{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                self._eta_column,
                console=console,
                auto_refresh=False,
                get_time=clock,
            )

        self._live = Live(
            console=console,
            auto_refresh=True,
            refresh_per_second=1,
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
        header.append(self._command, style="bold blue")
        label_str = ", ".join(self._target_labels) if self._target_labels else ""
        sample_str = f"{self._sample_count} samples" if self._sample_count is not None else ""
        dim_parts = [p for p in (label_str, sample_str) if p]
        if dim_parts:
            header.append(" ")
            header.append(" · ".join(dim_parts), style="dim")
        return header

    def _live_detail_text(self) -> str:
        """Compute the detail line with live elapsed for the running pass."""
        if self._current_pass_event is None:
            return ""
        event = self._current_pass_event
        parts = [f"round {event.round}"]

        if self._target_count > 1:
            parts.append(event.label)

        if self._clock is not None:
            elapsed_ms = self._clock() * 1000 - event.at_ms
            elapsed_seconds = max(0, int(elapsed_ms / 1000))
            minutes, seconds = divmod(elapsed_seconds, _SECONDS_PER_MINUTE)
            parts.append(f"running {minutes}:{seconds:02d}")

        if self._last_pass_duration_ms > 0:
            parts.append(f"last pass {format_duration(self._last_pass_duration_ms)}")
        return " · ".join(parts)

    def frame(self) -> Group:
        """Return the renderable the live display paints from."""
        parts: list[RenderableType] = []
        header = self._header_text()
        if header is not None:
            parts.append(header)
        if self._prepare_progress is not None:
            parts.append(self._prepare_progress)
        if self._pass_progress is not None:
            parts.append(self._pass_progress)
        detail = self._live_detail_text()
        if detail:
            parts.append(Text(detail, style="dim"))
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

        self._labels.add(event.label)
        self._track_timestamp(event.at_ms)

        if isinstance(event, PrepareStarted):
            self._on_prepare_started(event)
        elif isinstance(event, PrepareFinished):
            self._on_prepare_finished(event)
        elif isinstance(event, PassStarted):
            self._on_pass_started(event)
        elif isinstance(event, PassFinished):
            self._on_pass_finished(event)

    def _track_timestamp(self, at_ms: float) -> None:
        if self._run_start_ms is None:
            self._run_start_ms = at_ms
        self._run_end_ms = at_ms

    def _on_prepare_started(self, event: PrepareStarted) -> None:
        self._prepare_start_ms = event.at_ms
        if self._is_live and self._prepare_progress is not None:
            self._prepare_task_id = self._prepare_progress.add_task(event.label)
            self._refresh_live()

    def _on_prepare_finished(self, event: PrepareFinished) -> None:
        elapsed_ms = event.at_ms - self._prepare_start_ms
        self._prepare_elapsed_ms += elapsed_ms

        if not self._is_live:
            elapsed = format_duration(elapsed_ms)
            self._print_plain(event.at_ms, f"prepared {event.label} ({elapsed})")
            return

        if self._prepare_progress is not None and self._prepare_task_id is not None:
            self._prepare_progress.update(
                self._prepare_task_id,
                description=f"✔ {event.label}",
                total=1,
                completed=1,
            )
        self._refresh_live()

    def _on_pass_started(self, event: PassStarted) -> None:
        self._pass_start_ms = event.at_ms

        if self._total == 0 and self._sample_count is None:
            self._total = event.total_rounds * self._target_count

        if not self._is_live or self._pass_progress is None:
            return

        if self._compact:
            description = f"sample {event.round}/{event.total_rounds}"
            if self._pass_task_id is None:
                self._pass_task_id = self._pass_progress.add_task(
                    description, total=self._total, target=event.label
                )
            self._pass_progress.update(
                self._pass_task_id,
                description=description,
                target=event.label,
                completed=self._completed,
            )
        else:
            if self._pass_task_id is None:
                self._pass_task_id = self._pass_progress.add_task("sampling", total=self._total)
            self._pass_progress.update(self._pass_task_id, completed=self._completed)
            self._current_pass_event = event

        self._refresh_live()

    def _on_pass_finished(self, event: PassFinished) -> None:
        duration_ms = event.at_ms - self._pass_start_ms
        self._last_pass_duration_ms = duration_ms
        self._total_pass_time_ms += duration_ms
        self._completed += 1
        self._pass_finish_count += 1

        remaining = self._total - self._completed
        if remaining > 0:
            avg_ms = self._total_pass_time_ms / self._pass_finish_count
            eta_ms = avg_ms * remaining
            if self._eta_column is not None:
                self._eta_column.set_eta(eta_ms)

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
        elapsed_ms = at_ms - (self._run_start_ms or at_ms)
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

    def set_metric_count(self, count: int) -> None:
        """Record the number of distinct metrics for the summary line."""
        self._metric_count = count

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
        elapsed_ms = (self._run_end_ms or 0) - (self._run_start_ms or 0)
        elapsed = format_duration(elapsed_ms)
        prepare = format_duration(self._prepare_elapsed_ms)
        samples = self._sample_count or (
            self._total // self._target_count if self._target_count else 0
        )
        parts = [f"{samples} samples"]
        if self._metric_count is not None:
            parts.append(f"{self._metric_count} metrics")
        parts.append(f"{elapsed} (prepare {prepare})")
        suffix = " · " + " · ".join(parts)

        if self._target_count > 1:
            headline = f"✔ compared {self._target_count} targets"
        else:
            headline = f"✔ measured {next(iter(self._labels), 'bench')}"

        self._console.print(headline + suffix, highlight=False)


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
