"""Rich-based progress renderer for the ``gymrat iterate`` command.

Live mode shows a flat checklist of the iteration's phases; plain mode prints
timestamped milestone lines. Row rendering helpers live in :mod:`.rows`.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal

from rich.console import Console, Group, RenderableType
from rich.live import Live
from rich.text import Text

from gymrat.cli.iterate.rows import (
    NodeState,
    PhaseCounters,
    build_judge_detail,
    build_nodes,
    format_judge_plain,
    render_row,
)
from gymrat.cli.progress import compact_progress, passes_progress
from gymrat.cli.style import (
    COMPACT_HEIGHT_THRESHOLD,
    LIVE_REFRESH_PER_SECOND,
    STYLE_LABEL,
    STYLE_META,
    STYLE_TIMER_RUNNING,
    LiveDisplayMixin,
)
from gymrat.eta import MS_PER_SECOND, format_clock, format_duration, format_timestamp
from gymrat.plural import pluralize
from gymrat.progress_events import (
    ConfirmFinished,
    ConfirmStarted,
    HookFinished,
    HookStarted,
    IterationRecorded,
    JudgeFinished,
    JudgeStarted,
    PassFinished,
    PassStarted,
    PrepareFinished,
    PrepareStarted,
    ProgressEvent,
)
from gymrat.signals import install_termination_cleanup

if TYPE_CHECKING:
    from collections.abc import Callable

    from rich.progress import Progress, TaskID

    from gymrat.cli.progress import _ClockColumn

logger = logging.getLogger(__name__)


class IterateRenderer(LiveDisplayMixin):
    """Single-use progress renderer for ``gymrat iterate``."""

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
        self._primary_metric = primary_metric
        self._verbose = verbose
        self._clock = clock
        self._checks_cmd = checks_cmd
        self._total = sample_count * 2
        self._start_clock_time: float | None = None

        self._is_live = mode == "live" and console.width > 0
        self._compact = False
        self._stopped = False
        self._live: Live | None = None
        self._uninstall_cleanup: Callable[[], None] = lambda: None

        self._run_start_ms: float | None = None

        nodes = build_nodes(
            primary_metric,
            metric_count,
            has_before_hook=has_before_hook,
            has_after_hook=has_after_hook,
        )
        self._before_hook = nodes.before_hook
        self._prepare = nodes.prepare
        self._passes = nodes.passes
        self._judge = nodes.judge
        self._confirm = nodes.confirm
        self._record = nodes.record
        self._nodes = nodes.all_nodes

        self._prepare_current_start_ms: float = 0.0
        self._pass = PhaseCounters()
        self._confirm_phase = PhaseCounters()

        self._compact_progress: Progress | None = None
        self._compact_clock_col: _ClockColumn | None = None
        self._compact_task_id: TaskID | None = None

        self._handlers = self._build_handlers()

        if self._is_live:
            self._init_live()

    def _build_handlers(self) -> dict[type[ProgressEvent], Callable[..., None]]:
        return {
            HookStarted: self._on_hook_started,
            HookFinished: self._on_hook_finished,
            PrepareStarted: self._on_prepare_started,
            PrepareFinished: self._on_prepare_finished,
            PassStarted: self._on_pass_started,
            PassFinished: self._on_pass_finished,
            JudgeStarted: self._on_judge_started,
            JudgeFinished: self._on_judge_finished,
            ConfirmStarted: self._on_confirm_started,
            ConfirmFinished: self._on_confirm_finished,
            IterationRecorded: self._on_iteration_recorded,
        }

    def _init_live(self) -> None:
        self._compact = self._console.height < COMPACT_HEIGHT_THRESHOLD

        if self._compact:
            self._compact_progress, self._compact_clock_col = compact_progress(
                self._console, clock=self._clock
            )
        else:
            self._passes.bar, self._pass.clock_col = passes_progress(
                self._console, clock=self._clock
            )
            self._confirm.bar, self._confirm_phase.clock_col = passes_progress(
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
        for node in self._nodes:
            if node.status == "skipped":
                continue
            if node.status == "running" and node.bar is not None:
                rows.append(node.bar)
            else:
                rows.append(render_row(node, self._running_elapsed_ms(node)))
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
        eta_ms = self._eta_ms(
            self._pass.completed, self._pass.finish_count, self._pass.total_time_ms
        )
        if eta_ms is None:
            header.append(f"{format_duration(elapsed_ms)} elapsed", style=STYLE_META)
        else:
            header.append(format_clock(elapsed_ms), style=STYLE_TIMER_RUNNING)
            header.append(f"/{format_clock(elapsed_ms + eta_ms)}", style=STYLE_META)
        return header

    def _running_elapsed_ms(self, node: NodeState) -> float | None:
        if node is not self._judge or node.status != "running":
            return None
        if self._clock is None or node.start_ms <= 0:
            return None
        return self._clock() * MS_PER_SECOND - node.start_ms

    # -----------------------------------------------------------------------
    # Timing helpers
    # -----------------------------------------------------------------------

    def _eta_ms(self, completed: int, finish_count: int, total_time_ms: float) -> float | None:
        remaining = self._total - completed
        if remaining <= 0 or finish_count == 0:
            return None
        return (total_time_ms / finish_count) * remaining

    def _clock_elapsed_ms(self, start_clock: float | None) -> float | None:
        if start_clock is None or self._clock is None:
            return None
        return (self._clock() - start_clock) * MS_PER_SECOND

    def _track_timestamp(self, at_ms: float) -> None:
        if self._run_start_ms is None:
            self._run_start_ms = at_ms
            if self._clock is not None:
                self._start_clock_time = self._clock()

    def _print_plain(self, at_ms: float, message: str) -> None:
        ts = format_timestamp(at_ms, self._run_start_ms)
        self._console.print(f"{ts} {message}", highlight=False, markup=False)

    # -----------------------------------------------------------------------
    # Event handlers
    # -----------------------------------------------------------------------

    def report(self, event: ProgressEvent) -> None:
        """Dispatch ``event`` to its handler."""
        self._track_timestamp(event.at_ms)
        self._handlers[type(event)](event)

    def _on_hook_started(self, event: HookStarted) -> None:
        if event.stage == "before":
            self._before_hook.status = "running"
            self._before_hook.start_ms = event.at_ms
            self._refresh_live()

    def _on_hook_finished(self, event: HookFinished) -> None:
        if event.stage == "before":
            self._before_hook.status = "done"
            self._before_hook.elapsed_ms = event.at_ms - self._before_hook.start_ms
            self._refresh_live()

    def _on_prepare_started(self, event: PrepareStarted) -> None:
        self._prepare.status = "running"
        self._prepare_current_start_ms = event.at_ms
        self._prepare.target = event.label
        self._refresh_live()

    def _on_prepare_finished(self, event: PrepareFinished) -> None:
        elapsed_ms = event.at_ms - self._prepare_current_start_ms
        self._prepare.elapsed_ms += elapsed_ms
        self._prepare.status = "done"
        self._prepare.target = ""
        if self._is_live:
            self._refresh_live()
        else:
            self._print_plain(
                event.at_ms, f"prepare {event.label} done ({format_duration(elapsed_ms)})"
            )

    def _on_pass_started(self, event: PassStarted) -> None:
        is_confirm = event.phase == "confirm"
        counters = self._confirm_phase if is_confirm else self._pass
        node = self._confirm if is_confirm else self._passes
        counters.start_ms = event.at_ms

        if not is_confirm:
            node.status = "running"

        if self._compact and self._compact_progress is not None:
            if self._compact_task_id is None:
                self._compact_task_id = self._compact_progress.add_task(
                    "sampling",
                    total=self._total,
                    target=event.label,
                )
            elif is_confirm:
                self._compact_progress.update(self._compact_task_id, target=event.label)
            else:
                self._compact_progress.update(
                    self._compact_task_id,
                    target=event.label,
                    completed=counters.completed,
                )
        elif node.bar is not None:
            if counters.task_id is None:
                task_desc = "confirming" if is_confirm else "sampling"
                counters.task_id = node.bar.add_task(
                    task_desc, total=self._total, target=event.label
                )
            else:
                node.bar.update(counters.task_id, target=event.label)
        self._refresh_live()

    def _on_pass_finished(self, event: PassFinished) -> None:
        is_confirm = event.phase == "confirm"
        counters = self._confirm_phase if is_confirm else self._pass
        node = self._confirm if is_confirm else self._passes

        duration_ms = event.at_ms - counters.start_ms
        counters.total_time_ms += duration_ms
        counters.completed += 1
        counters.finish_count += 1

        eta_ms = self._eta_ms(counters.completed, counters.finish_count, counters.total_time_ms)
        if eta_ms is not None:
            for column in (counters.clock_col, self._compact_clock_col):
                if column is not None:
                    column.set_eta(eta_ms)

        if not is_confirm and counters.completed >= self._total:
            node.status = "done"
            node.elapsed_ms = counters.total_time_ms
            node.detail = f"{self._total} passes"

        self._advance_bar(counters, node)

        if not is_confirm:
            if self._is_live:
                self._refresh_live()
            elif counters.completed >= self._total:
                self._print_plain(
                    event.at_ms,
                    f"passes done ({format_duration(counters.total_time_ms)})",
                )
        else:
            self._refresh_live()

    def _advance_bar(self, counters: PhaseCounters, node: NodeState) -> None:
        if self._compact:
            if self._compact_progress is not None and self._compact_task_id is not None:
                self._compact_progress.update(self._compact_task_id, completed=counters.completed)
        elif node.bar is not None and counters.task_id is not None:
            node.bar.update(counters.task_id, completed=counters.completed)

    def _on_judge_started(self, event: JudgeStarted) -> None:
        self._judge.status = "running"
        self._judge.start_ms = event.at_ms
        self._refresh_live()

    def _on_judge_finished(self, event: JudgeFinished) -> None:
        self._judge.status = "done"
        self._judge.note = ""
        self._judge.elapsed_ms = (
            event.at_ms - self._judge.start_ms if self._judge.start_ms > 0 else 0
        )

        self._judge.detail = build_judge_detail(
            self._primary_metric, event.primary_delta_pct, event.regressed
        )

        if not event.regressed:
            self._confirm.status = "skipped"

        if self._is_live:
            self._refresh_live()
        else:
            breakdown = format_judge_plain(
                event.primary_delta_pct, event.regressed, event.metric_count
            )
            self._print_plain(event.at_ms, f"judge {breakdown}")

    def _on_confirm_started(self, event: ConfirmStarted) -> None:
        self._confirm.status = "running"
        self._confirm.start_ms = event.at_ms
        self._judge.alert = True

        if event.filtered_metrics is not None:
            self._confirm.note = pluralize(len(event.filtered_metrics), "metric")
        else:
            self._confirm.note = "full suite"

        if self._compact and self._compact_progress is not None:
            if self._compact_task_id is not None:
                self._compact_progress.remove_task(self._compact_task_id)
            self._compact_task_id = self._compact_progress.add_task("confirming", total=self._total)
            if self._compact_clock_col is not None:
                self._compact_clock_col.set_eta(0)
        elif self._confirm.bar is not None and self._confirm_phase.task_id is None:
            self._confirm_phase.task_id = self._confirm.bar.add_task(
                "confirming", total=self._total, note=self._confirm.note
            )

        self._refresh_live()

    def _on_confirm_finished(self, event: ConfirmFinished) -> None:
        self._confirm.status = "done"
        self._confirm.elapsed_ms = event.at_ms - self._confirm.start_ms
        self._judge.alert = False

        status = "regressions reproduced" if event.reproduced else "regressions not reproduced"
        self._confirm.detail = f"{self._confirm_phase.completed}/{self._total} · {status}"
        self._confirm.note = ""

        if self._is_live:
            self._refresh_live()
        else:
            self._print_plain(event.at_ms, f"confirm {self._confirm.detail}")

    def _on_iteration_recorded(self, event: IterationRecorded) -> None:
        self._record.status = "done"

        detail = f"{event.outcome} suggested"
        if self._checks_cmd is not None:
            detail += f" — checks ({self._checks_cmd}) run at gymrat keep"
        self._record.detail = detail

        if self._is_live:
            self._refresh_live()
        else:
            self._print_plain(event.at_ms, f"recorded {self._record.detail}")

    def stop(self) -> None:
        """Stop the renderer and clean up any live display."""
        if self._stopped:
            return
        self._stopped = True
        self._uninstall_cleanup()
        if self._live is not None:
            self._live.stop()
            self._live = None
