"""Rich-based progress renderer for the ``gymrat iterate`` command.

Live mode (TTY) shows a header line and a flat checklist of the iteration's
phases — before hook, prepare, passes, judge, confirm, record. Each row carries
its state through a glyph, a verb form, and a timer color, following the
conventions in :mod:`gymrat.cli.style`; the running sampling and confirm rows
are replaced by a progress bar with a count and an elapsed-over-total clock.
Plain mode prints timestamped milestone lines without ANSI escape codes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from rich.console import Console, Group, RenderableType
from rich.live import Live
from rich.spinner import Spinner
from rich.text import Text

from gymrat.cli.progress import compact_progress, passes_progress
from gymrat.cli.style import (
    COMPACT_HEIGHT_THRESHOLD,
    GLYPH_ALERT,
    GLYPH_DONE,
    GLYPH_PENDING,
    LIVE_REFRESH_PER_SECOND,
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
    LiveDisplayMixin,
)
from gymrat.eta import MS_PER_SECOND, format_clock, format_duration, format_timestamp
from gymrat.metric_name import format_inline, parse
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
    ProgressCallback,
    ProgressEvent,
)
from gymrat.report.format import pluralize
from gymrat.signals import install_termination_cleanup

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from rich.progress import Progress, TaskID

    from gymrat.cli.progress import _ClockColumn

logger = logging.getLogger(__name__)

# How many regressed metric names the judge's done line spells out before
# trailing off; the leading count already says how many there are.
_REGRESSED_NAME_CAP = 3


# ---------------------------------------------------------------------------
# Row state tracking
# ---------------------------------------------------------------------------


@dataclass
class _NodeState:
    """Status, wording, and timing of a single checklist row.

    The three verb forms are what the row's state reads as: ``noun`` while
    pending, ``gerund`` while running, ``past`` once done. ``hint`` is the dim
    explanation shown behind a pending row, ``note`` the dim context and
    ``target`` the in-flight target label shown while running, and ``detail``
    the dim outcome shown when done. A ``skipped`` row is dropped from the
    checklist entirely.
    """

    noun: str
    gerund: str
    past: str
    hint: str = ""
    note: str = ""
    target: str = ""
    detail: str | Text = ""
    status: Literal["pending", "running", "done", "skipped"] = "pending"
    start_ms: float = 0.0
    elapsed_ms: float = 0.0
    alert: bool = False
    bar: Progress | None = field(default=None, repr=False)
    # One spinner per row, kept across frames: recreating it every render
    # would reset its animation clock and freeze it on the first frame.
    spinner: Spinner | None = field(default=None, repr=False)

    @property
    def glyph(self) -> str:
        if self.alert:
            return GLYPH_ALERT
        match self.status:
            case "done":
                return GLYPH_DONE
            case _:
                return GLYPH_PENDING


# ---------------------------------------------------------------------------
# Iterate renderer
# ---------------------------------------------------------------------------


class IterateRenderer(LiveDisplayMixin):
    """Single-use progress renderer for ``gymrat iterate``.

    Call ``stop`` exactly once after the iteration ends. The renderer writes to
    the given ``console`` using either a rich ``Live`` block (live mode) or plain
    timestamped lines (plain mode).
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
        self._sample_count = sample_count
        self._metric_count = metric_count
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

        # The metric count is unknown (0) until the first pass reports, so the
        # hint names only the primary metric rather than claiming "0 metrics".
        judge_hint = f"{primary_metric} primary"
        if metric_count > 0:
            judge_hint = f"{pluralize(metric_count, 'metric')} · {judge_hint}"
        self._before_hook = _NodeState(noun="before hook", gerund="before hook", past="before hook")
        self._prepare = _NodeState(noun="prepare", gerund="preparing", past="prepared")
        self._passes = _NodeState(noun="passes", gerund="sampling", past="sampled")
        self._judge = _NodeState(
            noun="judge",
            gerund="judging",
            past="judged",
            hint=judge_hint,
            note=judge_hint,
        )
        self._confirm = _NodeState(
            noun="confirm",
            gerund="confirming",
            past="confirmed",
            hint="only if a gating metric regresses",
        )
        # The header already names the iteration number, so the record hint
        # carries only what follows the write — nothing when no after hook.
        self._record = _NodeState(
            noun="record",
            gerund="recording",
            past="recorded",
            hint="then after hook" if has_after_hook else "",
        )
        # A before hook that isn't configured is absence, not a step: its row
        # never joins the checklist rather than sitting pending forever.
        hook_rows = (self._before_hook,) if has_before_hook else ()
        self._nodes = (
            *hook_rows,
            self._prepare,
            self._passes,
            self._judge,
            self._confirm,
            self._record,
        )

        self._prepare_current_start_ms: float = 0.0

        self._pass_completed = 0
        self._pass_finish_count = 0
        self._total_pass_time_ms: float = 0.0
        self._pass_start_ms: float = 0.0

        self._confirm_completed = 0
        self._confirm_finish_count = 0
        self._confirm_total_pass_time_ms: float = 0.0
        self._confirm_pass_start_ms: float = 0.0

        self._pass_task_id: TaskID | None = None
        self._pass_clock_col: _ClockColumn | None = None
        self._confirm_task_id: TaskID | None = None
        self._confirm_clock_col: _ClockColumn | None = None

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
            self._passes.bar, self._pass_clock_col = passes_progress(
                self._console, clock=self._clock
            )
            self._confirm.bar, self._confirm_clock_col = passes_progress(
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

        self._uninstall_cleanup = install_termination_cleanup(self._clear_on_signal)

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
                rows.append(self._render_row(node))
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
            self._pass_completed, self._pass_finish_count, self._total_pass_time_ms
        )
        if eta_ms is None:
            header.append(f"{format_duration(elapsed_ms)} elapsed", style=STYLE_META)
        else:
            header.append(format_clock(elapsed_ms), style=STYLE_TIMER_RUNNING)
            header.append(f"/{format_clock(elapsed_ms + eta_ms)}", style=STYLE_META)
        return header

    def _render_row(self, node: _NodeState) -> RenderableType:
        match node.status:
            case "running":
                return self._render_running_row(node)
            case "done":
                return self._render_done_row(node)
            case _:
                return self._render_idle_row(node)

    def _render_running_row(self, node: _NodeState) -> RenderableType:
        style = STYLE_ALERT if node.alert else STYLE_RUNNING
        text = Text()
        text.append(node.gerund, style=STYLE_VERB)
        if node.note:
            text.append(f" {node.note}", style=STYLE_META)
        if node.target:
            text.append(" · ", style=STYLE_META)
            text.append(node.target, style=STYLE_LABEL)
        running_ms = self._running_elapsed_ms(node)
        if running_ms is not None:
            text.append(f" {format_duration(running_ms)}", style=STYLE_TIMER_RUNNING)
        if node.alert:
            return Text.assemble((f"{GLYPH_ALERT} ", style), text)
        if node.spinner is None:
            node.spinner = Spinner(SPINNER_NAME, text=text, style=style)
        else:
            node.spinner.update(text=text, style=style)
        return node.spinner

    def _render_done_row(self, node: _NodeState) -> Text:
        text = Text()
        text.append(f"{node.glyph} ", style=STYLE_ALERT if node.alert else STYLE_DONE)
        text.append(node.past)
        if node.detail:
            if isinstance(node.detail, Text):
                text.append(" ")
                text.append_text(node.detail)
            else:
                text.append(f" {node.detail}", style=STYLE_META)
        if node.elapsed_ms > 0:
            text.append(f" {format_duration(node.elapsed_ms)}", style=STYLE_TIMER_DONE)
        return text

    def _render_idle_row(self, node: _NodeState) -> Text:
        """Render a row that has not run yet: pending with its hint."""
        text = Text()
        text.append(f"{node.glyph} {node.noun}", style=STYLE_PENDING)
        if node.hint:
            text.append(f" ({node.hint})", style=STYLE_PENDING)
        return text

    def _running_elapsed_ms(self, node: _NodeState) -> float | None:
        """Live elapsed time for ``node``, or ``None`` if it doesn't tick live.

        Only the judge row ticks while running — it's the one long step that
        cannot be interrupted and lacks its own progress events. The rows that
        do have events tick through their own progress bar instead.
        """
        if node is not self._judge or node.status != "running":
            return None
        if self._clock is None or node.start_ms <= 0:
            return None
        return self._clock() * MS_PER_SECOND - node.start_ms

    # -----------------------------------------------------------------------
    # Timing helpers
    # -----------------------------------------------------------------------

    def _eta_ms(self, completed: int, finish_count: int, total_time_ms: float) -> float | None:
        """Milliseconds left for the given completion state, or ``None`` if too early."""
        remaining = self._total - completed
        if remaining <= 0 or finish_count == 0:
            return None
        return (total_time_ms / finish_count) * remaining

    def _clock_elapsed_ms(self, start_clock: float | None) -> float | None:
        """Milliseconds elapsed since ``start_clock``, or ``None`` if unavailable."""
        if start_clock is None or self._clock is None:
            return None
        return (self._clock() - start_clock) * MS_PER_SECOND

    def _track_timestamp(self, at_ms: float) -> None:
        if self._run_start_ms is None:
            self._run_start_ms = at_ms
            if self._clock is not None:
                self._start_clock_time = self._clock()

    def _format_timestamp(self, at_ms: float) -> str:
        return format_timestamp(at_ms, self._run_start_ms)

    def _print_plain(self, at_ms: float, message: str) -> None:
        ts = self._format_timestamp(at_ms)
        self._console.print(f"{ts} {message}", highlight=False, markup=False)

    # -----------------------------------------------------------------------
    # Event handlers
    # -----------------------------------------------------------------------

    def report(self, event: ProgressEvent) -> None:  # noqa: C901 -- dispatch table
        """Dispatch ``event`` to its handler after anchoring the run timestamp."""
        self._track_timestamp(event.at_ms)

        match event:
            case HookStarted():
                self._on_hook_started(event)
            case HookFinished():
                self._on_hook_finished(event)
            case PrepareStarted():
                self._on_prepare_started(event)
            case PrepareFinished():
                self._on_prepare_finished(event)
            case PassStarted():
                self._on_pass_started(event)
            case PassFinished():
                self._on_pass_finished(event)
            case JudgeStarted():
                self._on_judge_started(event)
            case JudgeFinished():
                self._on_judge_finished(event)
            case ConfirmStarted():
                self._on_confirm_started(event)
            case ConfirmFinished():
                self._on_confirm_finished(event)
            case IterationRecorded():
                self._on_iteration_recorded(event)

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
        if event.phase == "confirm":
            self._on_confirm_pass_started(event)
            return

        self._passes.status = "running"
        self._pass_start_ms = event.at_ms

        if self._compact and self._compact_progress is not None:
            if self._compact_task_id is None:
                self._compact_task_id = self._compact_progress.add_task(
                    "sampling",
                    total=self._total,
                    target=event.label,
                )
            else:
                self._compact_progress.update(
                    self._compact_task_id,
                    target=event.label,
                    completed=self._pass_completed,
                )
        elif self._passes.bar is not None:
            if self._pass_task_id is None:
                self._pass_task_id = self._passes.bar.add_task(
                    "sampling", total=self._total, target=event.label
                )
            else:
                self._passes.bar.update(self._pass_task_id, target=event.label)
        self._refresh_live()

    def _on_pass_finished(self, event: PassFinished) -> None:
        if event.phase == "confirm":
            self._on_confirm_pass_finished(event)
            return

        duration_ms = event.at_ms - self._pass_start_ms
        self._total_pass_time_ms += duration_ms
        self._pass_completed += 1
        self._pass_finish_count += 1

        eta_ms = self._eta_ms(
            self._pass_completed, self._pass_finish_count, self._total_pass_time_ms
        )
        if eta_ms is not None:
            for column in (self._pass_clock_col, self._compact_clock_col):
                if column is not None:
                    column.set_eta(eta_ms)

        if self._pass_completed >= self._total:
            self._passes.status = "done"
            self._passes.elapsed_ms = self._total_pass_time_ms
            self._passes.detail = f"{self._total} passes"

        if self._is_live:
            self._advance_bar(self._pass_completed)
            self._refresh_live()
        elif self._pass_completed >= self._total:
            self._print_plain(
                event.at_ms,
                f"passes done ({format_duration(self._total_pass_time_ms)})",
            )

    def _advance_bar(self, completed: int) -> None:
        """Push the sampling count into whichever bar the live display shows."""
        if self._compact:
            if self._compact_progress is not None and self._compact_task_id is not None:
                self._compact_progress.update(self._compact_task_id, completed=completed)
        elif self._passes.bar is not None and self._pass_task_id is not None:
            self._passes.bar.update(self._pass_task_id, completed=completed)

    def _on_confirm_pass_started(self, event: PassStarted) -> None:
        self._confirm_pass_start_ms = event.at_ms
        if self._confirm.bar is not None and self._confirm_task_id is not None:
            self._confirm.bar.update(self._confirm_task_id, target=event.label)
        self._refresh_live()

    def _on_confirm_pass_finished(self, event: PassFinished) -> None:
        duration_ms = event.at_ms - self._confirm_pass_start_ms
        self._confirm_total_pass_time_ms += duration_ms
        self._confirm_completed += 1
        self._confirm_finish_count += 1

        eta_ms = self._eta_ms(
            self._confirm_completed, self._confirm_finish_count, self._confirm_total_pass_time_ms
        )
        if eta_ms is not None and self._confirm_clock_col is not None:
            self._confirm_clock_col.set_eta(eta_ms)

        if self._confirm.bar is not None and self._confirm_task_id is not None:
            self._confirm.bar.update(self._confirm_task_id, completed=self._confirm_completed)

        self._refresh_live()

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

        delta_str = (
            f"{event.primary_delta_pct:+.1f}%" if event.primary_delta_pct is not None else "—"
        )
        # The live row carries only the primary verdict and the regressed
        # names; the per-metric breakdown would crowd the checklist, so it
        # stays in the plain log below.
        primary = (
            f"{delta_str} on {self._primary_metric}"
            if event.primary_delta_pct is not None
            else delta_str
        )
        # No regression means confirm never fires: its row leaves the
        # checklist and the verdict rides on the judge's done line instead.
        verdict = [] if event.regressed else ["no gating regression"]

        # Build the live detail as rich.Text so each regressed metric name
        # carries its own format_inline styling (dim group/kind, normal case).
        detail = Text()
        detail.append(primary, style=STYLE_META)
        if event.regressed:
            detail.append(" · ", style=STYLE_META)
            detail.append(f"{len(event.regressed)} regressed: ", style=STYLE_META)
            for i, name in enumerate(event.regressed[:_REGRESSED_NAME_CAP]):
                if i > 0:
                    detail.append(", ", style=STYLE_META)
                detail.append_text(Text.from_markup(format_inline(parse(name), color=True)))
            if len(event.regressed) > _REGRESSED_NAME_CAP:
                detail.append(", …", style=STYLE_META)
        for v in verdict:
            detail.append(" · ", style=STYLE_META)
            detail.append(v, style=STYLE_META)
        self._judge.detail = detail

        if not event.regressed:
            self._confirm.status = "skipped"

        if self._is_live:
            self._refresh_live()
        else:
            # Plain mode: raw names suffice (no ANSI escapes on plain consoles).
            # The confirm row right below already announces the hand-off, so the
            # regressed list carries no arrow. The leading count is the real
            # information; at most three names follow so the row cannot balloon —
            # the final iteration report carries the full list.
            handoff: list[str] = []
            if event.regressed:
                shown = ", ".join(event.regressed[:_REGRESSED_NAME_CAP])
                if len(event.regressed) > _REGRESSED_NAME_CAP:
                    shown += ", …"
                handoff = [f"{len(event.regressed)} regressed: {shown}"]
            improve_noise = self._metric_count - len(event.regressed)
            breakdown = " · ".join([delta_str, f"{improve_noise} improve/noise", *handoff])
            self._print_plain(event.at_ms, f"judge {breakdown}")

    def _on_confirm_started(self, event: ConfirmStarted) -> None:
        self._confirm.status = "running"
        self._confirm.start_ms = event.at_ms

        # While confirm reruns the suite, the judge verdict is provisional, so
        # its row swaps the done glyph for the alert glyph.
        self._judge.alert = True

        if event.filtered_metrics is not None:
            self._confirm.note = pluralize(len(event.filtered_metrics), "metric")
        else:
            self._confirm.note = "full suite"

        if self._confirm.bar is not None and self._confirm_task_id is None:
            self._confirm_task_id = self._confirm.bar.add_task(
                "confirming", total=self._total, note=self._confirm.note
            )

        self._refresh_live()

    def _on_confirm_finished(self, event: ConfirmFinished) -> None:
        self._confirm.status = "done"
        self._confirm.elapsed_ms = event.at_ms - self._confirm.start_ms
        self._judge.alert = False

        status = "regressions reproduced" if event.reproduced else "regressions not reproduced"
        self._confirm.detail = f"{self._confirm_completed}/{self._total} · {status}"
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


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_iterate_renderer(  # noqa: PLR0913, PLR0917 -- mirrors the renderer constructor
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
) -> IterateRenderer:
    """Build an :class:`IterateRenderer` wired to ``console``.

    Kept as a thin wrapper around the constructor so callers can monkeypatch
    ``create_iterate_renderer`` in tests without touching ``IterateRenderer``
    itself.

    Args:
        mode: ``"live"`` for a rich animated checklist, ``"plain"`` for
            timestamped milestone lines.
        console: The ``Console`` to render to.
        seq: Iteration sequence number.
        session_id: Session identifier.
        sample_count: Number of samples per side.
        metric_count: Number of metrics being tracked.
        primary_metric: Name of the primary metric.
        verbose: When ``True`` and live, the checklist stays on stderr after stop.
        clock: Optional deterministic clock for ``Progress(get_time=...)``.
        checks_cmd: Shell command for the project's checks, shown in the record
            row as ``checks (cmd) run at gymrat keep``.
        has_before_hook: Whether a before hook is configured; without one the
            hook row never joins the checklist.
        has_after_hook: Whether an after hook is configured; without one the
            record hint drops ``then after hook``.
    """
    return IterateRenderer(
        mode=mode,
        console=console,
        seq=seq,
        session_id=session_id,
        sample_count=sample_count,
        metric_count=metric_count,
        primary_metric=primary_metric,
        verbose=verbose,
        clock=clock,
        checks_cmd=checks_cmd,
        has_before_hook=has_before_hook,
        has_after_hook=has_after_hook,
    )


# ---------------------------------------------------------------------------
# Fan-out
# ---------------------------------------------------------------------------


def create_fan_out(subscribers: Sequence[ProgressCallback]) -> ProgressCallback:
    """Fan out each event to every subscriber, isolating failures.

    One subscriber failing (raising an exception) never silences the others:
    exceptions are logged and swallowed so the remaining subscribers always run.
    """
    subs = list(subscribers)

    def fan_out(event: ProgressEvent) -> None:
        for subscriber in subs:
            try:
                subscriber(event)
            except Exception:
                logger.exception("fan-out subscriber failed")

    return fan_out
