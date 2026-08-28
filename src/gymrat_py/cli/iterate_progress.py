"""Rich-based progress renderer for the ``gymrat iterate`` command.

Live mode (TTY) shows a tree of phase nodes — before hook, prepare, passes,
judge, confirm, record — each annotated with glyphs (``○`` pending, ``⠹``
running, ``✔`` done) and elapsed time. Plain mode prints timestamped milestone
lines without ANSI escape codes.
"""

from __future__ import annotations

import logging
import sys
from typing import TYPE_CHECKING, Literal

from rich.console import Console, Group, RenderableType
from rich.live import Live
from rich.text import Text
from rich.tree import Tree

from gymrat_py.eta import format_duration, format_eta
from gymrat_py.progress_events import (
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
from gymrat_py.signals import install_termination_cleanup

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from rich.progress import Progress, TaskID

    from gymrat_py.cli.progress import _EtaColumn

logger = logging.getLogger(__name__)

_COMPACT_HEIGHT_THRESHOLD = 12
_CLEAR_LINE = "\r\x1b[K"
_MS_PER_SECOND = 1000
_SECONDS_PER_MINUTE = 60
_SECONDS_PER_HOUR = 3600

_PENDING = "○"
_RUNNING = "⠹"
_DONE = "✔"
_ALERT = "!"


# ---------------------------------------------------------------------------
# Node state tracking
# ---------------------------------------------------------------------------


class _NodeState:
    """Tracks the status and timing of a single tree node."""

    def __init__(self, label: str, *, subtext: str = "") -> None:
        self.label = label
        self.subtext = subtext
        self.status: Literal["pending", "running", "done"] = "pending"
        self.start_ms: float = 0.0
        self.elapsed_ms: float = 0.0
        self.detail: str = ""
        self.alert: bool = False

    @property
    def glyph(self) -> str:
        if self.alert:
            return _ALERT
        if self.status == "pending":
            return _PENDING
        if self.status == "running":
            return _RUNNING
        return _DONE


# ---------------------------------------------------------------------------
# Iterate renderer
# ---------------------------------------------------------------------------


class IterateRenderer:
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

        self._before_hook = _NodeState("before hook")
        self._prepare = _NodeState("prepare")
        self._passes = _NodeState("passes")
        self._judge = _NodeState(
            "judge",
            subtext=f"{metric_count} metrics · {primary_metric} primary",
        )
        self._confirm = _NodeState(
            "confirm",
            subtext="only if a gating metric regresses — reruns the suite",
        )
        self._record = _NodeState(
            "record",
            subtext=f"seq {seq} · then after hook",
        )

        self._prepare_labels: list[tuple[str, float]] = []
        self._prepare_current_start_ms: float = 0.0

        self._pass_completed = 0
        self._pass_finish_count = 0
        self._total_pass_time_ms: float = 0.0
        self._last_pass_duration_ms: float = 0.0
        self._last_pass_label: str = ""
        self._pass_start_ms: float = 0.0
        self._pass_start_clock: float | None = None
        self._pass_eta_text = "estimating time left…"  # noqa: S105 -- not a password
        self._pass_detail = ""

        self._confirm_completed = 0
        self._confirm_finish_count = 0
        self._confirm_total_pass_time_ms: float = 0.0
        self._confirm_pass_start_ms: float = 0.0
        self._confirm_pass_start_clock: float | None = None
        self._confirm_current_pass_round = 0
        self._confirm_current_pass_label = ""
        self._confirm_eta_text = "estimating time left…"

        self._judge_delta: float | None = None
        self._judge_regressed: tuple[str, ...] = ()

        self._confirm_reproduced: bool | None = None

        self._compact_progress: Progress | None = None
        self._compact_eta_col: _EtaColumn | None = None
        self._compact_task_id: TaskID | None = None

        if self._is_live:
            self._init_live()

    def _init_live(self) -> None:
        self._compact = self._console.height < _COMPACT_HEIGHT_THRESHOLD

        if self._compact:
            from gymrat_py.cli.progress import (  # noqa: PLC0415
                compact_progress,
            )

            self._compact_progress, self._compact_eta_col = compact_progress(
                self._console, clock=self._clock
            )

        self._live = Live(
            console=self._console,
            auto_refresh=True,
            refresh_per_second=1,
            transient=not self._verbose,
            redirect_stderr=False,
            get_renderable=self.frame,
        )
        self._live.start()

        self._uninstall_cleanup = install_termination_cleanup(self._clear_on_signal)

    def frame(self) -> RenderableType:
        """Return the renderable the live display paints from."""
        if self._compact and self._compact_progress is not None:
            return self._compact_progress

        header = self._build_header()
        tree = Tree(header)

        for node_state in (
            self._before_hook,
            self._prepare,
            self._passes,
            self._judge,
            self._confirm,
            self._record,
        ):
            node_text = self._render_node(node_state)
            node = tree.add(node_text)

            if node_state is self._prepare and self._prepare_labels:
                for label, elapsed_ms in self._prepare_labels:
                    node.add(
                        Text(f"  {_DONE} {label} ({format_duration(elapsed_ms)})", style="dim")
                    )

            if node_state is self._passes and node_state.status in ("running", "done"):
                detail = self._pass_detail
                elapsed_ms = self._clock_elapsed_ms(self._pass_start_clock)
                if elapsed_ms is not None:
                    detail = detail.replace(
                        " running", f" running {format_duration(elapsed_ms)}", 1
                    )
                self._add_pass_bar(node, self._pass_completed, self._pass_eta_text, detail)

            if node_state is self._confirm and node_state.status == "running":
                node.add(
                    self._render_passes_bar(
                        self._confirm_completed,
                        self._total,
                        self._confirm_eta_text,
                    )
                )
                elapsed_ms = self._clock_elapsed_ms(self._confirm_pass_start_clock)
                if elapsed_ms is not None:
                    confirm_detail = (
                        f"round {self._confirm_current_pass_round} · "
                        f"{self._confirm_current_pass_label} "
                        f"running {format_duration(elapsed_ms)}"
                    )
                    node.add(Text(f"  {confirm_detail}", style="dim"))

        return Group(tree)

    def _add_pass_bar(self, node: Tree, completed: int, eta_text: str, note: str | None) -> None:
        node.add(self._render_passes_bar(completed, self._total, eta_text))
        if note:
            node.add(Text(f"  {note}", style="dim"))

    def _build_header(self) -> str:
        parts = [f"iterate #{self._seq} session {self._session_id}"]

        elapsed_ms = self._clock_elapsed_ms(self._start_clock_time)
        if elapsed_ms is not None:
            parts.append(f"{format_duration(elapsed_ms)} elapsed")

        eta_text = self._eta_text(
            self._pass_completed, self._pass_finish_count, self._total_pass_time_ms
        )
        if eta_text is not None:
            parts.append(eta_text)

        return " · ".join(parts)

    def _running_elapsed_ms(self, node: _NodeState) -> float | None:
        """Live elapsed time for ``node``, or ``None`` if it doesn't tick live.

        Only the judge node ticks while running — it's the one long step
        that cannot be interrupted and lacks its own progress events.
        """
        if node is not self._judge or node.status != "running":
            return None
        if self._clock is None or node.start_ms <= 0:
            return None
        return self._clock() * _MS_PER_SECOND - node.start_ms

    def _render_node(self, node: _NodeState) -> Text:
        glyph = node.glyph
        text = Text()
        if node.status == "done" and not node.alert:
            style = "green"
        elif node.status == "running":
            style = "yellow"
        else:
            style = "dim"

        text.append(f"{glyph} ", style=style)
        text.append(node.label, style="" if node.status != "pending" else "dim")

        if node.status == "done" and node.elapsed_ms > 0:
            text.append(f" ({format_duration(node.elapsed_ms)})", style="dim")
        else:
            running_ms = self._running_elapsed_ms(node)
            if running_ms is not None:
                text.append(f" ({format_duration(running_ms)})", style="dim")

        if node.detail:
            text.append(f" {node.detail}")

        if node.status == "pending" and node.subtext:
            text.append(f"  {node.subtext}", style="dim")

        return text

    def _render_passes_bar(
        self,
        completed: int,
        total: int,
        eta_text: str,
    ) -> Text:
        fraction = completed / total if total > 0 else 0.0
        pct = int(fraction * 100)
        text = Text()
        text.append(f"  {completed}/{total}", style="bold")
        text.append(f" ({pct}%)", style="dim")
        text.append(f" {eta_text}", style="dim")
        return text

    def _eta_text(self, completed: int, finish_count: int, total_time_ms: float) -> str | None:
        """ETA string for the given completion state, or ``None`` if too early to estimate."""
        remaining = self._total - completed
        if remaining <= 0 or finish_count == 0:
            return None
        avg_ms = total_time_ms / finish_count
        return format_eta(avg_ms * remaining)

    def _refresh_live(self) -> None:
        if self._live is not None:
            self._live.refresh()

    def _now_clock(self) -> float | None:
        """Current clock reading, or ``None`` if no clock was configured."""
        return self._clock() if self._clock is not None else None

    def _clock_elapsed_ms(self, start_clock: float | None) -> float | None:
        """Milliseconds elapsed since ``start_clock``, or ``None`` if unavailable."""
        if start_clock is None or self._clock is None:
            return None
        return (self._clock() - start_clock) * _MS_PER_SECOND

    def _track_timestamp(self, at_ms: float) -> None:
        if self._run_start_ms is None:
            self._run_start_ms = at_ms
            if self._clock is not None:
                self._start_clock_time = self._clock()

    def _format_timestamp(self, at_ms: float) -> str:
        elapsed_ms = at_ms - (self._run_start_ms or at_ms)
        total_seconds = int(elapsed_ms / _MS_PER_SECOND)
        hours = total_seconds // _SECONDS_PER_HOUR
        minutes = (total_seconds % _SECONDS_PER_HOUR) // _SECONDS_PER_MINUTE
        seconds = total_seconds % _SECONDS_PER_MINUTE
        return f"[{hours:02d}:{minutes:02d}:{seconds:02d}]"

    def _print_plain(self, at_ms: float, message: str) -> None:
        ts = self._format_timestamp(at_ms)
        self._console.print(f"{ts} {message}", highlight=False, markup=False)

    # -----------------------------------------------------------------------
    # Event handlers
    # -----------------------------------------------------------------------

    def report(self, event: ProgressEvent) -> None:  # noqa: C901 -- dispatch chain
        """Dispatch ``event`` to its handler after anchoring the run timestamp."""
        self._track_timestamp(event.at_ms)

        if isinstance(event, HookStarted):
            self._on_hook_started(event)
        elif isinstance(event, HookFinished):
            self._on_hook_finished(event)
        elif isinstance(event, PrepareStarted):
            self._on_prepare_started(event)
        elif isinstance(event, PrepareFinished):
            self._on_prepare_finished(event)
        elif isinstance(event, PassStarted):
            self._on_pass_started(event)
        elif isinstance(event, PassFinished):
            self._on_pass_finished(event)
        elif isinstance(event, JudgeStarted):
            self._on_judge_started(event)
        elif isinstance(event, JudgeFinished):
            self._on_judge_finished(event)
        elif isinstance(event, ConfirmStarted):
            self._on_confirm_started(event)
        elif isinstance(event, ConfirmFinished):
            self._on_confirm_finished(event)
        elif isinstance(event, IterationRecorded):
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
        self._prepare.detail = event.label
        self._refresh_live()

    def _on_prepare_finished(self, event: PrepareFinished) -> None:
        elapsed_ms = event.at_ms - self._prepare_current_start_ms
        self._prepare_labels.append((event.label, elapsed_ms))
        self._prepare.elapsed_ms += elapsed_ms
        self._prepare.status = "done"
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
        self._pass_start_clock = self._now_clock()

        parts = [f"round {event.round}", f"{event.label} running"]
        if self._last_pass_duration_ms > 0:
            last_pass = f"last pass {format_duration(self._last_pass_duration_ms)}"
            if self._last_pass_label:
                last_pass += f" ({self._last_pass_label})"
            parts.append(last_pass)
        self._pass_detail = " · ".join(parts)

        if self._compact and self._compact_progress is not None:
            if self._compact_task_id is None:
                desc = f"sample {event.round}/{event.total_rounds}"
                self._compact_task_id = self._compact_progress.add_task(
                    desc,
                    total=self._total,
                    target=event.label,
                )
            else:
                self._compact_progress.update(
                    self._compact_task_id,
                    description=f"sample {event.round}/{event.total_rounds}",
                    target=event.label,
                    completed=self._pass_completed,
                )
        self._refresh_live()

    def _on_pass_finished(self, event: PassFinished) -> None:
        if event.phase == "confirm":
            self._on_confirm_pass_finished(event)
            return

        duration_ms = event.at_ms - self._pass_start_ms
        self._pass_start_clock = None
        self._last_pass_duration_ms = duration_ms
        self._last_pass_label = event.label
        self._total_pass_time_ms += duration_ms
        self._pass_completed += 1
        self._pass_finish_count += 1

        eta_text = self._eta_text(
            self._pass_completed, self._pass_finish_count, self._total_pass_time_ms
        )
        if eta_text is not None:
            self._pass_eta_text = eta_text
            if self._compact_eta_col is not None:
                remaining = self._total - self._pass_completed
                avg_ms = self._total_pass_time_ms / self._pass_finish_count
                self._compact_eta_col.set_eta(avg_ms * remaining)

        if self._pass_completed >= self._total:
            self._passes.status = "done"
            self._passes.elapsed_ms = self._total_pass_time_ms

        if self._is_live:
            if (
                self._compact
                and self._compact_progress is not None
                and self._compact_task_id is not None
            ):
                self._compact_progress.update(self._compact_task_id, completed=self._pass_completed)
            self._refresh_live()
        elif self._pass_completed >= self._total:
            self._print_plain(
                event.at_ms,
                f"passes done ({format_duration(self._total_pass_time_ms)})",
            )

    def _on_confirm_pass_started(self, event: PassStarted) -> None:
        self._confirm_pass_start_ms = event.at_ms
        self._confirm_current_pass_round = event.round
        self._confirm_current_pass_label = event.label
        self._confirm_pass_start_clock = self._now_clock()
        self._refresh_live()

    def _on_confirm_pass_finished(self, event: PassFinished) -> None:
        duration_ms = event.at_ms - self._confirm_pass_start_ms
        self._confirm_pass_start_clock = None
        self._confirm_total_pass_time_ms += duration_ms
        self._confirm_completed += 1
        self._confirm_finish_count += 1

        eta_text = self._eta_text(
            self._confirm_completed, self._confirm_finish_count, self._confirm_total_pass_time_ms
        )
        if eta_text is not None:
            self._confirm_eta_text = eta_text

        self._refresh_live()

    def _on_judge_started(self, event: JudgeStarted) -> None:
        self._judge.status = "running"
        self._judge.start_ms = event.at_ms
        self._judge.subtext = ""
        self._refresh_live()

    def _on_judge_finished(self, event: JudgeFinished) -> None:
        self._judge.status = "done"
        self._judge.elapsed_ms = (
            event.at_ms - self._judge.start_ms if self._judge.start_ms > 0 else 0
        )

        self._judge_delta = event.primary_delta_pct
        self._judge_regressed = event.regressed

        delta_str = (
            f"{event.primary_delta_pct:+.1f}%" if event.primary_delta_pct is not None else "—"
        )
        improve_noise = self._metric_count - len(event.regressed)
        detail_parts = [delta_str, f"{improve_noise} improve/noise"]
        if event.regressed:
            detail_parts.append(f"regressed: {', '.join(event.regressed)} → confirm")
        self._judge.detail = " · ".join(detail_parts)

        if self._is_live:
            self._refresh_live()
        else:
            self._print_plain(event.at_ms, f"judge {self._judge.detail}")

    def _on_confirm_started(self, event: ConfirmStarted) -> None:
        self._confirm.status = "running"
        self._confirm.start_ms = event.at_ms
        self._confirm.subtext = ""

        # While confirm is running, judge shows ! instead of the done glyph
        self._judge.alert = True

        if event.filtered_metrics is not None:
            self._confirm.label = f"confirm {len(event.filtered_metrics)} metrics"
        else:
            self._confirm.label = "confirm (full suite)"

        self._refresh_live()

    def _on_confirm_finished(self, event: ConfirmFinished) -> None:
        self._confirm.status = "done"
        self._confirm.elapsed_ms = event.at_ms - self._confirm.start_ms
        self._confirm_reproduced = event.reproduced
        self._judge.alert = False

        status = "regressions reproduced" if event.reproduced else "regressions not reproduced"
        self._confirm.detail = f"{self._confirm_completed}/{self._total} · {status}"

        if self._is_live:
            self._refresh_live()
        else:
            self._print_plain(event.at_ms, f"confirm {self._confirm.detail}")

    def _on_iteration_recorded(self, event: IterationRecorded) -> None:
        self._record.status = "done"
        self._record.subtext = ""

        detail = f"seq {event.seq} · {event.outcome} suggested"
        if self._checks_cmd is not None:
            detail += f" — checks ({self._checks_cmd}) run at gymrat keep"
        self._record.detail = detail

        if self._is_live:
            self._refresh_live()
        else:
            self._print_plain(event.at_ms, f"recorded {self._record.detail}")

    # -----------------------------------------------------------------------
    # Signal cleanup
    # -----------------------------------------------------------------------

    def _clear_on_signal(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        sys.stderr.write(_CLEAR_LINE)
        sys.stderr.flush()

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
) -> IterateRenderer:
    """Build an :class:`IterateRenderer` wired to ``console``.

    Args:
        mode: ``"live"`` for a rich animated tree, ``"plain"`` for timestamped
            milestone lines.
        console: The ``Console`` to render to.
        seq: Iteration sequence number.
        session_id: Session identifier.
        sample_count: Number of samples per side.
        metric_count: Number of metrics being tracked.
        primary_metric: Name of the primary metric.
        verbose: When ``True`` and live, the tree stays on stderr after stop.
        clock: Optional deterministic clock for ``Progress(get_time=...)``.
        checks_cmd: Shell command for the project's checks, shown in the record
            node as ``checks (cmd) run at gymrat keep``.
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
