"""Progress reporter for a supervised optimization run.

This module is the shell around :mod:`gymrat.cli.supervise.reducer`: it owns the
terminal, the clock, and the session read, and holds the reducer state.  Every
event goes through :func:`handle_event`, which reads the session when the reducer
asks for it, replaces the state, and renders the result.
"""

from __future__ import annotations

import contextlib
import functools
import logging
import operator
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Literal

from gymrat.cli.console import stderr_console
from gymrat.cli.live_display import ErasableLive, mount_live
from gymrat.cli.supervise.frame import build_frame
from gymrat.cli.supervise.reducer import (
    ReporterState,
    advance,
    exit_phase,
    plain_line,
    wants_session_refresh,
)
from gymrat.cli.supervise.text import exit_phase_text
from gymrat.cli.supervise.types import BestIteration, ReadSessionResult
from gymrat.clock import now_ms
from gymrat.session.paths import session_jsonl_path
from gymrat.session.progress_file import read_progress as _default_read_progress
from gymrat.session.records import IterationRecord, KeepRecord, StopRecord
from gymrat.session.store import fold_session, latest_baseline, read_records
from gymrat.utils import MS_PER_SECOND, warn_to_stderr

if TYPE_CHECKING:
    from collections.abc import Callable
    from datetime import tzinfo

    from rich.console import RenderableType

    from gymrat.config import Effort
    from gymrat.session.progress_file import ProgressSnapshot
    from gymrat.session.records import SessionLogRecord
    from gymrat.supervisor.events import SessionEvent, SessionObserver
    from gymrat.supervisor.exit_sequence import ExitPhase

_logger = logging.getLogger(__name__)

REFRESH_MS = 1000
"""Default Live dashboard refresh interval in milliseconds."""

IDLE_WARN_MS = 30_000
"""After 30 seconds of no tool activity, the liveness line escalates to alert styling."""


# ---------------------------------------------------------------------------
# Session read
# ---------------------------------------------------------------------------


def _find_best_kept_iteration(
    records: list[SessionLogRecord], committed_seqs: set[int]
) -> BestIteration | None:
    """Return the committed keep with the best primary delta, if any has one."""
    candidates = [
        (r, delta)
        for r in records
        if isinstance(r, IterationRecord)
        and r.seq in committed_seqs
        and (delta := r.primary.delta_pct) is not None
    ]
    if not candidates:
        return None

    best, best_delta = min(candidates, key=operator.itemgetter(1))
    return BestIteration(
        delta_pct=best_delta, seq=best.seq, label=best.primary.name or best.primary.kind
    )


def _find_stop_message(records: list[SessionLogRecord]) -> str | None:
    """Return the newest stop record's message, or ``None`` if there is none."""
    return next((r.message for r in reversed(records) if isinstance(r, StopRecord)), None)


def read_live_session(root: str) -> ReadSessionResult:
    """Read and fold the live session log at ``root``.

    Args:
        root: The repository root whose session log to read.

    Returns:
        The folded session with its baseline presence, best kept iteration, and
        trailing stop message.
    """
    records = read_records(session_jsonl_path(root))
    state = fold_session(records)
    has_baseline = latest_baseline(records) is not None

    committed_seqs = {
        r.seq for r in records if isinstance(r, KeepRecord) and r.status == "committed"
    }

    return ReadSessionResult(
        state=state,
        has_baseline=has_baseline,
        best=_find_best_kept_iteration(records, committed_seqs),
        baseline_sha=state.session.baseline.sha if state.session is not None else None,
        stop_message=_find_stop_message(records) if state.ends_on_stop else None,
    )


# ---------------------------------------------------------------------------
# Reporter shell
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ReporterCtx:
    """The terminal, clock, and I/O handles the reporter shell owns.

    Everything the dashboard renders lives in ``state``, which the shell
    replaces after each event.  Only the shell writes to this object; the
    reducer never sees it.
    """

    state: ReporterState
    now: Callable[[], int]
    read_session_fn: Callable[[], ReadSessionResult]
    read_progress_fn: Callable[[str], ProgressSnapshot | None]
    plain_write_fn: Callable[[str], None]
    warn_fn: Callable[[str], None]
    live: ErasableLive | None
    tz: tzinfo | None
    idle_warn_ms: int


@dataclass(frozen=True, slots=True)
class SuperviseReporter:
    """The observer/stop/frame/warn surface that drives the supervise progress display.

    ``session_result`` hands back the session state as of the last re-read, which
    is what the closing summary reports once the display has stopped.

    Live mode's display and its refresh timer run from construction until
    ``stop``, and for that span a termination signal erases the display: its
    cleanup is installed at construction and uninstalled by ``stop``.

    ``exit_phase`` shows the run-end exit sequence's current phase: live mode
    repaints the frame, plain mode writes the phase line once per phase change.

    ``refresh_session`` re-reads the session so ``session_result`` reflects
    writes that no event announced, such as an exit-sequence step that failed after
    writing to the session log. A successful re-read writes no plain line; a
    failed read keeps the previous result and warns, as the event-driven
    re-read does.
    """

    observer: SessionObserver
    stop: Callable[[], None]
    frame: Callable[[], RenderableType]
    warn: Callable[[str], None]
    session_result: Callable[[], ReadSessionResult | None]
    final_text: Callable[[], str | None]
    exit_phase: Callable[[ExitPhase], None]
    refresh_session: Callable[[], None]


# ---------------------------------------------------------------------------
# Reporter factory
# ---------------------------------------------------------------------------


def _stop_live(live: ErasableLive | None) -> None:
    if live is not None:
        try:
            # stderr closed or broken at shutdown raises OSError on write
            with contextlib.suppress(OSError):
                live.stop()
        except ValueError as exc:
            # A closed text stream raises ValueError("I/O operation on closed
            # file") on write, not OSError; any other ValueError is unexpected
            # and must propagate.
            if "closed file" not in str(exc):
                raise


def _nothing_installed() -> None:
    pass


def _stop(ctx: ReporterCtx, uninstall_erase: Callable[[], None]) -> None:
    # Uninstalled first: once the display starts stopping, a signal must not
    # erase rows that Live.stop() has already cleared, or the output above them.
    uninstall_erase()
    _stop_live(ctx.live)


# ---------------------------------------------------------------------------
# Rendering and event dispatch
# ---------------------------------------------------------------------------


def _frame(ctx: ReporterCtx) -> RenderableType:
    # Runs on rich's refresh thread while events replace ctx.state on the
    # caller's thread. Read ctx.state exactly once: the state is immutable and
    # swapped by a single assignment, so one read is a consistent snapshot and
    # no lock is needed. A second read could mix two states in one frame.
    return build_frame(
        ctx.state,
        ctx.now(),
        tz=ctx.tz,
        idle_warn_ms=ctx.idle_warn_ms,
        read_progress=ctx.read_progress_fn,
    )


@dataclass(slots=True)
class _LiveFrame:
    """The live display's ``get_renderable``, which keeps the display up when a frame fails.

    rich builds a frame while the display is set up (in its constructor and on
    the mount's first paint), on its refresh thread, on every event-driven
    repaint, and once more inside ``Live.stop()``. An error escaping the build
    would end the refresh thread for good, freezing elapsed time and the idle
    indicator, or escape ``stop``. A frame that fails to build is replaced by the last one
    that built, and only the first failure is warned about, so a persistent
    fault does not print a line every second. The warning goes through
    ``ctx.warn_fn``, which is the dashboard's channel before rich builds its
    first frame.
    """

    ctx: ReporterCtx
    last: RenderableType = ""
    warned: bool = False

    def __call__(self) -> RenderableType:
        try:
            self.last = _frame(self.ctx)
        except Exception as exc:
            # Not logged at warning level: with no handler configured, logging
            # would print the traceback over the dashboard on every refresh.
            _logger.debug("dashboard frame failed to render", exc_info=True)
            if not self.warned:
                self.warned = True
                self.ctx.warn_fn(
                    f"warning: dashboard frame failed to render: {type(exc).__name__}: {exc}"
                )
        return self.last


def _read_session(ctx: ReporterCtx) -> ReadSessionResult | None:
    try:
        return ctx.read_session_fn()
    except Exception as exc:
        _logger.exception("session read failed")
        ctx.warn_fn(f"session read failed: {exc}")
        return None


def handle_event(ctx: ReporterCtx, event: SessionEvent) -> None:
    """Fold one session event into the reporter state and render the result.

    When the reducer hands back the state it was given — a text delta or a tool
    progress ping changes nothing — live mode skips the repaint, since the only
    part of the frame that could have moved is elapsed time and the Live
    display's refresh timer already keeps that current.

    Args:
        ctx: The reporter shell whose state is replaced with the reduced one.
        event: The incoming session event.
    """
    session = _read_session(ctx) if wants_session_refresh(ctx.state, event) else None
    before = ctx.state
    ctx.state = advance(before, event, session)

    if ctx.live is not None:
        if ctx.state is not before:
            ctx.live.refresh()
        return
    line = plain_line(before, ctx.state, event)
    if line is not None:
        ctx.plain_write_fn(line)


def _refresh_session(ctx: ReporterCtx) -> None:
    session = _read_session(ctx)
    if session is None:
        return
    ctx.state = replace(ctx.state, session_result=session)
    if ctx.live is not None:
        ctx.live.refresh()


def _report_exit_phase(ctx: ReporterCtx, phase: ExitPhase) -> None:
    before = ctx.state
    ctx.state = exit_phase(before, phase, ctx.now())
    if ctx.state is before:
        return
    if ctx.live is None:
        ctx.plain_write_fn(exit_phase_text(phase))
    else:
        ctx.live.refresh()


def create_supervise_reporter(  # noqa: PLR0913 - one parameter per reporter knob
    *,
    root: str,
    max_minutes: float,
    max_usd: float | None = None,
    max_iterations: int | None = None,
    mode: Literal["live", "plain"],
    log_path: str = "",
    now: Callable[[], int] | None = None,
    read_session: Callable[[], ReadSessionResult] | None = None,
    session_id: str = "",
    branch: str = "",
    plain_write: Callable[[str], None] | None = None,
    read_progress: Callable[[str], ProgressSnapshot | None] | None = None,
    color: bool | None = None,
    tz: tzinfo | None = None,
    model: str | None = None,
    effort: Effort | None = None,
    refresh_ms: int = REFRESH_MS,
    idle_warn_ms: int = IDLE_WARN_MS,
) -> SuperviseReporter:
    """Build the observer/stop/frame/warn surface for the supervise dashboard.

    In live mode the Live display starts here, taking over stderr, and rich
    repaints the frame on its own refresh thread every ``refresh_ms``, so
    elapsed time and the idle indicator advance without events. A frame that
    fails to render never raises: while the display is being set up, on the
    refresh thread, or on an event-driven repaint, the last frame that rendered
    stays on screen (none yet during setup) and the refresh thread keeps
    running; on the final paint inside ``stop``, ``stop`` returns normally. The
    first such failure is reported once, through ``warn``, as a warning line
    naming the error. The reporter's ``stop`` stops the display and its refresh
    thread, and must always run once the reporter exists.

    While the live display is up, a termination signal erases it and shows the
    cursor again, through a cleanup installed here and removed by ``stop``.

    Args:
        root: Project root whose session directory is monitored.
        max_minutes: Wall-clock cap in minutes.
        max_usd: Spend cap in USD, or ``None`` for uncapped.
        max_iterations: Iteration cap, or ``None`` for uncapped.
        mode: ``"live"`` for a Rich Live dashboard, ``"plain"`` for line-by-line
            stderr output.
        log_path: Path to the supervisor event log, shown on the frame's
            ``log:`` line.
        now: Wall-clock source returning epoch milliseconds.  Defaults to
            :func:`~gymrat.clock.now_ms`; override in tests.
        read_session: Callable that reads the current session state.  Defaults to
            :func:`read_live_session`; override in tests.
        session_id: Session identifier propagated to the frame.
        branch: Git branch name shown in the frame header.
        plain_write: Line writer for plain mode, called once per line without a
            trailing newline.  Defaults to writing the line plus a newline to
            stderr; override in tests.
        read_progress: Callable that reads the iterate progress sidecar.
            Defaults to the standard reader; override in tests.
        color: Tri-state color override: ``True`` forces color, ``False``
            disables it, ``None`` auto-detects.
        tz: Timezone for wall-clock timestamps.  ``None`` uses the local zone.
        model: Model name shown as a labelled row when set.
        effort: Effort level shown as a labelled row when set.
        refresh_ms: Live dashboard refresh interval in milliseconds.
        idle_warn_ms: Milliseconds of inactivity before the liveness line
            escalates to alert styling.

    Returns:
        A fully wired reporter whose callbacks drive the dashboard lifecycle.
    """
    if plain_write is None:
        plain_write = warn_to_stderr
    ctx = ReporterCtx(
        state=ReporterState(
            root=root,
            max_minutes=max_minutes,
            max_usd=max_usd,
            max_iterations=max_iterations,
            session_id=session_id,
            branch=branch,
            model=model,
            effort=effort,
            log_path=log_path,
        ),
        now=now if now is not None else now_ms,
        read_session_fn=(
            read_session if read_session is not None else functools.partial(read_live_session, root)
        ),
        read_progress_fn=read_progress if read_progress is not None else _default_read_progress,
        plain_write_fn=plain_write,
        warn_fn=plain_write,
        live=None,
        tz=tz,
        idle_warn_ms=idle_warn_ms,
    )

    uninstall_erase: Callable[[], None] = _nothing_installed
    if mode == "live":
        console = stderr_console(color_flag=color)
        # Set before the display exists: rich builds a frame while it is being
        # set up, and a failure then must already reach the dashboard's channel.
        # Messages carry arbitrary text (paths, command output); markup would
        # swallow anything in square brackets.
        ctx.warn_fn = functools.partial(console.print, markup=False)
        live = ErasableLive(
            console=console,
            auto_refresh=True,
            refresh_per_second=MS_PER_SECOND / refresh_ms,
            get_renderable=_LiveFrame(ctx),
            transient=True,
        )
        ctx.live = live
        # A signal exits through os._exit, which skips stop(). The mount's erase
        # waits on the display lock only for a bounded time, unlike Live.stop(),
        # so the cleanups installed after it always get to run. No reporter
        # exists for the caller to stop until this returns, so the mount itself
        # stops the display when starting or writing the first paint fails.
        uninstall_erase = mount_live(live)

    return SuperviseReporter(
        observer=lambda event: handle_event(ctx, event),
        stop=lambda: _stop(ctx, uninstall_erase),
        frame=lambda: _frame(ctx),
        warn=ctx.warn_fn,
        session_result=lambda: ctx.state.session_result,
        final_text=lambda: ctx.state.last_agent_text,
        exit_phase=lambda phase: _report_exit_phase(ctx, phase),
        refresh_session=lambda: _refresh_session(ctx),
    )
