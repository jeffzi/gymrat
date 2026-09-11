"""Progress reporter for a supervised optimization run.

Event dispatching lives in :mod:`gymrat.cli.supervise.handlers`.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import sys
from collections import deque
from typing import TYPE_CHECKING, Literal

from rich.live import Live

from gymrat.cli.console import stderr_console
from gymrat.cli.supervise.frame import build_frame
from gymrat.cli.supervise.handlers import handle_event, render_live
from gymrat.cli.supervise.session_read import make_default_read
from gymrat.cli.supervise.state import (
    IDLE_WARN_MS,
    ReadSessionResult,
    ReporterCtx,
    Starting,
    SuperviseReporter,
)
from gymrat.eta import MS_PER_SECOND
from gymrat.session.clock import now_ms
from gymrat.session.progress_file import read_progress as _default_read_progress

if TYPE_CHECKING:
    from collections.abc import Callable
    from datetime import tzinfo

    from gymrat.config import Effort
    from gymrat.session.progress_file import ProgressSnapshot

_tick_logger = logging.getLogger(__name__)

_MAX_FINISHED_TOOLS = 3

REFRESH_MS = 1000
"""Default Live dashboard refresh interval in milliseconds."""


# ---------------------------------------------------------------------------
# Reporter factory
# ---------------------------------------------------------------------------


def _stderr_write(text: str) -> None:
    sys.stderr.write(f"{text}\n")


def _stop_live(live: Live | None) -> None:
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


def _new_ctx(  # noqa: PLR0913 - one field per reporter knob
    *,
    root: str,
    max_minutes: float,
    max_usd: float | None,
    max_iterations: int | None,
    is_plain: bool,
    now: Callable[[], int],
    read_session: Callable[[], ReadSessionResult],
    read_progress: Callable[[str], ProgressSnapshot | None],
    plain_write: Callable[[str], None],
    label: str,
    session_id: str,
    branch: str,
    tz: tzinfo | None,
    no_color: bool,
    log_path: str,
    model: str | None,
    effort: Effort | None,
    idle_warn_ms: int,
    refresh_ms: int,
) -> ReporterCtx:
    return ReporterCtx(
        now=now,
        read_session_fn=read_session,
        read_progress_fn=read_progress,
        root=root,
        max_minutes=max_minutes,
        max_usd=max_usd,
        max_iterations=max_iterations,
        is_plain=is_plain,
        label=label,
        session_id=session_id,
        branch=branch,
        in_flight_tools={},
        finished_tools=deque(maxlen=_MAX_FINISHED_TOOLS),
        launch_timestamp=None,
        cost_usd=None,
        session_result=None,
        liveness=Starting(),
        last_loop_text="",
        tz=tz,
        plain_write_fn=plain_write,
        warn_fn=plain_write,
        live=None,
        nested={},
        nested_tool_ids={},
        no_color=no_color,
        log_path=log_path,
        last_agent_text=None,
        turn_count=0,
        last_decision=None,
        model=model,
        effort=effort,
        idle_warn_ms=idle_warn_ms,
        refresh_ms=refresh_ms,
    )


async def _tick(ctx: ReporterCtx) -> None:
    """Periodically refresh the Live display so elapsed time stays current.

    Runs until cancelled. If a render raises, the failure is reported once
    through ``ctx.warn_fn`` and the task exits — event-driven renders are
    unaffected.

    Args:
        ctx: The shared reporter context supplying the refresh interval and
            the state rendered on each tick.
    """
    interval = ctx.refresh_ms / MS_PER_SECOND
    while True:
        await asyncio.sleep(interval)
        try:
            render_live(ctx)
        except Exception as exc:
            # warn_fn may write to the same failed console; never let it mask
            # the original failure or skip the log below.
            with contextlib.suppress(Exception):
                ctx.warn_fn(f"tick render failed: {exc}")
            _tick_logger.exception("tick render failed")
            return


def _on_tick_done(task: asyncio.Task[None]) -> None:
    """Log unhandled tick-task failures so they are never silently swallowed."""
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        _tick_logger.error("tick task failed", exc_info=exc)


def _start_tick(ctx: ReporterCtx) -> None:
    """Create the tick task if in live mode; no-op in plain mode."""
    if ctx.is_plain:
        return
    task = asyncio.create_task(_tick(ctx))
    task.add_done_callback(_on_tick_done)
    ctx.tick_task = task


def _stop_tick_and_live(ctx: ReporterCtx) -> None:
    """Cancel the tick task (if running) then stop Live."""
    if ctx.tick_task is not None:
        ctx.tick_task.cancel()
        ctx.tick_task = None
    _stop_live(ctx.live)


def _make_warn(ctx: ReporterCtx) -> Callable[[str], None]:
    def warn(message: str) -> None:
        if ctx.is_plain:
            ctx.plain_write_fn(message)
        elif ctx.live is not None:
            ctx.live.console.print(message)

    return warn


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
    label: str = "",
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

    The returned reporter's ``start`` must be called from within a running
    event loop; it is a no-op in plain mode.

    Args:
        root: Project root whose session directory is monitored.
        max_minutes: Wall-clock cap in minutes.
        max_usd: Spend cap in USD, or ``None`` for uncapped.
        max_iterations: Iteration cap, or ``None`` for uncapped.
        mode: ``"live"`` for a Rich Live dashboard, ``"plain"`` for line-by-line
            stderr output.
        log_path: Path to the supervisor event log, shown in the closing summary.
        now: Monotonic-clock source returning milliseconds.  Defaults to
            :func:`~gymrat.session.clock.now_ms`; override in tests.
        read_session: Callable that reads the current session state.  Defaults to
            :func:`make_default_read`; override in tests.
        label: Human label for the run, shown in the frame header.
        session_id: Session identifier propagated to the frame.
        branch: Git branch name shown in the frame header.
        plain_write: Stderr writer for plain mode.  Defaults to
            ``sys.stderr.write``; override in tests.
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
    ctx = _new_ctx(
        root=root,
        max_minutes=max_minutes,
        max_usd=max_usd,
        max_iterations=max_iterations,
        is_plain=mode == "plain",
        now=now if now is not None else now_ms,
        read_session=read_session if read_session is not None else make_default_read(root),
        read_progress=read_progress if read_progress is not None else _default_read_progress,
        plain_write=plain_write if plain_write is not None else _stderr_write,
        label=label,
        session_id=session_id,
        branch=branch,
        tz=tz,
        no_color=color is False,
        log_path=log_path,
        model=model,
        effort=effort,
        idle_warn_ms=idle_warn_ms,
        refresh_ms=refresh_ms,
    )

    if not ctx.is_plain:
        ctx.live = Live(
            console=stderr_console(color_flag=color),
            auto_refresh=False,
            transient=True,
        )
        ctx.live.start()
        ctx.live.update(build_frame(ctx), refresh=True)

    warn = _make_warn(ctx)
    ctx.warn_fn = warn
    return SuperviseReporter(
        observer=lambda event: handle_event(ctx, event),
        start=lambda: _start_tick(ctx),
        stop=lambda: _stop_tick_and_live(ctx),
        frame=lambda: build_frame(ctx),
        warn=warn,
        session_result=lambda: ctx.session_result,
        final_text=lambda: ctx.last_agent_text,
    )
