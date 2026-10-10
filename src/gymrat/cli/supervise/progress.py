"""Progress reporter for a supervised optimization run.

This module is the shell around :mod:`gymrat.cli.supervise.reducer`: it owns the
terminal, the clock, and the session read, and holds the reducer state.  Every
event goes through the shell's ``handle_event``, which reads the session when the
reducer asks for it, replaces the state, and renders the result.  The live
display starts, refreshes, warns, and stops through
:class:`~gymrat.cli.live_display.LiveDisplayMixin`, as every CLI progress
renderer's does.
"""

from __future__ import annotations

import functools
import logging
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Literal, override

from gymrat.cli.console import stderr_console
from gymrat.cli.live_display import LiveDisplayMixin
from gymrat.cli.supervise.frame import build_frame
from gymrat.cli.supervise.reducer import (
    ReporterState,
    advance,
    exit_phase,
    plain_line,
    wants_session_refresh,
)
from gymrat.cli.supervise.text import exit_phase_text
from gymrat.clock import now_ms
from gymrat.session.progress_file import read_progress as _default_read_progress
from gymrat.session.store import read_live_session
from gymrat.utils import MS_PER_SECOND, warn_to_stderr

if TYPE_CHECKING:
    from collections.abc import Callable
    from datetime import tzinfo

    from rich.console import Console, RenderableType

    from gymrat.cli.live_display import ErasableLive
    from gymrat.config import Effort
    from gymrat.model import Direction
    from gymrat.session.progress_file import ProgressSnapshot
    from gymrat.session.store import ReadSessionResult
    from gymrat.supervisor.events import SessionEvent, SessionObserver
    from gymrat.supervisor.exit_sequence import ExitPhase

_logger = logging.getLogger(__name__)

REFRESH_MS = 1000
"""Default Live dashboard refresh interval in milliseconds."""

IDLE_WARN_MS = 30_000
"""After 30 seconds of no tool activity, the liveness line escalates to alert styling."""


# ---------------------------------------------------------------------------
# Reporter shell
# ---------------------------------------------------------------------------


class _DashboardShell(LiveDisplayMixin):
    """The terminal, clock, and I/O handles the dashboard owns, and the reducer state.

    Only the shell replaces the state; the reducer never sees the shell. Every
    event goes through :meth:`handle_event`. In live mode the display and its
    refresh timer run from construction until ``stop()``.
    """

    def __init__(  # noqa: PLR0913 - one parameter per handle the shell owns
        self,
        *,
        console: Console,
        state: ReporterState,
        now: Callable[[], int],
        read_session: Callable[[], ReadSessionResult],
        read_progress: Callable[[str], ProgressSnapshot | None],
        plain_write: Callable[[str], None],
        tz: tzinfo | None,
        idle_warn_ms: int,
    ) -> None:
        self._console = console
        self.state = state
        self._now = now
        self._read_session_fn = read_session
        self._read_progress = read_progress
        self._plain_write = plain_write
        self._tz = tz
        self._idle_warn_ms = idle_warn_ms
        self._session_read_warned = False
        self._last_frame: RenderableType = ""
        self._frame_warned = False

    def mount(self, mode: Literal["live", "plain"], refresh_ms: int) -> None:
        """Start the live display when *mode* and the console call for one.

        Args:
            mode: The render mode the caller asked for.
            refresh_ms: Live dashboard refresh interval in milliseconds.
        """
        if self._resolve_live(mode):
            # rich's stderr redirect lands a stray stderr write above the frame.
            self._mount_live(
                transient=True,
                get_renderable=self._live_frame,
                refresh_per_second=MS_PER_SECOND / refresh_ms,
                redirect_stderr=True,
            )

    def frame(self) -> RenderableType:
        """Render the current dashboard frame.

        Returns:
            The frame built from the current state.
        """
        # Runs on rich's refresh thread while events replace self.state on the
        # caller's thread. Read self.state exactly once: the state is immutable
        # and swapped by a single assignment, so one read is a consistent
        # snapshot and no lock is needed. A second read could mix two states in
        # one frame.
        return build_frame(
            self.state,
            self._now(),
            tz=self._tz,
            idle_warn_ms=self._idle_warn_ms,
            read_progress=self._read_progress,
        )

    def _live_frame(self) -> RenderableType:
        # rich builds a frame while the display is set up (in its constructor
        # and on the mount's first paint), on its refresh thread, on every
        # event-driven repaint, and once more inside Live.stop(). An error
        # escaping the build would end the refresh thread for good, freezing
        # elapsed time and the idle indicator, or escape stop. A frame that
        # fails to build is replaced by the last one that built, and only the
        # first failure is warned about, so a persistent fault does not print a
        # line every second.
        try:
            self._last_frame = self.frame()
        except Exception as exc:
            # Not logged at warning level: with no handler configured, logging
            # would print the traceback over the dashboard on every refresh.
            _logger.debug("dashboard frame failed to render", exc_info=True)
            if not self._frame_warned:
                self._frame_warned = True
                self.warn(f"warning: dashboard frame failed to render: {type(exc).__name__}: {exc}")
        return self._last_frame

    @override
    def warn(self, message: str) -> None:
        """Report a warning: above the live frame when one is up, else as a plain line.

        Args:
            message: The warning text, written as plain text.
        """
        if self._is_live:
            super().warn(message)
        else:
            self._plain_write(message)

    def _read_session(self) -> ReadSessionResult | None:
        try:
            return self._read_session_fn()
        except Exception as exc:
            # Not logged above debug level: with no handler configured, logging
            # would print the traceback over the dashboard on every failed read.
            _logger.debug("session read failed", exc_info=True)
            # Only the first failure is warned about, so a session that stays
            # unreadable does not print a line on every re-read.
            if not self._session_read_warned:
                self._session_read_warned = True
                self.warn(f"session read failed: {exc}")
            return None

    def handle_event(self, event: SessionEvent) -> None:
        """Fold one session event into the reporter state and render the result.

        When the reducer hands back the state it was given — a text delta or a
        tool progress ping changes nothing — live mode skips the repaint, since
        the only part of the frame that could have moved is elapsed time and the
        Live display's refresh timer already keeps that current.

        Args:
            event: The incoming session event.
        """
        session = self._read_session() if wants_session_refresh(self.state, event) else None
        before = self.state
        self.state = advance(before, event, session)

        if self._is_live:
            if self.state is not before:
                self._refresh_live()
            return
        line = plain_line(before, self.state, event)
        if line is not None:
            self._plain_write(line)

    def refresh_session(self) -> None:
        """Re-read the session, keeping the previous result when the read fails."""
        session = self._read_session()
        if session is None:
            return
        self.state = replace(self.state, session_result=session)
        self._refresh_live()

    def exit_phase(self, phase: ExitPhase) -> None:
        """Show the run-end exit sequence's phase, once per phase change.

        Args:
            phase: The exit sequence's current phase.
        """
        before = self.state
        self.state = exit_phase(before, phase, self._now())
        if self.state is before:
            return
        if self._is_live:
            self._refresh_live()
        else:
            self._plain_write(exit_phase_text(phase))


@dataclass(frozen=True, slots=True)
class SuperviseReporter:
    """The observer/stop/frame/warn surface that drives the supervise progress display.

    Live mode's display and its refresh timer run from construction until
    ``stop``, and for that span a termination signal erases the display: its
    cleanup is installed at construction and uninstalled by ``stop``.

    Attributes:
        observer: Receives every session event.
        stop: Stops the display and uninstalls its signal cleanup; a second call
            does nothing.
        frame: Renders the current dashboard frame.
        warn: Reports a warning as plain text without tearing the display; only
            the first failed session read, event-driven or refreshed, is
            reported here.
        session_result: The session state as of the last re-read, which is what
            the closing summary reports once the display has stopped.
        final_text: The text of the agent's last finished turn, or ``None`` before
            any has ended.
        exit_phase: Shows the run-end exit sequence's current phase: live mode
            repaints the frame, plain mode writes the phase line once per phase
            change.
        refresh_session: Re-reads the session so ``session_result`` reflects
            writes no event announced, such as an exit-sequence step that failed
            after writing to the session log. A successful re-read writes no
            plain line; a failed read keeps the previous result.
        display: The live-display bookkeeping behind ``live``, or ``None`` for
            a reporter that paints nothing.
    """

    observer: SessionObserver
    stop: Callable[[], None]
    frame: Callable[[], RenderableType]
    warn: Callable[[str], None]
    session_result: Callable[[], ReadSessionResult | None]
    final_text: Callable[[], str | None]
    exit_phase: Callable[[ExitPhase], None]
    refresh_session: Callable[[], None]
    display: LiveDisplayMixin | None = None

    @property
    def live(self) -> ErasableLive | None:
        """The active live display, or ``None`` outside live mode or after ``stop()``."""
        return None if self.display is None else self.display.live


# ---------------------------------------------------------------------------
# Reporter factory
# ---------------------------------------------------------------------------


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
    primary_direction: Direction = "lower",
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
        primary_direction: Whether a lower or a higher primary is the better
            outcome, passed to the default session reader to pick the best
            kept iteration.  Unused when ``read_session`` is given.

    Returns:
        A fully wired reporter whose callbacks drive the dashboard lifecycle.
    """
    shell = _DashboardShell(
        console=stderr_console(color_flag=color),
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
        read_session=(
            read_session
            if read_session is not None
            else functools.partial(read_live_session, root, primary_direction)
        ),
        read_progress=read_progress if read_progress is not None else _default_read_progress,
        plain_write=plain_write if plain_write is not None else warn_to_stderr,
        tz=tz,
        idle_warn_ms=idle_warn_ms,
    )
    # No reporter exists for the caller to stop until this returns, so the mount
    # itself stops the display when starting or writing the first paint fails.
    shell.mount(mode, refresh_ms)

    return SuperviseReporter(
        observer=shell.handle_event,
        stop=shell.stop,
        frame=shell.frame,
        warn=shell.warn,
        session_result=lambda: shell.state.session_result,
        final_text=lambda: shell.state.last_agent_text,
        exit_phase=shell.exit_phase,
        refresh_session=shell.refresh_session,
        display=shell,
    )
