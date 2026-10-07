"""The lifecycle of the CLI's live progress displays: mount, erase on a signal, stop.

Every live renderer mounts a rich ``Live`` that a termination handler can erase
without removing a line above its frame, and shares the bookkeeping that makes
``stop()`` run once. The glyphs, styles and theme the frames are painted with
live in :mod:`gymrat.cli.style`.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, override

from rich.live import Live
from rich.segment import Segment

from gymrat.signals import install_termination_cleanup, write_on_exit
from gymrat.utils import format_timestamp

if TYPE_CHECKING:
    from collections.abc import Callable

    from rich.console import (
        Console,
        ConsoleOptions,
        ConsoleRenderable,
        RenderableType,
        RenderResult,
    )

# Spinner frames are 80ms apart; refreshing any slower makes them look frozen.
LIVE_REFRESH_PER_SECOND = 10

# The escape sequence that makes the terminal cursor visible again after
# ``Live.start()`` hid it.
SHOW_CURSOR = "\x1b[?25h"

# Escape sequences that erase a whole terminal line and move the cursor up one.
_ERASE_LINE = "\x1b[2K"
_CURSOR_UP = "\x1b[1A"

# How long a termination handler waits for an in-flight paint to finish. A
# paint takes milliseconds; the bound only matters when the paint is itself
# waiting on the thread the handler interrupted.
_PAINT_WAIT_SECONDS = 1.0


@dataclass(slots=True)
class _Paint:
    """One frame paint, from the display adding its frame to a print until the print's write.

    ``buffer`` is the painting thread's console buffer, holding a marker from
    the moment the frame has rendered; rich empties it once the write has
    returned, so a paint whose buffer is empty again has reached the terminal.
    A paint that dropped its frame or failed to render puts no frame on screen
    and is ``settled`` from the start.
    """

    height_before: int
    buffer: list[Segment] | None = None
    settled: bool = False

    def in_flight(self) -> bool:
        return not self.settled and (self.buffer is None or bool(self.buffer))


# pyrefly: ignore[invalid-inheritance] -- rich/live.py re-imports Live in its __main__ demo, which pyrefly reads as a rebinding
class ErasableLive(Live):
    """A rich ``Live`` that a termination handler can erase without removing a line above it.

    rich records a frame's height while rendering it into the console buffer,
    before the bytes reach the terminal, and a print writes its frame after
    releasing the display lock. While a paint is in flight the terminal shows
    either the frame it replaces or the new one, so :meth:`erase_for_exit`
    erases the shorter of the two: rows of the taller one may remain, but the
    line above the frame never goes.

    Once erased, the display paints nothing more and adds its frame to nothing
    printed through its console, so no frame lands after the erase however the
    termination handler and the painting threads interleave.
    """

    # Paints whose frame may not have reached the terminal yet, oldest first.
    _paints: tuple[_Paint, ...] = ()
    # Height of the frame left on screen by a paint the erase made drop its
    # frame; None while no paint has been dropped.
    _height_kept_by_drop: int | None = None
    _erased: bool = False

    @property
    def erased(self) -> bool:
        """Whether :meth:`erase_for_exit` has run."""
        return self._erased

    @override
    # pyrefly: ignore[bad-override] -- same __main__ rebinding hides Live.refresh from pyrefly
    def refresh(self) -> None:
        """Paint the current frame; paint nothing once :meth:`erase_for_exit` has run."""
        with self._lock:
            if self._erased:
                return
            super().refresh()

    @override
    # pyrefly: ignore[bad-override] -- same __main__ rebinding hides Live.process_renderables from pyrefly
    def process_renderables(self, renderables: list[ConsoleRenderable]) -> list[ConsoleRenderable]:
        """Add the cursor reset and the live frame to a print, until the display is erased.

        Once erased, a print goes out as it is, without waiting on the display
        lock that a stalled paint may hold. A paint already under way when the
        erase ran drops its cursor reset and frame once rendered, keeping only
        the printed renderables.

        Args:
            renderables: What the print renders, before the live display adds to it.

        Returns:
            The renderables to print.
        """
        if self._erased:
            return renderables
        with self._lock:
            paint = _Paint(self._live_render.last_render_height)
            self._paints = (*(held for held in self._paints if held.in_flight()), paint)
            painted = super().process_renderables(renderables)
        lands = functools.partial(self._frame_lands, paint)
        return [_PaintUnlessErased(painted, renderables, paint, lands)]

    def erase_for_exit(self) -> str:
        """Stop painting for good and return the escapes that erase the frame on screen.

        Built for a termination handler that exits right after writing the
        result. The auto-refresh thread is told to stop, and a paint it has in
        flight gets a bounded wait to finish, so that no frame lands after the
        erase. When a paint is still in flight, on the refresh thread after the
        wait or on the main thread the handler interrupted, the shorter of its
        old and new frames is erased.

        When the display redirected ``sys.stdout`` or ``sys.stderr``, the real
        streams are put back, so anything written after the erase reaches the
        terminal directly rather than printing through the display.

        ``Live.stop()`` is never called: it paints a final frame and waits on
        the display lock without a bound.

        Returns:
            The escapes that erase the frame on screen and show the cursor
            again; only the show-cursor escape when nothing was painted; an
            empty string when the frame was already erased.
        """
        if self._erased:
            return ""
        # Set before waiting on the lock, so a paint queued behind a stalled
        # one skips instead of landing a frame after the erase.
        self._erased = True
        self._disable_redirect_io()
        if self._refresh_thread is not None:
            self._refresh_thread.stop()
        if self._lock.acquire(timeout=_PAINT_WAIT_SECONDS):
            try:
                height = self._height_on_screen()
            finally:
                self._lock.release()
        else:
            height = self._height_on_screen()
        return _erase_rows(height) + SHOW_CURSOR

    def _height_on_screen(self) -> int:
        base = (
            self._height_kept_by_drop
            if self._height_kept_by_drop is not None
            else self._live_render.last_render_height
        )
        in_flight = [paint.height_before for paint in self._paints if paint.in_flight()]
        return min([base, *in_flight])

    def _frame_lands(self, paint: _Paint, buffer: list[Segment]) -> bool:
        # rich recorded the rendered frame's height already; a dropped frame
        # leaves the one before it on screen, which the erase must match.
        if self._erased:
            self._height_kept_by_drop = paint.height_before
            paint.settled = True
            return False
        # The marker keeps the buffer non-empty from here until rich's write
        # has returned, including before rich adds the print's own segments.
        buffer.append(Segment(""))
        paint.buffer = buffer
        return True


@dataclass(frozen=True, slots=True)
class _PaintUnlessErased:
    """A print with the live display's additions, reduced to the print alone if erased meanwhile.

    The frame renders before the erase check, so a paint whose render was still
    under way when the display was erased lands no frame after the erase.
    """

    painted: list[ConsoleRenderable]
    printed: list[ConsoleRenderable]
    paint: _Paint
    frame_lands: Callable[[list[Segment]], bool]

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        try:
            segments = [
                segment
                for renderable in self.painted
                for segment in console.render(renderable, options)
            ]
        except BaseException:
            # The print fails before writing anything, so its frame never lands.
            self.paint.settled = True
            raise
        if not self.frame_lands(console._buffer):  # noqa: SLF001 -- rich exposes no other view of the pending write
            yield from self.printed
            return
        yield from segments


def _erase_rows(height: int) -> str:
    if height == 0:
        return ""
    return f"\r{_ERASE_LINE}" + f"{_CURSOR_UP}{_ERASE_LINE}" * (height - 1)


def mount_live(live: ErasableLive) -> Callable[[], None]:
    """Start a live display and paint its first frame, erasing it on a termination signal.

    The erase is installed before the display starts, so a signal at any moment
    from the cursor being hidden onward erases what is on screen and shows the
    cursor again. When starting or the first paint fails, the display is
    stopped and the erase uninstalled before the error propagates.

    Args:
        live: The display to start, not yet started.

    Returns:
        Uninstalls the erase. Call it before stopping the display, so a signal
        from then on leaves the terminal to ``Live.stop()``.
    """
    # The erase escapes go through write_on_exit, which defers them until every
    # cleanup has run.
    uninstall = install_termination_cleanup(lambda: write_on_exit(live.erase_for_exit()))
    try:
        live.start()
        live.refresh()
    except BaseException as failure:
        uninstall()
        try:
            live.stop()
        except BaseException as stop_failure:  # noqa: BLE001 -- the mount failure propagates, carrying this one
            # A start that failed partway leaves rich unable to stop cleanly,
            # and an interrupt may land mid-stop; either way the mount failure
            # is the one the caller must see.
            failure.add_note(f"stopping the display also failed: {stop_failure!r}")
        raise
    return uninstall


class LiveDisplayMixin:
    """Shared live-display bookkeeping for ``ProgressReporter`` and ``IterateRenderer``.

    Both renderers own the ``Console`` they render to, an ``ErasableLive``
    display (``None`` in plain mode) and a ``_stopped`` guard that makes
    ``stop()`` run once. Mixing this in keeps the live-mode check, the plain
    milestone line, and the refresh, warning, and stop bookkeeping identical
    without giving the two renderers a shared base class.
    """

    _console: Console
    _is_live: bool = False
    _live: ErasableLive | None = None
    _uninstall_erase: Callable[[], None] | None = None
    _stopped: bool = False

    def _resolve_live(self, mode: Literal["live", "plain"]) -> bool:
        """Record whether the renderer paints a live display.

        Args:
            mode: The render mode the caller asked for.

        Returns:
            ``True`` for live mode on a console wide enough to hold a frame; a
            zero-width console renders plain whatever the mode.
        """
        self._is_live = mode == "live" and self._console.width > 0
        return self._is_live

    def _print_milestone(self, line: str, at_ms: float, run_start_ms: float | None) -> None:
        """Print a plain-mode milestone line behind its run-relative timestamp."""
        timestamp = format_timestamp(at_ms, run_start_ms)
        self._console.print(f"{timestamp} {line}", highlight=False, markup=False, emoji=False)

    def _mount_live(self, *, transient: bool, get_renderable: Callable[[], RenderableType]) -> None:
        live = ErasableLive(
            console=self._console,
            auto_refresh=True,
            refresh_per_second=LIVE_REFRESH_PER_SECOND,
            transient=transient,
            redirect_stderr=False,
            get_renderable=get_renderable,
        )
        self._uninstall_erase = mount_live(live)
        self._live = live

    @property
    def live(self) -> ErasableLive | None:
        """The active live display, or ``None`` outside live mode or after ``stop()``."""
        return self._live

    def _refresh_live(self) -> None:
        if self._live is not None:
            self._live.refresh()

    def _after_live_stopped(self) -> None:
        """Runs once the live display has stopped, for a renderer's closing line."""

    def stop(self) -> None:
        """Stop the renderer and clean up any live display; a second call does nothing."""
        # Nothing to do when stop() already ran, or when a termination signal
        # erased the display: Live.stop() would then restore the cursor over rows
        # the erase already cleared, taking lines above the frame with them.
        if self._stopped or (self._live is not None and self._live.erased):
            return
        self._stopped = True
        if self._uninstall_erase is not None:
            self._uninstall_erase()
        if self._live is not None:
            self._live.stop()
            self._after_live_stopped()
            self._live = None

    def warn(self, message: str) -> None:
        """Print ``message`` verbatim on its own line, above the live frame when one is up.

        Args:
            message: The warning text, printed without markup, highlighting, or
                emoji codes.
        """
        self._console.print(message, highlight=False, markup=False, emoji=False)
