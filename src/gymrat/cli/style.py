"""Shared glyphs, styles, and rich theme for the CLI's live progress displays.

Every live renderer signals the state of a step three ways at once — glyph, verb
form, and timer color — so a row can be read at a glance:

| State   | Glyph   | Verb form             | Timer     |
| ------- | ------- | --------------------- | --------- |
| running | spinner | gerund (``sampling``) | cyan      |
| done    | ``✓``   | past (``sampled``)    | dim green |
| pending | ``○``   | noun (``judge``)      | none      |
| error   | ``✗``   | —                    | none      |

A step that turns out not to apply (a skipped confirm, a hook that was not
configured) is dropped from the checklist rather than shown with a skip marker.

Counts (``1/4``) are bold in the default foreground; command and target labels
are bold blue; metadata, separators, and hints are dim. Yellow is reserved for
alert/warning surfaces (idle warnings, caps, alert glyphs).

The report-side style vocabulary lives in :mod:`gymrat.report.style`.

:data:`CLI_THEME` re-points the rich style names the progress columns hard-code
(``progress.spinner``, ``progress.elapsed``, ``progress.download``, ``bar.*``)
at those conventions, so ``SpinnerColumn``, ``TimeElapsedColumn``,
``MofNCompleteColumn``, and ``BarColumn`` need no per-call styling. It inherits
rich's defaults, so every style name it does not name keeps its stock value.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, override

from rich.live import Live
from rich.theme import Theme

if TYPE_CHECKING:
    from rich.console import Console

GLYPH_DONE = "✓"
GLYPH_PENDING = "○"
GLYPH_ERROR = "✗"

# The one spinner animation every renderer uses, for ``SpinnerColumn`` in
# progress bars and ``Spinner`` in checklist rows alike.
SPINNER_NAME = "dots"
GLYPH_ALERT = "!"

STYLE_RUNNING = "cyan"
STYLE_DONE = "green"
STYLE_PENDING = "dim"
STYLE_ALERT = "yellow"

STYLE_VERB = "bold"
STYLE_COUNT = "bold"
STYLE_LABEL = "bold blue"
STYLE_META = "dim"
STYLE_BAR = "dim"
STYLE_REGRESSED = "red"

STYLE_TIMER_RUNNING = "cyan"
STYLE_TIMER_DONE = "dim green"

# Spinner frames are 80ms apart; refreshing any slower makes them look frozen.
LIVE_REFRESH_PER_SECOND = 10

# The escape sequence a live renderer writes to blank the current terminal
# line before redrawing or clearing it.
CLEAR_LINE = "\r\x1b[K"

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

# Below this terminal height, a full checklist or header-plus-rows layout
# can't fit, so a renderer switches to a single-row compact bar.
COMPACT_HEIGHT_THRESHOLD = 12

CLI_THEME = Theme({
    "progress.spinner": STYLE_RUNNING,
    "progress.elapsed": STYLE_TIMER_RUNNING,
    # MofNCompleteColumn renders its count with "progress.download".
    "progress.download": STYLE_COUNT,
    # BarColumn draws its filled span with "bar.complete" until the task
    # completes and "bar.finished" after, and its indeterminate animation
    # with "bar.pulse"; "bar.back" keeps rich's default for the unfilled
    # span, which already reads as the dim track behind the bar.
    "bar.complete": STYLE_BAR,
    "bar.finished": STYLE_BAR,
    "bar.pulse": STYLE_BAR,
})


# pyrefly: ignore[invalid-inheritance] -- rich/live.py re-imports Live in its __main__ demo, which pyrefly reads as a rebinding
class ErasableLive(Live):
    """A rich ``Live`` that a termination handler can erase row for row.

    rich records a frame's height while rendering it into the console buffer,
    before the bytes reach the terminal, so mid-paint its bookkeeping runs one
    frame ahead of the screen. This subclass remembers the height the terminal
    still shows for as long as a paint is in flight, so
    :meth:`erase_for_exit` erases exactly the rows on screen.
    """

    # Height of the frame on screen while this display's own refresh() is
    # mid-paint; None when no paint is in flight.
    _height_before_paint: int | None = None

    @override
    # pyrefly: ignore[bad-override] -- same __main__ rebinding hides Live.refresh from pyrefly
    def refresh(self) -> None:
        """Paint the current frame, keeping the on-screen height readable until the write lands."""
        with self._lock:
            self._height_before_paint = self._live_render.last_render_height
            try:
                super().refresh()
            finally:
                self._height_before_paint = None

    def erase_for_exit(self) -> str:
        """Stop painting for good and return the escapes that erase the frame on screen.

        Built for a termination handler that exits right after writing the
        result. The auto-refresh thread is told to stop, and a paint it has in
        flight finishes first so that no frame lands after the erase. When the
        handler interrupted the main thread's own paint, the display lock
        re-enters and the frame that paint was replacing is the one erased.
        A paint that is still stalled after a bounded wait has not reached the
        terminal either, so the frame it is replacing is erased as well.

        ``Live.stop()`` is never called: it paints a final frame and waits on
        the display lock without a bound.

        Returns:
            The escapes that erase every row of the frame on screen and show
            the cursor again; only the show-cursor escape when nothing was
            painted.
        """
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
        if self._height_before_paint is not None:
            return self._height_before_paint
        return self._live_render.last_render_height


def _erase_rows(height: int) -> str:
    if height == 0:
        return ""
    return f"\r{_ERASE_LINE}" + f"{_CURSOR_UP}{_ERASE_LINE}" * (height - 1)


class LiveDisplayMixin:
    """Shared live-display bookkeeping for ``ProgressReporter`` and ``IterateRenderer``.

    Both renderers own the ``Console`` they render to, an ``ErasableLive``
    display (``None`` in plain mode) and a ``_stopped`` guard against a
    termination signal racing the normal ``stop()`` path. Mixing this in keeps
    the refresh, warning, and signal-clear behavior identical without giving the
    two renderers a shared base class.
    """

    _console: Console
    _live: ErasableLive | None = None
    _stopped: bool = False

    @property
    def live(self) -> ErasableLive | None:
        """The active live display, or ``None`` outside live mode or after ``stop()``."""
        return self._live

    def _refresh_live(self) -> None:
        if self._live is not None:
            self._live.refresh()

    def warn(self, message: str) -> None:
        """Print ``message`` verbatim on its own line, above the live frame when one is up.

        Args:
            message: The warning text, printed without markup or highlighting.
        """
        self._console.print(message, highlight=False, markup=False)

    def clear_on_signal(self) -> None:
        """Erase the display once, guarding against repeated signal delivery.

        In plain mode only the current line is cleared. With a live display up,
        every row of the frame on screen is erased and the cursor is shown
        again, because ``os._exit`` follows and ``Live.stop()`` never runs.
        """
        # os._exit skips buffer flushing, so the clear must be flushed explicitly
        # or it never reaches the terminal.
        if self._stopped:
            return
        self._stopped = True
        sys.stderr.write(CLEAR_LINE if self._live is None else self._live.erase_for_exit())
        sys.stderr.flush()
