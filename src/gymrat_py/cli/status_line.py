"""Content-agnostic status-line primitive for streaming progress to stderr.

Three output strategies share one interface: ``plain`` writes newline-terminated
lines, ``overwrite`` rewrites a single row in place, and ``spinner`` draws a
minimal animated glyph. The status line is content-agnostic — the progress layer
owns what text to show; this layer owns how to put it on the terminal.
"""

import os
import shutil
import sys
import threading
from collections.abc import Callable
from typing import Literal, Protocol

from gymrat_py.report.style import shorten_label
from gymrat_py.signals import install_termination_cleanup

# Carriage-return + clear-to-end-of-line: rewind to column zero and wipe the row.
CLEAR_LINE = "\r\x1b[K"

# How often the ``on_tick`` callback fires, in milliseconds, in the TTY modes.
TICK_INTERVAL_MS = 1000

# The braille frames the minimal spinner cycles through.
_SPINNER_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

type RenderMode = Literal["spinner", "overwrite", "plain"]


class StatusLine(Protocol):
    """Single-use status line: ``stop`` must be called exactly once."""

    def write(self, text: str) -> None:
        """Show ``text`` as the current status."""

    def warn(self, message: str) -> None:
        """Surface ``message`` without losing the status being shown."""

    def stop(self) -> None:
        """Tear the status line down; call exactly once."""


def _stderr_is_tty() -> bool:
    """Whether the current ``sys.stderr`` reports itself as a terminal."""
    isatty = getattr(sys.stderr, "isatty", None)
    return bool(isatty()) if callable(isatty) else False


def _terminal_columns() -> int | None:
    """The terminal width in columns, or ``None`` off a TTY (no width to fit).

    ``shutil.get_terminal_size`` silently substitutes its 80-column fallback for
    a width it cannot trust — ``COLUMNS`` set to ``0``, empty, or a non-positive
    or non-numeric value — which would let a genuinely zero-width terminal fit a
    full-length row and spill. Honor an explicit ``COLUMNS`` directly instead,
    collapsing a non-positive or unparseable value to zero so the fitted text
    comes out empty rather than 80 columns wide.
    """
    if not _stderr_is_tty():
        return None
    columns_env = os.environ.get("COLUMNS")
    if columns_env is not None:
        try:
            return max(int(columns_env), 0)
        except ValueError:
            return 0
    return shutil.get_terminal_size().columns


def _fit_to_terminal_width(line: str) -> str:
    """Fit ``line`` to one column short of the terminal so the cursor cannot wrap.

    Off a TTY there are no columns to fit, so the text falls through unchanged.
    """
    columns = _terminal_columns()
    if columns is None:
        return line
    return shorten_label(line, columns - 1)


class _PlainStatusLine:
    """Plain-mode status line: every write is its own newline-terminated line."""

    def write(self, text: str) -> None:
        sys.stderr.write(f"{text}\n")

    def warn(self, message: str) -> None:
        sys.stderr.write(f"{message}\n")

    def stop(self) -> None:
        """No line is held open in plain mode, so there is nothing to clear."""


class _TtyStatusLine:
    """In-place status line for the ``overwrite`` and ``spinner`` TTY modes."""

    def __init__(self, *, spinner: bool, on_tick: Callable[[], str] | None) -> None:
        self._spinner = spinner
        self._on_tick = on_tick
        self._last_text: str | None = None
        self._frame = 0
        self._timer: threading.Timer | None = None
        self._stopped = False
        # A termination signal exits via os._exit without unwinding the run's
        # finally, so the row this line holds open would strand its last progress
        # text on the terminal. Clearing it here — the one place that knows a row
        # is open — is what wipes it before the process dies.
        self._uninstall_cleanup = install_termination_cleanup(self._clear_on_signal)
        if on_tick is not None:
            self._schedule_tick()

    def _clear_on_signal(self) -> None:
        # os._exit skips buffer flushing, so the clear must be flushed explicitly
        # or it never reaches the terminal. Marking the line stopped first keeps a
        # racing tick from redrawing progress over the cleared row.
        if self._stopped:
            return
        self._stopped = True
        sys.stderr.write(CLEAR_LINE)
        sys.stderr.flush()

    def _schedule_tick(self) -> None:
        timer = threading.Timer(TICK_INTERVAL_MS / 1000, self._fire_tick)
        timer.daemon = True
        self._timer = timer
        timer.start()

    def _fire_tick(self) -> None:
        # A tick that lands after stop must neither redraw nor reschedule, or the
        # timer chain would outlive the run it was reporting on.
        if self._stopped or self._on_tick is None:
            return
        self._write_text(self._on_tick())
        if not self._stopped:
            self._schedule_tick()

    def _write_text(self, text: str) -> None:
        self._last_text = text
        if self._spinner:
            self._render_spinner(text)
        else:
            sys.stderr.write(f"{CLEAR_LINE}{_fit_to_terminal_width(text)}")

    def _render_spinner(self, text: str) -> None:
        # Spinner mode is only chosen when color is allowed (resolve_render_mode
        # gates it on NO_COLOR and TTY), so a raw yellow SGR glyph is safe here.
        glyph = _SPINNER_FRAMES[self._frame % len(_SPINNER_FRAMES)]
        self._frame += 1
        sys.stderr.write(f"{CLEAR_LINE}\x1b[33m{glyph}\x1b[39m {text}")

    def write(self, text: str) -> None:
        self._write_text(text)

    def warn(self, message: str) -> None:
        previous = self._last_text
        if previous is None:
            sys.stderr.write(f"{message}\n")
            return

        sys.stderr.write(CLEAR_LINE)
        sys.stderr.write(f"{message}\n")
        # The spinner redraws itself on its next frame; overwrite mode must
        # restore the row it just cleared.
        if not self._spinner:
            sys.stderr.write(f"{CLEAR_LINE}{_fit_to_terminal_width(previous)}")

    def stop(self) -> None:
        self._stopped = True
        self._uninstall_cleanup()
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
        sys.stderr.write(CLEAR_LINE)


def create_status_line(mode: RenderMode, on_tick: Callable[[], str] | None = None) -> StatusLine:
    """Build a status line for ``mode`` that writes to stderr.

    ``on_tick``, when given, is called every ``TICK_INTERVAL_MS`` milliseconds in
    the ``spinner`` and ``overwrite`` modes, and its return value is written as
    the next line — this is how a live countdown refreshes between emits.
    """
    if mode == "plain":
        return _PlainStatusLine()
    return _TtyStatusLine(spinner=mode == "spinner", on_tick=on_tick)
