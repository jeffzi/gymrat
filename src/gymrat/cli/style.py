"""Shared glyphs, styles, and rich theme for the CLI's live progress displays.

Every live renderer signals the state of a step three ways at once — glyph, verb
form, and timer color — so a row can be read at a glance:

| State   | Glyph   | Verb form             | Timer     |
| ------- | ------- | --------------------- | --------- |
| running | spinner | gerund (``sampling``) | yellow    |
| done    | ``✓``   | past (``sampled``)    | dim green |
| pending | ``○``   | noun (``judge``)      | none      |
| error   | ``✗``   | —                    | none      |

A step that turns out not to apply (a skipped confirm, a hook that was not
configured) is dropped from the checklist rather than shown with a skip marker.

Counts (``1/4``) are bold in the default foreground; command and target labels
are bold blue; metadata, separators, and hints are dim.

:data:`CLI_THEME` re-points the rich style names the progress columns hard-code
(``progress.spinner``, ``progress.elapsed``, ``progress.download``, ``bar.*``)
at those conventions, so ``SpinnerColumn``, ``TimeElapsedColumn``,
``MofNCompleteColumn``, and ``BarColumn`` need no per-call styling. It inherits
rich's defaults, so every style name it does not name keeps its stock value.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from rich.theme import Theme

if TYPE_CHECKING:
    from rich.live import Live

GLYPH_DONE = "✓"
GLYPH_PENDING = "○"

# The one spinner animation every renderer uses, for ``SpinnerColumn`` in
# progress bars and ``Spinner`` in checklist rows alike.
SPINNER_NAME = "dots"
GLYPH_ALERT = "!"

STYLE_RUNNING = "yellow"
STYLE_DONE = "green"
STYLE_PENDING = "dim"
STYLE_ALERT = "yellow"

STYLE_VERB = "bold"
STYLE_COUNT = "bold"
STYLE_LABEL = "bold blue"
STYLE_META = "dim"
STYLE_BAR = "dim"

STYLE_TIMER_RUNNING = "yellow"
STYLE_TIMER_DONE = "dim green"

# Spinner frames are 80ms apart; refreshing any slower makes them look frozen.
LIVE_REFRESH_PER_SECOND = 10

# The escape sequence a live renderer writes to blank the current terminal
# line before redrawing or clearing it.
CLEAR_LINE = "\r\x1b[K"

# Below this terminal height, a full checklist or header-plus-rows layout
# can't fit, so a renderer switches to a single-row compact bar.
COMPACT_HEIGHT_THRESHOLD = 12

CLI_THEME = Theme(
    {
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
    }
)


class LiveDisplayMixin:
    """Shared live-display bookkeeping for ``ProgressReporter`` and ``IterateRenderer``.

    Both renderers own a rich ``Live`` instance (``None`` in plain mode) and a
    ``_stopped`` guard against a termination signal racing the normal ``stop()``
    path. Mixing this in keeps the refresh and signal-clear behavior identical
    without giving the two renderers a shared base class.
    """

    _live: Live | None = None
    _stopped: bool = False

    @property
    def live(self) -> Live | None:
        """The active ``Live`` display, or ``None`` outside live mode or after ``stop()``."""
        return self._live

    def _refresh_live(self) -> None:
        if self._live is not None:
            self._live.refresh()

    def clear_on_signal(self) -> None:
        """Clear the current line once, guarding against repeated signal delivery."""
        # os._exit skips buffer flushing, so the clear must be flushed explicitly
        # or it never reaches the terminal.
        if self._stopped:
            return
        self._stopped = True
        sys.stderr.write(CLEAR_LINE)
        sys.stderr.flush()
