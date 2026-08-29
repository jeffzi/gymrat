"""Shared glyphs, styles, and rich theme for the CLI's live progress displays.

Every live renderer signals the state of a step three ways at once — glyph, verb
form, and timer color — so a row can be read at a glance:

| State   | Glyph   | Verb form             | Timer     |
| ------- | ------- | --------------------- | --------- |
| running | spinner | gerund (``sampling``) | yellow    |
| done    | ``✓``   | past (``sampled``)    | dim green |
| pending | ``○``   | noun (``judge``)      | none      |
| error   | ``✗``   | —                     | none      |

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

from rich.theme import Theme

GLYPH_DONE = "✓"
GLYPH_PENDING = "○"
GLYPH_ERROR = "✗"

# The one spinner animation every renderer uses, for ``SpinnerColumn`` in
# progress bars and ``Spinner`` in checklist rows alike.
SPINNER_NAME = "dots"
GLYPH_ALERT = "!"

STYLE_RUNNING = "yellow"
STYLE_DONE = "green"
STYLE_PENDING = "dim"
STYLE_ERROR = "red"
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
