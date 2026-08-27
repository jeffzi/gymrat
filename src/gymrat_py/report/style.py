"""Style and color primitives for the report package.

This module owns three concerns the text renderers share:

- **Label fitting** — middle-ellipsis truncation (:func:`shorten_label`) and
  set-aware widening (:func:`truncate_labels`) so long branch names fit a column
  without losing the ends that tell sibling branches apart.
- **Style vocabulary** — the rich style strings a verdict, a variant name, a
  group label or an aggregate label wears, plus the markup helpers
  (:func:`format_hint_label`, :func:`highlight_inline_code`) that carry inline
  styling in user-facing prose.
- **Color resolution** — :func:`make_capture_console` and :func:`render_lines`
  turn an explicit ``color`` choice into a rich :class:`~rich.console.Console`
  writing to an in-memory buffer, resolving the choice against the environment
  without ever mutating :data:`os.environ`.

Display widths are measured in terminal cells (:func:`rich.cells.cell_len`), so
a wide CJK character counts as two columns. Slicing is on code points: Python
has no standard-library grapheme segmentation, so a cut may land between the
code points of a combined emoji. Such clusters are out of scope here.
"""

from __future__ import annotations

import io
import os
import re
from typing import TYPE_CHECKING, cast

from rich.cells import cell_len
from rich.console import Console
from rich.markup import escape

if TYPE_CHECKING:
    from collections.abc import Sequence

    from rich.console import RenderableType

    from gymrat_py.report.format import DisplayClass

# ---------------------------------------------------------------------------
# Label truncation
# ---------------------------------------------------------------------------

# U+2026, one cell wide — three periods would cost two more columns.
_ELLIPSIS = "…"

#: Widest a variant label prints, ellipsis included. A branch name is free to be
#: as long as git allows, but every column it heads is sized from it, so an
#: unbounded one would push the figures off the right edge of the terminal.
LABEL_DISPLAY_WIDTH = 20


def _clip_to_cells(text: str, budget: int, *, from_end: bool) -> str:
    """Return the run of code points from one end of ``text`` fitting ``budget`` cells.

    Args:
        text: The source string.
        budget: The number of terminal cells the returned slice may occupy.
        from_end: Take code points from the end of ``text`` instead of the start.

    Returns:
        The longest prefix (or suffix) of ``text`` whose cell width does not
        exceed ``budget``. A wide character is kept only when it fits whole, so
        the result never overshoots the budget.
    """
    if budget <= 0:
        return ""

    chars = list(text)
    if from_end:
        chars.reverse()

    kept: list[str] = []
    used = 0
    for char in chars:
        width = cell_len(char)
        if used + width > budget:
            break
        kept.append(char)
        used += width

    if from_end:
        kept.reverse()
    return "".join(kept)


def shorten_label(text: str, max_width: int) -> str:
    """Fit ``text`` into ``max_width`` terminal cells, keeping both of its ends.

    The ends carry the identity: branch names that share a prefix differ in
    their tail, and a progress line names its step at the front and its target
    at the back. Cutting from the middle keeps both, where a plain slice would
    drop whichever end runs past the budget.

    Text already inside the budget comes back untouched, so widening the budget
    can never lengthen the result. When a cut is needed the budget spends one
    cell on the ellipsis; the remainder splits between head and tail, with the
    extra cell of an odd remainder going to the head.

    Width is measured in terminal cells, so a wide CJK character counts as two.
    The kept head and tail are sliced on code points and each is filled only
    with characters that fit whole, so a kept slice never overshoots its share.

    Args:
        text: The label to fit.
        max_width: The budget in terminal cells.

    Returns:
        ``text`` verbatim when it already fits, an empty string when
        ``max_width`` is zero or negative, ``"…"`` alone when only one cell is
        available, or ``head + "…" + tail`` otherwise.
    """
    if max_width <= 0:
        return ""

    if cell_len(text) <= max_width:
        return text

    kept = max_width - cell_len(_ELLIPSIS)
    if kept <= 0:
        return _ELLIPSIS

    head_budget = (kept + 1) // 2
    tail_budget = kept - head_budget
    head = _clip_to_cells(text, head_budget, from_end=False)
    tail = _clip_to_cells(text, tail_budget, from_end=True)
    return f"{head}{_ELLIPSIS}{tail}"


def truncate_labels(labels: Sequence[str]) -> list[str]:
    """Shorten every label to the narrowest width that keeps the set as distinct.

    Labels are shortened as a set rather than one at a time because the ends are
    what tell sibling branches apart: ``feature/experiment-one-fastpath`` and
    ``feature/exploration-two-fastpath`` share both of theirs, so the narrowest
    width that keeps them distinct is the one worth spending. The width grows one
    cell at a time from :data:`LABEL_DISPLAY_WIDTH` up to the longest label until
    the shortened set has as many unique strings as the originals — two labels
    that were already identical stay that way, which is the run's own doing, not
    the display's.

    Widening never lengthens a label that already fits the wider budget:
    :func:`shorten_label` returns such a label verbatim.

    Args:
        labels: The labels to fit, in order.

    Returns:
        A new list of shortened labels aligned with ``labels``. Every label is
        returned verbatim when they all fit :data:`LABEL_DISPLAY_WIDTH`.
    """
    longest = max((cell_len(label) for label in labels), default=0)
    distinct = len(set(labels))
    for max_width in range(LABEL_DISPLAY_WIDTH, longest):
        shortened = [shorten_label(label, max_width) for label in labels]
        if len(set(shortened)) == distinct:
            return shortened
    return list(labels)


# ---------------------------------------------------------------------------
# Style vocabulary
# ---------------------------------------------------------------------------

#: The color each display class wears wherever the report states a verdict.
#:
#: Every style here is worn by the verdict itself — a glyph, a delta, a tally —
#: never by the row or the values around it: within-noise and inconclusive recede
#: to dim, identical reads cyan for "measured the same", and unstable keeps its
#: amber warning.
VERDICT_STYLES: dict[DisplayClass, str] = {
    "improved": "green",
    "regressed": "red",
    "unstable": "yellow",
    "identical": "cyan",
    "within-noise": "dim",
    "inconclusive": "dim",
}

#: The style a variant name wears where the report names it as a name.
VARIANT_NAME_STYLE = "bold underline"

#: The style a group label wears where the report heads a group of metrics.
GROUP_LABEL_STYLE = "blue"

#: The style an aggregate label (a geomean row, a total) wears.
AGGREGATE_LABEL_STYLE = "bold"

# The word `Hint` carries the underline; the colon is colored with it but never
# underlined, since an underscore running under a colon reads as punctuation of
# its own.
_HINT_WORD_STYLE = "yellow underline"
_HINT_COLON_STYLE = "yellow"

# A backtick-wrapped inline code span; the capture group is its content.
_INLINE_CODE_PATTERN = re.compile(r"`([^`]+)`")


def markup(text: str, style: str) -> str:
    """Wrap ``text`` in a rich-markup span carrying ``style``, escaping the text.

    The single markup primitive the report package styles a run of plain text
    with; :func:`render_lines` resolves the span to ANSI (or strips it) once.
    """
    return f"[{style}]{escape(text)}[/]"


def format_hint_label() -> str:
    """Return the ``Hint:`` label as rich markup, styled as two spans.

    The word and the colon are separate spans so the underline stops at the
    word: an underscore running under a colon reads as punctuation of its own.
    The returned markup is rendered to ANSI (or left plain) by
    :func:`render_lines`.

    Returns:
        Rich markup for the ``Hint:`` label.
    """
    word = f"[{_HINT_WORD_STYLE}]Hint[/{_HINT_WORD_STYLE}]"
    colon = f"[{_HINT_COLON_STYLE}]:[/{_HINT_COLON_STYLE}]"
    return f"{word}{colon}"


def highlight_inline_code(text: str) -> str:
    """Replace every ``` `...` ``` span in ``text`` with its content styled yellow.

    Backticks are stripped whether or not color is active, so callers can embed
    backtick-marked command names in user-facing strings and let
    :func:`render_lines` decide how to present them. Each span is styled
    independently, and text with no backtick pair comes back unchanged.

    The code content is passed through :func:`rich.markup.escape` so that
    markup-significant characters in a command or path render literally rather
    than being parsed as further markup.

    Args:
        text: Prose that may contain backtick-wrapped inline code spans.

    Returns:
        Rich markup with each ``` `...` ``` span replaced by its escaped content
        wrapped in a yellow span.
    """

    def style_span(match: re.Match[str]) -> str:
        return f"[yellow]{escape(match.group(1))}[/yellow]"

    return _INLINE_CODE_PATTERN.sub(style_span, text)


# ---------------------------------------------------------------------------
# Color resolution and capture rendering
# ---------------------------------------------------------------------------

#: A render width wide enough that a captured line never soft-wraps, whatever the
#: real terminal is. Every report block and table renders against it.
RENDER_WIDTH = 200


def make_capture_console(*, color: bool | None, width: int) -> Console:
    """Build a rich console that captures its output to an in-memory buffer.

    Color is resolved without touching :data:`os.environ`:

    - ``color=True`` forces ANSI even when ``NO_COLOR`` is set, by declaring the
      capture a terminal with color enabled.
    - ``color=False`` suppresses ANSI even when ``FORCE_COLOR`` is set.
    - ``color=None`` reads the environment then TTY-ness, with this
      precedence: ``FORCE_COLOR`` (any value but ``0``/``false``/empty) forces
      color on even when ``NO_COLOR`` is also set, ``NO_COLOR`` alone forces it
      off, and with neither a captured buffer is not a TTY, so the output is
      plain. ``FORCE_COLOR`` winning over ``NO_COLOR`` is the one place rich's own
      detection differs, so that case is resolved here rather than deferred.

    Wide content is never wrapped or cropped (``soft_wrap``), so the width only
    bounds justification, never the text.

    Args:
        color: The explicit color choice, or ``None`` to defer to the
            environment and TTY detection.
        width: The terminal width the console renders against.

    Returns:
        A console whose ``file`` is an :class:`io.StringIO` holding everything
        printed through it.
    """
    buffer = io.StringIO()
    if color is True:
        return Console(
            file=buffer,
            width=width,
            force_terminal=True,
            no_color=False,
            soft_wrap=True,
        )
    if color is False:
        # `no_color=True` only strips colors, leaving bold/underline/dim intact
        # when FORCE_COLOR has forced a terminal; disabling the color system drops
        # every SGR code, so color=False fully overrides the environment.
        return Console(file=buffer, width=width, color_system=None, soft_wrap=True)
    env_color = color_from_env()
    if env_color is True:
        return Console(
            file=buffer,
            width=width,
            force_terminal=True,
            no_color=False,
            soft_wrap=True,
        )
    if env_color is False:
        return Console(file=buffer, width=width, color_system=None, soft_wrap=True)
    return Console(file=buffer, width=width, soft_wrap=True)


def _force_color_env() -> bool:
    """Whether ``FORCE_COLOR`` in the environment asks for color, without mutating it.

    ``FORCE_COLOR`` wins over ``NO_COLOR`` when both are set:
    any value other than ``0``, ``false`` or the empty string enables color. Only
    the ``color=None`` branch consults this; an explicit choice never does.
    """
    value = os.environ.get("FORCE_COLOR")
    if value is None:
        return False
    return value.lower() not in {"", "0", "false"}


def color_from_env() -> bool | None:
    """The color preference the environment declares, or ``None`` to defer to the caller.

    One precedence rule, shared by every color surface so they never disagree:
    ``FORCE_COLOR`` (any value but ``0``/``false``/empty) forces color on even
    when ``NO_COLOR`` is also present; ``FORCE_COLOR`` set to a rejected value
    (``0``/``false``/empty) explicitly disables color — this must override
    Rich's presence-based detection which treats any ``FORCE_COLOR`` as "on";
    ``NO_COLOR`` (present, any value) then forces it off; with neither the
    answer is ``None`` so the caller decides from the stream's own TTY state.
    """
    if _force_color_env():
        return True
    if os.environ.get("FORCE_COLOR") is not None:
        return False
    if "NO_COLOR" in os.environ:
        return False
    return None


def render_lines(
    *renderables: RenderableType,
    color: bool | None = None,
    width: int = 80,
) -> str:
    """Render ``renderables`` through a capture console and return the text.

    Each renderable is printed in turn through a console built by
    :func:`make_capture_console`, so color resolution and no-wrap behavior match
    that helper. A plain string is interpreted as rich markup, so dynamic text
    with markup-significant characters (a metric named ``[i]``, say) must be
    escaped by the caller with :func:`rich.markup.escape` to render literally.

    Trailing whitespace is stripped from every line and the trailing newline is
    dropped, so the result is exactly the visible text with no soft wrapping.

    Args:
        renderables: One or more rich renderables or markup strings to print.
        color: The explicit color choice, or ``None`` to defer to the
            environment and TTY detection.
        width: The terminal width the console renders against.

    Returns:
        The rendered text, lines joined by newlines, with no trailing
        whitespace on any line and no trailing newline.
    """
    console = make_capture_console(color=color, width=width)
    for renderable in renderables:
        # The report styles every span deliberately; rich's repr highlighter would
        # otherwise embolden bare numbers (a delta, a sample count) and split a
        # styled `-17.5%` at the `%`, so it is switched off here.
        console.print(renderable, soft_wrap=True, highlight=False)

    buffer = cast("io.StringIO", console.file)
    lines = buffer.getvalue().split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    return "\n".join(line.rstrip() for line in lines)
