"""Style and color primitives for the report package.

This module owns two concerns the text renderers share:

- **Style vocabulary** — the rich style strings a verdict, a variant name, a
  group label or an aggregate label wears, plus the markup helpers
  (:func:`format_hint`, :func:`highlight_inline_code`) that carry inline
  styling in user-facing prose.
- **Color resolution** — :func:`make_capture_console` and :func:`render_lines`
  turn an explicit ``color`` choice into a rich :class:`~rich.console.Console`
  writing to an in-memory buffer, resolving the choice against the environment
  without ever mutating :data:`os.environ`.

The CLI-side style vocabulary lives in :mod:`gymrat.cli.style`, and fitting a
label into a column lives in :mod:`gymrat.report.labels`.
"""

from __future__ import annotations

import io
import re
from typing import TYPE_CHECKING, Literal, cast

from rich.console import Console
from rich.markup import escape

from gymrat.utils import color_from_env

if TYPE_CHECKING:
    from collections.abc import Sequence

    from rich.console import RenderableType

    from gymrat.report.display import DisplayClass

# ---------------------------------------------------------------------------
# Style vocabulary
# ---------------------------------------------------------------------------

VERDICT_STYLES: dict[DisplayClass, str] = {
    "improved": "green",
    "regressed": "red",
    "unstable": "yellow",
    "identical": "cyan",
    "within-noise": "dim",
    "inconclusive": "dim",
}
"""The color each display class wears wherever the report states a verdict.

Every style here is worn by the verdict itself — a glyph, a delta, a tally —
never by the row or the values around it: within-noise and inconclusive recede
to dim, identical reads cyan for "measured the same", and unstable keeps its
amber warning.
"""

EMPHASIZED_DIRECTION_STYLES: dict[Literal["improved", "regressed"], str] = {
    "improved": f"bold {VERDICT_STYLES['improved']}",
    "regressed": f"bold {VERDICT_STYLES['regressed']}",
}
"""The style a headline figure wears when it reports a direction.

A headline figure (a geomean, an iteration's outcome) is emboldened whatever it
says; it takes its verdict color only where there is a direction to report, and
stays plain ``"bold"`` otherwise.
"""

VARIANT_NAME_STYLE = "bold underline"
"""The style a variant name wears where the report names it as a name."""

GROUP_LABEL_STYLE = "bold blue"
"""The style a group label wears where the report heads a group of metrics."""

AGGREGATE_LABEL_STYLE = "bold"
"""The style an aggregate label (a geomean row, a total) wears."""

SCOPE_SEPARATOR = "·"
"""The middle-dot separator between a scope and its sub-label."""

# A backtick-wrapped inline code span; the capture group is its content.
_INLINE_CODE_PATTERN = re.compile(r"`([^`]+)`")


def markup(text: str, style: str) -> str:
    """Wrap ``text`` in a rich-markup span carrying ``style``, escaping the text.

    The single markup primitive the report package styles a run of plain text
    with; :func:`render_lines` resolves the span to ANSI (or strips it) once.

    Args:
        text: The plain text to wrap.
        style: The rich style to apply.

    Returns:
        The text wrapped in rich-markup style tags.
    """
    return f"[{style}]{escape(text)}[/]"


def join_header_parts(parts: Sequence[str]) -> str:
    """Join report header parts with the dimmed :data:`SCOPE_SEPARATOR` every header shares."""
    return f" {markup(SCOPE_SEPARATOR, 'dim')} ".join(parts)


def highlight_inline_code(text: str) -> str:
    """Replace every ``` `...` ``` span in ``text`` with its content styled blue.

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
        wrapped in a blue span.
    """

    def style_span(match: re.Match[str]) -> str:
        return _code_span(match.group(1))

    return _INLINE_CODE_PATTERN.sub(style_span, text)


def _code_span(code: str) -> str:
    return f"[blue]{escape(code)}[/blue]"


def format_hint(text: str) -> str:
    """Render a hint line as rich markup: dim end to end, inline code blue.

    A hint recedes rather than announces itself, so the whole line is dim and
    carries no label; the commands and flags it names stay legible through the
    blue :func:`highlight_inline_code` spans, which read as dim blue inside the
    surrounding dim span.

    Prose outside the code spans is escaped here rather than by the caller, so a
    hint naming a metric such as ``[i]`` renders it literally instead of having
    rich parse it as a style tag. The spans are escaped on their own, so
    escaping the prose separately is what keeps both from being escaped twice.

    Args:
        text: The bare hint sentence, which may contain ``` `...` ``` spans.

    Returns:
        Rich markup for the hint line, for :func:`render_lines` to resolve.
    """
    # Splitting on a pattern with one capture group alternates prose and code.
    parts = [
        _code_span(piece) if index % 2 else escape(piece)
        for index, piece in enumerate(_INLINE_CODE_PATTERN.split(text))
    ]
    return f"[dim]{''.join(parts)}[/dim]"


# ---------------------------------------------------------------------------
# Color resolution and capture rendering
# ---------------------------------------------------------------------------

RENDER_WIDTH = 200
"""A render width wide enough that a captured line never soft-wraps, whatever the real terminal is.

Every report block and table renders against it.
"""


def make_capture_console(*, color: bool | None) -> Console:
    """Build a rich console that captures its output to an in-memory buffer.

    Wide content is never wrapped or cropped (``soft_wrap``), so
    :data:`RENDER_WIDTH` only bounds justification, never the text. Emoji codes
    are off, so a ``:word:`` sequence in a metric name, kind or label prints
    literally (``lat:100:p99`` stays as written).

    Args:
        color: The explicit color choice, resolved without touching
            :data:`os.environ`:

            - ``True`` forces ANSI even when ``NO_COLOR`` is set, by declaring
              the capture a terminal with color enabled.
            - ``False`` suppresses ANSI even when ``FORCE_COLOR`` is set.
            - ``None`` defers to :func:`gymrat.utils.color_from_env`, which owns
              the ``FORCE_COLOR``/``NO_COLOR`` precedence and the ways it differs
              from rich's own detection. With neither variable set, a captured
              buffer is not a TTY, so the output is plain.

    Returns:
        A console whose ``file`` is an :class:`io.StringIO` holding everything
        printed through it.
    """
    buffer = io.StringIO()
    resolved = color if color is not None else color_from_env()

    force_terminal: bool | None
    color_system: Literal["auto", "standard", "256", "truecolor", "windows"] | None
    if resolved is True:
        force_terminal, color_system = True, "truecolor"
    elif resolved is False:
        # `no_color=True` only strips colors, leaving bold/underline/dim intact
        # when FORCE_COLOR has forced a terminal; disabling the color system drops
        # every SGR code, so color=False fully overrides the environment.
        force_terminal, color_system = None, None
    else:
        force_terminal, color_system = None, "auto"

    # Both width and height must be explicit: Rich's Console.size only honors an
    # explicit width when _height is also set; otherwise a dumb terminal
    # (TERM=dumb, common in git hooks and CI) early-returns (80, 25).
    return Console(
        file=buffer,
        width=RENDER_WIDTH,
        height=25,
        force_terminal=force_terminal,
        color_system=color_system,
        legacy_windows=False,
        no_color=False,
        soft_wrap=True,
        emoji=False,
    )


def render_line_list(*renderables: RenderableType, color: bool | None = None) -> list[str]:
    """Render ``renderables`` through a capture console and return the text lines.

    Each renderable is printed in turn through a console built by
    :func:`make_capture_console`, so color resolution and no-wrap behavior match
    that helper.

    Args:
        *renderables: One or more rich renderables or markup strings to print.
            A plain string is interpreted as rich markup, so dynamic text with
            markup-significant characters (a metric named ``[i]``, say) must be
            escaped by the caller with :func:`rich.markup.escape` to render
            literally.
        color: The explicit color choice, or ``None`` to defer to the
            environment and TTY detection.

    Returns:
        Exactly the visible lines with no soft wrapping, with no trailing
        whitespace on any line and no trailing empty line.
    """
    console = make_capture_console(color=color)
    for renderable in renderables:
        # The report styles every span deliberately; rich's repr highlighter would
        # otherwise embolden bare numbers (a delta, a sample count) and split a
        # styled `-17.5%` at the `%`, so it is switched off here.
        console.print(renderable, soft_wrap=True, highlight=False)

    buffer = cast("io.StringIO", console.file)
    lines = buffer.getvalue().split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    return [line.rstrip() for line in lines]


def render_lines(*renderables: RenderableType, color: bool | None = None) -> str:
    """Render ``renderables`` as :func:`render_line_list` does, joined into one text.

    Args:
        *renderables: One or more rich renderables or markup strings to print.
        color: The explicit color choice, or ``None`` to defer to the
            environment and TTY detection.

    Returns:
        The rendered lines joined by newlines, with no trailing newline.
    """
    return "\n".join(render_line_list(*renderables, color=color))
