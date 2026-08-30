"""Stderr ``Console`` factory for rich output."""

import sys

from rich.console import Console

from gymrat.cli.shared import color_override_of, resolve_stream_color
from gymrat.cli.style import CLI_THEME


def stderr_console(*, color_flag: bool | None = None) -> Console:
    """Build a ``Console`` that writes to stderr with resolved color and width.

    ``color_flag`` is the Typer ``--color`` / ``--no-color`` option value:
    ``True`` defers to auto-detection, ``False`` vetoes color, and ``None``
    (the default) runs the full detection chain.

    When colorless the console uses ``color_system=None`` rather than
    ``no_color=True`` so that **all** SGR is suppressed — including bold and
    dim — matching the stdout report surface.

    Rich's own ``Console`` already reads ``COLUMNS`` through a guarded path
    that ignores non-numeric values, so we do not reimplement that lookup.
    """
    override = color_override_of(color_flag) if color_flag is not None else None
    colored = resolve_stream_color(override, sys.stderr)

    # no_color pins Rich's own NO_COLOR detection so the shared precedence
    # (flag > FORCE_COLOR > NO_COLOR > TTY) is the single decider.
    # color_system=None strips every SGR sequence (bold, dim, italic — not
    # just color); "auto" lets Rich pick the palette from the terminal.
    color_system = "auto" if colored else None

    return Console(
        stderr=True,
        no_color=not colored,
        color_system=color_system,
        legacy_windows=False,
        theme=CLI_THEME,
    )
