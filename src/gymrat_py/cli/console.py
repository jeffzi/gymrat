"""Stderr ``Console`` factory for rich output."""

import os
import sys

from rich.console import Console

from gymrat_py.cli.shared import color_override_of, resolve_stream_color
from gymrat_py.cli.style import CLI_THEME


def stderr_console(*, color_flag: bool | None = None) -> Console:
    """Build a ``Console`` that writes to stderr with resolved color and width.

    ``color_flag`` is the Typer ``--color`` / ``--no-color`` option value:
    ``True`` defers to auto-detection, ``False`` vetoes color, and ``None``
    (the default) runs the full detection chain.

    The console carries :data:`~gymrat_py.cli.style.CLI_THEME`, which is what
    makes rich's own progress columns follow the shared state conventions.
    """
    override = color_override_of(color_flag) if color_flag is not None else None
    colored = resolve_stream_color(override, sys.stderr)

    columns = os.environ.get("COLUMNS")
    width = int(columns) if columns is not None else None

    return Console(stderr=True, no_color=not colored, width=width, theme=CLI_THEME)
