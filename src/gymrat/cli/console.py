"""Stderr ``Console`` factory for rich output."""

import sys
from typing import override

from rich.console import Console

from gymrat.cli.shared import is_broken_pipe, point_stream_at_devnull, resolve_stream_color
from gymrat.cli.style import CLI_THEME


class _StderrConsole(Console):
    """A ``Console`` that silences its own stream, not stdout, on a broken pipe."""

    @override
    def _write_buffer(self) -> None:
        # Rich routes only BrokenPipeError to on_broken_pipe; a Windows broken
        # pipe arrives as OSError(EINVAL) and would escape the hook.
        try:
            super()._write_buffer()
        except BrokenPipeError:
            raise
        except OSError as error:
            if not is_broken_pipe(error):
                raise
            raise BrokenPipeError(error.errno, error.strerror) from error

    @override
    def on_broken_pipe(self) -> None:
        self.quiet = True
        point_stream_at_devnull(self.file)
        raise SystemExit(1)


def stderr_console(*, color_flag: bool | None = None) -> Console:
    """Build a ``Console`` that writes to stderr with resolved color and width.

    When colorless the console sets ``no_color=True`` and ``color_system=None``
    so all SGR is suppressed, including bold and dim, matching the stdout
    report surface.

    A write that hits a broken stderr pipe exits with status 1 and writes
    nothing further to either stream.

    Args:
        color_flag: The Typer ``--color`` / ``--no-color`` option value.

    Returns:
        A stderr ``Console`` configured with the resolved color system and
        CLI theme.
    """
    colored = resolve_stream_color(color_flag, sys.stderr)
    color_system = "auto" if colored else None

    # Do not pass `width=` or read `COLUMNS` here: Rich already sizes the
    # console to the terminal.
    return _StderrConsole(
        stderr=True,
        # Pins Rich's own NO_COLOR detection so the shared precedence
        # (flag > FORCE_COLOR > NO_COLOR > TTY) is the single decider.
        no_color=not colored,
        color_system=color_system,
        legacy_windows=False,
        theme=CLI_THEME,
    )
