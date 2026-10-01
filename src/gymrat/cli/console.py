"""CLI console state: debug mode, color resolution, stream helpers, the stderr console.

This module owns the process-wide ``--debug`` and ``--color`` / ``--no-color``
state every command reads, the stream classification the output paths share,
and the stderr ``Console`` factory built on top of them. It sits below
:mod:`gymrat.cli.shared` and must never import it.
"""

import errno
import io
import os
import sys
from typing import IO, override

from rich.console import Console

from gymrat.cli.style import CLI_THEME
from gymrat.report.style import color_from_env

# ---------------------------------------------------------------------------
# Debug mode
# ---------------------------------------------------------------------------


class _DebugState:
    """Holds the global ``--debug`` flag without reaching for a ``global`` statement."""

    enabled: bool = False


def set_debug_mode(value: bool) -> None:  # noqa: FBT001 -- 1:1 setter for the --debug flag
    """Set the module debug flag that governs stack traces in error output."""
    _DebugState.enabled = value


def is_debug_mode() -> bool:
    """Whether ``--debug`` is on, so error and warning output should carry stack traces."""
    return _DebugState.enabled


def apply_debug(debug: bool) -> None:  # noqa: FBT001 -- 1:1 pass-through of a command's --debug flag
    """Enable debug mode when a command's own ``--debug`` flag is set.

    Never disables debug mode: a command's local ``--debug`` defaulting to
    ``False`` must not undo the root ``--debug`` flag already applied by
    :func:`set_debug_mode`.

    Args:
        debug: The command's own ``--debug`` flag.
    """
    if debug:
        set_debug_mode(True)


# ---------------------------------------------------------------------------
# Stream helpers
# ---------------------------------------------------------------------------


def is_tty(stream: object) -> bool:
    """Whether ``stream`` reports itself as an interactive terminal."""
    isatty = getattr(stream, "isatty", None)
    return bool(isatty()) if callable(isatty) else False


def is_broken_pipe(error: BaseException) -> bool:
    """Whether ``error`` is a write to a pipe whose reading end has closed.

    POSIX reports it as ``BrokenPipeError``. Windows reports it as a plain
    ``OSError`` with ``EINVAL``: the C runtime maps the ``ERROR_NO_DATA`` a write
    to a closed pipe fails with onto that errno, so no ``BrokenPipeError`` is
    ever raised there.

    Args:
        error: The exception a stream write or flush raised.

    Returns:
        ``True`` when ``error`` means the pipe's reader is gone.
    """
    if isinstance(error, BrokenPipeError):
        return True
    return sys.platform == "win32" and isinstance(error, OSError) and error.errno == errno.EINVAL


def point_stream_at_devnull(stream: IO[str]) -> None:
    """Redirect ``stream``'s file descriptor to devnull; a stream without one is left alone.

    The interpreter flushes a stream's unwritten buffer at shutdown. After a
    failed write that flush would fail again and turn the exit status into 120;
    a devnull descriptor lets it succeed.

    Args:
        stream: The stream whose descriptor is redirected.
    """
    try:
        fd = stream.fileno()
    except io.UnsupportedOperation:
        return
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, fd)
    finally:
        os.close(devnull)


# ---------------------------------------------------------------------------
# Color control
# ---------------------------------------------------------------------------


class _ColorState:
    """Holds the ``--color`` / ``--no-color`` override for all color surfaces.

    The root callback and per-command callbacks write this through
    :func:`set_color_override` so every :func:`resolve_stream_color` call shares
    a single truth.
    """

    override: bool | None = None


def set_color_override(override: bool | None) -> None:  # noqa: FBT001 -- 1:1 setter for the --no-color flag
    """Set the module-level color override read by every color surface."""
    _ColorState.override = override


def apply_color_override(color: bool | None) -> bool | None:  # noqa: FBT001 -- 1:1 pass-through of the --color/--no-color flag
    """Install a subcommand's color override and return it for report rendering.

    Only writes when ``color`` is not ``None`` so a subcommand that declares no
    local ``--color`` flag does not erase a root flag already applied by
    :func:`set_color_override`.

    Args:
        color: The subcommand's ``--color``/``--no-color`` flag, or ``None`` when neither was given.

    Returns:
        The color override as passed in.
    """
    if color is not None:
        set_color_override(color)
    return color


def resolve_stream_color(override: bool | None, stream: object) -> bool:  # noqa: FBT001 -- the resolved --color/--no-color preference, never a bare literal
    """Resolve whether ``stream`` should carry color.

    Precedence, shared by every color surface (report on stdout, progress on
    stderr, error text on stderr): an explicit per-call ``override`` wins; then
    the module-level ``_ColorState.override`` set by the root or subcommand
    callback; then ``FORCE_COLOR``; then ``NO_COLOR``; then the stream's TTY
    status.

    Args:
        override: The explicit ``--color`` / ``--no-color`` flag, or ``None``
            when neither was given.
        stream: The output stream whose TTY status is the final fallback.

    Returns:
        Whether the stream should emit color.
    """
    if override is not None:
        return override
    if _ColorState.override is not None:
        return _ColorState.override
    from_env = color_from_env()
    if from_env is not None:
        return from_env
    return is_tty(stream)


# ---------------------------------------------------------------------------
# Stderr console
# ---------------------------------------------------------------------------


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
