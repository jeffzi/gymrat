"""Error rendering and exit routing for the CLI.

Every command reports a failure and leaves through here: the error formatter,
the exit path that survives a closed stderr, the stdout writer that classifies a
closed pipe, and the boundary that runs an async command body. Nothing here
depends on the heavy statistics stack or the command bodies, so importing it
stays cheap. The debug, color and stream state lives in
:mod:`gymrat.cli.console`.
"""

import asyncio
import contextlib
import sys
import traceback
from collections.abc import Callable, Coroutine
from typing import Any, NoReturn

import typer
from rich.markup import escape

from gymrat.adapters import AdapterError
from gymrat.cli import console
from gymrat.errors import TOOL_FAILURE_EXIT_CODE, GymratError
from gymrat.report.style import format_hint, highlight_inline_code, markup, render_lines
from gymrat.utils import is_broken_pipe, point_stream_at_devnull, write_and_flush

BUGS_URL = "https://github.com/jeffzi/gymrat/issues"


def format_cli_error(error: object, *, debug: bool = False) -> str:
    """Render ``error`` for stderr with a red ``Error:`` label and optional sections.

    The sections appear in order: the label, the message body (an
    :class:`AdapterError` keeps its class-name prefix), the stack trace when
    ``debug`` is set, a dim hint line for a :class:`GymratError` that carries
    one, and a report-a-bug footer for errors that are not :class:`GymratError`.

    Args:
        error: The error to render.
        debug: When set, include the full stack trace.

    Returns:
        The assembled error string with Rich markup.
    """
    error_label = f"{markup('Error', 'red')}: "

    body = f"{type(error).__name__}: {error!s}" if isinstance(error, AdapterError) else str(error)

    doc = f"{error_label}{escape(body)}"

    if debug and isinstance(error, BaseException) and error.__traceback__ is not None:
        stack = "".join(traceback.format_exception(type(error), error, error.__traceback__))
        doc += f"\n{escape(stack.rstrip())}"

    if isinstance(error, GymratError):
        if error.hint is not None:
            doc += f"\n{format_hint(error.hint)}"
    else:
        footer = (
            "\nRun with `gymrat --debug` for details. "
            "If this is a bug, please report it at\n"
            f"{BUGS_URL}"
        )
        doc += highlight_inline_code(footer)

    stderr_color = console.resolve_stream_color(None, sys.stderr)
    return render_lines(doc, color=stderr_color)


def exit_with_error(error: object, code: int = TOOL_FAILURE_EXIT_CODE) -> NoReturn:
    """Print a formatted error to stderr and exit on ``code``, even if the write fails.

    The stack trace is included when ``--debug`` is on.

    Args:
        error: The error to report.
        code: The process exit status.

    Raises:
        typer.Exit: Always, carrying ``code``.
    """
    rendered = f"{format_cli_error(error, debug=console.is_debug_mode())}\n"
    # stderr is the reporting channel, so a write failure (a closed or broken
    # pipe) has nowhere to be reported. Swallow it rather than let it change the
    # exit code this call was asked for.
    with contextlib.suppress(OSError):
        write_and_flush(sys.stderr, rendered)
    with contextlib.suppress(OSError):
        sys.stdout.flush()
    raise typer.Exit(code)


def write_stdout(data: str) -> None:
    """Write a command's result to stdout, returning silently if the reader has gone.

    This write is the only place a closed stdout pipe is classified, per
    :func:`~gymrat.utils.is_broken_pipe`: the same error from anywhere
    else in a command body is a real failure. The stderr console classifies its
    own closed pipe.

    After any failed write stdout is pointed at devnull, so the unwritten bytes
    left in its buffer cannot fail the interpreter's shutdown flush and replace
    the command's exit status with 120.

    Args:
        data: The text to write.

    Raises:
        OSError: When the write fails for any reason other than a closed pipe.
    """
    try:
        write_and_flush(sys.stdout, data)
    except OSError as error:
        point_stream_at_devnull(sys.stdout)
        if not is_broken_pipe(error):
            raise


def run_guarded(body: Callable[[], None]) -> None:
    """Run a CLI body, routing any failure through the shared error formatter.

    Args:
        body: The command body.

    Raises:
        typer.Exit: The body's own exit, or the tool-failure code once any other
            failure has been reported on stderr.
    """
    try:
        body()
    except typer.Exit:
        raise
    except Exception as error:  # noqa: BLE001 -- CLI boundary: route any failure through the formatter
        exit_with_error(error)


def run_cli(run: Callable[[], Coroutine[Any, Any, None]]) -> None:
    """Run an async CLI body, routing any failure through the shared error formatter.

    Args:
        run: Builds the command body's coroutine.

    Raises:
        typer.Exit: The body's own exit, or the tool-failure code once any other
            failure has been reported on stderr.
    """
    run_guarded(lambda: asyncio.run(run()))
