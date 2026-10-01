"""Shared CLI infrastructure: exit routing, error rendering, report output.

This module holds the pieces every benchmarking command reuses — the error
formatter and exit path, the render-mode resolution, and the report writers —
with no dependency on the heavy statistics stack or the command bodies, so
importing it stays cheap. The debug, color and stream state lives in
:mod:`gymrat.cli.console`; the flag parsers and option declarations live in
:mod:`gymrat.cli.options`; the session budget trailers and warnings live in
:mod:`gymrat.cli.budget_report`; the repository lock and command trace live in
:mod:`gymrat.command_run`.
"""

import asyncio
import contextlib
import sys
import traceback
from collections.abc import Awaitable, Callable, Coroutine
from dataclasses import dataclass, replace
from typing import Any, Literal, NoReturn, Protocol

import typer
from rich.markup import escape

from gymrat.adapters.types import AdapterError
from gymrat.cli import console
from gymrat.cli.options import OutputFormat
from gymrat.cli.progress import ProgressReporter
from gymrat.config import CliFlags, ResolvedConfig
from gymrat.errors import TOOL_FAILURE_EXIT_CODE, GymratError, hint_of
from gymrat.exec import kill_live_process_groups
from gymrat.report.json_doc import BudgetSummary
from gymrat.report.style import (
    RENDER_WIDTH,
    format_hint,
    highlight_inline_code,
    markup,
    render_lines,
)
from gymrat.report.types import FailOnCondition, ReportOptions
from gymrat.sampling import RunOptions
from gymrat.signals import install_termination_cleanup

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BUGS_URL = "https://github.com/jeffzi/gymrat/issues"


class _WritableStream(Protocol):
    """A text stream this module writes error and progress output to."""

    def write(self, data: str, /) -> object: ...

    def flush(self) -> object: ...


# ---------------------------------------------------------------------------
# Stream helpers
# ---------------------------------------------------------------------------


def write_and_flush(stream: _WritableStream, data: str) -> None:
    """Write ``data`` to ``stream`` and flush it so an immediate exit cannot truncate it."""
    stream.write(data)
    stream.flush()


# ---------------------------------------------------------------------------
# Error formatting
# ---------------------------------------------------------------------------


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

    hint = hint_of(error)
    if hint is not None:
        doc += f"\n{format_hint(hint)}"

    if not isinstance(error, GymratError):
        footer = (
            "\nRun with `gymrat --debug` for details. "
            "If this is a bug, please report it at\n"
            f"{BUGS_URL}"
        )
        doc += highlight_inline_code(footer)

    stderr_color = console.resolve_stream_color(None, sys.stderr)
    return render_lines(doc, color=stderr_color, width=RENDER_WIDTH)


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
    :func:`~gymrat.cli.console.is_broken_pipe`: the same error from anywhere
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
        console.point_stream_at_devnull(sys.stdout)
        if not console.is_broken_pipe(error):
            raise


def run_cli(run: Callable[[], Coroutine[Any, Any, None]]) -> None:
    """Run an async CLI body, routing any failure through the shared error formatter.

    Args:
        run: Builds the command body's coroutine.

    Raises:
        typer.Exit: The body's own exit, or the tool-failure code once any other
            failure has been reported on stderr.
    """
    try:
        asyncio.run(run())
    except typer.Exit:
        raise
    except Exception as error:  # noqa: BLE001 -- CLI boundary: route any failure through the formatter
        exit_with_error(error)


# ---------------------------------------------------------------------------
# Render mode
# ---------------------------------------------------------------------------


def resolve_render_mode() -> Literal["live", "plain"]:
    """Map the stderr TTY status to the output strategy the progress reporter uses.

    A non-TTY stderr always renders plain; a TTY gets the rich-based live
    layout regardless of color — styling is handled by the console's own
    color resolution.

    Returns:
        ``"live"`` when stderr is a TTY, ``"plain"`` otherwise.
    """
    return "live" if console.is_tty(sys.stderr) else "plain"


async def run_with_signal_abort[T](
    execute: Callable[[asyncio.Event], Awaitable[T]],
) -> T:
    """Run ``execute`` with an abort event a termination signal trips.

    ``execute`` receives an :class:`asyncio.Event` to hand the in-flight bench so
    a ``SIGINT`` / ``SIGTERM`` sets it and the current sample is abandoned rather
    than the process being torn down mid-command. The signal handler owns the
    exit itself (``128 +`` the signal number); this only wires the event and
    always removes the cleanup afterward, so a completed run leaves no handler
    behind.

    On the signal path the loop cannot resume to act on the abort before the
    process exits, so the cleanup kills any live exec-spawned group synchronously
    before setting the event; the event still drives the async abort race when
    the loop does keep running.

    Args:
        execute: An async callable that receives an abort event and returns the
            run result.

    Returns:
        The value returned by ``execute``.
    """
    abort = asyncio.Event()

    def terminate() -> None:
        kill_live_process_groups()
        abort.set()

    uninstall = install_termination_cleanup(terminate)
    try:
        return await execute(abort)
    finally:
        uninstall()


# ---------------------------------------------------------------------------
# Flag dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SharedFlags(CliFlags):
    """The flags every command carries: the config set plus how the report prints."""

    color: bool | None = None
    format: Literal["text", "json"] = "text"


@dataclass(frozen=True, slots=True)
class CompareFlags(SharedFlags):
    """The compare command's flags: the shared set plus the two only a verdict can answer."""

    verbose: bool = False
    fail_on: tuple[FailOnCondition, ...] = ()


@dataclass(frozen=True, slots=True)
class MeasureFlags(SharedFlags):
    """The measure command's flags: the shared set plus whether to record the run."""

    record: bool = False


# ---------------------------------------------------------------------------
# Run infrastructure
# ---------------------------------------------------------------------------


def begin_run(
    flags: SharedFlags,
    target_count: int,
    *,
    command: str | None = None,
    target_labels: list[str] | None = None,
) -> ProgressReporter:
    """Build the progress reporter a run prints through, sized and colored per the flags.

    Args:
        flags: The command's flags; ``color`` styles the stderr console and
            ``samples`` sizes the progress display.
        target_count: How many targets the run measures.
        command: The command name the reporter labels its output with.
        target_labels: The display label of each target, in run order.

    Returns:
        A progress reporter writing to a stderr console in the render mode
        stderr's TTY status selects.
    """
    mode = resolve_render_mode()
    progress_console = console.stderr_console(color_flag=flags.color)
    return ProgressReporter(
        mode,
        progress_console,
        target_count,
        flags.samples,
        command=command,
        target_labels=target_labels,
    )


def run_options_of(config: ResolvedConfig, progress: ProgressReporter) -> RunOptions:
    """Wire ``config``'s run settings and ``progress``'s callbacks into the shared run fields."""
    return RunOptions(
        bench=config.bench,
        prepare=config.prepare,
        adapter=config.adapter,
        samples=config.samples,
        timeout_seconds=config.timeout_seconds,
        config_metrics=config.metrics,
        config_kinds=config.kinds,
        on_progress=progress.report,
        warn=progress.warn,
    )


class JsonRenderer[T](Protocol):
    """A JSON renderer for a report result type.

    Implementations omit the ``budget`` field from the document entirely when *budget*
    is ``None``, rather than serializing it as ``null``.
    """

    def __call__(self, result: T, /, *, budget: BudgetSummary | None = None) -> str:
        """Render *result* as a JSON document.

        Args:
            result: The report result to serialize.
            budget: The session budget summary to embed, or ``None`` to omit the field.

        Returns:
            The JSON document text.
        """
        ...


@dataclass(frozen=True, slots=True)
class ReportRenderers[T]:
    """The text and JSON renderers a command hands ``emit_report`` for its result type."""

    text: Callable[[T, ReportOptions], str]
    json: JsonRenderer[T]


def wants_json(flags: SharedFlags) -> bool:
    """Whether ``flags.format`` selects the JSON document rather than the text report."""
    return flags.format == OutputFormat.json.value


def emit_report[T](  # noqa: PLR0913 -- keyword-only budget params extend a 4-positional surface
    result: T,
    flags: SharedFlags,
    renderers: ReportRenderers[T],
    render_opts: ReportOptions,
    *,
    budget_trailer: str = "",
    budget_summary: BudgetSummary | None = None,
) -> None:
    """Render ``result`` per ``flags.format`` and write it to stdout.

    The text renderer captures into an in-memory buffer that cannot see the real
    stdout, so a deferred color choice (``render_opts.color is None``) is
    resolved here against stdout's own TTY state through the shared precedence:
    a report printed to a terminal is styled, the same report piped away is
    plain. An explicit ``--color`` / ``--no-color`` veto in ``render_opts.color``
    is passed through untouched. The JSON document is never styled, so it ignores
    ``render_opts`` entirely.

    Args:
        result: The typed result to render.
        flags: CLI flags selecting ``text`` or ``json`` output format.
        renderers: The text and JSON render callables for the result type.
        render_opts: Styling options forwarded to the text renderer.
        budget_trailer: Text appended after the report (typically a time-left
            line).
        budget_summary: Forwarded to the JSON renderer for the ``budget`` key
            in machine-readable output.
    """
    if wants_json(flags):
        write_stdout(renderers.json(result, budget=budget_summary) + "\n")
        return
    color = console.resolve_stream_color(render_opts.color, sys.stdout)
    output = renderers.text(result, replace(render_opts, color=color))
    write_stdout(output + budget_trailer + "\n")
