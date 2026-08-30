"""Shared CLI infrastructure: parsing, exit routing, error rendering, locking.

This module holds the pieces every benchmarking command reuses — the argument
coercers, the error formatter and exit path, the repository single-flight lock,
and the render-mode resolution — with no dependency on the heavy statistics
stack or the command bodies, so importing it stays cheap.
"""

import asyncio
import contextlib
import math
import re
import sys
import traceback
from collections.abc import Awaitable, Callable, Coroutine
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Annotated, Any, Literal, NoReturn, Protocol

import typer
from rich.markup import escape

from gymrat.adapters.types import AdapterError
from gymrat.cli.progress import ProgressReporter, create_progress_reporter
from gymrat.config import MAX_SAFE_INTEGER, MAX_TIMEOUT_SECONDS, CliFlags, ResolvedConfig
from gymrat.errors import GymratError, hint_of
from gymrat.exec import kill_live_process_groups
from gymrat.git import NotAGitRepositoryError
from gymrat.report.style import (
    RENDER_WIDTH,
    color_from_env,
    format_hint,
    highlight_inline_code,
    markup,
    render_lines,
)
from gymrat.report.types import FailOnCondition, GeomeanFailOn, RegressedFailOn, ReportOptions
from gymrat.sampling import RunOptions, TargetSpec
from gymrat.session.lock import acquire_lock
from gymrat.session.paths import lockfile_path, repo_root
from gymrat.signals import install_termination_cleanup

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BUGS_URL = "https://github.com/jeffzi/gymrat/issues"

# The exit code of a gate trip: a run that did what it was asked and said no.
GATE_EXIT_CODE = 1

# The exit code of a tool failure, the convention every unhandled error exits on.
TOOL_FAILURE_EXIT_CODE = 2

_POSITIVE_INTEGER_RE = re.compile(r"\d+")
_POSITIVE_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")
_GEOMEAN_CONDITION_RE = re.compile(r"geomean:(-?\d+(?:\.\d+)?)")


class _WritableStream(Protocol):
    """A text stream this module writes error and progress output to."""

    def write(self, data: str, /) -> object: ...

    def flush(self) -> object: ...


# ---------------------------------------------------------------------------
# Debug mode
# ---------------------------------------------------------------------------


class _DebugState:
    """Holds the global ``--debug`` flag without reaching for a ``global`` statement."""

    enabled: bool = False


def set_debug_mode(value: bool) -> None:  # noqa: FBT001 -- 1:1 setter for the --debug flag
    """Set the module debug flag that governs stack traces in error output."""
    _DebugState.enabled = value


class _StderrColorState:
    """Holds the ``--no-color`` override for stderr error output.

    Commands set this at startup so ``_resolve_stderr_color`` reads the flag
    instead of always deferring to env+TTY detection.
    """

    override: bool | None = None


def set_stderr_color_override(override: bool | None) -> None:  # noqa: FBT001 -- 1:1 setter for the --no-color flag
    """Set the module-level color override that ``format_cli_error`` reads."""
    _StderrColorState.override = override


def apply_debug(debug: bool) -> None:  # noqa: FBT001 -- 1:1 pass-through of a command's --debug flag
    """Enable debug mode when a command's own ``--debug`` flag is set.

    Never disables debug mode: a command's local ``--debug`` defaulting to
    ``False`` must not undo the root ``--debug`` flag already applied by
    :func:`set_debug_mode`.
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


def write_and_flush(stream: _WritableStream, data: str) -> None:
    """Write ``data`` to ``stream`` and flush it so an immediate exit cannot truncate it."""
    stream.write(data)
    stream.flush()


# ---------------------------------------------------------------------------
# Color control
# ---------------------------------------------------------------------------


def color_override_of(color: bool) -> Literal[False] | None:  # noqa: FBT001 -- 1:1 map of the --color flag
    """Translate the ``--color`` flag into the renderer's color override.

    ``False`` vetoes color; ``None`` defers to auto-detection.
    """
    return None if color else False


def resolve_stream_color(override: bool | None, stream: object) -> bool:  # noqa: FBT001 -- the resolved --color/--no-color preference, never a bare literal
    """Resolve whether output to ``stream`` should carry color.

    One precedence rule, shared by every color surface — the report on stdout,
    the progress line on stderr, and the error text on stderr — so they never
    disagree: an explicit ``override`` (the ``--color`` / ``--no-color`` flag)
    wins; then ``FORCE_COLOR`` (any value but ``0``/``false``/empty); then
    ``NO_COLOR`` (present, any value); then the stream's own TTY detection.
    """
    if override is not None:
        return override
    from_env = color_from_env()
    if from_env is not None:
        return from_env
    return is_tty(stream)


def _resolve_stderr_color() -> bool:
    """Whether stderr error output should carry color, per the shared precedence."""
    return resolve_stream_color(_StderrColorState.override, sys.stderr)


# ---------------------------------------------------------------------------
# Error formatting
# ---------------------------------------------------------------------------


def format_cli_error(error: object, *, debug: bool = False) -> str:
    """Render ``error`` for stderr with a red ``Error:`` label and optional sections.

    The sections appear in order: the label, the message body (an
    :class:`AdapterError` keeps its class-name prefix), the stack trace when
    ``debug`` is set, a dim hint line for a :class:`GymratError` that carries
    one, and a report-a-bug footer for errors that are not :class:`GymratError`.
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

    return render_lines(doc, color=_resolve_stderr_color(), width=RENDER_WIDTH)


def exit_with_error(error: object, code: int = TOOL_FAILURE_EXIT_CODE) -> NoReturn:
    """Print a formatted error to stderr and exit on ``code``, even if the write fails."""
    rendered = f"{format_cli_error(error, debug=_DebugState.enabled)}\n"
    # stderr is the reporting channel, so a write failure (a closed or broken
    # pipe) has nowhere to be reported. Swallow it rather than let it change the
    # exit code this call was asked for.
    with contextlib.suppress(OSError):
        write_and_flush(sys.stderr, rendered)
    with contextlib.suppress(OSError):
        sys.stdout.flush()
    raise typer.Exit(code)


def run_cli(run: Callable[[], Coroutine[Any, Any, None]]) -> None:
    """Run an async CLI body, routing any failure through the shared error formatter."""
    try:
        asyncio.run(run())
    except typer.Exit:
        raise
    except BrokenPipeError:
        raise typer.Exit(0) from None
    except Exception as error:  # noqa: BLE001 -- CLI boundary: route any failure through the formatter
        exit_with_error(error)


# ---------------------------------------------------------------------------
# Flag parsers
# ---------------------------------------------------------------------------


def parse_positional(positional: str) -> TargetSpec:
    """Parse the ``label=target`` syntax of a positional, splitting on the first ``=`` only.

    A target containing its own ``=`` survives intact — ``a=b=c`` parses to label
    ``a``, target ``b=c``. An empty half is always a typo, so each raises its own
    usage error rather than resolving to a silent default.
    """
    head, sep, tail = positional.partition("=")
    label: str | None = head if sep else None
    target = tail if sep else positional

    if label == "":
        message = (
            'the label before "=" is empty; '
            'write the positional as "label=<ref|dir>" or drop the "=".'
        )
        raise typer.BadParameter(message)
    if target == "":
        message = 'the target is empty; write the positional as "[label=]<ref|dir>".'
        raise typer.BadParameter(message)

    return TargetSpec(label=label, target=target)


def parse_positive_integer_up_to(max_value: int) -> Callable[[str], int]:
    """Build a coercer accepting only a positive integer at or below ``max_value``."""

    def parse(value: str) -> int:
        if _POSITIVE_INTEGER_RE.fullmatch(value) is None or int(value) <= 0:
            message = "must be a positive integer."
            raise typer.BadParameter(message)
        parsed = int(value)
        if parsed > max_value:
            message = f"must be a positive integer no greater than {max_value}."
            raise typer.BadParameter(message)
        return parsed

    return parse


def parse_positive_number(value: str) -> float:
    """Parse a strictly positive finite decimal, rejecting negatives, zero, and trailing garbage."""
    if _POSITIVE_NUMBER_RE.fullmatch(value) is None:
        message = "must be a positive number."
        raise typer.BadParameter(message)
    parsed = float(value)
    if parsed <= 0 or not math.isfinite(parsed):
        message = "must be a positive number."
        raise typer.BadParameter(message)
    return parsed


def parse_max_minutes(value: str) -> float:
    """Parse a positive number of minutes bounded by the 32-bit timer ceiling."""
    parsed = parse_positive_number(value)
    max_minutes = MAX_TIMEOUT_SECONDS // 60
    if parsed > max_minutes:
        message = f"must be at most {max_minutes} minutes."
        raise typer.BadParameter(message)
    return parsed


def parse_fail_on(value: str) -> FailOnCondition:
    """Parse a fail-on condition: ``regressed`` or ``geomean:<number>``.

    Anything else raises a usage error naming the allowed grammar.
    """
    if value == "regressed":
        return RegressedFailOn()

    match = _GEOMEAN_CONDITION_RE.fullmatch(value)
    if match is not None:
        return GeomeanFailOn(pct=float(match.group(1)))

    message = 'allowed values are "regressed" or "geomean:<number>" (e.g. geomean:2).'
    raise typer.BadParameter(message)


# ---------------------------------------------------------------------------
# Render mode
# ---------------------------------------------------------------------------


def resolve_render_mode() -> Literal["live", "plain"]:
    """Map the stderr TTY status to the output strategy the progress reporter uses.

    A non-TTY stderr always renders plain; a TTY gets the rich-based live
    layout regardless of color — styling is handled by the console's own
    color resolution.
    """
    return "live" if is_tty(sys.stderr) else "plain"


# ---------------------------------------------------------------------------
# Repository lock
# ---------------------------------------------------------------------------


async def with_repo_lock[T](command: str, body: Callable[[], Awaitable[T]]) -> T:
    """Hold the repository's single-flight lock for the length of ``body``.

    Inside a git repository the lock is acquired around ``body`` and released
    however it settles — including on exception, and always before the caller
    renders its report. Outside every git repository the answer is to run
    ``body`` with no lock at all; any other git failure exits without
    benchmarking rather than running unlocked.
    """
    try:
        root = repo_root()
    except NotAGitRepositoryError:
        return await body()
    except GymratError as error:
        exit_with_error(error)

    release = acquire_lock(lockfile_path(root), command)
    try:
        return await body()
    finally:
        release()


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

    color: bool = True
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
# CLI option declarations
# ---------------------------------------------------------------------------


class OutputFormat(StrEnum):
    """The ``--format`` choices: a human report or a machine-readable document."""

    text = "text"
    json = "json"


# The config-bearing options every command shares, declared once as reusable
# annotations so ``compare`` and ``measure`` carry an identical surface.
BenchOption = Annotated[str | None, typer.Option("--bench", "-b", help="bench command")]
PrepareOption = Annotated[
    str | None,
    typer.Option("--prepare", "-p", help="preparation script to run before each revision"),
]
AdapterOption = Annotated[
    str | None, typer.Option("--adapter", "-a", help="adapter type for parsing benchmark output")
]
SamplesOption = Annotated[
    int | None,
    typer.Option(
        "--samples",
        "-s",
        parser=parse_positive_integer_up_to(MAX_SAFE_INTEGER),
        help="paired samples per target",
    ),
]
TimeoutOption = Annotated[
    int | None,
    typer.Option(
        "--timeout",
        "-t",
        parser=parse_positive_integer_up_to(MAX_TIMEOUT_SECONDS),
        help="timeout in seconds",
    ),
]
ConfigOption = Annotated[str | None, typer.Option("--config", "-c", help="configuration file path")]
NoColorOption = Annotated[
    bool, typer.Option("--no-color", help="print the report without ANSI styles")
]
FormatOption = Annotated[OutputFormat, typer.Option("--format", help="output format")]
DebugOption = Annotated[bool, typer.Option("--debug", "-d", help="show stack traces on errors")]
RecordOption = Annotated[
    bool,
    typer.Option("--record", "-r", help="append the run to the session log as a baseline"),
]
MessageOption = Annotated[
    str | None, typer.Option("--message", "-m", help="commit message for the settled edit")
]
BranchOption = Annotated[
    str | None,
    typer.Option("--branch", help="branch to point at the squash commit (default: <branch>-final)"),
]
ForceOption = Annotated[bool, typer.Option("--force", "-f", help="skip the confirmation prompt")]
VerboseOption = Annotated[
    bool, typer.Option("--verbose", "-v", help="keep the progress tree visible after the run")
]


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
    """Build the progress reporter a run prints through, sized and colored per the flags."""
    from gymrat.cli.console import (  # noqa: PLC0415 -- console.py imports from shared.py
        stderr_console,
    )

    mode = resolve_render_mode()
    console = stderr_console(color_flag=flags.color)
    extra: dict[str, object] = {}
    if command is not None:
        extra["command"] = command
    if target_labels is not None:
        extra["target_labels"] = target_labels
    return create_progress_reporter(
        mode,
        console,
        target_count,
        flags.samples,
        **extra,  # type: ignore[arg-type]
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


@dataclass(frozen=True, slots=True)
class ReportRenderers[T]:
    """The text and JSON renderers a command hands ``emit_report`` for its result type."""

    text: Callable[[T, ReportOptions], str]
    json: Callable[[T], str]


def wants_json(flags: SharedFlags) -> bool:
    """Whether ``flags.format`` selects the JSON document rather than the text report."""
    return flags.format == OutputFormat.json.value


def emit_report[T](
    result: T,
    flags: SharedFlags,
    renderers: ReportRenderers[T],
    render_opts: ReportOptions,
) -> None:
    """Render ``result`` per ``flags.format`` and write it to stdout.

    The text renderer captures into an in-memory buffer that cannot see the real
    stdout, so a deferred color choice (``render_opts.color is None``) is
    resolved here against stdout's own TTY state through the shared precedence:
    a report printed to a terminal is styled, the same report piped away is
    plain. An explicit ``--color`` / ``--no-color`` veto in ``render_opts.color``
    is passed through untouched. The JSON document is never styled, so it ignores
    ``render_opts`` entirely.
    """
    if wants_json(flags):
        write_and_flush(sys.stdout, renderers.json(result) + "\n")
        return
    color = resolve_stream_color(render_opts.color, sys.stdout)
    output = renderers.text(result, replace(render_opts, color=color))
    write_and_flush(sys.stdout, output + "\n")
