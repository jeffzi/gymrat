"""Shared CLI infrastructure: parsing, exit routing, error rendering.

This module holds the pieces every benchmarking command reuses — the argument
coercers, the error formatter and exit path, and the render-mode resolution —
with no dependency on the heavy statistics stack or the command bodies, so
importing it stays cheap. The repository lock and command trace live in
:mod:`gymrat.cli.lock`.
"""

import asyncio
import contextlib
import math
import re
import sys
import traceback
from collections.abc import Awaitable, Callable, Coroutine, Generator
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Annotated, Any, Literal, NoReturn, Protocol, override

import click
import typer
from rich.markup import escape

from gymrat import clock as _clock
from gymrat.adapters.types import AdapterError
from gymrat.cli.lock import TOOL_FAILURE_EXIT_CODE
from gymrat.cli.progress import ProgressReporter
from gymrat.config import MAX_SAFE_INTEGER, MAX_TIMEOUT_SECONDS, CliFlags, ResolvedConfig
from gymrat.errors import GymratError, hint_of
from gymrat.eta import format_duration
from gymrat.exec import kill_live_process_groups
from gymrat.report.json_doc import BudgetSummary
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
from gymrat.session.budget import (
    SIDES_PER_ITERATE,
    Budget,
    estimate_iterate_duration,
    format_budget_trailer,
    read_budget,
)
from gymrat.session.paths import repo_root, session_jsonl_path
from gymrat.session.store import read_records
from gymrat.signals import install_termination_cleanup
from gymrat.warn import warn_to_stderr

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BUGS_URL = "https://github.com/jeffzi/gymrat/issues"

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


def is_debug_mode() -> bool:
    """Whether ``--debug`` is on, so error and warning output should carry stack traces."""
    return _DebugState.enabled


class _ColorState:
    """Holds the ``--color`` / ``--no-color`` override for all color surfaces.

    The root callback and per-command callbacks write this through
    :func:`set_color_override` so :func:`resolve_stream_color` and
    :func:`_resolve_stderr_color` share a single truth.
    """

    override: bool | None = None


def set_color_override(override: bool | None) -> None:  # noqa: FBT001 -- 1:1 setter for the --no-color flag
    """Set the module-level color override read by every color surface."""
    _ColorState.override = override


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


def write_and_flush(stream: _WritableStream, data: str) -> None:
    """Write ``data`` to ``stream`` and flush it so an immediate exit cannot truncate it."""
    stream.write(data)
    stream.flush()


# ---------------------------------------------------------------------------
# Color control
# ---------------------------------------------------------------------------


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


def _resolve_stderr_color() -> bool:
    """Whether stderr error output should carry color, per the shared precedence."""
    return resolve_stream_color(_ColorState.override, sys.stderr)


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


@contextlib.contextmanager
def broken_pipe_guard() -> Generator[None]:
    """Catch BrokenPipeError from a stdout write and exit cleanly.

    Sync counterpart of the same mapping in ``run_cli``: a broken pipe from the
    reading end closing is not an error for a CLI that already produced its
    output.

    Raises:
        typer.Exit: With code 0 when a ``BrokenPipeError`` is caught.
    """
    try:
        yield
    except BrokenPipeError:
        raise typer.Exit(0) from None


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

    Args:
        positional: The raw positional argument in ``label=target`` or bare
            ``target`` form.

    Returns:
        The parsed label and target.

    Raises:
        typer.BadParameter: When the label or target half is empty.
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


class PositionalParamType(click.ParamType[TargetSpec, str]):
    """Click type for ``[label=]<ref|dir>`` positional arguments.

    Wraps :func:`parse_positional` in a proper :class:`click.ParamType` so the
    help panel shows ``<ref|dir>`` instead of the parser function repr.
    """

    name = "<ref|dir>"

    @override
    def convert(
        self,
        value: str | TargetSpec,
        param: click.Parameter | None,
        ctx: click.Context | None,
    ) -> TargetSpec:
        if isinstance(value, TargetSpec):
            return value
        return parse_positional(value)


def parse_positive_integer_up_to(max_value: int) -> Callable[[str], int]:
    """Build a coercer accepting only a positive integer at or below ``max_value``."""

    def parse(value: str) -> int:
        if _POSITIVE_INTEGER_RE.fullmatch(value) is None:
            message = "must be a positive integer."
            raise typer.BadParameter(message)
        parsed = int(value)
        if parsed <= 0:
            message = "must be a positive integer."
            raise typer.BadParameter(message)
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

    Args:
        value: The raw ``--fail-on`` flag value.

    Returns:
        The parsed fail-on condition.

    Raises:
        typer.BadParameter: When the value does not match the allowed grammar.
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

    Returns:
        ``"live"`` when stderr is a TTY, ``"plain"`` otherwise.
    """
    return "live" if is_tty(sys.stderr) else "plain"


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
        metavar="<int>",
        help="paired samples per target",
    ),
]
TimeoutOption = Annotated[
    int | None,
    typer.Option(
        "--timeout",
        "-t",
        parser=parse_positive_integer_up_to(MAX_TIMEOUT_SECONDS),
        metavar="<int>",
        help="timeout in seconds",
    ),
]
ConfigOption = Annotated[str | None, typer.Option("--config", "-c", help="configuration file path")]
ColorOption = Annotated[
    bool | None, typer.Option("--color/--no-color", help="force or suppress ANSI styles")
]
FormatOption = Annotated[OutputFormat, typer.Option("--format", help="output format")]
DebugOption = Annotated[bool, typer.Option("--debug", "-d", help="show stack traces on errors")]
RecordOption = Annotated[
    bool,
    typer.Option("--record", "-r", help="append the run to the session log as a baseline"),
]
BranchOption = Annotated[
    str | None,
    typer.Option("--branch", help="branch to point at the squash commit (default: <branch>-final)"),
]
BaselineOption = Annotated[
    str | None,
    typer.Option(
        "--baseline",
        metavar="<ref>",
        help=(
            "git ref that pins a freshly opened session; "
            "defaults to HEAD and is ignored when a session is resumed"
        ),
    ),
]
ForceOption = Annotated[bool, typer.Option("--force", "-f", help="skip the confirmation prompt")]
AllowUnimprovedOption = Annotated[
    bool,
    typer.Option(
        "--allow-unimproved",
        help="keep the edit even when the iteration was not improved",
    ),
]
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
    return ProgressReporter(
        mode,
        console,
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


def budget_summary_of(budget: Budget, current_ms: float) -> BudgetSummary:
    """The JSON ``budget`` field for an active budget at ``current_ms``."""
    return BudgetSummary(
        cap_minutes=budget.max_minutes,
        remaining_seconds=int(budget.remaining_ms(current_ms) // 1000),
    )


def budget_snapshot(root: str) -> tuple[str, BudgetSummary | None]:
    """Read the live budget at *root* and return the text trailer and JSON summary.

    Args:
        root: The repository root whose session budget to read.

    Returns:
        The text trailer and JSON summary, or ``("", None)`` when no budget
        is active.
    """
    current = _clock.now_ms()
    budget = read_budget(root, now_ms=current)
    if budget is None:
        return "", None
    return "\n" + format_budget_trailer(budget, current), budget_summary_of(budget, current)


def write_budget_report(
    root: str,
    *,
    use_json: bool,
    render_json: Callable[[BudgetSummary | None], str],
    text_report: str,
) -> None:
    """Render the JSON or text report from a single budget read, then write it once.

    Args:
        root: The repository root whose session budget to read.
        use_json: When true, delegate to *render_json*; otherwise concatenate
            *text_report* with the budget trailer.
        render_json: Callable that turns an optional ``BudgetSummary`` into a
            complete JSON string.
        text_report: Pre-rendered text body used in plain-text mode.
    """
    trailer, summary = budget_snapshot(root)
    report = render_json(summary) if use_json else text_report + trailer
    write_and_flush(sys.stdout, report + "\n")


def _repo_root_or_none() -> str | None:
    """The current repository root, or ``None`` outside a git repo or on a read failure."""
    try:
        return repo_root()
    except (GymratError, OSError):
        return None


def budget_for_report() -> tuple[str, BudgetSummary | None]:
    """Read the live budget rooted at the current repository.

    Returns:
        The text trailer and JSON summary, or ``("", None)`` outside a git
        repository or when no budget is active.
    """
    root = _repo_root_or_none()
    if root is None:
        return "", None
    return budget_snapshot(root)


def warn_duration_over_budget(*, halve: bool) -> None:
    """Warn on stderr when the estimated duration would outlast the budget.

    Nothing is written when the budget or the estimate is unknown.

    Args:
        halve: When ``True``, check half the last iterate estimate — one side,
            the shape ``measure`` runs — and name that per-side figure on its
            own. When ``False``, check the full estimate — both sides, the
            shape ``compare`` runs — and lead with the full cost, keeping the
            per-side figure in parentheses so a per-side number that still
            fits does not read as if nothing were wrong.
    """
    root = _repo_root_or_none()
    if root is None:
        return
    current = _clock.now_ms()
    budget = read_budget(root, now_ms=current)
    if budget is None:
        return
    try:
        records = read_records(session_jsonl_path(root))
    except (GymratError, OSError):
        return
    estimate = estimate_iterate_duration(records)
    if estimate is None:
        return
    per_side_ms = estimate.duration_ms / SIDES_PER_ITERATE
    threshold_ms = per_side_ms if halve else estimate.duration_ms
    remaining = budget.remaining_ms(current)
    if threshold_ms > remaining:
        per_side = format_duration(per_side_ms)
        cost = (
            f"{per_side} per side"
            if halve
            else f"{format_duration(estimate.duration_ms)} ({per_side} per side)"
        )
        left = format_duration(remaining)
        warn_to_stderr(f"warning: {left} left; the last full measurement took at most {cost}")


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
        write_and_flush(sys.stdout, renderers.json(result, budget=budget_summary) + "\n")
        return
    color = resolve_stream_color(render_opts.color, sys.stdout)
    output = renderers.text(result, replace(render_opts, color=color))
    write_and_flush(sys.stdout, output + budget_trailer + "\n")
