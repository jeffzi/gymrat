"""Shared CLI infrastructure: parsing, exit routing, error rendering, locking.

This module holds the pieces every benchmarking command reuses — the argument
coercers, the error formatter and exit path, the repository single-flight lock,
and the render-mode resolution — with no dependency on the heavy statistics
stack or the command bodies, so importing it stays cheap.
"""

import contextlib
import os
import re
import sys
import traceback
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Literal, NoReturn, Protocol

import typer
from rich.markup import escape

from gymrat_py.adapters.types import AdapterError
from gymrat_py.cli.status_line import RenderMode
from gymrat_py.config import MAX_TIMEOUT_SECONDS, CliFlags
from gymrat_py.errors import GymratError, hint_of, message_of
from gymrat_py.git import NotAGitRepositoryError
from gymrat_py.report.style import (
    RENDER_WIDTH,
    format_hint_label,
    highlight_inline_code,
    markup,
    render_lines,
)
from gymrat_py.report.types import FailOnCondition, GeomeanFailOn, RegressedFailOn
from gymrat_py.sampling import TargetSpec
from gymrat_py.session.lock import acquire_lock
from gymrat_py.session.paths import lockfile_path, repo_root

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


def suppress_color() -> None:
    """Veto color unconditionally by clearing ``FORCE_COLOR`` and setting ``NO_COLOR``.

    The style layer resolves ``FORCE_COLOR`` before ``NO_COLOR``, so a leftover
    ``FORCE_COLOR`` from the caller's shell would otherwise defeat ``--no-color``.
    """
    os.environ.pop("FORCE_COLOR", None)
    os.environ["NO_COLOR"] = "1"


def color_override_of(color: bool) -> Literal[False] | None:  # noqa: FBT001 -- 1:1 map of the --color flag
    """Translate the ``--color`` flag into the renderer's color override.

    ``False`` vetoes color; ``None`` defers to auto-detection.
    """
    return None if color else False


def _resolve_stderr_color() -> bool:
    """Resolve whether stderr error output should carry color.

    Mirror the ``styleText(process.stderr)`` precedence: an explicit
    ``FORCE_COLOR`` wins over ``NO_COLOR``, which wins over the stream's own TTY
    detection.
    """
    if os.environ.get("FORCE_COLOR"):
        return True
    if "NO_COLOR" in os.environ:
        return False
    return is_tty(sys.stderr)


# ---------------------------------------------------------------------------
# Error formatting
# ---------------------------------------------------------------------------


def format_cli_error(error: object, *, debug: bool = False) -> str:
    """Render ``error`` for stderr with a red ``Error:`` label and optional sections.

    The sections appear in order: the label, the message body (an
    :class:`AdapterError` keeps its class-name prefix), the stack trace when
    ``debug`` is set, a ``Hint:`` for a :class:`GymratError` that carries one,
    and a report-a-bug footer for errors that are not :class:`GymratError`.
    """
    error_label = f"{markup('Error', 'red')}: "

    if isinstance(error, AdapterError):
        body = f"{type(error).__name__}: {message_of(error)}"
    else:
        body = message_of(error)

    doc = f"{error_label}{escape(body)}"

    if debug and isinstance(error, BaseException) and error.__traceback__ is not None:
        stack = "".join(traceback.format_exception(type(error), error, error.__traceback__))
        doc += f"\n{escape(stack.rstrip())}"

    hint = hint_of(error)
    if hint is not None:
        doc += f"\n{format_hint_label()} {highlight_inline_code(hint)}"

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


# ---------------------------------------------------------------------------
# Flag parsers
# ---------------------------------------------------------------------------


def parse_positional(positional: str) -> TargetSpec:
    """Parse the ``label=target`` syntax of a positional, splitting on the first ``=`` only.

    A target containing its own ``=`` survives intact — ``a=b=c`` parses to label
    ``a``, target ``b=c``. An empty half is always a typo, so each raises its own
    usage error rather than resolving to a silent default.
    """
    eq_index = positional.find("=")
    if eq_index == -1:
        label: str | None = None
        target = positional
    else:
        label = positional[:eq_index]
        target = positional[eq_index + 1 :]

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


def collect_positional(value: str, previous: Sequence[TargetSpec]) -> list[TargetSpec]:
    """Accumulate parsed candidate positionals as the parser walks the variadic argument."""
    return [*previous, parse_positional(value)]


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
    if parsed <= 0:
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


def resolve_render_mode(color_flag: bool) -> RenderMode:  # noqa: FBT001 -- 1:1 map of the --color flag
    """Map the ``--color`` flag to the output strategy the progress reporter uses.

    A vetoed flag suppresses color first; a non-TTY stderr always renders plain;
    a TTY renders a spinner when color is allowed and falls back to in-place
    overwrite otherwise.
    """
    if not color_flag:
        suppress_color()
    if not is_tty(sys.stderr):
        return "plain"
    color_allowed = color_flag and os.environ.get("NO_COLOR") is None
    return "spinner" if color_allowed else "overwrite"


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
    """The measure command's flags: the shared set, with no extra fields."""
