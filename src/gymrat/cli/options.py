"""CLI flag parsers and the option declarations every command shares.

The parsers turn raw flag and positional text into typed values, raising
:class:`typer.BadParameter` so typer reports a bad value as a usage error. The
``*Option`` aliases declare each shared flag once, so every command that takes
it carries an identical surface.
"""

import math
import re
from enum import StrEnum
from typing import Annotated

import typer

from gymrat.config import (
    MAX_SAFE_INTEGER,
    MAX_TIMEOUT_SECONDS,
    is_positive_integer,
    parse_bounded_positive_int,
)
from gymrat.report.types import FailOnCondition, GeomeanFailOn, RegressedFailOn
from gymrat.sampling import TargetSpec
from gymrat.utils import SECONDS_PER_MINUTE

_POSITIVE_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")
_POSITIVE_NUMBER_MESSAGE = "must be a positive number."
_GEOMEAN_CONDITION_RE = re.compile(r"geomean:(-?\d+(?:\.\d+)?)")


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


class PositionalParamType:
    """Typer parser for ``[label=]<ref|dir>`` positional arguments.

    Wraps :func:`parse_positional` in a callable whose ``__name__`` typer
    renders as the help panel's type column, so it reads ``<ref|dir>`` instead
    of the parser function name. It deliberately has no ``click.ParamType``
    base: typer bundles its own click, so a real-click usage error raised here
    would escape typer's handling and exit 1 instead of 2.
    """

    __name__ = "ref|dir"

    def __call__(self, value: str | TargetSpec) -> TargetSpec:
        """Parse ``value`` into a target, passing an already-parsed one through.

        Args:
            value: The raw positional, or a target typer already converted.

        Returns:
            The parsed label and target.

        Raises:
            typer.BadParameter: When the label or target half is empty.
        """
        if isinstance(value, TargetSpec):
            return value
        return parse_positional(value)


def parse_positive_int(value: str, maximum: int) -> int:
    """Parse a positive integer flag bounded by ``maximum``.

    The value goes through :func:`~gymrat.config.parse_bounded_positive_int`,
    the rule the matching ``GYMRAT_*`` env var applies too.

    Args:
        value: The raw flag value.
        maximum: Largest accepted value.

    Returns:
        The parsed integer.

    Raises:
        typer.BadParameter: When the value is not a bare run of ASCII digits
            naming at least 1, or is above ``maximum``.
    """
    if not is_positive_integer(value):
        message = "must be a positive integer."
        raise typer.BadParameter(message)
    parsed = parse_bounded_positive_int(value, maximum)
    if parsed is None:
        message = f"must be at most {maximum}."
        raise typer.BadParameter(message)
    return parsed


def parse_samples(value: str) -> int:
    """Parse ``--samples``: a positive integer up to :data:`MAX_SAFE_INTEGER`.

    Args:
        value: The raw flag value.

    Returns:
        The parsed sample count.

    Raises:
        typer.BadParameter: When the value is not a positive integer within the ceiling.
    """
    return parse_positive_int(value, MAX_SAFE_INTEGER)


def parse_timeout(value: str) -> int:
    """Parse ``--timeout``: a positive whole number of seconds up to :data:`MAX_TIMEOUT_SECONDS`.

    Args:
        value: The raw flag value.

    Returns:
        The parsed timeout in seconds.

    Raises:
        typer.BadParameter: When the value is not a positive integer within the ceiling.
    """
    return parse_positive_int(value, MAX_TIMEOUT_SECONDS)


def parse_positive_number(value: str) -> float:
    """Parse a strictly positive finite decimal.

    Args:
        value: The raw flag value.

    Returns:
        The parsed number.

    Raises:
        typer.BadParameter: When the value is negative, zero, not finite, or has
            trailing garbage.
    """
    if _POSITIVE_NUMBER_RE.fullmatch(value) is None:
        raise typer.BadParameter(_POSITIVE_NUMBER_MESSAGE)
    parsed = float(value)
    if parsed <= 0 or not math.isfinite(parsed):
        raise typer.BadParameter(_POSITIVE_NUMBER_MESSAGE)
    return parsed


def parse_max_minutes(value: str) -> float:
    """Parse a positive number of minutes bounded by the 32-bit timer ceiling.

    Args:
        value: The raw flag value.

    Returns:
        The parsed number of minutes.

    Raises:
        typer.BadParameter: When the value is not a positive number, or is above
            the ceiling.
    """
    parsed = parse_positive_number(value)
    max_minutes = MAX_TIMEOUT_SECONDS // SECONDS_PER_MINUTE
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
# CLI option declarations
# ---------------------------------------------------------------------------


class OutputFormat(StrEnum):
    """The ``--format`` choices: a human report or a machine-readable document."""

    text = "text"
    json = "json"


# The config-bearing options every command shares, declared once as reusable
# annotations so ``compare`` and ``measure`` carry an identical surface.
BenchOption = Annotated[str | None, typer.Option("--bench", "-b", help="bench command")]
"""--bench/-b: the bench command; None defers to config."""
PrepareOption = Annotated[
    str | None,
    typer.Option("--prepare", "-p", help="preparation script to run before each revision"),
]
"""--prepare/-p: preparation script to run before each revision; None defers to config."""
AdapterOption = Annotated[
    str | None, typer.Option("--adapter", "-a", help="adapter type for parsing benchmark output")
]
"""--adapter/-a: adapter type for parsing benchmark output; None defers to config."""
SamplesOption = Annotated[
    int | None,
    typer.Option(
        "--samples",
        "-s",
        parser=parse_samples,
        metavar="<int>",
        help="paired samples per target",
    ),
]
"""--samples/-s: positive integer parsed by :func:`parse_samples`; None defers to config."""
TimeoutOption = Annotated[
    int | None,
    typer.Option(
        "--timeout",
        "-t",
        parser=parse_timeout,
        metavar="<int>",
        help="timeout in seconds",
    ),
]
"""--timeout/-t: positive integer seconds parsed by :func:`parse_timeout`; None defers to config."""
ConfigOption = Annotated[str | None, typer.Option("--config", "-c", help="configuration file path")]
"""--config/-c: configuration file path."""
ColorOption = Annotated[
    bool | None, typer.Option("--color/--no-color", help="force or suppress ANSI styles")
]
"""--color/--no-color: force or suppress ANSI styles; None when neither is given."""
FormatOption = Annotated[OutputFormat, typer.Option("--format", help="output format")]
"""--format: output format, an :class:`OutputFormat` choice."""
DebugOption = Annotated[bool, typer.Option("--debug", "-d", help="show stack traces on errors")]
"""--debug/-d: show stack traces on errors."""
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
"""--baseline: git ref that pins a freshly opened session; None defaults to HEAD.

Ignored when a session is resumed.
"""
ForceOption = Annotated[bool, typer.Option("--force", "-f", help="skip the confirmation prompt")]
"""--force/-f: skip the confirmation prompt."""
