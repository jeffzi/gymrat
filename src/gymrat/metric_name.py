"""Metric name grammar — parse, decompose, and format benchmark metric names.

A metric name follows the grammar ``segment(/segment)*( #kind)?`` where each
segment is a non-empty string of characters other than ``/`` and ``#``, and the
optional kind suffix is separated by exactly one ``#``. Multi-segment paths
expose a *group* (all segments but the last, joined with ``/``) and a *case*
(the last segment); single-segment paths have no group.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import ClassVar

from rich.markup import RE_TAGS, escape

from gymrat.errors import GymratError

LINE_TERMINATORS = re.compile("[\\n\\v\\f\\r\\x1c-\\x1e\\x85\\u2028\\u2029]")
"""Characters a metric name may not carry: every boundary :meth:`str.splitlines` breaks on.

That is LF, VT (U+000B), FF (U+000C), CR, U+001C to U+001E, U+0085, U+2028, and
U+2029. A name holding one would split across lines in any message or record line
that names it, and LF, CR, U+2028, and U+2029 are line terminators to a JavaScript
regular-expression engine, so an anchored name check on the session record could
never match such a name — gymrat must never write a record it cannot read back.
Written as escapes so the source stays plain ASCII.
"""


@dataclass(frozen=True, slots=True)
class MetricName:
    """Parsed representation of a benchmark metric name.

    Attributes:
        path: Non-empty tuple of path segments split on ``/``.
        kind: The kind suffix after ``#``, or ``None`` when absent.
    """

    path: tuple[str, ...]
    kind: str | None

    @property
    def group(self) -> str | None:
        """Path prefix minus the last segment, joined with ``/``.

        Returns:
            The group string, or ``None`` for single-segment paths.
        """
        if len(self.path) <= 1:
            return None
        return "/".join(self.path[:-1])

    @property
    def case(self) -> str:
        """Last segment of the path — the leaf benchmark name."""
        return self.path[-1]


class MetricNameError(GymratError):
    """A metric name that does not follow the grammar.

    :func:`parse` raises one subclass per rule of the grammar, so a caller that
    treats the rules differently catches the subclass it singles out.

    Attributes:
        flaw: The rule the name breaks, as a noun phrase that completes "a metric
            name with ...".
    """

    flaw: ClassVar[str]


class MultipleHashesError(MetricNameError):
    """A metric name carrying more than one ``#``."""

    flaw = "more than one '#'"


class EmptyKindError(MetricNameError):
    """A metric name whose ``#`` is followed by nothing."""

    flaw = "an empty kind"


class EmptyPathSegmentError(MetricNameError):
    """A metric name with nothing between two ``/``, or at either end of its path."""

    flaw = "an empty path segment"


def parse(name: str) -> MetricName:
    """Parse a metric name string into a :class:`MetricName`.

    Args:
        name: Raw metric name, e.g. ``"node/access.get_1field#time"``.

    Returns:
        A frozen :class:`MetricName` with path segments and optional kind.

    Raises:
        MultipleHashesError: When the name contains more than one ``#``.
        EmptyKindError: When nothing follows the ``#``.
        EmptyPathSegmentError: When a path segment is empty.
    """
    path_part, *kind_parts = name.split("#")
    if len(kind_parts) > 1:
        msg = f"metric name contains multiple '#': {name}"
        raise MultipleHashesError(msg)

    kind = kind_parts[0] if kind_parts else None
    if kind == "":
        msg = f"metric name has an empty kind after '#': {name}"
        raise EmptyKindError(msg)

    path = tuple(path_part.split("/"))
    if not all(path):
        msg = f"metric name has an empty path segment: {name}"
        raise EmptyPathSegmentError(msg)

    return MetricName(path=path, kind=kind)


_BACKSLASHES_BEFORE_BRACKET = re.compile(r"\\+(?=\[)")


def _guard_plain_bracket(match: re.Match[str]) -> str:
    # rich drops one backslash from every ``\[`` it finds in plain text, and
    # ``rich.markup.escape`` only rewrites a bracket that opens a tag, so a
    # backslash run before any other bracket gets the backslash rich will drop.
    run = match.group()
    opens_tag = RE_TAGS.match(match.string, match.start()) is not None
    return run if opens_tag else f"{run}\\"


def _escape(text: str, *, before_tag: bool) -> str:
    # rich reads a backslash as an escape only when a tag follows it, so trailing
    # backslashes are doubled exactly when the caller writes a tag right after
    # ``text``. ``rich.markup.escape`` cannot know what follows: it doubles a single
    # trailing backslash and leaves a longer run alone, so the run is set aside and
    # only the rest, which no longer ends in a backslash, goes through it.
    body = text.rstrip("\\")
    trailing = text[len(body) :]
    guarded = _BACKSLASHES_BEFORE_BRACKET.sub(_guard_plain_bracket, body)
    return escape(guarded) + (trailing * 2 if before_tag else trailing)


def format_inline(metric: MetricName) -> str:
    """Format a parsed metric name for inline display with rich dim markup.

    The group and the kind are dimmed around the case. Three things in the name
    are escaped so the markup renders the name exactly as it is written: a square
    bracket that would open a tag, a backslash directly before a square bracket,
    and a backslash directly before one of the dim tags.

    Args:
        metric: The parsed metric name.

    Returns:
        Rich markup that renders as the metric name.
    """
    group, kind = metric.group, metric.kind
    prefix = f"[dim]{_escape(group, before_tag=False)}/[/dim]" if group is not None else ""
    suffix = f"[dim]#{_escape(kind, before_tag=True)}[/dim]" if kind is not None else ""
    return f"{prefix}{_escape(metric.case, before_tag=kind is not None)}{suffix}"
