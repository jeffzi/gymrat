"""Metric name grammar — parse, decompose, and format benchmark metric names.

A metric name follows the grammar ``segment(/segment)*( #kind)?`` where each
segment is a non-empty string of characters other than ``/`` and ``#``, and the
optional kind suffix is separated by exactly one ``#``. Multi-segment paths
expose a *group* (all segments but the last, joined with ``/``) and a *case*
(the last segment); single-segment paths have no group.
"""

from __future__ import annotations

from dataclasses import dataclass

from rich.markup import escape

from gymrat.errors import GymratError


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

        Returns ``None`` for single-segment paths.
        """
        if len(self.path) <= 1:
            return None
        return "/".join(self.path[:-1])

    @property
    def case(self) -> str:
        """Last segment of the path — the leaf benchmark name."""
        return self.path[-1]


def parse(name: str) -> MetricName:
    """Parse a metric name string into a :class:`MetricName`.

    Args:
        name: Raw metric name, e.g. ``"node/access.get_1field#time"``.

    Returns:
        A frozen :class:`MetricName` with path segments and optional kind.

    Raises:
        GymratError: When the name contains more than one ``#``, has an empty
            path segment, or has an empty kind after ``#``.
    """
    path_part, *kind_parts = name.split("#")
    if len(kind_parts) > 1:
        msg = f"metric name contains multiple '#': {name}"
        raise GymratError(msg)

    kind = kind_parts[0] if kind_parts else None
    if kind == "":
        msg = f"metric name has an empty kind after '#': {name}"
        raise GymratError(msg)

    path = tuple(path_part.split("/"))
    if not all(path):
        msg = f"metric name has an empty path segment: {name}"
        raise GymratError(msg)

    return MetricName(path=path, kind=kind)


def format_inline(metric: MetricName) -> str:
    """Format a parsed metric name for inline display.

    The group prefix and kind suffix are wrapped in rich ``[dim]`` markup so
    the case segment stands out.

    Args:
        metric: A parsed :class:`MetricName`.

    Returns:
        The formatted string with rich markup.
    """
    group = metric.group
    prefix = f"[dim]{escape(group)}/[/dim]" if group is not None else ""
    suffix = f"[dim]#{escape(metric.kind)}[/dim]" if metric.kind is not None else ""
    return f"{prefix}{escape(metric.case)}{suffix}"
