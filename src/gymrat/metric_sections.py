"""Sorting a run's metrics into kinds, and each kind into the groups its names infer.

The comparison table draws one section per kind and one block per group, and the
geomean rows closing those sections and blocks are computed per kind and per
group. Both sides sort through :func:`section_metrics`, so a row and the geomean
that closes it always describe the same set of metrics.
"""

from collections.abc import Iterable

from gymrat.metric_name import parse as parse_metric_name


def section_metrics(kinds: Iterable[tuple[str, str]]) -> dict[str, list[tuple[str, str | None]]]:
    """Sort metrics into their kinds, inferring each one's group from its name.

    Kinds, and the metrics inside each kind, keep first-appearance order.

    Args:
        kinds: Each metric's name and kind, in the order the run reported them.

    Returns:
        Per kind, the ``(name, group)`` of every metric of that kind, the group
        ``None`` for a name whose path has a single segment.
    """
    sections: dict[str, list[tuple[str, str | None]]] = {}
    for name, kind in kinds:
        sections.setdefault(kind, []).append((name, parse_metric_name(name).group))
    return sections
