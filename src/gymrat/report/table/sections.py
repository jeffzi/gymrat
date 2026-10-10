"""The sectioned layout a table is drawn in: kinds, and the groups within them.

Sorting a run's metrics into kinds and groups lives here rather than inside a
renderer: it is what keeps a row and the geomean closing it describing the same
set of metrics. A comparison and a single-target measurement agree on nothing but
their metadata, so the planner is stated over that alone (:class:`SectionedMetric`)
and draws both in the same sections.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from gymrat.metric_name import parse as parse_metric_name

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from gymrat.model import ResolvedMetricMeta


class SectionedMetric(Protocol):
    """All a layout needs of a metric entry: the metadata that decides where it lands."""

    @property
    def meta(self) -> ResolvedMetricMeta:
        """The resolved metadata that sorts the metric into a kind and a group."""


@dataclass(frozen=True, slots=True)
class GroupBlock[Row]:
    """A group of one section's metrics, gathered under the prefix they share."""

    group: str
    metrics: list[Row]


@dataclass(frozen=True, slots=True)
class MetricBlock[Row]:
    """A single metric of a section that belongs to no group."""

    metric: Row


#: One block of a section: either a named group or a single ungrouped metric.
type SectionBlock[Row] = GroupBlock[Row] | MetricBlock[Row]


@dataclass(frozen=True, slots=True)
class SectionPlan[Row]:
    """One kind's slice of the table: what it holds, and whether the run is judged on it.

    Attributes:
        kind: The metric kind this section covers.
        has_gating: Whether any of the section's metrics gate the run.
        blocks: The section's groups and standalone metrics, in first-appearance
            order.
    """

    kind: str
    has_gating: bool
    blocks: list[SectionBlock[Row]]


@dataclass(frozen=True, slots=True)
class SectionLayout[Row]:
    """The run's metrics as sections, and as the flat list a single-kind run draws.

    Attributes:
        sections: One plan per kind, in first-appearance order.
        ordered: Every metric in the order the run reported it, whatever section
            it landed in.
    """

    sections: tuple[SectionPlan[Row], ...]
    ordered: tuple[Row, ...]


def plan_sections[Row, Metric: SectionedMetric](
    metrics: Mapping[str, Metric],
    measure: Callable[[str, str | None, Metric], Row],
) -> SectionLayout[Row]:
    """Sort the run's metrics into one section per kind, and each section into its groups.

    Kinds, groups and metrics keep first-appearance order — the order the
    aggregates were computed in — so a section reads in the same order as the rows
    its geomean covers. A group block sits where its first metric appeared and
    gathers the rest of the group with it, rather than letting a metric of another
    group split it.

    Rows are built here rather than looked up later, so every row a section names
    is the row the table draws. ``measure`` receives the inferred group rather
    than a finished label, since what a renderer does with the prefix is its own
    business.

    Args:
        metrics: Every metric of the run, keyed by name, in first-appearance order.
        measure: Builds a row from a metric's name, inferred group, and entry.

    Returns:
        The sectioned layout and the flat ordered rows.
    """
    blocks_by_kind: dict[str, list[SectionBlock[Row]]] = {}
    gating_kinds: set[str] = set()
    ordered: list[Row] = []

    for name, metric in metrics.items():
        meta = metric.meta
        blocks = blocks_by_kind.setdefault(meta.kind, [])
        if meta.gating:
            gating_kinds.add(meta.kind)

        group = parse_metric_name(name).group
        row = measure(name, group, metric)
        ordered.append(row)

        if group is None:
            blocks.append(MetricBlock(metric=row))
            continue

        opened = _open_group(blocks, group)
        if opened is not None:
            opened.metrics.append(row)
        else:
            blocks.append(GroupBlock(group=group, metrics=[row]))

    sections = tuple(
        SectionPlan(kind=kind, has_gating=kind in gating_kinds, blocks=blocks)
        for kind, blocks in blocks_by_kind.items()
    )
    return SectionLayout(sections=sections, ordered=tuple(ordered))


def _open_group[Row](blocks: list[SectionBlock[Row]], group: str) -> GroupBlock[Row] | None:
    """The already-opened block for ``group`` among ``blocks``, or ``None`` when none is open."""
    return next(
        (block for block in blocks if isinstance(block, GroupBlock) and block.group == group),
        None,
    )
