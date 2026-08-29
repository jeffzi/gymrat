"""The sectioned layout a multi-kind run is drawn in.

Sorting a run's metrics into kinds and groups, and reading a candidate's
aggregate back out for each scope, live here rather than inside a renderer: it is
what keeps a row and the geomean closing it describing the same set of metrics. A
comparison and a single-target measurement agree on nothing but their metadata,
so the planner is stated over that alone (:class:`SectionedMetric`) and draws
both in the same sections.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from gymrat.model import GeomeanResult
from gymrat.verdict import infer_group

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from gymrat.config import KindEntry
    from gymrat.model import ResolvedMetricMeta
    from gymrat.report.types import CandidateComparison
    from gymrat.verdict import KindAggregate


class SectionedMetric(Protocol):
    """All a layout needs of a metric entry: the metadata that decides where it lands."""

    @property
    def meta(self) -> ResolvedMetricMeta:
        """The resolved metadata that sorts the metric into a kind and a group."""


@dataclass(slots=True)
class GroupBlock[Row]:
    """A group of one section's metrics, gathered under the prefix they share."""

    group: str
    metrics: list[Row]


@dataclass(slots=True)
class MetricBlock[Row]:
    """A single metric of a section that belongs to no group."""

    metric: Row


type SectionBlock[Row] = GroupBlock[Row] | MetricBlock[Row]


@dataclass(slots=True)
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


def section_label(short_name: str, group: str | None) -> str:
    """A metric's short name as its section shows it, the group prefix stripped when it has one."""
    return short_name if group is None else short_name[len(group) + 1 :]


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
    sections: dict[str, SectionPlan[Row]] = {}
    ordered: list[Row] = []

    for name, metric in metrics.items():
        meta = metric.meta
        section = sections.get(meta.kind)
        if section is None:
            section = SectionPlan(kind=meta.kind, has_gating=False, blocks=[])
            sections[meta.kind] = section
        if meta.gating:
            section.has_gating = True

        group = infer_group(meta.short_name)
        row = measure(name, group, metric)
        ordered.append(row)

        if group is None:
            section.blocks.append(MetricBlock(metric=row))
            continue

        opened = _open_group(section, group)
        if opened is not None:
            opened.metrics.append(row)
        else:
            section.blocks.append(GroupBlock(group=group, metrics=[row]))

    return SectionLayout(sections=tuple(sections.values()), ordered=tuple(ordered))


def _open_group[Row](section: SectionPlan[Row], group: str) -> GroupBlock[Row] | None:
    """The section's already-opened block for ``group``, or ``None`` when none is open."""
    for block in section.blocks:
        if isinstance(block, GroupBlock) and block.group == group:
            return block
    return None


def spans_many_kinds(metrics: Mapping[str, SectionedMetric]) -> bool:
    """Whether the run spans several kinds, and so is reported in sections.

    Read straight off the metrics rather than off a :class:`SectionLayout`, so the
    parts of a report drawn outside the table can ask without building rows they
    have no use for.
    """
    return len({metric.meta.kind for metric in metrics.values()}) > 1


_INFORMATIONAL_TAG = "informational — gating off"


def informational_tag(kind: str, config_kinds: Mapping[str, KindEntry] | None) -> str:
    """The tag a non-gating kind's title carries, naming the config key that decided it.

    Gating is resolved per metric before the report sees it, so only the config
    distinguishes a kind switched off wholesale from one whose metrics were each
    switched off by name. Naming the key is what lets the reader switch it back.
    """
    entry = config_kinds.get(kind) if config_kinds is not None else None
    switched_off = entry is not None and entry.gating is False
    source = f" (config: kinds.{kind}.gating = false)" if switched_off else ""
    return f"{_INFORMATIONAL_TAG}{source}"


# The aggregate stated where a candidate reported none. Every section is drawn from
# the same metadata the aggregates were computed from, so this stands in for
# nothing the renderers can produce — and if one ever does, the row says it
# aggregated nothing rather than inventing a figure.
NO_AGGREGATE: GeomeanResult = GeomeanResult(value=math.nan, n=0, band=0, excluded=())


def _kind_aggregate_of(candidate: CandidateComparison, kind: str) -> KindAggregate | None:
    """The aggregate a candidate reported for one kind, or ``None`` when it reported none."""
    for aggregate in candidate.kinds:
        if aggregate.kind == kind:
            return aggregate
    return None


def kind_geomean_of(candidate: CandidateComparison, kind: str) -> GeomeanResult:
    """The geomean over every metric of ``kind``, gating or not."""
    aggregate = _kind_aggregate_of(candidate, kind)
    return NO_AGGREGATE if aggregate is None else aggregate.geomean


def group_geomean_of(candidate: CandidateComparison, kind: str, group: str) -> GeomeanResult:
    """The geomean over one group of ``kind``'s metrics."""
    aggregate = _kind_aggregate_of(candidate, kind)
    if aggregate is None:
        return NO_AGGREGATE
    for entry in aggregate.groups:
        if entry.group == group:
            return entry.geomean
    return NO_AGGREGATE


def flat_geomean_of(candidate: CandidateComparison) -> GeomeanResult:
    """The geomean a flat table closes on: the single kind's gating metrics.

    A run reporting one kind has no section to name, so its geomean row states
    what the run is judged on without saying which kind that was.
    """
    if not candidate.kinds:
        return NO_AGGREGATE
    gated = candidate.kinds[0].gated_geomean
    return NO_AGGREGATE if gated is None else gated
