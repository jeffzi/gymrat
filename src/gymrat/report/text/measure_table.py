"""The measured-value columns a measurement and a probe report share.

A probe report is a measurement report of the experiment worktree plus a
reference and a delta column, so both tables start with the same metric-name and
measured-value columns.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from rich.text import Text

from gymrat.report.format import format_metric_cell_parts
from gymrat.report.table.cells import (
    header_metric_cell,
    indented_section_label,
    variant_name_cell,
)

if TYPE_CHECKING:
    from gymrat.model import ResolvedMetricMeta
    from gymrat.report.format import MetricCellParts
    from gymrat.report.table.render import NamedRow, TableSkeleton


class MeasuredMetric(Protocol):
    """A metric reading with a median, a spread, and the metadata that shaped it."""

    @property
    def median(self) -> float | None:
        """The metric's median measurement, or ``None`` when none reported."""

    @property
    def spread(self) -> float | None:
        """The half-range around the median, or ``None``."""

    @property
    def meta(self) -> ResolvedMetricMeta:
        """The metadata that shaped the reading."""


@dataclass(frozen=True, slots=True)
class MeasuredRow:
    """One measured metric's name, its section label, and its padded value fields.

    Attributes:
        name: The metric's bare name.
        label: The metric's section label, indented under its group.
        value: The measured median and spread, split into padded fields.
    """

    name: str
    label: str
    value: MetricCellParts


def measured_row(name: str, group: str | None, metric: MeasuredMetric) -> MeasuredRow:
    """The name, label, and value cells one measured metric draws as.

    Args:
        name: The metric's bare name.
        group: The group the metric sits under, or ``None`` when ungrouped.
        metric: The metric's reading.

    Returns:
        The row's name, section label, and value cell parts.
    """
    return MeasuredRow(
        name=name,
        label=indented_section_label(metric.meta.short_name, group),
        value=format_metric_cell_parts(metric.median, metric.spread, metric.meta.unit),
    )


def measured_header_cells(title: str | None, label: str) -> tuple[Text, Text]:
    """The metric-name and measured-value header cells.

    Args:
        title: The section title the header row carries, or ``None``.
        label: The target's display label heading the value column.

    Returns:
        The metric header cell and the styled target label.
    """
    return header_metric_cell(title), variant_name_cell(label)


def measured_cells[Row: NamedRow](skeleton: TableSkeleton[Row], row: Row) -> tuple[Text, Text]:
    """The metric-name and measured-value cells of one metric row, as literal text.

    Args:
        skeleton: The table skeleton that settled the name and value cells.
        row: The metric row to draw.

    Returns:
        The name cell and value cell.
    """
    return Text(skeleton.name_cell(row)), Text(skeleton.value_cell(row))
