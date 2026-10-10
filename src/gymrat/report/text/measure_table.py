"""The run header and the measured-value columns a measurement and a probe report share.

A probe report is a measurement report of the experiment worktree plus a
reference and a delta column, so both open on the same run header and both
tables start with the same metric-name and measured-value columns.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from rich.markup import escape

from gymrat.report.format import format_metric_cell_parts
from gymrat.report.style import VARIANT_NAME_STYLE, join_header_parts, markup
from gymrat.report.table.cells import header_metric_cell, indented_section_label
from gymrat.utils import pluralize

if TYPE_CHECKING:
    from collections.abc import Sequence

    from rich.text import Text

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


def run_header(
    command: str, label: str, samples: int, adapter: str, extra: Sequence[str] = ()
) -> str:
    """The single-target run header as markup: the command, the target, and the run.

    Args:
        command: The subcommand the report belongs to, such as ``"measure"``.
        label: The target's display label, already truncated.
        samples: How many samples the run collected.
        adapter: The adapter that parsed the bench output.
        extra: Further markup parts appended after the adapter.

    Returns:
        The header parts joined into one markup line.
    """
    return join_header_parts([
        markup(f"gymrat {command}", "bold"),
        markup(label, VARIANT_NAME_STYLE),
        escape(pluralize(samples, "sample")),
        f"adapter: {escape(adapter)}",
        *extra,
    ])


def measured_header_cells(title: str | None, label: str) -> tuple[Text, str]:
    """The metric-name and measured-value header cells, as markup.

    Args:
        title: The section title the header row carries, or ``None``.
        label: The target's display label heading the value column.

    Returns:
        The metric header cell and the styled target label.
    """
    return header_metric_cell(title), markup(label, VARIANT_NAME_STYLE)


def measured_cells[Row: NamedRow](skeleton: TableSkeleton[Row], row: Row) -> tuple[str, str]:
    """The metric-name and measured-value cells of one metric row, escaped for markup.

    Args:
        skeleton: The table skeleton that settled the name and value cells.
        row: The metric row to draw.

    Returns:
        The escaped name cell and value cell.
    """
    return escape(skeleton.name_cell(row)), escape(skeleton.value_cell(row))
