"""The single-target measurement table: median and spread for each metric, no verdicts.

There is nothing to judge a measurement against, so the table carries no deltas
and no geomeans — it states what the target reported and how steady it was, laid
out in the same sections a comparison uses.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from rich.markup import escape

from gymrat.report.format import format_metric_cell_parts
from gymrat.report.sections import plan_sections
from gymrat.report.style import VARIANT_NAME_STYLE, markup
from gymrat.report.table import (
    build_cell_dispatcher,
    group_metric_cell,
    header_metric_cell,
    indented_section_label,
    plan_table_skeleton,
    render_body,
)

if TYPE_CHECKING:
    from gymrat.report.format import MetricCellParts
    from gymrat.report.types import MeasurementResult


@dataclass(frozen=True, slots=True)
class _MeasureRow:
    """One measured metric's name, its section label, and its padded value fields."""

    name: str
    label: str
    value: MetricCellParts


def render_measure_table(
    result: MeasurementResult,
    label: str,
    *,
    color: bool | None,
) -> list[str]:
    """Render a single-revision measurement table with a median and spread per metric.

    Args:
        result: The measurement to draw.
        label: The target's display label, already truncated, heading the value
            column.
        color: The explicit color choice, or ``None`` to defer to the environment.

    Returns:
        The rendered table lines.
    """
    layout = plan_sections(
        result.metrics,
        lambda name, group, metric: _MeasureRow(
            name=name,
            label=indented_section_label(metric.meta.short_name, group),
            value=format_metric_cell_parts(metric.median, metric.spread, metric.meta.unit),
        ),
    )
    skeleton = plan_table_skeleton(layout, result.config_kinds, lambda row: row.value, label)
    widths = [skeleton.metric_width, skeleton.value_width]

    def metric_cells(row: _MeasureRow) -> tuple[str, str]:
        return escape(skeleton.name_cell(row)), escape(skeleton.value_cell(row))

    to_cells = build_cell_dispatcher(
        header=lambda title: (header_metric_cell(title), markup(label, VARIANT_NAME_STYLE)),
        group=lambda group_label: (group_metric_cell(group_label), ""),
        metric=metric_cells,
    )

    return render_body(skeleton.body, widths, to_cells, color=color)
