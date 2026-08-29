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
from gymrat.report.style import GROUP_LABEL_STYLE, VARIANT_NAME_STYLE
from gymrat.report.table import (
    METRIC_COLUMN_HEADER,
    METRIC_COLUMN_MIN,
    VALUE_COLUMN_MIN,
    GroupLine,
    HeaderLine,
    MetricLine,
    aggregate_label_lengths,
    compute_column_width,
    indented_section_label,
    join_value_cell,
    markup,
    plan_body,
    render_body,
    section_annotation,
    value_widths,
    widest_header_label,
)

if TYPE_CHECKING:
    from gymrat.report.format import MetricCellParts
    from gymrat.report.table import BodyLine
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
    sectioned = len(layout.sections) > 1
    value_fields = value_widths([row.value for row in layout.ordered])

    body: list[BodyLine[_MeasureRow, object]] = plan_body(
        layout,
        None,
        lambda section: section_annotation(section, result.config_kinds),
    )

    def value_cell(row: _MeasureRow) -> str:
        return join_value_cell(row.value, value_fields)

    def name_cell(row: _MeasureRow) -> str:
        return row.label if sectioned else row.name

    widths = [
        compute_column_width(
            len(widest_header_label(body)),
            [len(name_cell(row)) for row in layout.ordered] + aggregate_label_lengths(body),
            METRIC_COLUMN_MIN,
        ),
        compute_column_width(
            len(label),
            [len(value_cell(row)) for row in layout.ordered],
            VALUE_COLUMN_MIN,
        ),
    ]

    def to_cells(line: BodyLine[_MeasureRow, object]) -> tuple[str, ...]:
        if isinstance(line, HeaderLine):
            title = line.title
            metric_cell = (
                markup(title, "bold") if title is not None else escape(METRIC_COLUMN_HEADER)
            )
            return (metric_cell, markup(label, VARIANT_NAME_STYLE))
        if isinstance(line, GroupLine):
            return (markup(line.label, GROUP_LABEL_STYLE), "")
        if isinstance(line, MetricLine):
            return (escape(name_cell(line.row)), escape(value_cell(line.row)))
        # A measurement plans no aggregate rows, so no other content line reaches here.
        msg = f"unexpected body line {line!r}"
        raise AssertionError(msg)

    return render_body(body, widths, to_cells, color=color)
