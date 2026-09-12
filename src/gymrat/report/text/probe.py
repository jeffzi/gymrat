"""The probe report: what the experiment worktree measures now, beside its baseline.

The probe table is the measurement table plus two columns — the median the newest
baseline record came to for each metric, and the signed percentage between the
two. It stops there. One unpaired run supports no verdict, no geomean and no
significance test, so the table states the gap and leaves the reading to the
reader.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from rich.cells import cell_len
from rich.markup import escape
from rich.text import Text

from gymrat.model import Effect
from gymrat.plural import pluralize
from gymrat.report.format import format_delta, format_metric_cell_parts, is_improvement
from gymrat.report.sections import plan_sections
from gymrat.report.style import (
    RENDER_WIDTH,
    VARIANT_NAME_STYLE,
    VERDICT_STYLES,
    join_header_parts,
    markup,
    render_lines,
    truncate_labels,
)
from gymrat.report.table import (
    METRIC_COLUMN_MIN,
    VALUE_COLUMN_MIN,
    VERDICT_COLUMN_MIN,
    GroupLine,
    HeaderLine,
    MetricLine,
    aggregate_label_lengths,
    compute_column_width,
    group_metric_cell,
    header_metric_cell,
    indented_section_label,
    join_value_cell,
    plan_body,
    render_body,
    section_annotation,
    value_widths,
    widest_header_label,
)
from gymrat.report.types import ReportOptions

if TYPE_CHECKING:
    from collections.abc import Callable

    from gymrat.loop.probe import ProbeMetric, ProbeResult
    from gymrat.model import Direction
    from gymrat.report.format import MetricCellParts
    from gymrat.report.sections import SectionLayout
    from gymrat.report.table import BodyLine

# The default presentation flags: detect color, no header override. Immutable, so
# one shared instance is safe as a default argument.
_DEFAULT_OPTIONS = ReportOptions()

# The header the column of baseline medians carries.
_REFERENCE_COLUMN_HEADER = "baseline"

# The header the column of signed percentages carries.
_DELTA_COLUMN_HEADER = "delta"

# What the delta column states for a metric the baseline never reported. There is
# no gap to state, and a blank cell would read as a gap of zero.
NO_REFERENCE = "no reference"

# Deltas the report refuses to paint: nothing measured, or a figure that rounds to
# zero and so points in no direction to call good or bad.
_UNPAINTED_DELTAS = frozenset({"", "0.0%"})


@dataclass(frozen=True, slots=True)
class _ProbeRow:
    """One probed metric's name, its section label, and the three cells it states."""

    name: str
    label: str
    value: MetricCellParts
    reference: MetricCellParts
    delta: str
    delta_style: str | None


@dataclass(frozen=True, slots=True)
class _ProbeCells:
    """The row-to-cell-text callables the probe table's width and body passes share."""

    name: Callable[[_ProbeRow], str]
    value: Callable[[_ProbeRow], str]
    reference: Callable[[_ProbeRow], str]


def _delta_style(delta: str, delta_pct: float | None, direction: Direction) -> str | None:
    """The style a rendered delta wears, or ``None`` where it points nowhere.

    Args:
        delta: The delta as :func:`~gymrat.report.format.format_delta` rendered
            it, so the paint agrees with the digits on screen.
        delta_pct: The signed percentage behind that text, or ``None`` when the
            baseline had nothing to pair the metric with.
        direction: Whether the metric is better lower or better higher.

    Returns:
        The improved or regressed style, or ``None`` for a delta with no
        reference behind it, none to state, or one that rounds to zero.
    """
    if delta_pct is None or delta in _UNPAINTED_DELTAS:
        return None
    # `is_improvement` reads a percentage as lower-is-better; a higher-is-better
    # metric improves on the opposite sign, so the two agree exactly when the
    # metric is itself lower-is-better.
    improved = is_improvement(Effect(value=delta_pct, unit="percent"))
    return VERDICT_STYLES["improved" if improved == (direction == "lower") else "regressed"]


def _probe_row(name: str, group: str | None, metric: ProbeMetric) -> _ProbeRow:
    """The row one probed metric draws as."""
    delta = (
        NO_REFERENCE
        if metric.delta_pct is None
        else format_delta(Effect(value=metric.delta_pct, unit="percent"))
    )
    return _ProbeRow(
        name=name,
        label=indented_section_label(metric.meta.short_name, group),
        value=format_metric_cell_parts(metric.median, metric.spread, metric.meta.unit),
        reference=format_metric_cell_parts(metric.reference_median, None, metric.meta.unit),
        delta=delta,
        delta_style=_delta_style(delta, metric.delta_pct, metric.meta.direction),
    )


def _probe_header(result: ProbeResult, label: str) -> str:
    """The probe report's run header as markup: the worktree, the run, and any scope."""
    parts = [
        markup("gymrat probe", "bold"),
        markup(label, VARIANT_NAME_STYLE),
        escape(pluralize(result.samples, "sample")),
        f"adapter: {escape(result.adapter)}",
    ]
    if result.scoped:
        parts.append(f"scoped: {escape(', '.join(result.names))}")
    return join_header_parts(parts)


def _probe_column_widths(
    layout: SectionLayout[_ProbeRow],
    body: list[BodyLine[_ProbeRow, object]],
    label: str,
    cells: _ProbeCells,
) -> list[int]:
    """The metric, value, baseline, and delta column widths for the probe table."""
    return [
        compute_column_width(
            cell_len(widest_header_label(body)),
            [cell_len(cells.name(row)) for row in layout.ordered] + aggregate_label_lengths(body),
            METRIC_COLUMN_MIN,
        ),
        compute_column_width(
            cell_len(label),
            [cell_len(cells.value(row)) for row in layout.ordered],
            VALUE_COLUMN_MIN,
        ),
        compute_column_width(
            cell_len(_REFERENCE_COLUMN_HEADER),
            [cell_len(cells.reference(row)) for row in layout.ordered],
            VALUE_COLUMN_MIN,
        ),
        compute_column_width(
            cell_len(_DELTA_COLUMN_HEADER),
            [cell_len(row.delta) for row in layout.ordered],
            VERDICT_COLUMN_MIN,
        ),
    ]


def _render_probe_table(result: ProbeResult, label: str, *, color: bool | None) -> list[str]:
    """Render the metric rows: measured value, baseline value, and the delta between them."""
    layout = plan_sections(
        {metric.name: metric for metric in result.metrics},
        _probe_row,
    )
    value_fields = value_widths([row.value for row in layout.ordered])
    reference_fields = value_widths([row.reference for row in layout.ordered])

    body: list[BodyLine[_ProbeRow, object]] = plan_body(
        layout,
        None,
        lambda section: section_annotation(section, None),
    )
    grouped = len(layout.sections) > 1 or any(isinstance(line, GroupLine) for line in body)

    def name_cell(row: _ProbeRow) -> str:
        return row.label if grouped else row.name

    def value_cell(row: _ProbeRow) -> str:
        return join_value_cell(row.value, value_fields)

    def reference_cell(row: _ProbeRow) -> str:
        return join_value_cell(row.reference, reference_fields)

    cells = _ProbeCells(name=name_cell, value=value_cell, reference=reference_cell)
    widths = _probe_column_widths(layout, body, label, cells)

    def delta_cell(row: _ProbeRow) -> str:
        return escape(row.delta) if row.delta_style is None else markup(row.delta, row.delta_style)

    def to_cells(line: BodyLine[_ProbeRow, object]) -> tuple[str, ...]:
        if isinstance(line, HeaderLine):
            return (
                header_metric_cell(line.title),
                markup(label, VARIANT_NAME_STYLE),
                escape(_REFERENCE_COLUMN_HEADER),
                escape(_DELTA_COLUMN_HEADER),
            )
        if isinstance(line, GroupLine):
            return (group_metric_cell(line.label), "", "", "")
        if isinstance(line, MetricLine):
            return (
                escape(name_cell(line.row)),
                escape(value_cell(line.row)),
                escape(reference_cell(line.row)),
                delta_cell(line.row),
            )
        msg = f"unexpected body line {line!r}"
        raise AssertionError(msg)

    return render_body(body, widths, to_cells, color=color)


def render_probe_report(result: ProbeResult, options: ReportOptions = _DEFAULT_OPTIONS) -> str:
    """Render a probe as the run header followed by the probe table.

    Args:
        result: The probe to draw.
        options: The presentation flags. ``options.color`` forces color on or off,
            or defers to the environment when ``None``.

    Returns:
        The rendered report.
    """
    color = options.color
    label = truncate_labels([result.label])[0]
    header = render_lines(
        Text.from_markup(_probe_header(result, label)), color=color, width=RENDER_WIDTH
    )
    return "\n".join([header, *_render_probe_table(result, label, color=color)])
