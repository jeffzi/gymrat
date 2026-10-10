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

from gymrat.model import is_improvement
from gymrat.report.format import (
    ZERO_PERCENT_DELTA,
    format_metric_cell_parts,
    format_percent_delta,
)
from gymrat.report.style import (
    VARIANT_NAME_STYLE,
    VERDICT_STYLES,
    markup,
    render_lines,
    truncate_labels,
)
from gymrat.report.table.cells import VERDICT_COLUMN_MIN, group_metric_cell
from gymrat.report.table.render import (
    VALUE_COLUMN_MIN,
    build_cell_dispatcher,
    compute_column_width,
    join_value_cell,
    plan_sections,
    plan_table_skeleton,
    render_body,
    value_widths,
)
from gymrat.report.text.measure_table import (
    MeasuredRow,
    measured_cells,
    measured_header_cells,
    measured_row,
)
from gymrat.report.text.run_header import run_header
from gymrat.report.types import DEFAULT_REPORT_OPTIONS, ReportOptions
from gymrat.utils import pluralize

if TYPE_CHECKING:
    from gymrat.loop.probe import ProbeMetric, ProbeResult
    from gymrat.model import Direction
    from gymrat.report.format import MetricCellParts

# The header the column of baseline medians carries.
_REFERENCE_COLUMN_HEADER = "baseline"

# The header the column of signed percentages carries.
_DELTA_COLUMN_HEADER = "delta"

# What the delta column states for a metric the baseline never reported. A zero
# reference leaves the cell blank instead: the baseline exists, but no percentage
# of zero is defined.
NO_REFERENCE = "no reference"

# Deltas the report refuses to paint: nothing measured, or a figure that rounds to
# zero and so points in no direction to call good or bad.
_UNPAINTED_DELTAS = frozenset({"", ZERO_PERCENT_DELTA})


@dataclass(frozen=True, slots=True)
class _ProbeRow:
    """One probed metric's measured cells, plus its baseline value and the delta between."""

    measured: MeasuredRow
    reference: MetricCellParts
    delta: str
    delta_style: str | None

    @property
    def name(self) -> str:
        """The metric's bare name."""
        return self.measured.name

    @property
    def label(self) -> str:
        """The metric's section label, indented under its group."""
        return self.measured.label


def _delta_style(delta: str, delta_pct: float | None, direction: Direction) -> str | None:
    """The style a rendered delta wears, or ``None`` where it points nowhere.

    Args:
        delta: The delta as :func:`~gymrat.report.format.format_percent_delta` rendered
            it, so the paint agrees with the digits on screen.
        delta_pct: The signed percentage behind that text, or ``None`` when the
            probe has no delta to state.
        direction: Whether the metric is better lower or better higher.

    Returns:
        The improved or regressed style, or ``None`` for a delta with no
        reference behind it, none to state, or one that rounds to zero.
    """
    if delta_pct is None or delta in _UNPAINTED_DELTAS:
        return None
    return VERDICT_STYLES["improved" if is_improvement(delta_pct, direction) else "regressed"]


def _probe_row(name: str, group: str | None, metric: ProbeMetric) -> _ProbeRow:
    """The row one probed metric draws as."""
    if metric.reference_median is None:
        delta = NO_REFERENCE
    elif metric.delta_pct is None:
        delta = ""
    else:
        delta = format_percent_delta(metric.delta_pct)
    return _ProbeRow(
        measured=measured_row(name, group, metric),
        reference=format_metric_cell_parts(metric.reference_median, None, metric.meta.unit),
        delta=delta,
        delta_style=_delta_style(delta, metric.delta_pct, metric.meta.direction),
    )


def _probe_header(result: ProbeResult, label: str) -> str:
    """The probe report's run header as markup: the worktree, the run, and any scope."""
    scope = [f"scoped: {escape(', '.join(result.names))}"] if result.names else []
    return run_header(
        "probe",
        markup(label, VARIANT_NAME_STYLE),
        pluralize(result.samples, "sample"),
        result.adapter,
        scope,
    )


def _render_probe_table(result: ProbeResult, label: str, *, color: bool | None) -> list[str]:
    """Render the metric rows: measured value, baseline value, and the delta between them."""
    layout = plan_sections(
        {metric.name: metric for metric in result.metrics},
        _probe_row,
    )
    skeleton = plan_table_skeleton(layout, None, lambda row: row.measured.value, label)
    reference_fields = value_widths([row.reference for row in layout.ordered])

    def reference_cell(row: _ProbeRow) -> str:
        return join_value_cell(row.reference, reference_fields)

    widths = [
        skeleton.metric_width,
        skeleton.value_width,
        compute_column_width(
            cell_len(_REFERENCE_COLUMN_HEADER),
            [cell_len(reference_cell(row)) for row in layout.ordered],
            VALUE_COLUMN_MIN,
        ),
        compute_column_width(
            cell_len(_DELTA_COLUMN_HEADER),
            [cell_len(row.delta) for row in layout.ordered],
            VERDICT_COLUMN_MIN,
        ),
    ]

    def metric_cells(row: _ProbeRow) -> tuple[Text, Text, Text, Text]:
        return (
            *measured_cells(skeleton, row),
            Text(reference_cell(row)),
            Text().append(row.delta, row.delta_style),
        )

    to_cells = build_cell_dispatcher(
        header=lambda title: (
            *measured_header_cells(title, label),
            Text(_REFERENCE_COLUMN_HEADER),
            Text(_DELTA_COLUMN_HEADER),
        ),
        group=lambda group_label: (group_metric_cell(group_label), "", "", ""),
        metric=metric_cells,
    )

    return render_body(skeleton.body, widths, to_cells, color=color)


def render_probe_report(
    result: ProbeResult, options: ReportOptions = DEFAULT_REPORT_OPTIONS
) -> str:
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
    header = render_lines(_probe_header(result, label), color=color)
    return "\n".join([header, *_render_probe_table(result, label, color=color)])
