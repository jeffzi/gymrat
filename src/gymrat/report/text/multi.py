"""The multi-candidate comparison table: one baseline against two or more candidates.

Each metric row states the baseline's figure once and pairs every candidate's own
figure with the verdict between it and the baseline; each candidate column sizes
its value and verdict fields on its own cells, so a candidate's figures align
within its column without dragging the others wider. The aggregate rows state one
geomean per candidate column, each labelled once in the name column. The grid
around the cells — the ``│`` separators, the ``─`` rules and their junctions — is
drawn by :mod:`gymrat.report.table`, so the columns line up across every
section.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from rich.cells import cell_len
from rich.text import Text

from gymrat.report.style import SCOPE_SEPARATOR, VERDICT_STYLES
from gymrat.report.table.cells import (
    CELL_GUTTER,
    GEOMEAN_LABEL,
    NO_GEOMEAN_FIGURE,
    NO_GEOMEAN_FIGURE_STYLE,
    NO_STABLE_METRICS,
    NO_STABLE_METRICS_STYLE,
    aggregate_label_cell,
    geomean_parts,
    geomean_scope_label,
    geomean_value_style,
    group_metric_cell,
    header_metric_cell,
    variant_name_cell,
    verdict_cell,
    verdict_widths,
)
from gymrat.report.table.render import (
    VALUE_COLUMN_MIN,
    AggregateLine,
    AggregateRows,
    build_cell_dispatcher,
    compute_column_width,
    is_grouped,
    join_value_cell,
    metric_column_width,
    name_cell,
    plan_body,
    plan_sections,
    render_body,
    section_annotation,
    value_widths,
)
from gymrat.report.text.comparison_rows import (
    CandidateCell,
    ComparisonRow,
    ScopeAggregate,
    comparison_aggregate_rows,
    comparison_row_builder,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from gymrat.model import GeomeanResult
    from gymrat.report.display import DisplayClass
    from gymrat.report.table.cells import VerdictWidths
    from gymrat.report.table.render import BodyLine, ValueWidths
    from gymrat.report.types import (
        CandidateComparison,
        ComparisonResult,
    )

type _AggregateCells = tuple[Text, ...]
type _MetricCells = tuple[Text, ...]

# The name and baseline columns precede a table's candidate columns.
_LEADING_COLUMNS = 2


@dataclass(frozen=True, slots=True)
class _ColumnFields:
    """Pre-measured field widths for each candidate column and the baseline."""

    values: list[ValueWidths]
    verdicts: list[VerdictWidths]
    baseline: ValueWidths


def _measure_columns(
    ordered: Sequence[ComparisonRow],
    candidate_count: int,
) -> _ColumnFields:
    """Measure each candidate column's value and verdict widths, plus the baseline's."""
    col_values = [
        value_widths([row.candidates[index].value for row in ordered])
        for index in range(candidate_count)
    ]
    col_verdicts = [
        verdict_widths([
            v.parts for row in ordered if (v := row.candidates[index].verdict) is not None
        ])
        for index in range(candidate_count)
    ]
    baseline = value_widths([row.baseline for row in ordered])
    return _ColumnFields(values=col_values, verdicts=col_verdicts, baseline=baseline)


def _column_widths(
    body: Sequence[BodyLine[ComparisonRow, _AggregateCells]],
    cells_by_name: dict[str, _MetricCells],
    baseline_header: str,
    candidates: Sequence[CandidateComparison],
) -> list[int]:
    """The column widths, measured over the rows, headers and aggregates' plain text."""
    rows = list(cells_by_name.values())
    aggregate_lines = [line for line in body if isinstance(line, AggregateLine)]
    metric_width = metric_column_width(body, [row[0].cell_len for row in rows])
    baseline_width = compute_column_width(
        cell_len(baseline_header), [row[1].cell_len for row in rows], VALUE_COLUMN_MIN
    )
    candidate_widths = [
        compute_column_width(
            cell_len(candidate.label),
            [row[_LEADING_COLUMNS + index].cell_len for row in rows]
            + [line.cell[index].cell_len for line in aggregate_lines],
            VALUE_COLUMN_MIN,
        )
        for index, candidate in enumerate(candidates)
    ]
    return [metric_width, baseline_width, *candidate_widths]


def _metric_cells(row: ComparisonRow, fields: _ColumnFields, *, grouped: bool) -> _MetricCells:
    """One metric row's cells: its name, the baseline figure, and each candidate's side."""
    return (
        Text(name_cell(row, grouped=grouped)),
        Text(join_value_cell(row.baseline, fields.baseline)),
        *(
            _candidate_cell(cell, fields.values[index], fields.verdicts[index])
            for index, cell in enumerate(row.candidates)
        ),
    )


def render_comparison_table(result: ComparisonResult, *, color: bool | None) -> list[str]:
    """Render a multi-candidate comparison table (one baseline vs. two or more candidates).

    Args:
        result: The comparison to draw, its candidate labels already shortened.
        color: The explicit color choice, or ``None`` to defer to the environment.

    Returns:
        The rendered table lines.
    """
    baseline_header = result.baseline_label
    candidates = result.candidates

    layout = plan_sections(
        result.metrics,
        comparison_row_builder(len(candidates), result.samples, with_band=False),
    )
    fields = _measure_columns(layout.ordered, len(candidates))

    aggregates = _aggregate_rows(candidates)
    body: list[BodyLine[ComparisonRow, _AggregateCells]] = plan_body(
        layout,
        aggregates,
        lambda section: section_annotation(section, result.config_kinds),
    )
    grouped = is_grouped(layout, body)
    cells_by_name = {
        row.name: _metric_cells(row, fields, grouped=grouped) for row in layout.ordered
    }
    widths = _column_widths(body, cells_by_name, baseline_header, candidates)

    def cells_of(row: ComparisonRow) -> _MetricCells:
        return cells_by_name[row.name]

    to_cells = build_cell_dispatcher(
        header=lambda title: (
            header_metric_cell(title),
            variant_name_cell(baseline_header),
            *(variant_name_cell(candidate.label) for candidate in candidates),
        ),
        group=lambda label: (group_metric_cell(label), "", *("" for _ in candidates)),
        metric=cells_of,
        aggregate=lambda line: (aggregate_label_cell(line.label), "", *line.cell),
    )

    return render_body(body, widths, to_cells, color=color)


def _aggregate_rows(
    candidates: Sequence[CandidateComparison],
) -> AggregateRows[ComparisonRow, _AggregateCells]:
    """The per-candidate geomean builders, one aggregate cell per candidate column."""

    def line(scope: str | None, aggregates: list[ScopeAggregate]) -> AggregateLine[_AggregateCells]:
        return AggregateLine(
            label=GEOMEAN_LABEL if scope is None else geomean_scope_label(scope),
            cell=tuple(
                geomean_column_cell(aggregate.geomean, aggregate.outcomes)
                for aggregate in aggregates
            ),
        )

    return comparison_aggregate_rows(candidates, line)


def geomean_column_cell(
    geomean: GeomeanResult,
    outcomes: Sequence[DisplayClass | None],
) -> Text:
    """The geomean of one candidate column: the aggregate, then how many metrics back it.

    The multi-candidate table names the scope once in its label column and states
    each candidate's own figure and count in the candidate columns, so this builds
    one column's cell.

    Args:
        geomean: The candidate's aggregate over the scope's metrics.
        outcomes: The display class of each metric behind the figure, for vetoing
            the figure's color when every one is quiet.

    Returns:
        The styled cell: the delta by
        :func:`geomean_value_style`, the provenance
        dimmed. An empty geomean shows the ``no stable metrics`` stand-in rather
        than the ``0.0%`` it computes to.
    """
    parts = geomean_parts(geomean)
    if parts is None:
        return Text.assemble(
            (NO_GEOMEAN_FIGURE, NO_GEOMEAN_FIGURE_STYLE),
            CELL_GUTTER,
            (NO_STABLE_METRICS, NO_STABLE_METRICS_STYLE),
        )
    return Text.assemble(
        (parts.delta, geomean_value_style(geomean, outcomes)),
        f" {SCOPE_SEPARATOR} ",
        (parts.provenance, "dim"),
    )


def _candidate_cell(
    cell: CandidateCell,
    values: ValueWidths,
    verdicts: VerdictWidths,
) -> Text:
    """One candidate cell: the value left plain, the glyph and delta in the verdict color.

    The band is dropped from the multi-candidate cell, so only the glyph and the
    delta (or the ``unstable`` word standing in for it) carry the verdict's color;
    an unstable cell paints them amber, a quiet cell recedes them to dim, and the
    figure itself stays plain whatever the verdict.

    Args:
        cell: The candidate's figure and optional verdict.
        values: The column's value field widths.
        verdicts: The column's verdict field widths.

    Returns:
        The styled cell.
    """
    value = join_value_cell(cell.value, values)
    if cell.verdict is None:
        return Text(value)
    style = VERDICT_STYLES[cell.verdict.outcome]
    verdict = verdict_cell(
        cell.verdict.parts,
        verdicts,
        glyph_style=style,
        delta_style=style,
        band_style=None,
    )
    return Text.assemble(value, CELL_GUTTER, verdict)
