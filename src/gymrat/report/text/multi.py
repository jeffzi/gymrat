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

from gymrat.report.display import display_class
from gymrat.report.format import baseline_cell_parts, candidate_cell_parts
from gymrat.report.geomean_label import GEOMEAN_LABEL, geomean_scope_label
from gymrat.report.sections import (
    flat_geomean_of,
    group_geomean_of,
    kind_geomean_of,
    plan_sections,
)
from gymrat.report.style import VERDICT_STYLES
from gymrat.report.table import (
    CELL_GUTTER,
    VALUE_COLUMN_MIN,
    AggregateLine,
    AggregateRow,
    AggregateRows,
    GroupLine,
    HeaderLine,
    MetricLine,
    compute_column_width,
    geomean_column_cell,
    group_metric_cell,
    header_metric_cell,
    indented_section_label,
    is_grouped,
    join_value_cell,
    plan_body,
    render_body,
    section_annotation,
    value_widths,
    verdict_cell,
    verdict_parts,
    verdict_widths,
)
from gymrat.report.table.markup import aggregate_label_cell, variant_name_cell
from gymrat.report.table.render import metric_column_width, row_name_cell
from gymrat.report.types import candidate_at

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from gymrat.model import GeomeanResult
    from gymrat.report.display import DisplayClass
    from gymrat.report.format import MetricCellParts
    from gymrat.report.table import (
        BodyLine,
        TableCell,
        ValueWidths,
        VerdictParts,
        VerdictWidths,
    )
    from gymrat.report.types import (
        CandidateComparison,
        ComparisonResult,
        MetricComparison,
    )

type _AggregateCells = tuple[Text, ...]
type _MetricCells = tuple[Text, ...]

# The name and baseline columns precede a table's candidate columns.
_LEADING_COLUMNS = 2


@dataclass(frozen=True, slots=True)
class _CandidateVerdict:
    """A candidate's verdict parts and display outcome, always present together.

    Bundled so ``_CandidateCell.verdict`` being None means both are absent.
    """

    parts: VerdictParts
    outcome: DisplayClass


@dataclass(frozen=True, slots=True)
class _CandidateCell:
    """One candidate's side of a metric row: its figure and optional verdict."""

    value: MetricCellParts
    verdict: _CandidateVerdict | None


@dataclass(frozen=True, slots=True)
class _ComparisonRow:
    """One metric row: its names, the baseline figure, and each candidate's side."""

    name: str
    label: str
    baseline: MetricCellParts
    candidates: tuple[_CandidateCell, ...]
    gating: bool


@dataclass(frozen=True, slots=True)
class _ColumnFields:
    """Pre-measured field widths for each candidate column and the baseline."""

    values: list[ValueWidths]
    verdicts: list[VerdictWidths]
    baseline: ValueWidths


@dataclass(frozen=True, slots=True)
class _TableContext:
    """Shared parameters for column width measurement and cell rendering."""

    baseline_header: str
    candidates: Sequence[CandidateComparison]
    fields: _ColumnFields
    grouped: bool


def _measure_columns(
    ordered: Sequence[_ComparisonRow],
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
    body: Sequence[BodyLine[_ComparisonRow, _AggregateCells]],
    cells_by_name: dict[str, _MetricCells],
    table: _TableContext,
) -> list[int]:
    """The column widths, measured over the rows, headers and aggregates' plain text."""
    rows = list(cells_by_name.values())
    aggregate_lines = [line for line in body if isinstance(line, AggregateLine)]
    metric_width = metric_column_width(body, [row[0].cell_len for row in rows])
    baseline_width = compute_column_width(
        cell_len(table.baseline_header), [row[1].cell_len for row in rows], VALUE_COLUMN_MIN
    )
    candidate_widths = [
        compute_column_width(
            cell_len(candidate.label),
            [row[_LEADING_COLUMNS + index].cell_len for row in rows]
            + [line.cell[index].cell_len for line in aggregate_lines],
            VALUE_COLUMN_MIN,
        )
        for index, candidate in enumerate(table.candidates)
    ]
    return [metric_width, baseline_width, *candidate_widths]


def _metric_cells(row: _ComparisonRow, table: _TableContext) -> _MetricCells:
    """One metric row's cells: its name, the baseline figure, and each candidate's side."""
    return (
        Text(row_name_cell(row, grouped=table.grouped)),
        Text(join_value_cell(row.baseline, table.fields.baseline)),
        *(
            _candidate_cell(cell, table.fields.values[index], table.fields.verdicts[index])
            for index, cell in enumerate(row.candidates)
        ),
    )


def _to_cells(
    line: BodyLine[_ComparisonRow, _AggregateCells],
    table: _TableContext,
    cells_by_name: dict[str, _MetricCells],
) -> tuple[TableCell, ...]:
    """The cells one content row renders to."""
    if isinstance(line, HeaderLine):
        return (
            header_metric_cell(line.title),
            variant_name_cell(table.baseline_header),
            *(variant_name_cell(candidate.label) for candidate in table.candidates),
        )
    if isinstance(line, GroupLine):
        return (group_metric_cell(line.label), "", *("" for _ in table.candidates))
    if isinstance(line, MetricLine):
        return cells_by_name[line.row.name]
    if isinstance(line, AggregateLine):
        return (aggregate_label_cell(line.label), "", *line.cell)
    msg = f"unexpected body line {line!r}"
    raise AssertionError(msg)


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
        lambda name, group, metric: _build_row(
            metric, name, group, len(candidates), result.samples
        ),
    )
    fields = _measure_columns(layout.ordered, len(candidates))

    aggregates = _aggregate_rows(candidates)
    body: list[BodyLine[_ComparisonRow, _AggregateCells]] = plan_body(
        layout,
        aggregates,
        lambda section: section_annotation(section, result.config_kinds),
    )
    table = _TableContext(
        baseline_header=baseline_header,
        candidates=candidates,
        fields=fields,
        grouped=is_grouped(layout, body),
    )
    cells_by_name = {row.name: _metric_cells(row, table) for row in layout.ordered}
    widths = _column_widths(body, cells_by_name, table)

    return render_body(
        body,
        widths,
        lambda line: _to_cells(line, table, cells_by_name),
        color=color,
    )


def _build_row(
    metric: MetricComparison,
    name: str,
    group: str | None,
    candidate_count: int,
    samples: int,
) -> _ComparisonRow:
    """Split one metric into the baseline figure and one cell per candidate column."""
    cells: list[_CandidateCell] = []
    for index in range(candidate_count):
        side = candidate_at(metric, index)
        metric_verdict = side.verdict if side is not None else None
        cell_verdict: _CandidateVerdict | None = None
        if metric_verdict is not None:
            cell_verdict = _CandidateVerdict(
                parts=verdict_parts(metric_verdict, samples, with_band=False),
                outcome=display_class(metric_verdict),
            )
        cells.append(
            _CandidateCell(
                value=candidate_cell_parts(side, metric.meta.unit),
                verdict=cell_verdict,
            )
        )
    return _ComparisonRow(
        name=name,
        label=indented_section_label(metric.meta.short_name, group),
        baseline=baseline_cell_parts(metric),
        candidates=tuple(cells),
        gating=metric.meta.gating,
    )


def _aggregate_rows(
    candidates: Sequence[CandidateComparison],
) -> AggregateRows[_ComparisonRow, _AggregateCells]:
    """The per-candidate geomean builders, one aggregate cell per candidate column."""

    def column_cells(
        geomean_of: Callable[[CandidateComparison], GeomeanResult],
        rows: Sequence[_ComparisonRow],
    ) -> _AggregateCells:
        return tuple(
            geomean_column_cell(
                geomean_of(candidate),
                [
                    v.outcome if (v := row.candidates[index].verdict) is not None else None
                    for row in rows
                ],
            )
            for index, candidate in enumerate(candidates)
        )

    return AggregateRows(
        group=lambda kind, group, rows: AggregateRow(
            label=geomean_scope_label(group),
            cell=column_cells(lambda candidate: group_geomean_of(candidate, kind, group), rows),
        ),
        kind=lambda kind, rows: AggregateRow(
            label=geomean_scope_label(kind),
            cell=column_cells(lambda candidate: kind_geomean_of(candidate, kind), rows),
        ),
        flat=lambda rows: AggregateRow(
            label=GEOMEAN_LABEL,
            cell=column_cells(flat_geomean_of, [row for row in rows if row.gating]),
        ),
    )


def _candidate_cell(
    cell: _CandidateCell,
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
