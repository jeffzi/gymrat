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
from rich.markup import escape

from gymrat.report.display import display_class
from gymrat.report.format import baseline_cell_parts, candidate_cell_parts
from gymrat.report.geomean_label import GEOMEAN_LABEL, geomean_scope_label
from gymrat.report.sections import (
    flat_geomean_of,
    group_geomean_of,
    kind_geomean_of,
    plan_sections,
)
from gymrat.report.style import (
    AGGREGATE_LABEL_STYLE,
    VARIANT_NAME_STYLE,
    VERDICT_STYLES,
    markup,
)
from gymrat.report.table import (
    CELL_GUTTER,
    METRIC_COLUMN_MIN,
    VALUE_COLUMN_MIN,
    AggregateColumnCell,
    AggregateLine,
    AggregateRow,
    AggregateRows,
    GroupLine,
    HeaderLine,
    MetricLine,
    aggregate_label_lengths,
    compute_column_width,
    geomean_column_cell,
    group_metric_cell,
    header_metric_cell,
    indented_section_label,
    join_value_cell,
    join_verdict_cell,
    plan_body,
    render_body,
    section_annotation,
    style_verdict_cell,
    value_widths,
    verdict_parts,
    verdict_widths,
    widest_header_label,
)
from gymrat.report.types import candidate_at

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from gymrat.model import GeomeanResult
    from gymrat.report.display import DisplayClass
    from gymrat.report.format import MetricCellParts
    from gymrat.report.table import BodyLine, ValueWidths, VerdictParts, VerdictWidths
    from gymrat.report.types import (
        CandidateComparison,
        ComparisonResult,
        MetricComparison,
    )

type _AggregateCells = tuple[AggregateColumnCell, ...]


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
        verdict_widths(
            [v.parts for row in ordered if (v := row.candidates[index].verdict) is not None]
        )
        for index in range(candidate_count)
    ]
    baseline = value_widths([row.baseline for row in ordered])
    return _ColumnFields(values=col_values, verdicts=col_verdicts, baseline=baseline)


def _column_widths(
    body: Sequence[BodyLine[_ComparisonRow, _AggregateCells]],
    ordered: Sequence[_ComparisonRow],
    table: _TableContext,
) -> list[int]:
    """The column widths, measured over the plain rows, headers and aggregates."""
    aggregate_lines = [line for line in body if isinstance(line, AggregateLine)]

    def candidate_text(row: _ComparisonRow, index: int) -> str:
        cell = row.candidates[index]
        value = join_value_cell(cell.value, table.fields.values[index])
        if cell.verdict is None:
            return value
        verdict = join_verdict_cell(cell.verdict.parts, table.fields.verdicts[index])
        return f"{value}{CELL_GUTTER}{verdict}"

    metric_width = compute_column_width(
        cell_len(widest_header_label(body)),
        [cell_len(row.label if table.grouped else row.name) for row in ordered]
        + aggregate_label_lengths(body),
        METRIC_COLUMN_MIN,
    )
    baseline_width = compute_column_width(
        cell_len(table.baseline_header),
        [cell_len(join_value_cell(row.baseline, table.fields.baseline)) for row in ordered],
        VALUE_COLUMN_MIN,
    )
    candidate_widths = [
        compute_column_width(
            cell_len(candidate.label),
            [cell_len(candidate_text(row, index)) for row in ordered]
            + [cell_len(line.cell[index].text) for line in aggregate_lines],
            VALUE_COLUMN_MIN,
        )
        for index, candidate in enumerate(table.candidates)
    ]
    return [metric_width, baseline_width, *candidate_widths]


def _to_cells(
    line: BodyLine[_ComparisonRow, _AggregateCells],
    table: _TableContext,
) -> tuple[str, ...]:
    """The markup cells one content row renders to."""
    if isinstance(line, HeaderLine):
        return (
            header_metric_cell(line.title),
            markup(table.baseline_header, VARIANT_NAME_STYLE),
            *(markup(candidate.label, VARIANT_NAME_STYLE) for candidate in table.candidates),
        )
    if isinstance(line, GroupLine):
        return (group_metric_cell(line.label), "", *("" for _ in table.candidates))
    if isinstance(line, MetricLine):
        row = line.row
        return (
            escape(row.label if table.grouped else row.name),
            escape(join_value_cell(row.baseline, table.fields.baseline)),
            *(
                _candidate_markup(cell, table.fields.values[index], table.fields.verdicts[index])
                for index, cell in enumerate(row.candidates)
            ),
        )
    if isinstance(line, AggregateLine):
        return (
            markup(line.label, AGGREGATE_LABEL_STYLE),
            "",
            *(_aggregate_cell_markup(cell) for cell in line.cell),
        )
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
    grouped = len(layout.sections) > 1 or any(isinstance(line, GroupLine) for line in body)
    table = _TableContext(
        baseline_header=baseline_header,
        candidates=candidates,
        fields=fields,
        grouped=grouped,
    )
    widths = _column_widths(body, layout.ordered, table)

    return render_body(
        body,
        widths,
        lambda line: _to_cells(line, table),
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


def _candidate_markup(
    cell: _CandidateCell,
    values: ValueWidths,
    verdicts: VerdictWidths,
) -> str:
    """One candidate cell's markup: the value left plain, the glyph and delta in the verdict color.

    The band is dropped from the multi-candidate cell, so only the glyph and the
    delta (or the ``unstable`` word standing in for it) carry the verdict's color;
    an unstable cell paints them amber, a quiet cell recedes them to dim, and the
    figure itself stays plain whatever the verdict.
    """
    value = join_value_cell(cell.value, values)
    if cell.verdict is None:
        return escape(value)
    style = VERDICT_STYLES[cell.verdict.outcome]
    verdict_cell = style_verdict_cell(
        cell.verdict.parts,
        verdicts,
        glyph_style=style,
        delta_style=style,
        band_style=None,
    )
    return f"{escape(value)}{CELL_GUTTER}{verdict_cell}"


def _aggregate_cell_markup(cell: AggregateColumnCell) -> str:
    """A geomean column cell rendered to markup, each span wrapped where it sits in the text."""
    pieces: list[str] = []
    cursor = 0
    for span in cell.spans:
        start = cell.text.index(span.text, cursor)
        pieces.append(escape(cell.text[cursor:start]))
        pieces.append(markup(span.text, span.style))
        cursor = start + len(span.text)
    pieces.append(escape(cell.text[cursor:]))
    return "".join(pieces)
