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

from rich.markup import escape

from gymrat.report.format import (
    GEOMEAN_LABEL,
    baseline_cell_parts,
    candidate_cell_parts,
    geomean_scope_label,
    shown_class,
)
from gymrat.report.sections import (
    flat_geomean_of,
    group_geomean_of,
    kind_geomean_of,
    plan_sections,
)
from gymrat.report.style import (
    AGGREGATE_LABEL_STYLE,
    GROUP_LABEL_STYLE,
    VARIANT_NAME_STYLE,
    VERDICT_STYLES,
)
from gymrat.report.table import (
    CELL_GUTTER,
    METRIC_COLUMN_HEADER,
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
    indented_section_label,
    join_value_cell,
    join_verdict_cell,
    markup,
    plan_body,
    render_body,
    section_annotation,
    style_verdict_cell,
    value_widths,
    verdict_parts,
    verdict_widths,
    widest_header_label,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from gymrat.model import GeomeanResult
    from gymrat.report.format import DisplayClass, MetricCellParts
    from gymrat.report.table import BodyLine, ValueWidths, VerdictParts, VerdictWidths
    from gymrat.report.types import (
        CandidateComparison,
        ComparisonResult,
        MetricComparison,
    )

# The per-candidate aggregate cell of one geomean row, one entry per candidate.
type _AggregateCells = tuple[AggregateColumnCell, ...]


@dataclass(frozen=True, slots=True)
class _CandidateCell:
    """One candidate's side of a metric row: its figure, verdict, and outcome."""

    value: MetricCellParts
    verdict: VerdictParts | None
    outcome: DisplayClass | None


@dataclass(frozen=True, slots=True)
class _ComparisonRow:
    """One metric row: its names, the baseline figure, and each candidate's side."""

    name: str
    label: str
    baseline: MetricCellParts
    candidates: tuple[_CandidateCell, ...]
    gating: bool


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
        lambda name, group, metric: _build_row(metric, name, group, candidates, result.samples),
    )
    column_values = [
        value_widths([row.candidates[index].value for row in layout.ordered])
        for index in range(len(candidates))
    ]
    column_verdicts = [
        verdict_widths(
            [
                parts
                for row in layout.ordered
                if (parts := row.candidates[index].verdict) is not None
            ]
        )
        for index in range(len(candidates))
    ]
    baseline_fields = value_widths([row.baseline for row in layout.ordered])

    def candidate_text(row: _ComparisonRow, index: int) -> str:
        cell = row.candidates[index]
        value = join_value_cell(cell.value, column_values[index])
        if cell.verdict is None:
            return value
        return f"{value}{CELL_GUTTER}{join_verdict_cell(cell.verdict, column_verdicts[index])}"

    def baseline_text(row: _ComparisonRow) -> str:
        return join_value_cell(row.baseline, baseline_fields)

    aggregates = _aggregate_rows(candidates)
    body: list[BodyLine[_ComparisonRow, _AggregateCells]] = plan_body(
        layout,
        aggregates,
        lambda section: section_annotation(section, result.config_kinds),
    )
    aggregate_lines = [line for line in body if isinstance(line, AggregateLine)]
    grouped = len(layout.sections) > 1 or any(isinstance(line, GroupLine) for line in body)

    metric_width = compute_column_width(
        len(widest_header_label(body)),
        [len(row.label if grouped else row.name) for row in layout.ordered]
        + aggregate_label_lengths(body),
        METRIC_COLUMN_MIN,
    )
    baseline_width = compute_column_width(
        len(baseline_header),
        [len(baseline_text(row)) for row in layout.ordered],
        VALUE_COLUMN_MIN,
    )
    candidate_widths = [
        compute_column_width(
            len(candidate.label),
            [len(candidate_text(row, index)) for row in layout.ordered]
            + [len(line.cell[index].text) for line in aggregate_lines],
            VALUE_COLUMN_MIN,
        )
        for index, candidate in enumerate(candidates)
    ]
    widths = [metric_width, baseline_width, *candidate_widths]

    def metric_name(row: _ComparisonRow) -> str:
        return row.label if grouped else row.name

    def to_cells(line: BodyLine[_ComparisonRow, _AggregateCells]) -> tuple[str, ...]:
        if isinstance(line, HeaderLine):
            title = line.title
            metric_cell = (
                markup(title, "bold") if title is not None else escape(METRIC_COLUMN_HEADER)
            )
            return (
                metric_cell,
                markup(baseline_header, VARIANT_NAME_STYLE),
                *(markup(candidate.label, VARIANT_NAME_STYLE) for candidate in candidates),
            )
        if isinstance(line, GroupLine):
            return (markup(line.label, GROUP_LABEL_STYLE), "", *("" for _ in candidates))
        if isinstance(line, MetricLine):
            row = line.row
            return (
                escape(metric_name(row)),
                escape(baseline_text(row)),
                *(
                    _candidate_markup(cell, column_values[index], column_verdicts[index])
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

    return render_body(body, widths, to_cells, color=color)


def _build_row(
    metric: MetricComparison,
    name: str,
    group: str | None,
    candidates: Sequence[CandidateComparison],
    samples: int,
) -> _ComparisonRow:
    """Split one metric into the baseline figure and one cell per candidate column."""
    cells: list[_CandidateCell] = []
    for index in range(len(candidates)):
        side = metric.candidates[index] if index < len(metric.candidates) else None
        verdict = side.verdict if side is not None else None
        parts = None if verdict is None else verdict_parts(verdict, samples, with_band=False)
        cells.append(
            _CandidateCell(
                value=candidate_cell_parts(side, metric.meta.unit),
                verdict=parts,
                outcome=shown_class(verdict),
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
                [row.candidates[index].outcome for row in rows],
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
    if cell.verdict is None or cell.outcome is None:
        return escape(value)
    style = VERDICT_STYLES[cell.outcome]
    verdict_cell = style_verdict_cell(
        cell.verdict,
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
