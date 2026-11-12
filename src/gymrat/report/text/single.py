"""The single-candidate comparison table: one baseline against one candidate.

Each metric row states the baseline's figure, the candidate's, and the verdict
between them; each scope closes on the geomean of the metrics above it. The row
cells are pre-aligned here — magnitude, spread, glyph, delta and band each
padded to their column's width — and built once as styled rich ``Text``, whose
plain text sizes the columns; the grid around them is drawn by
:mod:`gymrat.report.table`.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from rich.cells import cell_len
from rich.text import Text

from gymrat.report.display import QUIET_VERDICTS
from gymrat.report.style import VERDICT_STYLES
from gymrat.report.table.cells import (
    GEOMEAN_LABEL,
    NO_GEOMEAN_FIGURE,
    NO_GEOMEAN_FIGURE_STYLE,
    NO_STABLE_METRICS,
    NO_STABLE_METRICS_STYLE,
    VERDICT_COLUMN_MIN,
    VerdictParts,
    aggregate_label_cell,
    geomean_count,
    geomean_parts,
    geomean_scope_label,
    geomean_value_style,
    group_metric_cell,
    header_metric_cell,
    variant_name_cell,
    verdict_cell,
    verdict_widths,
)
from gymrat.report.table.grid import (
    VALUE_COLUMN_MIN,
    AggregateLine,
    AggregateRows,
    build_cell_dispatcher,
    compute_column_width,
    join_value_cell,
    plan_sections,
    plan_table_skeleton,
    render_body,
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
    from gymrat.report.table.grid import TableSkeleton
    from gymrat.report.types import CandidateComparison, ComparisonResult

# The glyph slot a geomean figure fills — a blank, since the row states a mean,
# not an outcome.
_GEOMEAN_GLYPH_SLOT = " "

type _MetricCells = tuple[Text, Text, Text, Text]


@dataclass(frozen=True, slots=True)
class _AggregateCell:
    """A geomean row's verdict cell: its fields, and the style each field wears."""

    parts: VerdictParts
    glyph_style: str | None
    delta_style: str | None
    band_style: str | None


def geomean_label(geomean: GeomeanResult) -> str:
    """The flat geomean row's label, carrying the count of metrics behind the figure.

    A table with one candidate names the count here, which frees its cells of
    everything but the aggregate itself.

    Args:
        geomean: The aggregate the row closes on.

    Returns:
        The label with the metric count, or the bare :data:`GEOMEAN_LABEL` when
        no metric stands behind the figure.
    """
    return GEOMEAN_LABEL if geomean.n == 0 else f"{GEOMEAN_LABEL} {geomean_count(geomean)}"


def scoped_geomean_label(scope: str, geomean: GeomeanResult) -> str:
    """A sectioned table's aggregate label with the count behind its figure.

    Args:
        scope: The group or kind the aggregate covers.
        geomean: The scope's aggregate.

    Returns:
        ``<scope label> (n/total)``, the count as :func:`geomean_count` phrases it.
    """
    return f"{geomean_scope_label(scope)} {geomean_count(geomean)}"


def _geomean_cell(
    geomean: GeomeanResult,
    outcomes: Sequence[DisplayClass | None],
) -> _AggregateCell:
    """A geomean's verdict cell, or the ``— no stable metrics`` stand-in for an empty one."""
    parts = geomean_parts(geomean)
    if parts is None:
        return _AggregateCell(
            parts=VerdictParts(
                glyph=NO_GEOMEAN_FIGURE, delta="", word=NO_STABLE_METRICS, band="", pairs=""
            ),
            glyph_style=NO_GEOMEAN_FIGURE_STYLE,
            delta_style=NO_STABLE_METRICS_STYLE,
            band_style=None,
        )
    return _AggregateCell(
        parts=VerdictParts(
            glyph=_GEOMEAN_GLYPH_SLOT, delta=parts.delta, word="", band=parts.band, pairs=""
        ),
        glyph_style=None,
        delta_style=geomean_value_style(geomean, outcomes),
        band_style="dim",
    )


def _side(row: ComparisonRow) -> CandidateCell:
    """The one candidate's side of a single-candidate row."""
    return row.candidates[0]


def render_table(result: ComparisonResult, *, color: bool | None) -> list[str]:
    """Render a two-revision comparison table (one baseline vs. one candidate).

    Args:
        result: The comparison to draw; its first candidate is the one shown.
        color: The explicit color choice, or ``None`` to defer to the environment.

    Returns:
        The rendered table lines.
    """
    candidate = result.candidates[0]
    baseline = result.baseline_label
    headers = (baseline, candidate.label, f"vs {baseline}")

    layout = plan_sections(
        result.metrics,
        comparison_row_builder(1, result.samples, with_band=True),
    )
    skeleton = plan_table_skeleton(
        layout,
        _aggregate_rows(candidate),
        result.config_kinds,
        lambda row: row.baseline,
        baseline,
    )
    body = skeleton.body
    candidate_fields = value_widths([_side(row).value for row in layout.ordered])

    aggregate_lines = [line for line in body if isinstance(line, AggregateLine)]
    verdict_fields = _verdict_fields(layout.ordered, [line.cell.parts for line in aggregate_lines])

    def metric_cells(row: ComparisonRow) -> _MetricCells:
        return (
            Text(skeleton.name_cell(row)),
            Text(skeleton.value_cell(row)),
            Text(join_value_cell(_side(row).value, candidate_fields)),
            _metric_verdict_cell(row, verdict_fields),
        )

    cells_by_name = {row.name: metric_cells(row) for row in layout.ordered}
    aggregate_cells = {
        line.cell: _aggregate_verdict_cell(line.cell, verdict_fields) for line in aggregate_lines
    }
    widths = _column_widths(
        headers, list(cells_by_name.values()), skeleton, list(aggregate_cells.values())
    )

    def cells_of(row: ComparisonRow) -> _MetricCells:
        return cells_by_name[row.name]

    to_cells = build_cell_dispatcher(
        header=lambda title: (
            header_metric_cell(title),
            variant_name_cell(headers[0]),
            variant_name_cell(headers[1]),
            Text.assemble("vs ", variant_name_cell(headers[0])),
        ),
        group=lambda label: (group_metric_cell(label), "", "", ""),
        metric=cells_of,
        aggregate=lambda line: (
            aggregate_label_cell(line.label),
            "",
            "",
            aggregate_cells[line.cell],
        ),
    )

    return render_body(body, widths, to_cells, color=color)


def _aggregate_rows(
    candidate: CandidateComparison,
) -> AggregateRows[ComparisonRow, _AggregateCell]:
    """The three aggregate-row builders for a single-candidate table."""

    def line(scope: str | None, aggregates: list[ScopeAggregate]) -> AggregateLine[_AggregateCell]:
        (aggregate,) = aggregates
        geomean = aggregate.geomean
        label = geomean_label(geomean) if scope is None else scoped_geomean_label(scope, geomean)
        return AggregateLine(label=label, cell=_geomean_cell(geomean, aggregate.outcomes))

    return comparison_aggregate_rows([candidate], line)


def _column_widths(
    headers: tuple[str, str, str],
    rows: Sequence[_MetricCells],
    skeleton: TableSkeleton[ComparisonRow, _AggregateCell],
    aggregate_verdicts: Sequence[Text],
) -> list[int]:
    """The four column widths: the skeleton's name and baseline, then the candidate and verdict."""
    verdict_lengths = [row[3].cell_len for row in rows] + [
        verdict.cell_len for verdict in aggregate_verdicts
    ]
    return [
        skeleton.metric_width,
        skeleton.value_width,
        compute_column_width(
            cell_len(headers[1]), [row[2].cell_len for row in rows], VALUE_COLUMN_MIN
        ),
        compute_column_width(cell_len(headers[2]), verdict_lengths, VERDICT_COLUMN_MIN),
    ]


def _verdict_fields(
    rows: Sequence[ComparisonRow], aggregate_parts: Sequence[VerdictParts]
) -> VerdictWidths:
    """The verdict column's widths, its delta field wide enough for any metric row's word.

    A metric row's word, such as ``unstable``, is measured so the fields after
    it line up with the other rows'. The geomean's ``no stable metrics``
    stand-in is not: nothing follows it in its cell, so widening for it would
    only push every metric row's delta right.

    Args:
        rows: The table's metric rows.
        aggregate_parts: The verdict fields of the table's geomean rows.

    Returns:
        The delta, band, pair-count and detectable-change widths every verdict
        cell pads to.
    """
    metric_parts = [v.parts for row in rows if (v := _side(row).verdict) is not None]
    fields = verdict_widths([*metric_parts, *aggregate_parts])
    word_width = max((len(parts.word) for parts in metric_parts), default=0)
    return replace(fields, delta=max(fields.delta, word_width))


def _metric_verdict_cell(row: ComparisonRow, verdict_fields: VerdictWidths) -> Text:
    """One metric row's verdict cell: its glyph in the verdict color, the band dimmed."""
    verdict = _side(row).verdict
    if verdict is None:
        return Text()
    outcome = verdict.outcome
    quiet = outcome in QUIET_VERDICTS
    return verdict_cell(
        verdict.parts,
        verdict_fields,
        glyph_style=VERDICT_STYLES[outcome],
        delta_style=VERDICT_STYLES[outcome] if quiet else None,
        band_style="dim",
    )


def _aggregate_verdict_cell(cell: _AggregateCell, verdict_fields: VerdictWidths) -> Text:
    """A geomean row's verdict cell, each field in the style the aggregate chose for it."""
    return verdict_cell(
        cell.parts,
        verdict_fields,
        glyph_style=cell.glyph_style,
        delta_style=cell.delta_style,
        band_style=cell.band_style,
    )
