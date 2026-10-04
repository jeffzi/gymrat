"""The single-candidate comparison table: one baseline against one candidate.

Each metric row states the baseline's figure, the candidate's, and the verdict
between them; each scope closes on the geomean of the metrics above it. The row
cells are pre-aligned here — magnitude, spread, glyph, delta and band each
padded to their column's width — and built once as styled rich ``Text``, whose
plain text sizes the columns; the grid around them is drawn by
:mod:`gymrat.report.table`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from rich.cells import cell_len
from rich.text import Text

from gymrat.report.display import QUIET_VERDICTS, display_class, shown_class
from gymrat.report.format import baseline_cell_parts, candidate_cell_parts
from gymrat.report.geomean_label import (
    NO_GEOMEAN_FIGURE,
    NO_STABLE_METRICS,
    geomean_label,
    geomean_parts,
    geomean_value_style,
    scoped_geomean_label,
)
from gymrat.report.sections import (
    flat_geomean_of,
    group_geomean_of,
    kind_geomean_of,
    plan_sections,
)
from gymrat.report.style import VERDICT_STYLES
from gymrat.report.table.markup import (
    VALUE_COLUMN_MIN,
    VERDICT_COLUMN_MIN,
    VerdictParts,
    aggregate_label_cell,
    group_metric_cell,
    header_metric_cell,
    indented_section_label,
    join_value_cell,
    value_widths,
    variant_name_cell,
    verdict_cell,
    verdict_parts,
    verdict_widths,
)
from gymrat.report.table.render import (
    AggregateLine,
    AggregateRow,
    AggregateRows,
    GroupLine,
    HeaderLine,
    MetricLine,
    compute_column_width,
    is_grouped,
    metric_column_width,
    plan_body,
    render_body,
    row_name_cell,
    section_annotation,
)
from gymrat.report.types import candidate_at

if TYPE_CHECKING:
    from collections.abc import Sequence

    from gymrat.model import GeomeanResult, MetricVerdict
    from gymrat.report.display import DisplayClass
    from gymrat.report.format import MetricCellParts
    from gymrat.report.table.markup import VerdictWidths
    from gymrat.report.table.render import BodyLine, TableCell
    from gymrat.report.types import CandidateComparison, ComparisonResult, MetricComparison

# The glyph slot a geomean figure fills — a blank, since the row states a mean,
# not an outcome.
_GEOMEAN_GLYPH_SLOT = " "

type _MetricCells = tuple[Text, Text, Text, Text]


@dataclass(frozen=True, slots=True)
class _MeasuredVerdict:
    """A metric's verdict and the pre-split parts that render it.

    Bundled so the two are always present together — a None
    ``_MeasuredRow.verdict`` means both are absent, never one without the other.
    """

    metric_verdict: MetricVerdict
    parts: VerdictParts


@dataclass(frozen=True, slots=True)
class _MeasuredRow:
    """One metric row's figures and the verdict between them, pre-split for padding."""

    name: str
    label: str
    baseline: MetricCellParts
    candidate: MetricCellParts
    verdict: _MeasuredVerdict | None
    gating: bool


@dataclass(frozen=True, slots=True)
class _AggregateCell:
    """A geomean row's verdict cell: its fields, and the style each field wears."""

    parts: VerdictParts
    glyph_style: str | None
    delta_style: str | None
    band_style: str | None


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
            glyph_style="bold",
            delta_style="dim",
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


def _measured_outcomes(rows: Sequence[_MeasuredRow]) -> list[DisplayClass | None]:
    """The display class of each row's verdict, for vetoing a geomean's color."""
    return [
        shown_class(row.verdict.metric_verdict) if row.verdict is not None else None for row in rows
    ]


def render_table(
    result: ComparisonResult,
    candidate: CandidateComparison,
    candidate_index: int,
    *,
    color: bool | None,
) -> list[str]:
    """Render a two-revision comparison table (one baseline vs. one candidate).

    Args:
        result: The comparison to draw.
        candidate: The candidate column's run-level aggregates.
        candidate_index: The candidate's position in each metric's slices.
        color: The explicit color choice, or ``None`` to defer to the environment.

    Returns:
        The rendered table lines.
    """
    baseline = result.baseline_label
    headers = ("metric", baseline, candidate.label, f"vs {baseline}")

    layout = plan_sections(
        result.metrics,
        lambda name, group, metric: _build_row(
            metric, name, group, candidate_index, result.samples
        ),
    )
    baseline_fields = value_widths([row.baseline for row in layout.ordered])
    candidate_fields = value_widths([row.candidate for row in layout.ordered])

    aggregates = _aggregate_rows(candidate)
    body: list[BodyLine[_MeasuredRow, _AggregateCell]] = plan_body(
        layout,
        aggregates,
        lambda section: section_annotation(section, result.config_kinds),
    )
    grouped = is_grouped(layout, body)

    aggregate_lines = [line for line in body if isinstance(line, AggregateLine)]
    aggregate_parts = [line.cell.parts for line in aggregate_lines]
    verdict_fields = verdict_widths(
        [row.verdict.parts for row in layout.ordered if row.verdict is not None] + aggregate_parts
    )

    def metric_cells(row: _MeasuredRow) -> _MetricCells:
        return (
            Text(row_name_cell(row, grouped=grouped)),
            Text(join_value_cell(row.baseline, baseline_fields)),
            Text(join_value_cell(row.candidate, candidate_fields)),
            _metric_verdict_cell(row, verdict_fields),
        )

    cells_by_name = {row.name: metric_cells(row) for row in layout.ordered}
    aggregate_cells = {
        line.cell: _aggregate_verdict_cell(line.cell, verdict_fields) for line in aggregate_lines
    }
    widths = _column_widths(
        headers, list(cells_by_name.values()), body, list(aggregate_cells.values())
    )

    def to_cells(line: BodyLine[_MeasuredRow, _AggregateCell]) -> tuple[TableCell, ...]:
        return _to_cells(line, headers, cells_by_name, aggregate_cells)

    return render_body(body, widths, to_cells, color=color)


def _build_row(
    metric: MetricComparison,
    name: str,
    group: str | None,
    candidate_index: int,
    samples: int,
) -> _MeasuredRow:
    """Split one metric into the fields a comparison row pads."""
    side = candidate_at(metric, candidate_index)
    metric_verdict = side.verdict if side is not None else None
    verdict: _MeasuredVerdict | None = None
    if metric_verdict is not None:
        verdict = _MeasuredVerdict(
            metric_verdict=metric_verdict,
            parts=verdict_parts(metric_verdict, samples, with_band=True),
        )
    return _MeasuredRow(
        name=name,
        label=indented_section_label(metric.meta.short_name, group),
        baseline=baseline_cell_parts(metric),
        candidate=candidate_cell_parts(side, metric.meta.unit),
        verdict=verdict,
        gating=metric.meta.gating,
    )


def _aggregate_rows(
    candidate: CandidateComparison,
) -> AggregateRows[_MeasuredRow, _AggregateCell]:
    """The three aggregate-row builders for a single-candidate table."""

    def scoped(
        scope: str, geomean: GeomeanResult, rows: Sequence[_MeasuredRow]
    ) -> AggregateRow[_AggregateCell]:
        return AggregateRow(
            label=scoped_geomean_label(scope, geomean),
            cell=_geomean_cell(geomean, _measured_outcomes(rows)),
        )

    return AggregateRows(
        group=lambda kind, group, rows: scoped(
            group, group_geomean_of(candidate, kind, group), rows
        ),
        kind=lambda kind, rows: scoped(kind, kind_geomean_of(candidate, kind), rows),
        flat=lambda rows: _flat_aggregate(candidate, rows),
    )


def _flat_aggregate(
    candidate: CandidateComparison,
    rows: Sequence[_MeasuredRow],
) -> AggregateRow[_AggregateCell]:
    """The single geomean a flat table closes on, over the run's gating metrics."""
    geomean = flat_geomean_of(candidate)
    gating = [row for row in rows if row.gating]
    return AggregateRow(
        label=geomean_label(geomean.n),
        cell=_geomean_cell(geomean, _measured_outcomes(gating)),
    )


def _column_widths(
    headers: tuple[str, str, str, str],
    rows: Sequence[_MetricCells],
    body: Sequence[BodyLine[_MeasuredRow, _AggregateCell]],
    aggregate_verdicts: Sequence[Text],
) -> list[int]:
    """The four column widths, measured over the rows, headers and aggregates' plain text."""

    def value_width(index: int) -> int:
        return compute_column_width(
            cell_len(headers[index]), [row[index].cell_len for row in rows], VALUE_COLUMN_MIN
        )

    verdict_lengths = [row[3].cell_len for row in rows] + [
        verdict.cell_len for verdict in aggregate_verdicts
    ]
    return [
        metric_column_width(body, [row[0].cell_len for row in rows]),
        value_width(1),
        value_width(2),
        compute_column_width(cell_len(headers[3]), verdict_lengths, VERDICT_COLUMN_MIN),
    ]


def _to_cells(
    line: BodyLine[_MeasuredRow, _AggregateCell],
    headers: tuple[str, str, str, str],
    cells_by_name: dict[str, _MetricCells],
    aggregate_cells: dict[_AggregateCell, Text],
) -> tuple[TableCell, ...]:
    """The cells one content row renders to."""
    if isinstance(line, HeaderLine):
        return (
            header_metric_cell(line.title),
            variant_name_cell(headers[1]),
            variant_name_cell(headers[2]),
            Text.assemble("vs ", variant_name_cell(headers[1])),
        )
    if isinstance(line, GroupLine):
        return (group_metric_cell(line.label), "", "", "")
    if isinstance(line, MetricLine):
        return cells_by_name[line.row.name]
    if isinstance(line, AggregateLine):
        return (
            aggregate_label_cell(line.label),
            "",
            "",
            aggregate_cells[line.cell],
        )
    msg = f"unexpected body line {line!r}"
    raise AssertionError(msg)


def _metric_verdict_cell(row: _MeasuredRow, verdict_fields: VerdictWidths) -> Text:
    """One metric row's verdict cell: its glyph in the verdict color, the band dimmed."""
    if row.verdict is None:
        return Text()
    outcome = display_class(row.verdict.metric_verdict)
    quiet = outcome in QUIET_VERDICTS
    return verdict_cell(
        row.verdict.parts,
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
