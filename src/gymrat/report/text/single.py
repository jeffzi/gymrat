"""The single-candidate comparison table: one baseline against one candidate.

Each metric row states the baseline's figure, the candidate's, and the verdict
between them; each scope closes on the geomean of the metrics above it. The row
strings are pre-aligned here — magnitude, spread, glyph, delta and band each
padded to their column's width — and the grid around them is drawn by
:mod:`gymrat.report.table`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from rich.cells import cell_len
from rich.markup import escape

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
from gymrat.report.style import (
    AGGREGATE_LABEL_STYLE,
    VARIANT_NAME_STYLE,
    VERDICT_STYLES,
    markup,
)
from gymrat.report.table import (
    METRIC_COLUMN_MIN,
    VALUE_COLUMN_MIN,
    VERDICT_COLUMN_MIN,
    AggregateLine,
    AggregateRow,
    AggregateRows,
    GroupLine,
    HeaderLine,
    MetricLine,
    VerdictParts,
    aggregate_label_lengths,
    compute_column_width,
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
    from collections.abc import Sequence

    from gymrat.model import GeomeanResult, MetricVerdict
    from gymrat.report.display import DisplayClass
    from gymrat.report.format import MetricCellParts
    from gymrat.report.table import BodyLine, VerdictWidths
    from gymrat.report.types import CandidateComparison, ComparisonResult, MetricComparison

# The glyph slot a geomean figure fills — a blank, since the row states a mean,
# not an outcome.
_GEOMEAN_GLYPH_SLOT = " "


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
    grouped = len(layout.sections) > 1 or any(isinstance(line, GroupLine) for line in body)

    aggregate_parts = [line.cell.parts for line in body if isinstance(line, AggregateLine)]
    verdict_fields = verdict_widths(
        [row.verdict.parts for row in layout.ordered if row.verdict is not None] + aggregate_parts
    )

    def metric_cells(row: _MeasuredRow) -> tuple[str, str, str, str]:
        return (
            row.label if grouped else row.name,
            join_value_cell(row.baseline, baseline_fields),
            join_value_cell(row.candidate, candidate_fields),
            "" if row.verdict is None else join_verdict_cell(row.verdict.parts, verdict_fields),
        )

    cells_by_name = {row.name: metric_cells(row) for row in layout.ordered}
    rows = [cells_by_name[row.name] for row in layout.ordered]
    widths = _column_widths(headers, rows, body, aggregate_parts, verdict_fields)

    def to_cells(line: BodyLine[_MeasuredRow, _AggregateCell]) -> tuple[str, ...]:
        return _to_cells(line, headers, cells_by_name, verdict_fields)

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
    rows: Sequence[tuple[str, str, str, str]],
    body: Sequence[BodyLine[_MeasuredRow, _AggregateCell]],
    aggregate_parts: Sequence[VerdictParts],
    verdict_fields: VerdictWidths,
) -> list[int]:
    """The four column widths, measured over the plain rows, headers and aggregates."""

    def value_width(index: int) -> int:
        return compute_column_width(
            cell_len(headers[index]), [cell_len(row[index]) for row in rows], VALUE_COLUMN_MIN
        )

    verdict_lengths = [cell_len(row[3]) for row in rows] + [
        cell_len(join_verdict_cell(parts, verdict_fields)) for parts in aggregate_parts
    ]
    return [
        compute_column_width(
            cell_len(widest_header_label(body)),
            [cell_len(row[0]) for row in rows] + aggregate_label_lengths(body),
            METRIC_COLUMN_MIN,
        ),
        value_width(1),
        value_width(2),
        compute_column_width(cell_len(headers[3]), verdict_lengths, VERDICT_COLUMN_MIN),
    ]


def _to_cells(
    line: BodyLine[_MeasuredRow, _AggregateCell],
    headers: tuple[str, str, str, str],
    cells_by_name: dict[str, tuple[str, str, str, str]],
    verdict_fields: VerdictWidths,
) -> tuple[str, ...]:
    """The markup cells one content row renders to."""
    if isinstance(line, HeaderLine):
        return (
            header_metric_cell(line.title),
            markup(headers[1], VARIANT_NAME_STYLE),
            markup(headers[2], VARIANT_NAME_STYLE),
            f"vs {markup(headers[1], VARIANT_NAME_STYLE)}",
        )
    if isinstance(line, GroupLine):
        return (group_metric_cell(line.label), "", "", "")
    if isinstance(line, MetricLine):
        return _metric_cells(line.row, cells_by_name, verdict_fields)
    if isinstance(line, AggregateLine):
        cell = line.cell
        return (
            markup(line.label, AGGREGATE_LABEL_STYLE),
            "",
            "",
            style_verdict_cell(
                cell.parts,
                verdict_fields,
                glyph_style=cell.glyph_style,
                delta_style=cell.delta_style,
                band_style=cell.band_style,
            ),
        )
    msg = f"unexpected body line {line!r}"
    raise AssertionError(msg)


def _metric_cells(
    row: _MeasuredRow,
    cells_by_name: dict[str, tuple[str, str, str, str]],
    verdict_fields: VerdictWidths,
) -> tuple[str, ...]:
    """One metric row's markup cells: plain figures, and the styled verdict cell."""
    cells = cells_by_name[row.name]
    if row.verdict is None:
        verdict_cell = ""
    else:
        outcome = display_class(row.verdict.metric_verdict)
        quiet = outcome in QUIET_VERDICTS
        verdict_cell = style_verdict_cell(
            row.verdict.parts,
            verdict_fields,
            glyph_style=VERDICT_STYLES[outcome],
            delta_style=VERDICT_STYLES[outcome] if quiet else None,
            band_style="dim",
        )
    return (escape(cells[0]), escape(cells[1]), escape(cells[2]), verdict_cell)
