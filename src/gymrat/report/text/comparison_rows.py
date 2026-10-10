"""The metric row a comparison table draws: the baseline's figure and each candidate's side.

The single-candidate table is the multi-candidate table with one candidate, so both
build their rows here and differ only in whether a verdict carries its band. Both
close a scope on the same geomean too, so which aggregate a scope reads, and which
rows decide its color, are settled here as well.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from gymrat.report.format import baseline_cell_parts, candidate_cell_parts
from gymrat.report.table.cells import (
    flat_geomean_of,
    group_geomean_of,
    indented_section_label,
    kind_geomean_of,
    shown_verdict,
)
from gymrat.report.table.render import AggregateRows
from gymrat.report.types import candidate_at

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from gymrat.model import GeomeanResult
    from gymrat.report.display import DisplayClass
    from gymrat.report.format import MetricCellParts
    from gymrat.report.table.cells import ShownVerdict
    from gymrat.report.table.render import AggregateLine
    from gymrat.report.types import CandidateComparison, MetricComparison


@dataclass(frozen=True, slots=True)
class CandidateCell:
    """One candidate's side of a metric row: its figure and optional verdict.

    Attributes:
        value: The candidate's median and spread, split into padded fields.
        verdict: The verdict shown beside it, or ``None`` when it has none.
    """

    value: MetricCellParts
    verdict: ShownVerdict | None


@dataclass(frozen=True, slots=True)
class ComparisonRow:
    """One metric row: its names, the baseline figure, and each candidate's side.

    Attributes:
        name: The metric's bare name.
        label: The metric's section label, indented under its group.
        baseline: The baseline's median and spread, split into padded fields.
        candidates: One cell per candidate column, in column order.
        gating: Whether the metric gates the run.
    """

    name: str
    label: str
    baseline: MetricCellParts
    candidates: tuple[CandidateCell, ...]
    gating: bool


def comparison_row_builder(
    candidate_count: int, samples: int, *, with_band: bool
) -> Callable[[str, str | None, MetricComparison], ComparisonRow]:
    """A row builder that splits one metric into the baseline figure and its candidate cells.

    Args:
        candidate_count: How many candidate columns the table draws.
        samples: How many samples each side collected, which a verdict's
            evidence is judged against.
        with_band: Whether each verdict carries its noise band.

    Returns:
        A builder taking a metric's bare name, its group (``None`` when
        ungrouped) and its comparison, in the order section planning hands them.
    """

    def build(name: str, group: str | None, metric: MetricComparison) -> ComparisonRow:
        cells: list[CandidateCell] = []
        for index in range(candidate_count):
            side = candidate_at(metric, index)
            cells.append(
                CandidateCell(
                    value=candidate_cell_parts(side, metric.meta.unit),
                    verdict=shown_verdict(
                        side.verdict if side is not None else None, samples, with_band=with_band
                    ),
                )
            )
        return ComparisonRow(
            name=name,
            label=indented_section_label(metric.meta.short_name, group),
            baseline=baseline_cell_parts(metric),
            candidates=tuple(cells),
            gating=metric.meta.gating,
        )

    return build


def candidate_outcomes(rows: Sequence[ComparisonRow], index: int) -> list[DisplayClass | None]:
    """The display class of one candidate's verdict on each row, for vetoing a geomean's color.

    Args:
        rows: The metric rows a geomean covers.
        index: The candidate column to read.

    Returns:
        Each row's verdict outcome for that candidate, or ``None`` where it has none.
    """
    return [
        verdict.outcome if (verdict := row.candidates[index].verdict) is not None else None
        for row in rows
    ]


@dataclass(frozen=True, slots=True)
class ScopeAggregate:
    """One candidate's geomean over a scope, and the outcomes that may veto its color.

    Attributes:
        geomean: The candidate's aggregate over the scope's metrics.
        outcomes: The display class of each metric behind the figure.
    """

    geomean: GeomeanResult
    outcomes: list[DisplayClass | None]


def comparison_aggregate_rows[Cell](
    candidates: Sequence[CandidateComparison],
    line: Callable[[str | None, list[ScopeAggregate]], AggregateLine[Cell]],
) -> AggregateRows[ComparisonRow, Cell]:
    """The aggregate-row builders of a comparison table, one aggregate per candidate.

    A group reads its group geomean and a kind its kind geomean, each vetoed by
    every row it covers. The flat table closes on the gated geomean, so only the
    gating rows decide its color.

    Args:
        candidates: The candidates, in column order.
        line: Builds the table's aggregate line from the scope's name (``None``
            for a flat table) and each candidate's aggregate, in column order.

    Returns:
        The group, kind and flat aggregate builders.
    """

    def resolve(
        geomean_of: Callable[[CandidateComparison], GeomeanResult],
        rows: Sequence[ComparisonRow],
    ) -> list[ScopeAggregate]:
        return [
            ScopeAggregate(geomean=geomean_of(candidate), outcomes=candidate_outcomes(rows, index))
            for index, candidate in enumerate(candidates)
        ]

    return AggregateRows(
        group=lambda kind, group, rows: line(
            group, resolve(lambda candidate: group_geomean_of(candidate, kind, group), rows)
        ),
        kind=lambda kind, rows: line(
            kind, resolve(lambda candidate: kind_geomean_of(candidate, kind), rows)
        ),
        flat=lambda rows: line(None, resolve(flat_geomean_of, [row for row in rows if row.gating])),
    )
