"""The metric row a comparison table draws: the baseline's figure and each candidate's side.

The single-candidate table is the multi-candidate table with one candidate, so both
build their rows here and differ only in whether a verdict carries its band.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from gymrat.report.format import baseline_cell_parts, candidate_cell_parts
from gymrat.report.table.cells import indented_section_label, shown_verdict
from gymrat.report.types import candidate_at

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from gymrat.report.display import DisplayClass
    from gymrat.report.format import MetricCellParts
    from gymrat.report.table.cells import ShownVerdict
    from gymrat.report.types import MetricComparison


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
