"""The metric row a comparison table draws: the baseline's figure and each candidate's side.

The single-candidate table is the multi-candidate table with one candidate, so both
build their rows here and differ only in whether a verdict carries its band. Both
close a scope on the same geomean too, so which aggregate a scope reads, and which
rows decide its color, are settled here as well.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from gymrat.model import GeomeanResult
from gymrat.report.display import GLYPHS, VERDICT_GLOSSES, display_class
from gymrat.report.format import (
    baseline_cell_parts,
    candidate_cell_parts,
    format_noise_band_value,
    format_pair_count,
    format_percent_delta,
)
from gymrat.report.table.cells import VerdictParts, indented_section_label
from gymrat.report.table.grid import AggregateRows
from gymrat.report.types import candidate_at

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from gymrat.model import MetricVerdict
    from gymrat.report.display import DisplayClass
    from gymrat.report.format import MetricCellParts
    from gymrat.report.table.grid import AggregateLine
    from gymrat.report.types import CandidateComparison, MetricComparison
    from gymrat.verdict import KindAggregate


# The aggregate stated where a candidate reported none. Every section is drawn from
# the same metadata the aggregates were computed from, so this stands in for
# nothing the renderers can produce — and if one ever does, the row says it
# aggregated nothing rather than inventing a figure.
NO_AGGREGATE: GeomeanResult = GeomeanResult(value=math.nan, n=0, band=0, excluded=())


def _kind_aggregate_of(candidate: CandidateComparison, kind: str) -> KindAggregate | None:
    """The aggregate a candidate reported for one kind, or ``None`` when it reported none."""
    return next((aggregate for aggregate in candidate.kinds if aggregate.kind == kind), None)


def kind_geomean_of(candidate: CandidateComparison, kind: str) -> GeomeanResult:
    """The geomean over every metric of ``kind``, gating or not."""
    aggregate = _kind_aggregate_of(candidate, kind)
    return NO_AGGREGATE if aggregate is None else aggregate.geomean


def group_geomean_of(candidate: CandidateComparison, kind: str, group: str) -> GeomeanResult:
    """The geomean over one group of ``kind``'s metrics.

    Args:
        candidate: The candidate whose geomean to read.
        kind: The metric kind the group belongs to.
        group: The group whose geomean to read.

    Returns:
        The group's geomean, or :data:`NO_AGGREGATE` when the candidate
        reported no aggregate for that kind or group.
    """
    aggregate = _kind_aggregate_of(candidate, kind)
    if aggregate is None:
        return NO_AGGREGATE
    return next((entry.geomean for entry in aggregate.groups if entry.group == group), NO_AGGREGATE)


def flat_geomean_of(candidate: CandidateComparison) -> GeomeanResult:
    """The geomean a flat table closes on: the single kind's gating metrics.

    A run reporting one kind has no section to name, so its geomean row states
    what the run is judged on without saying which kind that was.

    Args:
        candidate: The candidate whose geomean to read.

    Returns:
        The gated geomean of the single kind, or :data:`NO_AGGREGATE` when
        none is available.
    """
    if not candidate.kinds:
        return NO_AGGREGATE
    gated = candidate.kinds[0].gated_geomean
    return NO_AGGREGATE if gated is None else gated


@dataclass(frozen=True, slots=True)
class ShownVerdict:
    """A verdict's pre-split parts and display class, always present together.

    Attributes:
        parts: The fields the verdict column pads and styles.
        outcome: The display class the verdict presents as.
    """

    parts: VerdictParts
    outcome: DisplayClass


def shown_verdict(
    verdict: MetricVerdict | None, samples: int, *, with_band: bool
) -> ShownVerdict | None:
    """Take a verdict apart into the fields a verdict column pads and styles.

    Args:
        verdict: The verdict to render, or ``None`` when the metric has none.
        samples: The run's sample count, so a full-count verdict drops its ``n=N``.
        with_band: Whether the caller shows a noise band (the compact
            multi-candidate table drops it).

    Returns:
        The verdict's fields with its display class, or ``None`` when there is
        no verdict to show.
    """
    if verdict is None:
        return None
    shown = display_class(verdict)
    unstable = verdict.verdict == "unstable"
    band = ""
    if with_band and not unstable and shown != "inconclusive" and verdict.method != "exact":
        band = format_noise_band_value(verdict.noise_pct)
    return ShownVerdict(
        parts=VerdictParts(
            glyph=GLYPHS[shown],
            delta="" if unstable else format_percent_delta(verdict.delta),
            word=VERDICT_GLOSSES["unstable"] if unstable else "",
            band=band,
            pairs="" if verdict.n == samples else format_pair_count(verdict.n),
        ),
        outcome=shown,
    )


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
