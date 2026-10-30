"""Tests for reading a candidate's geomeans back out and pairing them with each scope's outcomes."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from gymrat.report.table.grid import AggregateLine
from gymrat.report.text.comparison_rows import (
    NO_AGGREGATE,
    ComparisonRow,
    ScopeAggregate,
    comparison_aggregate_rows,
    comparison_row_builder,
    group_geomean_of,
    kind_geomean_of,
)
from tests.report._comparisons import (
    NWayCandidate,
    create_candidate,
    gating_kind,
    memory_kind,
    n_way_kind_metric,
    time_kind,
)
from tests.report._verdicts import geomean_of

if TYPE_CHECKING:
    from collections.abc import Callable

    from gymrat.model import GeomeanResult
    from gymrat.report.table.grid import AggregateRows

# ---------------------------------------------------------------------------
# kind_geomean_of / group_geomean_of — reading a candidate's aggregate back out
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kind", "expected"),
    [
        pytest.param("memory", memory_kind().geomean, id="reported-kind"),
        pytest.param("cpu", NO_AGGREGATE, id="unreported-kind"),
    ],
)
def test_kind_geomean_of_when_looking_up_a_kind_does_return_its_geomean_or_no_aggregate(
    kind: str, expected: GeomeanResult
):
    candidate = create_candidate(kinds=[time_kind(), memory_kind()])

    geomean = kind_geomean_of(candidate, kind)

    assert geomean == expected


@pytest.mark.parametrize(
    ("kind", "group", "expected"),
    [
        pytest.param("time", "entity", time_kind().groups[0].geomean, id="reported-group"),
        pytest.param("time", "render", NO_AGGREGATE, id="unreported-group"),
        pytest.param("cpu", "entity", NO_AGGREGATE, id="unreported-kind"),
    ],
)
def test_group_geomean_of_when_looking_up_a_group_does_return_its_geomean_or_no_aggregate(
    kind: str, group: str, expected: GeomeanResult
):
    candidate = create_candidate(kinds=[time_kind(), memory_kind()])

    geomean = group_geomean_of(candidate, kind, group)

    assert geomean == expected


# ---------------------------------------------------------------------------
# comparison_aggregate_rows — each scope's geomean and the outcomes vetoing it
# ---------------------------------------------------------------------------

type _Recorded = tuple[str | None, list[ScopeAggregate]]

_CANDIDATES = (
    create_candidate(
        label="candidate-a",
        kinds=[gating_kind("time", geomean_of(-9, 2), {"entity": geomean_of(-8, 1)})],
    ),
    create_candidate(
        label="candidate-b",
        kinds=[gating_kind("time", geomean_of(-12, 2), {"entity": geomean_of(-11, 1)})],
    ),
)


def _rows() -> list[ComparisonRow]:
    build = comparison_row_builder(len(_CANDIDATES), 10, with_band=True)
    gating = n_way_kind_metric(
        kind="time",
        short_name="entity.spawn",
        candidates=[
            NWayCandidate(verdict="no-signal", delta=-9, median=91),
            NWayCandidate(verdict="improved", delta=-12, median=88),
        ],
    )
    informational = n_way_kind_metric(
        kind="time",
        short_name="encode",
        gating=False,
        candidates=[
            NWayCandidate(verdict="improved", delta=-2, median=98),
            NWayCandidate(verdict="regressed", delta=3, median=103),
        ],
    )
    return [
        build("entity/spawn#time", "entity", gating),
        build("encode#time", None, informational),
    ]


def _record(label: str | None, aggregates: list[ScopeAggregate]) -> AggregateLine[_Recorded]:
    return AggregateLine(label=label or "", cell=(label, aggregates))


def _group_line(
    rows: AggregateRows[ComparisonRow, _Recorded], all_rows: list[ComparisonRow]
) -> AggregateLine[_Recorded]:
    return rows.group("time", "entity", all_rows)


def _kind_line(
    rows: AggregateRows[ComparisonRow, _Recorded], all_rows: list[ComparisonRow]
) -> AggregateLine[_Recorded]:
    return rows.kind("time", all_rows)


def _flat_line(
    rows: AggregateRows[ComparisonRow, _Recorded], all_rows: list[ComparisonRow]
) -> AggregateLine[_Recorded]:
    return rows.flat(all_rows)


@pytest.mark.parametrize(
    ("build_line", "expected"),
    [
        pytest.param(
            _group_line,
            (
                "entity",
                [
                    ScopeAggregate(geomean_of(-8, 1), ["within-noise", "improved"]),
                    ScopeAggregate(geomean_of(-11, 1), ["improved", "regressed"]),
                ],
            ),
            id="group-reads-its-group-geomean-over-every-row",
        ),
        pytest.param(
            _kind_line,
            (
                "time",
                [
                    ScopeAggregate(geomean_of(-9, 2), ["within-noise", "improved"]),
                    ScopeAggregate(geomean_of(-12, 2), ["improved", "regressed"]),
                ],
            ),
            id="kind-reads-its-kind-geomean-over-every-row",
        ),
        pytest.param(
            _flat_line,
            (
                None,
                [
                    ScopeAggregate(geomean_of(-9, 2), ["within-noise"]),
                    ScopeAggregate(geomean_of(-12, 2), ["improved"]),
                ],
            ),
            id="flat-reads-the-gated-geomean-over-gating-rows-only",
        ),
    ],
)
def test_comparison_aggregate_rows_when_a_scope_closes_does_pair_each_candidate_with_its_outcomes(
    build_line: Callable[
        [AggregateRows[ComparisonRow, _Recorded], list[ComparisonRow]], AggregateLine[_Recorded]
    ],
    expected: _Recorded,
):
    rows = comparison_aggregate_rows(_CANDIDATES, _record)

    line = build_line(rows, _rows())

    assert line.cell == expected
