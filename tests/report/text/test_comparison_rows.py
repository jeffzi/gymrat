"""Tests for a verdict's shown parts, a candidate's geomeans, and each scope's outcomes."""

from __future__ import annotations

from dataclasses import replace
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
    shown_verdict,
)
from tests.report._comparisons import (
    NWayCandidate,
    create_candidate,
    gating_kind,
    memory_kind,
    n_way_kind_metric,
    time_kind,
)
from tests.report._verdicts import geomean_of, permutation_verdict

if TYPE_CHECKING:
    from collections.abc import Callable

    from gymrat.model import GeomeanResult, PermutationVerdict
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
# shown_verdict — which rows show a noise band and a detectable change
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("verdict", "with_band", "expected"),
    [
        pytest.param(
            permutation_verdict(verdict="no-signal"), True, (True, "2.1%"), id="within-noise"
        ),
        pytest.param(permutation_verdict(verdict="improved"), True, (True, ""), id="improved"),
        pytest.param(permutation_verdict(verdict="unstable"), True, (False, ""), id="unstable"),
        pytest.param(permutation_verdict(n=3), True, (False, ""), id="inconclusive"),
        pytest.param(
            permutation_verdict(verdict="no-signal"), False, (False, ""), id="without-band"
        ),
    ],
)
def test_shown_verdict_when_verdict_carries_a_detectable_change_does_state_it_only_on_banded_within_noise_rows(
    verdict: PermutationVerdict, with_band: bool, expected: tuple[bool, str]
):
    detecting = replace(verdict, detectable_pct=2.1)

    shown = shown_verdict(detecting, 10, with_band=with_band)

    assert shown is not None
    assert (shown.parts.band != "", shown.parts.detects) == expected


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
