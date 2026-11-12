"""Tests for a verdict's shown parts, a candidate's geomeans, and each scope's outcomes."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import pytest

from gymrat.errors import GymratError
from gymrat.report.table.grid import AggregateLine
from gymrat.report.text.comparison_rows import (
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
    informational_kind,
    memory_kind,
    n_way_kind_metric,
    time_kind,
)
from tests.report._verdicts import geomean_of, permutation_verdict

if TYPE_CHECKING:
    from collections.abc import Callable

    from gymrat.model import PermutationVerdict
    from gymrat.report.table.grid import AggregateRows

# ---------------------------------------------------------------------------
# kind_geomean_of / group_geomean_of — reading a candidate's aggregate back out
# ---------------------------------------------------------------------------


def test_kind_geomean_of_when_kind_reported_does_return_its_geomean():
    candidate = create_candidate(kinds=[time_kind(), memory_kind()])

    geomean = kind_geomean_of(candidate, "memory")

    assert geomean == memory_kind().geomean


def test_kind_geomean_of_when_kind_unreported_does_raise_an_internal_error_naming_it():
    candidate = create_candidate(kinds=[time_kind(), memory_kind()])

    with pytest.raises(Exception, match="cpu") as exc_info:
        kind_geomean_of(candidate, "cpu")

    assert not isinstance(exc_info.value, GymratError)


@pytest.mark.parametrize(
    ("kind", "group"),
    [
        pytest.param("time", "render", id="unreported-group"),
        pytest.param("cpu", "entity", id="unreported-kind"),
    ],
)
def test_group_geomean_of_when_group_unreported_does_raise_an_internal_error_naming_kind_and_group(
    kind: str, group: str
):
    candidate = create_candidate(kinds=[time_kind(), memory_kind()])

    with pytest.raises(Exception, match=group) as exc_info:
        group_geomean_of(candidate, kind, group)

    assert kind in str(exc_info.value)
    assert not isinstance(exc_info.value, GymratError)


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

# Each kind's gated geomean differs from its whole-kind geomean, so a scope that
# read the wrong one would show.
_CANDIDATES = (
    create_candidate(
        label="candidate-a",
        kinds=[
            replace(
                gating_kind("time", geomean_of(-9, 2), {"entity": geomean_of(-8, 1)}),
                gated_geomean=geomean_of(-7, 1),
            )
        ],
    ),
    create_candidate(
        label="candidate-b",
        kinds=[
            replace(
                gating_kind("time", geomean_of(-12, 2), {"entity": geomean_of(-11, 1)}),
                gated_geomean=geomean_of(-10, 1),
            )
        ],
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
                    ScopeAggregate(geomean_of(-7, 1), ["within-noise"]),
                    ScopeAggregate(geomean_of(-10, 1), ["improved"]),
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


_INFORMATIONAL_CANDIDATES = (
    create_candidate(label="candidate-a", kinds=[informational_kind("time", geomean_of(-5, 2))]),
    create_candidate(label="candidate-b", kinds=[informational_kind("time", geomean_of(4, 2))]),
)


def _informational_rows() -> list[ComparisonRow]:
    build = comparison_row_builder(len(_INFORMATIONAL_CANDIDATES), 10, with_band=True)
    warmup = n_way_kind_metric(
        kind="time",
        short_name="warmup",
        gating=False,
        candidates=[
            NWayCandidate(verdict="no-signal", delta=-1, median=99),
            NWayCandidate(verdict="improved", delta=-6, median=94),
        ],
    )
    encode = n_way_kind_metric(
        kind="time",
        short_name="encode",
        gating=False,
        candidates=[
            NWayCandidate(verdict="improved", delta=-9, median=91),
            NWayCandidate(verdict="regressed", delta=14, median=114),
        ],
    )
    return [build("warmup#time", None, warmup), build("encode#time", None, encode)]


def test_comparison_aggregate_rows_when_flat_kind_gates_nothing_does_close_on_its_kind_geomean_over_every_row():
    rows = comparison_aggregate_rows(_INFORMATIONAL_CANDIDATES, _record)

    line = rows.flat(_informational_rows())

    assert line.cell == (
        None,
        [
            ScopeAggregate(geomean_of(-5, 2), ["within-noise", "improved"]),
            ScopeAggregate(geomean_of(4, 2), ["improved", "regressed"]),
        ],
    )
