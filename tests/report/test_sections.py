"""Tests for section planning: contract-derived group lookup via metric names.

The ``plan_sections`` function groups metrics into kind sections.  Group
membership derives from ``gymrat.metric_name.parse`` applied to the full metric
name (the dict key), not from the ``short_name`` field.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pytest

from gymrat.model import GeomeanResult, ResolvedMetricMeta
from gymrat.report.sections import (
    NO_AGGREGATE,
    GroupBlock,
    group_geomean_of,
    kind_geomean_of,
    plan_sections,
)
from tests.report._inputs import create_candidate, memory_kind, time_kind

if TYPE_CHECKING:
    from gymrat.report.sections import SectionLayout


@dataclass(frozen=True, slots=True)
class _Row:
    """Minimal row the measure callback produces, carrying enough to assert on."""

    name: str
    group: str | None


@dataclass(frozen=True, slots=True)
class _FakeMetric:
    """Satisfies the ``SectionedMetric`` protocol: one ``.meta`` attribute."""

    meta: ResolvedMetricMeta


def _measure(name: str, group: str | None, metric: _FakeMetric) -> _Row:
    return _Row(name=name, group=group)


def _metric(*, kind: str = "time", short_name: str = "x") -> _FakeMetric:
    return _FakeMetric(
        meta=ResolvedMetricMeta(
            direction="lower",
            gating=True,
            exact=False,
            unit=None,
            kind=kind,
            short_name=short_name,
        ),
    )


def _group_blocks(layout: SectionLayout[_Row]) -> list[GroupBlock[_Row]]:
    """The ``GroupBlock``s in the layout's first section."""
    section = layout.sections[0]
    return [block for block in section.blocks if isinstance(block, GroupBlock)]


# ---------------------------------------------------------------------------
# plan_sections — contract-derived groups from metric name
# ---------------------------------------------------------------------------


def test_plan_sections_when_deeper_path_does_use_full_prefix_as_group():
    layout = plan_sections(
        {
            "node/access/get_1field#time": _metric(short_name="get_1field"),
            "node/access/get_2field#time": _metric(short_name="get_2field"),
        },
        _measure,
    )

    groups = _group_blocks(layout)
    assert [g.group for g in groups] == ["node/access"]
    assert len(groups[0].metrics) == 2


def test_plan_sections_when_single_segment_name_does_produce_no_group():
    layout = plan_sections(
        {
            "fib#time": _metric(short_name="fib"),
            "warmup#time": _metric(short_name="warmup"),
        },
        _measure,
    )

    groups = _group_blocks(layout)
    assert groups == []


def test_plan_sections_when_measure_callback_does_receive_contract_derived_group():
    # The group derives from the metric name key ("entity/alive_check#time" →
    # "entity"), not from short_name ("alive_check" → None).
    layout = plan_sections(
        {
            "entity/alive_check#time": _metric(short_name="alive_check"),
            "fib#time": _metric(short_name="fib"),
        },
        _measure,
    )

    rows_by_name = {row.name: row for row in layout.ordered}
    assert rows_by_name["entity/alive_check#time"].group == "entity"
    assert rows_by_name["fib#time"].group is None


def test_plan_sections_when_group_members_interleave_does_gather_them_in_the_first_block():
    # infer_group derives the group from the metric name key, not short_name:
    # "entity/spawn#time" → group "entity".
    layout = plan_sections(
        {
            "entity/spawn#time": _metric(),
            "render/frame#time": _metric(),
            "entity/remove#time": _metric(),
        },
        _measure,
    )

    (section,) = layout.sections
    assert section.blocks == [
        GroupBlock(
            group="entity",
            metrics=[
                _Row(name="entity/spawn#time", group="entity"),
                _Row(name="entity/remove#time", group="entity"),
            ],
        ),
        GroupBlock(group="render", metrics=[_Row(name="render/frame#time", group="render")]),
    ]


def test_plan_sections_when_group_spans_kinds_does_open_one_block_per_section():
    layout = plan_sections(
        {
            "entity/spawn#time": _metric(kind="time"),
            "entity/spawn#memory": _metric(kind="memory"),
        },
        _measure,
    )

    assert [section.blocks for section in layout.sections] == [
        [GroupBlock(group="entity", metrics=[_Row(name="entity/spawn#time", group="entity")])],
        [GroupBlock(group="entity", metrics=[_Row(name="entity/spawn#memory", group="entity")])],
    ]


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
