"""Tests for a table's section planning.

The ``plan_sections`` function groups metrics into kind sections.  Group
membership derives from ``gymrat.metric_name.parse`` applied to the full metric
name (the dict key), not from the ``short_name`` field.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from gymrat.report.table.render import GroupBlock, MetricBlock, plan_sections
from tests.report._verdicts import metric_meta

if TYPE_CHECKING:
    from gymrat.model import ResolvedMetricMeta

# ---------------------------------------------------------------------------
# plan_sections — contract-derived groups from metric name
# ---------------------------------------------------------------------------


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


def _metric(*, kind: str = "time") -> _FakeMetric:
    return _FakeMetric(meta=metric_meta("x", kind=kind))


def test_plan_sections_when_group_members_interleave_does_gather_them_in_the_first_block():
    # The group derives from the metric name key, not short_name:
    # "entity/spawn#time" → group "entity".
    layout = plan_sections(
        {
            "entity/spawn#time": _metric(),
            "fib#time": _metric(),
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
        MetricBlock(metric=_Row(name="fib#time", group=None)),
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
