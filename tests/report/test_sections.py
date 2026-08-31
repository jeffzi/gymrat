"""Tests for section planning: contract-derived group lookup via metric names.

The ``plan_sections`` function groups metrics into kind sections.  Group
membership derives from ``gymrat.metric_name.parse`` applied to the full metric
name (the dict key), not from the ``short_name`` field.
"""

from __future__ import annotations

from dataclasses import dataclass

from gymrat.model import ResolvedMetricMeta
from gymrat.report.sections import GroupBlock, plan_sections


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


# ---------------------------------------------------------------------------
# plan_sections — contract-derived groups from metric name
# ---------------------------------------------------------------------------


def test_plan_sections_when_multi_segment_name_does_group_by_metric_name_path():
    """``infer_group`` must derive the group from the metric name key, not short_name.

    Metric name ``entity/alive_check#time`` has path ``("entity", "alive_check")``,
    so its group is ``"entity"``.  ``short_name="alive_check"`` has no dot and
    would yield ``None`` — confirming the group comes from the key, not the label.
    """
    layout = plan_sections(
        {
            "entity/alive_check#time": _metric(short_name="alive_check"),
            "entity/spawn#time": _metric(short_name="spawn"),
            "render/frame#time": _metric(short_name="frame"),
        },
        _measure,
    )

    section = layout.sections[0]
    groups = [block for block in section.blocks if isinstance(block, GroupBlock)]
    group_names = [g.group for g in groups]
    assert group_names == ["entity", "render"]
    assert len(groups[0].metrics) == 2
    assert len(groups[1].metrics) == 1


def test_plan_sections_when_deeper_path_does_use_full_prefix_as_group():
    layout = plan_sections(
        {
            "node/access/get_1field#time": _metric(short_name="get_1field"),
            "node/access/get_2field#time": _metric(short_name="get_2field"),
        },
        _measure,
    )

    section = layout.sections[0]
    groups = [block for block in section.blocks if isinstance(block, GroupBlock)]
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

    section = layout.sections[0]
    groups = [block for block in section.blocks if isinstance(block, GroupBlock)]
    assert groups == []


def test_plan_sections_when_measure_callback_does_receive_contract_derived_group():
    """The measure callback receives the group derived from the metric name key.

    ``"entity/alive_check#time"`` has group ``"entity"`` (path prefix), while
    ``short_name="alive_check"`` has no dot.  Deriving from the key yields
    ``"entity"``; deriving from short_name would yield ``None``.
    """
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
