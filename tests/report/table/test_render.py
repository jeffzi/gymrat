"""Tests for grouped table layout in single-section (flat) comparison reports.

A single-kind comparison uses the flat (single-section) layout path.  These
tests verify that the flat body groups metrics under group headers, uses case
names for member rows, places ungrouped rows after groups, preserves
first-appearance ordering, and renders the full path prefix for deeper groups.

The body-rendering section pins the rules a planned body draws between a header
and its rows and ahead of an aggregate row, including two rules in a row.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from rich.text import Text

from gymrat.report.table.render import (
    AggregateLine,
    GroupLine,
    HeaderLine,
    MetricLine,
    RuleLine,
    render_body,
)
from gymrat.report.text.render import render_report
from gymrat.verdict import GroupAggregate, KindAggregate

if TYPE_CHECKING:
    from gymrat.report.table.render import BodyLine
from tests.report._assertions import table_region
from tests.report._comparisons import (
    create_candidate,
    create_comparison_result,
    grouped_flat_result,
    kind_metric,
)
from tests.report._verdicts import geomean_of

# ---------------------------------------------------------------------------
# group headers and case names
# ---------------------------------------------------------------------------


def test_table_region_when_flat_body_with_groups_does_emit_group_headers_and_case_names():
    region = table_region(render_report(grouped_flat_result()))

    assert region == [
        "gymrat compare · baseline main ↔ perf/faster-decode · 10 paired samples · adapter: mitata",
        "metric",
        "<rule>",
        "entity · time",
        "alive_check",
        "spawn",
        "",
        "warmup",
        "<rule>",
        "geomean (3 stable metrics)",
    ]


# ---------------------------------------------------------------------------
# ungrouped trailing rows
# ---------------------------------------------------------------------------


def test_table_region_when_flat_body_has_ungrouped_rows_does_trail_after_groups():
    region = table_region(render_report(grouped_flat_result()))

    group_member_indices = [
        i for i, entry in enumerate(region) if entry in ("alive_check", "spawn")
    ]
    warmup_indices = [
        i for i, entry in enumerate(region) if "warmup" in entry and "geomean" not in entry.lower()
    ]

    assert group_member_indices
    assert warmup_indices
    assert max(group_member_indices) < min(warmup_indices)


# ---------------------------------------------------------------------------
# first-appearance order
# ---------------------------------------------------------------------------


def test_table_region_when_flat_body_with_multiple_groups_does_preserve_first_appearance_order():
    geomean = geomean_of(-2, 4)
    result = create_comparison_result(
        metrics={
            "node/get#time": kind_metric(
                kind="time", short_name="node.get", verdict="improved", delta=-5
            ),
            "entity/spawn#time": kind_metric(
                kind="time",
                short_name="entity.spawn",
                verdict="regressed",
                delta=4,
            ),
            "node/set#time": kind_metric(
                kind="time",
                short_name="node.set",
                verdict="no-signal",
                delta=0.1,
            ),
            "entity/check#time": kind_metric(
                kind="time",
                short_name="entity.check",
                verdict="improved",
                delta=-3,
            ),
        },
        candidates=[
            create_candidate(
                kinds=[
                    KindAggregate(
                        kind="time",
                        geomean=geomean,
                        groups=(
                            GroupAggregate(group="node", geomean=geomean_of(-2.5, 2)),
                            GroupAggregate(group="entity", geomean=geomean_of(0.5, 2)),
                        ),
                        gated_geomean=geomean,
                    )
                ]
            )
        ],
    )

    region = table_region(render_report(result))

    assert region == [
        "gymrat compare · baseline main ↔ perf/faster-decode · 10 paired samples · adapter: mitata",
        "metric",
        "<rule>",
        "node · time",
        "get",
        "set",
        "",
        "entity · time",
        "spawn",
        "check",
        "<rule>",
        "geomean (4 stable metrics)",
    ]


# ---------------------------------------------------------------------------
# deeper paths
# ---------------------------------------------------------------------------


def test_table_region_when_flat_body_with_deeper_path_does_use_full_prefix_as_group():
    geomean = geomean_of(-3.2, 2)
    result = create_comparison_result(
        metrics={
            "node/access/get_1field#time": kind_metric(
                kind="time",
                short_name="node_access.get_1field",
                verdict="improved",
                delta=-5,
            ),
            "node/access/get_2field#time": kind_metric(
                kind="time",
                short_name="node_access.get_2field",
                verdict="no-signal",
                delta=0.1,
            ),
        },
        candidates=[
            create_candidate(
                kinds=[
                    KindAggregate(
                        kind="time",
                        geomean=geomean,
                        groups=(GroupAggregate(group="node/access", geomean=geomean),),
                        gated_geomean=geomean,
                    )
                ]
            )
        ],
    )

    region = table_region(render_report(result))

    group_headers = [e for e in region if e.startswith("node/access")]
    assert group_headers, "expected a group header starting with 'node/access'"
    assert "get_1field" in region
    assert "get_2field" in region
    assert "node/access/get_1field#time" not in region
    assert "node/access/get_2field#time" not in region


# ---------------------------------------------------------------------------
# body rules
# ---------------------------------------------------------------------------


def _text_cells(line: BodyLine[str, str]) -> tuple[Text, ...]:
    """Two plain ``Text`` cells for a header, group, metric, or aggregate line."""
    match line:
        case HeaderLine():
            return (Text("metric"), Text("value"))
        case GroupLine(label=label):
            return (Text(label), Text(""))
        case MetricLine(row=row):
            return (Text(row), Text("1"))
        case AggregateLine(label=label, cell=cell):
            return (Text(label), Text(cell))
        case _:
            msg = f"no cells for {line!r}"
            raise AssertionError(msg)


@pytest.mark.parametrize("color", [False, True])
@pytest.mark.parametrize(
    ("body", "expected"),
    [
        pytest.param(
            [
                HeaderLine(),
                RuleLine(),
                MetricLine("decode"),
                RuleLine(),
                AggregateLine("geomean", "-3.2%"),
            ],
            [
                "metric   │ value",
                "─────────┼───────",
                "decode   │ 1",
                "─────────┼───────",
                "geomean  │ -3.2%",
            ],
            id="rule-after-header-and-before-aggregate",
        ),
        pytest.param(
            [HeaderLine(), RuleLine(), RuleLine(), AggregateLine("geomean", "-3.2%")],
            [
                "metric   │ value",
                "─────────┼───────",
                "─────────┼───────",
                "geomean  │ -3.2%",
            ],
            id="two-rules-with-no-row-between",
        ),
        pytest.param(
            [HeaderLine(), RuleLine()],
            ["metric   │ value", "─────────┼───────"],
            id="rule-closing-the-body",
        ),
        pytest.param(
            [
                HeaderLine(),
                RuleLine(),
                GroupLine("[bold]"),
                MetricLine("[dim]"),
                AggregateLine("geomean", "-3.2%"),
            ],
            [
                "metric   │ value",
                "─────────┼───────",
                "[bold]   │",
                "[dim]    │ 1",
                "geomean  │ -3.2%",
            ],
            id="bracketed-text-cells-render-literally",
        ),
    ],
)
def test_render_body_when_text_cells_does_draw_rows_and_rules(
    body: list[BodyLine[str, str]], expected: list[str], color: bool
):
    lines = render_body(body, [8, 6], _text_cells, color=color)

    assert lines == expected
