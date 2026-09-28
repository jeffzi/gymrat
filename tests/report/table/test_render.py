"""Tests for grouped table layout in single-section (flat) comparison reports.

A single-kind comparison uses the flat (single-section) layout path.  These
tests verify that the flat body groups metrics under group headers, uses case
names for member rows, places ungrouped rows after groups, preserves
first-appearance ordering, and renders the full path prefix for deeper groups.

The verdict-cell section pins the styled cell a verdict column is built from:
its plain text is the column's width source, and its styles sit on the glyph,
the delta (or the word standing in for it) and the band, never on padding.

The body-rendering section pins the rules a planned body draws between a header
and its rows and ahead of an aggregate row, including two rules in a row.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from rich.text import Text

from gymrat.report.table import (
    AggregateLine,
    GroupLine,
    HeaderLine,
    MetricLine,
    VerdictParts,
    VerdictWidths,
    render_body,
    verdict_cell,
)
from gymrat.report.table.render import RuleLine
from gymrat.report.text import render_report
from gymrat.verdict import GroupAggregate, KindAggregate

if TYPE_CHECKING:
    from gymrat.report.table import BodyLine
    from gymrat.report.types import ComparisonResult
from tests.report._inputs import (
    create_candidate,
    create_comparison_result,
    geomean_of,
    kind_metric,
    table_region,
)


def _grouped_flat_result() -> ComparisonResult:
    """Single ``time`` kind: ``entity`` group (2 members) + ungrouped ``warmup``."""
    geomean = geomean_of(-3.2, 3)
    return create_comparison_result(
        metrics={
            "entity/alive_check#time": kind_metric(
                kind="time",
                short_name="entity.alive_check",
                verdict="improved",
                delta=-10,
            ),
            "entity/spawn#time": kind_metric(
                kind="time",
                short_name="entity.spawn",
                verdict="regressed",
                delta=4,
            ),
            "warmup#time": kind_metric(
                kind="time",
                short_name="warmup",
                verdict="no-signal",
                delta=0.3,
            ),
        },
        candidates=[
            create_candidate(
                kinds=[
                    KindAggregate(
                        kind="time",
                        geomean=geomean,
                        groups=(
                            GroupAggregate(
                                group="entity",
                                geomean=geomean_of(-3.1, 2),
                            ),
                        ),
                        gated_geomean=geomean,
                    )
                ]
            )
        ],
    )


# ---------------------------------------------------------------------------
# group headers and case names
# ---------------------------------------------------------------------------


def test_table_region_when_flat_body_with_groups_does_emit_group_headers_and_case_names():
    region = table_region(render_report(_grouped_flat_result()))

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
    region = table_region(render_report(_grouped_flat_result()))

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
# styled verdict cell
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("parts", "widths", "delta_style", "band_style", "expected"),
    [
        pytest.param(
            VerdictParts(glyph="~", delta="", word="", band="2.5%", pairs=""),
            VerdictWidths(delta=7, band=4),
            None,
            "dim",
            ("~           ±2.5%", [("~", "green"), ("±2.5%", "dim")]),
            id="delta-empty-band-present",
        ),
        pytest.param(
            VerdictParts(glyph="✓", delta="-10.0%", word="", band="2.5%", pairs="n=8"),
            VerdictWidths(delta=7, band=4),
            "red",
            "dim",
            ("✓   -10.0%  ±2.5%  n=8", [("✓", "green"), ("-10.0%", "red"), ("±2.5%", "dim")]),
            id="all-fields-present",
        ),
        pytest.param(
            VerdictParts(glyph="✓", delta="-10.0%", word="", band="", pairs="n=8"),
            VerdictWidths(delta=7, band=4),
            "red",
            "dim",
            ("✓   -10.0%         n=8", [("✓", "green"), ("-10.0%", "red")]),
            id="band-absent-with-pairs-reserves-band-slot",
        ),
        pytest.param(
            VerdictParts(glyph="≈", delta="", word="unstable", band="", pairs=""),
            VerdictWidths(delta=6, band=0),
            "red",
            None,
            ("≈  unstable", [("≈", "green"), ("unstable", "red")]),
            id="word-stands-in-for-delta",
        ),
        pytest.param(
            VerdictParts(glyph="~", delta="+4.0%", word="", band="", pairs=""),
            VerdictWidths(delta=6, band=0),
            None,
            None,
            ("~   +4.0%", [("~", "green")]),
            id="delta-unstyled-band-absent",
        ),
    ],
)
def test_verdict_cell_when_fields_padded_to_widths_does_style_only_field_text(
    parts: VerdictParts,
    widths: VerdictWidths,
    delta_style: str | None,
    band_style: str | None,
    expected: tuple[str, list[tuple[str, str]]],
):
    plain, styled = expected

    cell = verdict_cell(
        parts, widths, glyph_style="green", delta_style=delta_style, band_style=band_style
    )

    assert isinstance(cell, Text)
    assert cell.plain == plain
    assert [(cell.plain[span.start : span.end], span.style) for span in cell.spans] == styled


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
