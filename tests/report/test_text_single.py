"""Tests for grouped rendering in single-candidate, single-section comparison reports.

A single-kind comparison renders flat (no section borders).  These tests verify
that the flat renderer shows group headers with a kind suffix, indents member
rows by the group indent, and uses case names rather than full metric names.
"""

from __future__ import annotations

from gymrat.report.text import render_report
from gymrat.verdict import GroupAggregate, KindAggregate
from tests.report._inputs import (
    cells_of,
    create_candidate,
    create_comparison_result,
    geomean_of,
    kind_metric,
    line_starting_with,
    strip_ansi,
)


def _grouped_flat_result():
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
# indented member rows
# ---------------------------------------------------------------------------


def test_render_report_when_flat_grouped_does_indent_member_rows():
    """Grouped metrics show their case name indented, not the full metric name."""
    report = render_report(_grouped_flat_result())

    line = line_starting_with(report, "  alive_check")

    assert cells_of(line)[0].rstrip() == "  alive_check"


# ---------------------------------------------------------------------------
# kind suffix on group header
# ---------------------------------------------------------------------------


def test_render_report_when_flat_grouped_does_show_kind_on_group_header():
    """The group header carries the kind when it is uniform across the group.

    In the flat (single-section) layout there is no section header to carry the
    kind, so it is stated on the group header instead (e.g. ``entity  time``).
    """
    report = strip_ansi(render_report(_grouped_flat_result()))

    entity_header = line_starting_with(report, "entity ")

    assert "time" in cells_of(entity_header)[0]
