"""Tests for grouped rendering in single-section (flat) measurement reports.

A single-kind measurement renders flat (no section borders).  This test
verifies that the flat measurement table groups metrics under group headers,
using case names for member rows rather than full metric names.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from gymrat.report.text.render import render_measure_report
from gymrat.report.types import ReportOptions
from tests.report._assertions import table_region
from tests.report._measurements import (
    create_measurement_result,
    measured_metric,
    two_kind_measurement,
)

if TYPE_CHECKING:
    from syrupy.assertion import SnapshotAssertion

# ---------------------------------------------------------------------------
# flat measurement grouping
# ---------------------------------------------------------------------------


def test_render_measure_report_when_single_kind_grouped_does_show_group_headers():
    result = create_measurement_result(
        metrics={
            "entity/alive_check#time": measured_metric(
                kind="time",
                short_name="entity.alive_check",
                unit="ns",
            ),
            "entity/spawn#time": measured_metric(
                kind="time",
                short_name="entity.spawn",
                median=104,
                unit="ns",
            ),
        },
    )

    region = table_region(render_measure_report(result))

    assert region == [
        "gymrat measure · main · 10 samples · adapter: mitata",
        "metric",
        "<rule>",
        "entity · time",
        "alive_check",
        "spawn",
    ]


# ---------------------------------------------------------------------------
# multi-kind measurement with no groups
# ---------------------------------------------------------------------------

_LONG_KIND = "allocation_throughput"


def _ungrouped_two_kind_rows() -> list[str]:
    """The table rows of a two-kind measurement whose metrics sit in no group."""
    result = create_measurement_result(
        metrics={
            "decode#time": measured_metric(kind="time", short_name="decode", unit="ns"),
            f"encode#{_LONG_KIND}": measured_metric(kind=_LONG_KIND, short_name="encode"),
        },
    )
    report = render_measure_report(result, ReportOptions(color=False))
    return [line for line in report.split("\n") if "│" in line]


def test_render_measure_report_when_kinds_differ_without_groups_does_show_short_names():
    names = [line.split("│")[0].rstrip() for line in _ungrouped_two_kind_rows() if "100" in line]

    assert names == ["decode", "encode"]


def test_render_measure_report_when_section_title_is_widest_does_size_the_metric_column_to_it():
    separators = {line.index("│") for line in _ungrouped_two_kind_rows()}

    assert separators == {len(_LONG_KIND) + 1}


# ---------------------------------------------------------------------------
# whole report — golden
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("color", [False, True], ids=["plain", "colored"])
def test_render_measure_report_when_rendered_does_match_its_golden(
    color: bool, snapshot: SnapshotAssertion
):
    report = render_measure_report(two_kind_measurement(), ReportOptions(color=color))

    assert report.split("\n") == snapshot
