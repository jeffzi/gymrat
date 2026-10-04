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
from tests.report._inputs import (
    create_measurement_result,
    measured_metric,
    table_region,
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

    assert "alive_check" in region
    assert "spawn" in region
    assert "entity/alive_check#time" not in region
    assert "entity/spawn#time" not in region


# ---------------------------------------------------------------------------
# whole report — golden
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("color", [False, True], ids=["plain", "colored"])
def test_render_measure_report_when_rendered_does_match_its_golden(
    color: bool, snapshot: SnapshotAssertion
):
    report = render_measure_report(two_kind_measurement(), ReportOptions(color=color))

    assert report.split("\n") == snapshot
