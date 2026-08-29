"""Tests for grouped rendering in single-section (flat) measurement reports.

A single-kind measurement renders flat (no section borders).  This test
verifies that the flat measurement table groups metrics under group headers,
using case names for member rows rather than full metric names.
"""

from __future__ import annotations

from gymrat.report.text import render_measure_report
from tests.report._inputs import (
    create_measurement_result,
    measured_metric,
    table_region,
)

# ---------------------------------------------------------------------------
# flat measurement grouping
# ---------------------------------------------------------------------------


def test_render_measure_report_when_single_kind_grouped_does_show_group_headers():
    """A single-kind measurement with grouped metrics must group under headers."""
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
