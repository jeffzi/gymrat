"""Tests for the comparison report's table and the measurement report.

These cover the comparison table's run header and label truncation.

They also cover the measurement report's golden output.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from gymrat.report.labels import truncate_labels
from gymrat.report.text.render import render_measure_report, render_report
from gymrat.report.types import ReportOptions
from tests.report._assertions import (
    line_starting_with,
    stripped_cells,
    table_region,
)
from tests.report._comparisons import (
    create_candidate,
    create_comparison_result,
)
from tests.report._measurements import two_kind_measurement

if TYPE_CHECKING:
    from syrupy.assertion import SnapshotAssertion


# ---------------------------------------------------------------------------
# run header
# ---------------------------------------------------------------------------


def test_render_report_when_header_override_given_does_replace_the_compare_header():
    result = create_comparison_result()

    report = render_report(result, ReportOptions(header="iteration 3 · experiment vs baseline"))

    assert table_region(report)[:2] == ["iteration 3 · experiment vs baseline", "metric"]


# ---------------------------------------------------------------------------
# label truncation
# ---------------------------------------------------------------------------


def test_render_report_when_variant_label_overflows_does_truncate_it():
    result = create_comparison_result(
        baseline_label="main",
        candidates=[create_candidate(label="feature/entity-spawn-fastpath")],
    )
    shown = truncate_labels(["main", "feature/entity-spawn-fastpath"])[1]

    output = render_report(result)

    header = output.split("\n")[0]
    assert shown in header
    assert stripped_cells(line_starting_with(output, "metric")) == [
        "metric",
        "main",
        shown,
        "vs main",
    ]


# ---------------------------------------------------------------------------
# whole report — golden
# ---------------------------------------------------------------------------


def test_render_measure_report_when_rendered_does_match_its_golden(snapshot: SnapshotAssertion):
    report = render_measure_report(two_kind_measurement(), ReportOptions(color=False))

    assert report.split("\n") == snapshot
