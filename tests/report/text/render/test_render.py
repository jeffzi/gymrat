"""Tests for the comparison report's table and the measurement report.

These cover the comparison table's run header and label truncation. The
single-candidate table's pair counts and column alignment, CJK names included,
are pinned in ``test_single``.

They also cover the measurement report: its flat and grouped tables, multi-kind
sections, its golden output, and how its header label is styled.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from gymrat.report.labels import truncate_labels
from gymrat.report.style import VARIANT_NAME_STYLE
from gymrat.report.text.render import render_measure_report, render_report
from gymrat.report.types import ReportOptions
from tests.report._assertions import (
    line_containing,
    line_starting_with,
    sgr_for,
    stripped_cells,
    styles_at,
    table_region,
    table_rows,
)
from tests.report._comparisons import (
    create_candidate,
    create_comparison_result,
    permutation_metric,
)
from tests.report._measurements import (
    create_measurement_result,
    entity_time_measurements,
    measured_metric,
    two_kind_measurement,
)

if TYPE_CHECKING:
    from syrupy.assertion import SnapshotAssertion

    from gymrat.report.types import MeasurementResult


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


def test_render_report_when_variant_label_overflows_does_truncate_leaving_metric_names_whole():
    result = create_comparison_result(
        baseline_label="main",
        candidates=[create_candidate(label="feature/entity-spawn-fastpath")],
        metrics={
            "decode/an-extremely-long-metric-name/time": permutation_metric(
                verdict="improved", delta=-10, unit="ns"
            ),
        },
    )

    shown = truncate_labels(["main", "feature/entity-spawn-fastpath"])[1]

    output = render_report(result)

    assert output.split("\n")[0] == (
        f"gymrat compare · baseline main ↔ {shown} · 10 paired samples · adapter: mitata"
    )
    assert stripped_cells(line_starting_with(output, "metric")) == [
        "metric",
        "main",
        shown,
        "vs main",
    ]
    assert stripped_cells(table_rows(output)[1])[0] == "decode/an-extremely-long-metric-name/time"


# ---------------------------------------------------------------------------
# flat measurement table
# ---------------------------------------------------------------------------


def _flat_measurement() -> MeasurementResult:
    """A flat single-kind run of two metrics measured in nanoseconds."""
    return create_measurement_result(
        metrics={
            "decode/time": measured_metric(median=100, spread=1, unit="ns"),
            "encode/time": measured_metric(median=2048, spread=2, unit="ns"),
        }
    )


def test_render_measure_report_when_one_kind_does_draw_one_flat_table_of_medians():
    report = render_measure_report(_flat_measurement())

    assert table_region(report) == [
        "gymrat measure · main · 10 samples · adapter: mitata",
        "metric",
        "<rule>",
        "decode/time",
        "encode/time",
    ]
    assert stripped_cells(line_starting_with(report, "metric")) == ["metric", "main"]
    assert [stripped_cells(row) for row in table_rows(report)[1:]] == [
        ["decode/time", "100ns ± 1%"],
        ["encode/time", "2.0µs ± 2%"],
    ]


def test_render_measure_report_when_metric_has_no_spread_does_state_the_bare_median():
    result = create_measurement_result(
        metrics={"decode/time": measured_metric(median=100, spread=None, unit="ns")}
    )

    row = line_starting_with(render_measure_report(result), "decode/time")

    assert stripped_cells(row)[-1] == "100ns"


# ---------------------------------------------------------------------------
# flat measurement grouping
# ---------------------------------------------------------------------------


def test_render_measure_report_when_single_kind_grouped_does_show_group_headers():
    result = create_measurement_result(metrics=entity_time_measurements())

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


def test_render_measure_report_when_kinds_differ_without_groups_does_show_short_names_under_titles():
    result = create_measurement_result(
        metrics={
            "decode#time": measured_metric(kind="time", short_name="decode", unit="ns"),
            "encode#allocation_throughput": measured_metric(
                kind="allocation_throughput", short_name="encode"
            ),
        },
    )

    rows = table_rows(render_measure_report(result, ReportOptions(color=False)))

    assert [stripped_cells(row) for row in rows] == [
        ["time", "main"],
        ["decode", "100ns ± 1%"],
        ["allocation_throughput", "main"],
        ["encode", "100 ± 1%"],
    ]


# ---------------------------------------------------------------------------
# whole report — golden
# ---------------------------------------------------------------------------


def test_render_measure_report_when_rendered_does_match_its_golden(snapshot: SnapshotAssertion):
    report = render_measure_report(two_kind_measurement(), ReportOptions(color=False))

    assert report.split("\n") == snapshot


# ---------------------------------------------------------------------------
# measurement styling
# ---------------------------------------------------------------------------


def test_render_measure_report_when_colored_does_underline_the_header_label():
    report = render_measure_report(two_kind_measurement(), ReportOptions(color=True))

    assert styles_at(line_containing(report, "gymrat measure"), "main") == sgr_for(
        VARIANT_NAME_STYLE
    )
