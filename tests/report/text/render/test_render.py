"""Tests for the comparison report's table and the measurement report.

These cover the comparison table's run header, label truncation, and the
terminal-cell widths of CJK names. The single-candidate table's pair counts and
column alignment are pinned in ``test_single``.

They also cover the measurement report: its flat and grouped tables, multi-kind
sections, its golden output, and how its header and section titles are styled.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from rich.cells import cell_len

from gymrat.report.text.render import render_measure_report, render_report
from gymrat.report.types import ReportOptions
from tests._ansi import strip_ansi
from tests.report._assertions import (
    line_containing,
    line_starting_with,
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

    output = render_report(result, ReportOptions(header="iteration 3 · experiment vs baseline"))

    assert strip_ansi(output).split("\n")[:2] == [
        "iteration 3 · experiment vs baseline",
        "metric                      │ main         │ perf/faster-decode │ vs main",
    ]


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

    output = render_report(result)

    assert output.split("\n")[0] == (
        "gymrat compare · baseline main ↔ feature/en…-fastpath · 10 paired samples · adapter: mitata"
    )
    assert stripped_cells(line_starting_with(output, "metric")) == [
        "metric",
        "main",
        "feature/en…-fastpath",
        "vs main",
    ]
    assert stripped_cells(table_rows(output)[1])[0] == "decode/an-extremely-long-metric-name/time"


# ---------------------------------------------------------------------------
# CJK metric names — column widths measured in terminal cells
# ---------------------------------------------------------------------------


def test_render_report_when_metric_name_has_cjk_does_align_separator_columns():
    result = create_comparison_result(
        metrics={
            "ascii-name": permutation_metric(
                verdict="improved", delta=-10, baseline_median=1000, unit="ns"
            ),
            "测试指标": permutation_metric(
                verdict="regressed", delta=5, baseline_median=2000, unit="ns"
            ),
        }
    )

    report = render_report(result)
    ascii_row = line_starting_with(report, "ascii-name")
    cjk_row = line_starting_with(report, "测试指标")

    # Column separators must sit at the same terminal cell position in both rows,
    # meaning the label column accounted for double-width CJK cells.
    ascii_sep = cell_len(ascii_row[: ascii_row.index("│")])
    cjk_sep = cell_len(cjk_row[: cjk_row.index("│")])
    assert ascii_sep == cjk_sep


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

    assert rows == [
        "time                  │ main",
        "decode                │ 100ns ± 1%",
        "allocation_throughput │ main",
        "encode                │   100 ± 1%",
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


@pytest.mark.parametrize(
    ("needle", "marker", "expected"),
    [
        pytest.param("gymrat measure", "gymrat measure", ["1"], id="title-bold"),
        pytest.param("gymrat measure", "main", ["1", "4"], id="header-label-underlined"),
    ],
)
def test_render_measure_report_when_colored_does_style_each_element(
    needle: str, marker: str, expected: list[str]
):
    report = render_measure_report(two_kind_measurement(), ReportOptions(color=True))

    assert styles_at(line_containing(report, needle), marker) == expected
