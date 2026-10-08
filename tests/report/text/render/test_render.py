"""Tests for the comparison report's table and the measurement report.

These cover the comparison table's run header, label truncation, pair-count
annotations, and column alignment: the ``±`` offset shared across value cells,
the delta right-aligned across verdict cells, aggregate bands, and terminal-cell
widths for CJK names. Alignment is asserted *within* a parsed cell, since the box
chrome is rich's rather than a hand-spliced grid.

They also cover the measurement report: its flat and grouped tables, multi-kind
sections, its golden output, and how its header and section titles are styled.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from rich.cells import cell_len

from gymrat.report.text.render import render_measure_report, render_report
from gymrat.report.types import CandidateMetric, MetricComparison, ReportOptions
from tests._ansi import strip_ansi
from tests.report._assertions import (
    cells_of,
    delta_cell,
    line_containing,
    line_starting_with,
    rule_lines,
    stripped_cells,
    styles_at,
    table_region,
    table_rows,
)
from tests.report._comparisons import (
    create_candidate,
    create_comparison_result,
    exact_metric,
    metric_meta,
    other_kind,
    permutation_metric,
)
from tests.report._measurements import (
    create_measurement_result,
    measured_metric,
    two_kind_measurement,
)
from tests.report._verdicts import band_metric, exact_verdict

if TYPE_CHECKING:
    from syrupy.assertion import SnapshotAssertion

    from gymrat.report.types import MeasurementResult


# ---------------------------------------------------------------------------
# run header
# ---------------------------------------------------------------------------


def test_render_report_when_header_override_given_does_replace_the_compare_header():
    result = create_comparison_result()

    output = render_report(result, ReportOptions(header="iteration 3 · experiment vs baseline"))

    assert strip_ansi(output).split("\n")[0] == "iteration 3 · experiment vs baseline"
    assert "gymrat compare" not in strip_ansi(output)


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

    assert "feature/entity-spawn-fastpath" not in output
    assert "feature/en…-fastpath" in output
    assert stripped_cells(table_rows(output)[1])[0] == "decode/an-extremely-long-metric-name/time"


# ---------------------------------------------------------------------------
# metric rows
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("metric", "expected"),
    [
        pytest.param(
            permutation_metric(verdict="improved", delta=-10, n=8),
            "✓  -10.0%  ±2.5%  n=8",
            id="permutation",
        ),
        pytest.param(
            band_metric(verdict="improved", delta=-5, n=8),
            "✓  -5.0%  ±2.5%  n=8",
            id="band",
        ),
        pytest.param(
            exact_metric(delta=-7.9, n=6, unit="ns"),
            "✓  -7.9%  n=6",
            id="exact",
        ),
    ],
)
def test_render_report_when_metric_paired_fewer_rounds_does_annotate_with_pair_count(
    metric: MetricComparison, expected: str
):
    result = create_comparison_result(samples=10, metrics={"decode/time": metric})

    row = line_starting_with(render_report(result), "decode/time")

    assert stripped_cells(row)[-1] == expected


# ---------------------------------------------------------------------------
# value column alignment
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("metrics", "first", "second"),
    [
        pytest.param(
            {
                "first/metric": permutation_metric(
                    verdict="improved",
                    delta=-10,
                    baseline_median=162000,
                    baseline_spread=9,
                    unit="ns",
                ),
                "second/metric": permutation_metric(
                    verdict="improved",
                    delta=-10,
                    baseline_median=29200,
                    baseline_spread=12,
                    unit="ns",
                ),
            },
            "162.0µs ±  9%",
            " 29.2µs ± 12%",
            id="percentage-spreads",
        ),
        pytest.param(
            {
                "first/metric": permutation_metric(
                    verdict="improved",
                    delta=-10,
                    baseline_median=5,
                    baseline_spread=7620,
                    unit="bytes",
                ),
                "second/metric": permutation_metric(
                    verdict="improved",
                    delta=-10,
                    baseline_median=49152,
                    baseline_spread=1,
                    unit="bytes",
                ),
            },
            "    5B ± 381B",
            "49.2KB ±   1%",
            id="absolute-beside-percentage",
        ),
    ],
)
def test_render_report_when_aligning_value_columns_does_stack_magnitude_and_spread(
    metrics: dict[str, MetricComparison], first: str, second: str
):
    report = render_report(create_comparison_result(metrics=metrics))
    first_cell = cells_of(line_starting_with(report, "first/metric"))[1]
    second_cell = cells_of(line_starting_with(report, "second/metric"))[1]

    assert first in first_cell
    assert second in second_cell
    assert first_cell.index("±") == second_cell.index("±")


def test_render_report_when_a_magnitude_has_no_spread_does_keep_it_in_the_magnitude_field():
    report = render_report(
        create_comparison_result(
            metrics={
                "first/metric": permutation_metric(
                    verdict="improved",
                    delta=-10,
                    baseline_median=2048,
                    baseline_spread=2,
                    unit="ns",
                ),
                "second/metric": MetricComparison(
                    baseline_median=120,
                    baseline_spread=None,
                    candidates=(CandidateMetric(median=120, verdict=exact_verdict()),),
                    meta=metric_meta("second/metric", exact=True),
                ),
            }
        )
    )
    first_cell = cells_of(line_starting_with(report, "first/metric"))[1]
    second_cell = cells_of(line_starting_with(report, "second/metric"))[1]

    assert first_cell.index("2.0µs") + len("2.0µs") == second_cell.index("120") + len("120")


# ---------------------------------------------------------------------------
# verdict column alignment
# ---------------------------------------------------------------------------


def test_render_report_when_aligning_the_verdict_column_does_lay_out_glyph_delta_and_band():
    result = create_comparison_result(
        metrics={
            "regressed/time": permutation_metric(
                verdict="regressed", delta=0.4, noise_pct=2.5, unit="ns"
            ),
            "flat/time": permutation_metric(verdict="no-signal", delta=0, noise_pct=100, unit="ns"),
            "improved/time": permutation_metric(
                verdict="improved", delta=-12.4, noise_pct=30, unit="ns"
            ),
        }
    )

    report = render_report(result)

    assert stripped_cells(line_starting_with(report, "regressed/time"))[-1] == "✗   +0.4%  ±  2.5%"
    assert stripped_cells(line_starting_with(report, "flat/time"))[-1] == "~    0.0%  ±100.0%"
    assert stripped_cells(line_starting_with(report, "improved/time"))[-1] == "✓  -12.4%  ± 30.0%"


# ---------------------------------------------------------------------------
# aggregate noise band
# ---------------------------------------------------------------------------


def test_render_report_when_flat_geomean_carries_a_band_does_state_it_behind_the_delta():
    result = create_comparison_result(
        metrics={"faster/time": permutation_metric(verdict="improved", delta=-17.5)},
        candidates=[create_candidate(kinds=[other_kind(-5.8, 1, band=1.2)])],
    )

    row = line_starting_with(render_report(result), "geomean")

    assert delta_cell(row).strip() == "-5.8%  ±1.2%"


def test_render_report_when_only_the_aggregate_carries_a_band_does_widen_the_verdict_column():
    result = create_comparison_result(
        samples=1,
        metrics={"decode/time": band_metric(delta=-0.4, noise_pct=0.5, n=1, unit="ns")},
        candidates=[create_candidate(kinds=[other_kind(-0.1, 1, band=0.5)])],
    )

    report = render_report(result)
    row = line_starting_with(report, "geomean")
    rule = rule_lines(report)[0]

    assert delta_cell(row).strip() == "-0.1%  ±0.5%"
    assert len(strip_ansi(row).rstrip()) <= len(rule)


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
        pytest.param("gymrat measure", "·", ["2"], id="header-separator-dim"),
        pytest.param("gymrat measure", "main", ["1", "4"], id="header-label-underlined"),
        pytest.param("time", "time", ["1"], id="section-title-bold"),
        pytest.param("entity", "entity", ["1", "34"], id="group-header-blue"),
        pytest.param("informational", "informational", ["2"], id="informational-dim"),
    ],
)
def test_render_measure_report_when_colored_does_style_each_element(
    needle: str, marker: str, expected: list[str]
):
    report = render_measure_report(two_kind_measurement(), ReportOptions(color=True))

    assert styles_at(line_containing(report, needle), marker) == expected
