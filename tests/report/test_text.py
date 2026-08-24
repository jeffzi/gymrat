"""Tests for the single-candidate comparison text report.

These port ``text.test.ts`` (the single-candidate cases). The two-candidate
blocks — the candidate-column sub-field alignment and the compact
multi-candidate aggregate cell — belong to the multi-candidate task and are left
out here. Where the TypeScript suite asserted column alignment by byte offset
(``separatorOffsets``), the port asserts alignment *within* a parsed cell — the
``±`` offset shared across value cells, the delta right-aligned across verdict
cells — since the box chrome is rich's rather than a hand-spliced grid.
"""

from __future__ import annotations

import math

import pytest

from gymrat_py.model import ApproximateVerdict, Exclusion
from gymrat_py.report.text import render_report
from gymrat_py.report.types import CandidateMetric, MetricComparison, ReportOptions
from tests.report._inputs import (
    band_metric,
    cells_of,
    create_candidate,
    create_comparison_result,
    delta_cell,
    exact_metric,
    exact_verdict,
    last_table_row,
    line_containing,
    line_starting_with,
    metric_meta,
    other_kind,
    signed_rank_metric,
    strip_ansi,
    styles_at,
    two_kind_result,
)

# ---------------------------------------------------------------------------
# run header
# ---------------------------------------------------------------------------


def test_render_report_when_rendering_header_does_name_roles_variants_samples_adapter():
    result = create_comparison_result(
        baseline_label="main",
        candidates=[create_candidate(label="experiment")],
        samples=10,
        adapter="mitata",
    )

    output = render_report(result)

    assert (
        "gymrat compare · baseline main ↔ experiment · 10 paired samples · adapter: mitata"
        in output
    )


@pytest.mark.parametrize(
    ("samples", "expected"),
    [
        pytest.param(1, "· 1 paired sample ·", id="one"),
        pytest.param(2, "· 2 paired samples ·", id="many"),
    ],
)
def test_render_report_when_header_counts_samples_does_match_noun(samples: int, expected: str):
    output = render_report(create_comparison_result(samples=samples))

    assert expected in output


def test_render_report_when_several_candidates_does_list_all_against_the_baseline():
    result = create_comparison_result(
        baseline_label="main",
        candidates=[
            create_candidate(label="candidate-a"),
            create_candidate(label="candidate-b"),
        ],
    )

    output = render_report(result)

    assert "gymrat compare · baseline main ↔ candidate-a, candidate-b ·" in output


def test_render_report_when_header_override_given_does_replace_the_compare_header():
    result = create_comparison_result()

    output = render_report(result, ReportOptions(header="iteration 3 · experiment vs baseline"))

    assert strip_ansi(output).split("\n")[0] == "iteration 3 · experiment vs baseline"
    assert "gymrat compare" not in strip_ansi(output)


# ---------------------------------------------------------------------------
# table header
# ---------------------------------------------------------------------------


def test_render_report_when_rendering_table_header_does_label_columns():
    result = create_comparison_result()

    output = render_report(result)
    header_line = line_starting_with(output, "metric")

    assert [cell.strip() for cell in cells_of(header_line)] == [
        "metric",
        "main",
        "perf/faster-decode",
        "vs main",
    ]
    assert "gymrat compare" in output
    assert "geomean" in output


# ---------------------------------------------------------------------------
# label truncation
# ---------------------------------------------------------------------------


def test_render_report_when_variant_label_overflows_does_truncate_leaving_metric_names_whole():
    result = create_comparison_result(
        baseline_label="main",
        candidates=[create_candidate(label="feature/entity-spawn-fastpath")],
        metrics={
            "decode/an-extremely-long-metric-name/time": signed_rank_metric(
                verdict="improved", delta=-10, unit="ns"
            ),
        },
    )

    output = render_report(result)

    assert "feature/entity-spawn-fastpath" not in output
    assert "feature/en…-fastpath" in output
    assert "decode/an-extremely-long-metric-name/time" in output


# ---------------------------------------------------------------------------
# metric rows
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("verdict", "delta", "noise_pct", "expected"),
    [
        pytest.param("improved", -17.5, 2.5, "✓  -17.5%  ±2.5%", id="improved-with-band"),
        pytest.param("unstable", -50, 30, "≈  unstable", id="unstable-word-alone"),
    ],
)
def test_render_report_when_rendering_a_metric_row_does_pair_glyph_delta_and_band(
    verdict: ApproximateVerdict, delta: float, noise_pct: float, expected: str
):
    result = create_comparison_result(
        metrics={
            "decode/time": signed_rank_metric(
                verdict=verdict, delta=delta, noise_pct=noise_pct, unit="ns"
            )
        }
    )

    row = line_starting_with(render_report(result), "decode/time")

    assert cells_of(row)[-1].strip() == expected


def test_render_report_when_rendering_a_metric_row_does_drop_pair_count_and_pvalue():
    result = create_comparison_result(
        metrics={"decode/time": signed_rank_metric(verdict="improved", delta=-10, p=0.002)}
    )

    row = line_starting_with(render_report(result), "decode/time")

    assert "n=" not in row
    assert "p=" not in row


def test_render_report_when_metric_is_one_sided_does_show_only_the_measured_side():
    result = create_comparison_result(
        metrics={
            "old-only/time": MetricComparison(
                baseline_median=2048,
                baseline_spread=2,
                candidates=(CandidateMetric(),),
                meta=metric_meta("old-only/time", gating=False, unit="ns"),
            ),
        }
    )

    row = line_starting_with(render_report(result), "old-only/time")

    assert "2.0µs ± 2%" in row
    assert row.endswith("│")


def test_render_report_when_delta_is_undefined_arithmetic_does_keep_the_glyph():
    result = create_comparison_result(
        metrics={
            "nan-delta/count": MetricComparison(
                baseline_median=0,
                baseline_spread=None,
                candidates=(CandidateMetric(median=120, verdict=exact_verdict(delta=math.nan)),),
                meta=metric_meta("nan-delta/count", exact=True),
            ),
        }
    )

    row = line_starting_with(render_report(result), "nan-delta/count")

    assert cells_of(row)[-1].strip() == "~"


def test_render_report_when_spread_exceeds_the_median_does_state_it_in_absolute_units():
    result = create_comparison_result(
        metrics={
            "jittery/heap": signed_rank_metric(
                verdict="unstable",
                delta=5,
                baseline_median=5,
                baseline_spread=7620,
                unit="bytes",
            ),
        }
    )

    row = line_starting_with(render_report(result), "jittery/heap")

    assert cells_of(row)[1].strip() == "5B ± 381B"


@pytest.mark.parametrize(
    ("metric", "expected"),
    [
        pytest.param(
            signed_rank_metric(verdict="improved", delta=-10, n=8),
            "✓  -10.0%  ±2.5%  n=8",
            id="signed-rank",
        ),
        pytest.param(
            band_metric(verdict="improved", delta=-5, n=4),
            "✓  -5.0%  ±2.5%  n=4",
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

    assert cells_of(row)[-1].strip() == expected


# ---------------------------------------------------------------------------
# value column alignment
# ---------------------------------------------------------------------------


def test_render_report_when_columns_differ_in_width_does_right_align_value_cells():
    result = create_comparison_result(
        metrics={
            "short": signed_rank_metric(
                verdict="improved", delta=-50, baseline_median=914, unit="ns"
            ),
            "very-long-metric-name": signed_rank_metric(
                verdict="improved", delta=-10, baseline_median=49152, unit="bytes"
            ),
        }
    )

    report = render_report(result)
    short_cell = cells_of(line_starting_with(report, "short"))[1]
    long_cell = cells_of(line_starting_with(report, "very-long-metric-name"))[1]

    assert len(short_cell.rstrip()) == len(long_cell.rstrip())
    assert short_cell.strip() == "914ns ± 1%"
    assert long_cell.strip() == "49.2KB ± 1%"


def test_render_report_when_widths_differ_does_stack_separators_across_header_rows_and_geomean():
    result = create_comparison_result(
        metrics={
            "a/time": signed_rank_metric(verdict="improved", delta=-5, unit="ns"),
            "a-much-longer-metric/time": signed_rank_metric(
                verdict="improved", delta=-5, baseline_median=100000, unit="ns"
            ),
        },
        candidates=[create_candidate(kinds=[other_kind(-5, 2)])],
    )

    report = render_report(result)
    short_value = cells_of(line_starting_with(report, "a/time"))[1]
    long_value = cells_of(line_starting_with(report, "a-much-longer-metric/time"))[1]
    metric_delta = cells_of(line_starting_with(report, "a/time"))[-1]
    geomean_delta = cells_of(line_starting_with(report, "geomean"))[-1]

    assert short_value.index("±") == long_value.index("±")
    assert geomean_delta.index("-5.0%") + len("-5.0%") == metric_delta.index("-5.0%") + len("-5.0%")


@pytest.mark.parametrize(
    ("metrics", "first", "second"),
    [
        pytest.param(
            {
                "first/metric": signed_rank_metric(
                    verdict="improved",
                    delta=-10,
                    baseline_median=162000,
                    baseline_spread=9,
                    unit="ns",
                ),
                "second/metric": signed_rank_metric(
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
                "first/metric": signed_rank_metric(
                    verdict="improved",
                    delta=-10,
                    baseline_median=5,
                    baseline_spread=7620,
                    unit="bytes",
                ),
                "second/metric": signed_rank_metric(
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
                "first/metric": signed_rank_metric(
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


def test_render_report_when_aligning_the_verdict_column_does_right_align_deltas_and_pin_band():
    result = create_comparison_result(
        metrics={
            "regressed/time": signed_rank_metric(
                verdict="regressed", delta=0.4, noise_pct=2.5, unit="ns"
            ),
            "flat/time": signed_rank_metric(verdict="no-signal", delta=0, noise_pct=100, unit="ns"),
            "improved/time": signed_rank_metric(
                verdict="improved", delta=-12.4, noise_pct=30, unit="ns"
            ),
        }
    )

    report = render_report(result)

    assert (
        cells_of(line_starting_with(report, "regressed/time"))[-1].strip() == "✗   +0.4%  ±  2.5%"
    )
    assert cells_of(line_starting_with(report, "flat/time"))[-1].strip() == "~    0.0%  ±100.0%"
    assert cells_of(line_starting_with(report, "improved/time"))[-1].strip() == "✓  -12.4%  ± 30.0%"


def test_render_report_when_a_verdict_is_unstable_does_seat_the_word_without_widening_others():
    result = create_comparison_result(
        metrics={
            "improved/time": signed_rank_metric(verdict="improved", delta=-12.4, unit="ns"),
            "jittery/time": signed_rank_metric(
                verdict="unstable", delta=-50, noise_pct=30, unit="ns"
            ),
        }
    )

    report = render_report(result)

    assert cells_of(line_starting_with(report, "improved/time"))[-1].strip() == "✓  -12.4%  ±2.5%"
    assert cells_of(line_starting_with(report, "jittery/time"))[-1].strip() == "≈  unstable"


# ---------------------------------------------------------------------------
# geomean row
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("n", "expected_label"),
    [
        pytest.param(4, "geomean (4 stable metrics)", id="plural"),
        pytest.param(1, "geomean (1 stable metric)", id="singular"),
    ],
)
def test_render_report_when_rendering_the_geomean_row_does_label_with_the_metric_count(
    n: int, expected_label: str
):
    result = create_comparison_result(
        metrics={"a/time": signed_rank_metric(verdict="improved", delta=-6)},
        candidates=[create_candidate(kinds=[other_kind(-5.8, n)])],
    )

    row = line_starting_with(render_report(result), "geomean")

    assert cells_of(row)[0].strip() == expected_label


def test_render_report_when_rendering_the_geomean_row_does_align_its_delta_with_the_column():
    result = create_comparison_result(
        metrics={"a/time": signed_rank_metric(verdict="improved", delta=-17.9)},
        candidates=[create_candidate(kinds=[other_kind(-6, 1)])],
    )

    report = render_report(result)
    metric_cell = cells_of(line_starting_with(report, "a/time"))[-1]
    geomean_cell = cells_of(line_starting_with(report, "geomean"))[-1]

    assert geomean_cell.index("-6.0%") + len("-6.0%") == metric_cell.index("-17.9%") + len("-17.9%")


def test_render_report_when_rendering_the_geomean_row_does_leave_excluded_to_the_summary():
    result = create_comparison_result(
        metrics={"a/time": signed_rank_metric(verdict="improved", delta=-6)},
        candidates=[
            create_candidate(
                kinds=[
                    other_kind(
                        0,
                        1,
                        excluded=[Exclusion(metric="nan-delta/count", reason="undefined-ratio")],
                    )
                ]
            )
        ],
    )

    row = line_starting_with(render_report(result), "geomean")

    assert "excluded" not in row


def test_render_report_when_every_metric_excluded_does_report_no_stable_metrics():
    result = create_comparison_result(
        metrics={"jittery/time": signed_rank_metric(verdict="unstable", delta=-50)},
        candidates=[
            create_candidate(
                kinds=[
                    other_kind(
                        math.nan,
                        0,
                        excluded=[Exclusion(metric="jittery/time", reason="unstable")],
                    )
                ]
            )
        ],
    )

    row = line_starting_with(render_report(result), "geomean")

    assert [cell.strip() for cell in cells_of(row)] == [
        "geomean",
        "",
        "",
        "—  no stable metrics",
    ]


# ---------------------------------------------------------------------------
# aggregate noise band
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        pytest.param("geomean · entity", "-3.1%  ±1.5%", id="group"),
        pytest.param("geomean · time", "-3.2%  ±2.0%", id="kind"),
    ],
)
def test_render_report_when_an_aggregate_carries_a_band_does_state_it_behind_the_delta(
    label: str, expected: str
):
    row = line_starting_with(render_report(two_kind_result()), label)

    assert delta_cell(row).strip() == expected


def test_render_report_when_flat_geomean_carries_a_band_does_state_it_behind_the_delta():
    result = create_comparison_result(
        metrics={"faster/time": signed_rank_metric(verdict="improved", delta=-17.5)},
        candidates=[create_candidate(kinds=[other_kind(-5.8, 1, band=1.2)])],
    )

    row = line_starting_with(render_report(result), "geomean")

    assert delta_cell(row).strip() == "-5.8%  ±1.2%"


def test_render_report_when_an_aggregate_has_no_band_does_print_the_delta_alone():
    row = line_starting_with(render_report(two_kind_result()), "geomean · memory")

    assert delta_cell(row).strip() == "-7.0%"


def test_render_report_when_only_the_aggregate_carries_a_band_does_widen_the_verdict_column():
    result = create_comparison_result(
        samples=1,
        metrics={"decode/time": band_metric(delta=-0.4, noise_pct=0.5, n=1, unit="ns")},
        candidates=[create_candidate(kinds=[other_kind(-0.1, 1, band=0.5)])],
    )

    report = render_report(result)
    row = line_starting_with(report, "geomean")
    rule = next(
        bare
        for line in report.split("\n")
        if (bare := strip_ansi(line)) and set(bare) <= set("─┼┬")
    )

    assert delta_cell(row).strip() == "-0.1%  ±0.5%"
    assert len(strip_ansi(row).rstrip()) <= len(rule)


def test_render_report_when_an_aggregate_carries_a_band_does_line_it_up_with_metric_rows():
    report = render_report(two_kind_result())

    assert delta_cell(line_starting_with(report, "geomean · time")).index("±") == delta_cell(
        line_starting_with(report, "  alive_check")
    ).index("±")


def test_render_report_when_rendering_with_color_does_dim_the_aggregate_band(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("FORCE_COLOR", "1")

    line = line_containing(render_report(two_kind_result()), "geomean · time")

    assert "2" in styles_at(line, "±2.0%")


# ---------------------------------------------------------------------------
# closing the table
# ---------------------------------------------------------------------------


def test_render_report_when_closing_the_table_does_end_on_the_geomean_row():
    result = create_comparison_result(
        metrics={"a/time": signed_rank_metric(verdict="improved", delta=-6)}
    )

    row = last_table_row(render_report(result))

    assert cells_of(row)[0].strip() == "geomean (10 stable metrics)"
