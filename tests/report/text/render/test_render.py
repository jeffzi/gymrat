"""Tests for the single-candidate comparison text report.

These cover the single-candidate cases. The two-candidate blocks — the
candidate-column sub-field alignment and the compact multi-candidate aggregate
cell — belong to the multi-candidate task and are left out here. Column
alignment is asserted *within* a parsed cell — the ``±`` offset shared across
value cells, the delta right-aligned across verdict cells — since the box chrome
is rich's rather than a hand-spliced grid.

They also cover the flat measurement report's grouped rows (metrics under group
headers, members named by case), selecting and labelling the metrics a report
highlights, and the footer lines that explain noise bands and suggest more
samples. Styling is asserted by intent rather than exact escape bytes: markup
is rendered through :func:`gymrat.report.style.render_lines` with color off to
check the plain content, and with color on to check that the expected SGR
attribute code is present.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import pytest
from rich.cells import cell_len

from gymrat.model import PERMUTATION_MIN_N
from gymrat.report.style import format_hint
from gymrat.report.text.render import (
    footer_lines,
    render_measure_report,
    render_report,
    select_highlights,
)
from gymrat.report.types import CandidateMetric, MetricComparison, ReportOptions
from tests._ansi import (
    sgr_codes,
    strip_ansi,
)
from tests.report._assertions import (
    cells_of,
    delta_cell,
    line_containing,
    line_starting_with,
    render_colored,
    render_plain,
    styles_at,
    table_region,
    table_rows,
)
from tests.report._comparisons import (
    NWayCandidate,
    create_candidate,
    create_comparison_result,
    exact_metric,
    metric_meta,
    n_way_metric,
    other_kind,
    permutation_metric,
    two_kind_result,
)
from tests.report._measurements import (
    create_measurement_result,
    measured_metric,
    two_kind_measurement,
)
from tests.report._verdicts import (
    CandidateSpec,
    approximate_metric,
    band_metric,
    exact_verdict,
    metric_for,
    one_sided_metric,
)

if TYPE_CHECKING:
    from syrupy.assertion import SnapshotAssertion

    from gymrat.report.types import MetricComparisons


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
    assert cells_of(table_rows(output)[1])[0].strip() == "decode/an-extremely-long-metric-name/time"


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

    assert cells_of(row)[-1].strip() == expected


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

    assert (
        cells_of(line_starting_with(report, "regressed/time"))[-1].strip() == "✗   +0.4%  ±  2.5%"
    )
    assert cells_of(line_starting_with(report, "flat/time"))[-1].strip() == "~    0.0%  ±100.0%"
    assert cells_of(line_starting_with(report, "improved/time"))[-1].strip() == "✓  -12.4%  ± 30.0%"


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
    rule = next(
        bare
        for line in report.split("\n")
        if (bare := strip_ansi(line)) and set(bare) <= set("─┼┬")
    )

    assert delta_cell(row).strip() == "-0.1%  ±0.5%"
    assert len(strip_ansi(row).rstrip()) <= len(rule)


def test_render_report_when_ascii_labels_differ_in_length_does_pad_inside_the_bold_label(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("FORCE_COLOR", "1")
    result = create_comparison_result(
        candidates=[
            create_candidate(label="fast", kinds=[other_kind(-10, 1)]),
            create_candidate(label="slower", kinds=[other_kind(4, 1)]),
        ],
        metrics={
            "decode/time": n_way_metric([
                NWayCandidate(verdict="improved", delta=-10, median=90),
                NWayCandidate(verdict="regressed", delta=4, median=104),
            ])
        },
    )

    report = render_report(result)

    assert styles_at(line_containing(report, "fast  "), "fast  ") == ["1"]


def test_render_report_when_rendering_with_color_does_dim_the_aggregate_band(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("FORCE_COLOR", "1")

    line = line_containing(render_report(two_kind_result()), "geomean · time")

    assert "2" in styles_at(line, "±2.0%")


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


@pytest.mark.parametrize(
    ("labels", "summary_column"),
    [
        pytest.param(("fast", "测试指标"), 10, id="cjk-label-widest"),
        pytest.param(("a-longer-name", "测试"), 15, id="ascii-label-widest"),
        pytest.param(("fast", "slower"), 8, id="ascii-only"),
    ],
)
def test_render_report_when_candidate_labels_differ_in_width_does_start_every_summary_at_one_column(
    labels: tuple[str, str],
    summary_column: int,
):
    result = create_comparison_result(
        candidates=[
            create_candidate(label=labels[0], kinds=[other_kind(-10, 1)]),
            create_candidate(label=labels[1], kinds=[other_kind(4, 1)]),
        ],
        metrics={
            "decode/time": n_way_metric([
                NWayCandidate(verdict="improved", delta=-10, median=90),
                NWayCandidate(verdict="regressed", delta=4, median=104),
            ])
        },
    )

    report = strip_ansi(render_report(result))
    summaries = [line for line in report.split("\n") if re.search(r"✓ \d+ improved", line)]

    assert [line.split("  ✓")[0].rstrip() for line in summaries] == list(labels)
    assert [cell_len(line[: line.index("✓")]) for line in summaries] == [summary_column] * 2


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


@pytest.mark.parametrize("color", [False, True], ids=["plain", "colored"])
def test_render_measure_report_when_rendered_does_match_its_golden(
    color: bool, snapshot: SnapshotAssertion
):
    report = render_measure_report(two_kind_measurement(), ReportOptions(color=color))

    assert report.split("\n") == snapshot


# ---------------------------------------------------------------------------
# select_highlights
# ---------------------------------------------------------------------------


def test_select_highlights_when_mixed_verdicts_does_rank_movers_only():
    metrics: MetricComparisons = {
        "small-improvement/time": approximate_metric(verdict="improved", delta=-4),
        "quiet-unstable/time": approximate_metric(verdict="unstable", delta=6, noise_pct=210),
        "small-regression/time": approximate_metric(verdict="regressed", delta=3),
        "big-regression/ops": approximate_metric(
            verdict="regressed", delta=-12, direction="higher"
        ),
        "within-noise/time": approximate_metric(verdict="no-signal", delta=0.4),
        "big-improvement/time": approximate_metric(verdict="improved", delta=-20),
        "one-sided/time": one_sided_metric(),
        "loud-unstable/time": approximate_metric(verdict="unstable", delta=5, noise_pct=300),
        "tied/heap": band_metric(n=10, usable_n=0),
        "single-pair/time": band_metric(n=1, noise_pct=0.5),
        "short-improved/time": band_metric(verdict="improved", delta=-10, n=4),
    }

    highlights = select_highlights(metrics, 0)

    assert [highlight.name for highlight in highlights] == [
        "big-regression/ops",
        "small-regression/time",
        "big-improvement/time",
        "small-improvement/time",
        "loud-unstable/time",
        "quiet-unstable/time",
    ]


@pytest.mark.parametrize(
    ("delta", "baseline_median"),
    [
        pytest.param(0, 0, id="zero-baseline-median"),
        pytest.param(-100, 100, id="zero-candidate-median"),
        pytest.param(0, 1e-310, id="overflowing-noise-ratio"),
    ],
)
def test_select_highlights_when_unstable_around_zero_median_does_rank_by_absolute_noise(
    delta: float, baseline_median: float
):
    metrics: MetricComparisons = {
        "loud-unstable/time": approximate_metric(verdict="unstable", delta=5, noise_pct=30),
        "zero-median/heap": permutation_metric(
            verdict="unstable",
            delta=delta,
            baseline_median=baseline_median,
            noise_pct=0.5,
            noise_abs=381,
            unit="bytes",
        ),
    }

    highlights = select_highlights(metrics, 0)

    assert [highlight.name for highlight in highlights] == [
        "zero-median/heap",
        "loud-unstable/time",
    ]


def test_select_highlights_when_equal_magnitude_does_keep_declaration_order():
    metrics: MetricComparisons = {
        "second-listed/ops": approximate_metric(verdict="regressed", delta=-5, direction="higher"),
        "third-listed/time": approximate_metric(verdict="regressed", delta=5),
        "first-listed/time": approximate_metric(verdict="regressed", delta=9),
    }

    highlights = select_highlights(metrics, 0)

    assert [highlight.name for highlight in highlights] == [
        "first-listed/time",
        "second-listed/ops",
        "third-listed/time",
    ]


def test_select_highlights_when_selected_does_carry_the_candidate_verdict_of_its_metric():
    metrics: MetricComparisons = {
        "slower/time": metric_for([
            CandidateSpec(verdict="improved", delta=-10),
            CandidateSpec(verdict="regressed", delta=8),
        ]),
    }

    highlights = select_highlights(metrics, 1)

    (highlight,) = highlights
    assert highlight.name == "slower/time"
    assert highlight.metric is metrics["slower/time"]
    assert highlight.verdict is metrics["slower/time"].candidates[1].verdict


@pytest.mark.parametrize(
    ("candidate_index", "expected"),
    [
        pytest.param(0, ["b/time", "a/time"], id="c0"),
        pytest.param(1, ["a/time"], id="c1"),
    ],
)
def test_select_highlights_when_multiple_candidates_does_rank_each_by_its_own_verdicts(
    candidate_index: int, expected: list[str]
):
    metrics: MetricComparisons = {
        "a/time": metric_for([
            CandidateSpec(verdict="improved", delta=-4),
            CandidateSpec(verdict="regressed", delta=3),
        ]),
        "b/time": metric_for([
            CandidateSpec(verdict="regressed", delta=6),
            CandidateSpec(verdict="no-signal", delta=0.2),
        ]),
    }

    highlights = select_highlights(metrics, candidate_index)

    assert [highlight.name for highlight in highlights] == expected


# ---------------------------------------------------------------------------
# footer_lines
# ---------------------------------------------------------------------------


#: The one hint the footer offers, in the prose ``format_hint`` renders it from.
SAMPLE_SHORTAGE_HINT = "re-run with `gymrat compare --samples 6` or more for statistical verdicts"

#: The hint after ``format_hint`` → ``render_plain`` round-trips (backticks stripped).
SAMPLE_SHORTAGE_HINT_PLAIN = (
    "re-run with gymrat compare --samples 6 or more for statistical verdicts"
)


def _verbose_lines(metrics: MetricComparisons) -> list[str]:
    return [
        line
        for line in footer_lines(metrics, verbose=True, command="compare", samples=4)
        if SAMPLE_SHORTAGE_HINT_PLAIN not in render_plain(line)
    ]


def _band_lines_for(metrics: MetricComparisons) -> list[str]:
    return [
        render_plain(line)
        for line in _verbose_lines(metrics)
        if render_plain(line).startswith("noise band")
    ]


def test_footer_lines_when_colored_does_dim_the_descriptive_verdict_line():
    metrics: MetricComparisons = {"a/time": approximate_metric(verdict="improved", delta=-10)}

    verdict_line = next(
        line for line in _verbose_lines(metrics) if "permutation" in render_plain(line)
    )

    assert "2" in sgr_codes(render_colored(verdict_line))


def test_footer_lines_when_verbose_does_close_on_the_sample_shortage_hint():
    metrics: MetricComparisons = {"a/time": band_metric(n=4)}

    lines = footer_lines(metrics, verbose=True, command="compare", samples=4)

    assert lines[-1] == format_hint(SAMPLE_SHORTAGE_HINT)


@pytest.mark.parametrize(
    ("metrics", "expected"),
    [
        pytest.param(
            {"decode/time": band_metric(n=3), "encode/time": band_metric(n=5)},
            ["noise band ±(half-range × K) — n=5 below permutation floor (6 pairs)"],
            id="run-too-short",
        ),
        pytest.param(
            {
                "entity.alive_check/heap": band_metric(n=10, usable_n=3),
                "iteration.columns/heap": band_metric(n=8, usable_n=2),
            },
            ["noise band ±(half-range × K) — ties left n=2 usable pairs (6 needed)"],
            id="ties-starved",
        ),
        pytest.param(
            {"decode/time": band_metric(n=3), "tied/heap": band_metric(n=10, usable_n=3)},
            [
                "noise band ±(half-range × K) — n=3 below permutation floor (6 pairs)",
                "noise band ±(half-range × K) — ties left n=3 usable pairs (6 needed)",
            ],
            id="each-cause-a-different-metric",
        ),
    ],
)
def test_footer_lines_when_cause_varies_does_phrase_band_line_accordingly(
    metrics: MetricComparisons, expected: list[str]
):
    lines = _band_lines_for(metrics)

    assert lines == expected


_DROPPED_ROUNDS = (
    "some rounds were dropped — not all samples produced paired measurements for every metric"
)


@pytest.mark.parametrize(
    ("metrics", "samples", "expected"),
    [
        pytest.param(
            {
                "decode/time": band_metric(n=3),
                "encode/time": band_metric(n=5),
                "parse/time": approximate_metric(verdict="improved", delta=-10),
            },
            4,
            [SAMPLE_SHORTAGE_HINT_PLAIN],
            id="every-band-metric-short",
        ),
        pytest.param(
            {
                "entity.alive_check/heap": band_metric(n=10, usable_n=3),
                "iteration.columns/heap": band_metric(n=8, usable_n=2),
                "parse/time": approximate_metric(verdict="improved", delta=-10),
            },
            4,
            [],
            id="ties-alone",
        ),
        pytest.param(
            {
                "decode/time": band_metric(n=3),
                "entity.alive_check/heap": band_metric(n=10, usable_n=3),
                "parse/time": approximate_metric(verdict="improved", delta=-10),
            },
            4,
            [SAMPLE_SHORTAGE_HINT_PLAIN],
            id="shortage-and-ties-different-metrics",
        ),
        pytest.param(
            {"parse/time": approximate_metric(verdict="improved", delta=-10)},
            4,
            [],
            id="permutation-carried-every-metric",
        ),
        pytest.param(
            {"a/time": band_metric(n=PERMUTATION_MIN_N - 1)},
            PERMUTATION_MIN_N - 1,
            [SAMPLE_SHORTAGE_HINT_PLAIN],
            id="fewer-samples-than-floor",
        ),
        pytest.param(
            {"a/time": band_metric(n=1)}, 1, [SAMPLE_SHORTAGE_HINT_PLAIN], id="single-sample"
        ),
        pytest.param(
            {
                "a/time": band_metric(n=3),
                "b/time": approximate_metric(verdict="improved", delta=-10),
            },
            10,
            [_DROPPED_ROUNDS],
            id="enough-samples-but-rounds-dropped",
        ),
        pytest.param(
            {"a/time": band_metric(n=PERMUTATION_MIN_N - 1)},
            PERMUTATION_MIN_N,
            [_DROPPED_ROUNDS],
            id="floor-reached-but-fewer-paired",
        ),
        pytest.param(
            {"a/time": approximate_metric(verdict="improved", delta=-10)},
            10,
            [],
            id="enough-samples-and-every-metric-tested",
        ),
    ],
)
def test_footer_lines_when_cause_varies_does_hint_accordingly(
    metrics: MetricComparisons, samples: int, expected: list[str]
):
    lines = footer_lines(metrics, verbose=False, command="compare", samples=samples)

    assert [render_plain(line) for line in lines] == expected
