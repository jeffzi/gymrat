"""Tests for the multi-candidate comparison text report.

These cover the candidate-per-column table: each candidate's figures paired with
its own verdict, the identical cell, how each cell is colored, bracketed metric
names and candidate labels, the per-candidate summary lines, and the
per-candidate highlights.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from rich.cells import cell_len

from gymrat.report.text.render import render_report
from gymrat.report.types import CandidateMetric, MetricComparison, ReportOptions
from tests._ansi import (
    TRAILING_SGR_RUN,
    strip_ansi,
)
from tests.report._assertions import (
    cells_of,
    highlight_lines,
    line_containing,
    line_starting_with,
    rule_lines,
    stripped_cells,
    styles_at,
)
from tests.report._comparisons import (
    NWayCandidate,
    create_candidate,
    create_comparison_result,
    metric_meta,
    multi_candidate_result,
    n_way_metric,
    other_kind,
)
from tests.report._verdicts import band_verdict, permutation_verdict

if TYPE_CHECKING:
    from collections.abc import Callable

    from gymrat.report.types import ComparisonResult


# ---------------------------------------------------------------------------
# candidate columns
# ---------------------------------------------------------------------------


def test_render_report_when_many_candidates_does_pair_each_figure_with_its_own_verdict():
    row = line_starting_with(render_report(multi_candidate_result()), "decode/time")

    assert stripped_cells(row) == [
        "decode/time",
        "100ns ± 1%",
        "90ns ± 1%  ✓  -10.0%",
        "104ns ± 1%  ✗  +4.0%",
        "150ns ± 3%  ≈  unstable",
    ]


def test_render_report_when_many_candidates_does_size_the_last_column_to_fit_its_aggregate():
    report = render_report(multi_candidate_result(2))
    rules = rule_lines(report)
    geomean_line = line_starting_with(strip_ansi(report), "geomean")

    assert rules
    for rule in rules:
        assert len(rule) >= len(geomean_line)


def _bracketed_result() -> ComparisonResult:
    """Two candidates whose labels, like the metric names, read as markup tags."""
    return create_comparison_result(
        candidates=[
            create_candidate(label="[bold]fast"),
            create_candidate(label="[dim]slow"),
        ],
        metrics={
            "[italic]decode/time": n_way_metric([
                NWayCandidate(verdict="improved", delta=-10, median=90),
                NWayCandidate(verdict="regressed", delta=4, median=104),
            ]),
        },
    )


def test_render_report_when_names_carry_brackets_does_print_them_literally():
    report = strip_ansi(render_report(_bracketed_result()))

    assert stripped_cells(line_starting_with(report, "metric")) == [
        "metric",
        "main",
        "[bold]fast",
        "[dim]slow",
    ]
    assert stripped_cells(line_starting_with(report, "[italic]decode/time")) == [
        "[italic]decode/time",
        "100ns ± 1%",
        "90ns ± 1%  ✓  -10.0%",
        "104ns ± 1%  ✗  +4.0%",
    ]


def test_render_report_when_ties_starve_the_test_does_mark_the_candidate_cell_identical():
    result = create_comparison_result(
        candidates=[
            create_candidate(label="candidate-a"),
            create_candidate(label="candidate-b"),
        ],
        metrics={
            "tied/time": MetricComparison(
                baseline_median=100,
                baseline_spread=1,
                candidates=(
                    CandidateMetric(median=100, spread=1, verdict=band_verdict(usable_n=0)),
                    CandidateMetric(
                        median=90,
                        spread=1,
                        verdict=permutation_verdict(verdict="improved", delta=-10, p=0.002),
                    ),
                ),
                meta=metric_meta("tied/time", unit="ns"),
            ),
        },
    )

    row = line_starting_with(render_report(result), "tied/time")

    assert stripped_cells(row) == [
        "tied/time",
        "100ns ± 1%",
        "100ns ± 1%  =  -0.5%",
        "90ns ± 1%  ✓  -10.0%",
    ]


# ---------------------------------------------------------------------------
# candidate column color
# ---------------------------------------------------------------------------


def _dimming_result() -> ComparisonResult:
    """A two-candidate run with a quiet row and a row where one candidate moved."""
    return create_comparison_result(
        candidates=[
            create_candidate(label="candidate-a"),
            create_candidate(label="candidate-b"),
        ],
        metrics={
            "flat/time": n_way_metric([
                NWayCandidate(verdict="no-signal", delta=0.3, median=100),
                NWayCandidate(verdict="unstable", delta=-50, median=50),
            ]),
            "mixed/time": n_way_metric([
                NWayCandidate(verdict="no-signal", delta=0.3, median=100),
                NWayCandidate(verdict="improved", delta=-17.5, median=83),
            ]),
        },
    )


@pytest.mark.parametrize(
    ("make_result", "row", "token", "code"),
    [
        pytest.param(_dimming_result, "flat/time", "~", "2", id="quiet-glyph-dim"),
        pytest.param(_dimming_result, "flat/time", "≈", "33", id="unstable-glyph-amber"),
        pytest.param(
            multi_candidate_result, "decode/time", "unstable", "33", id="unstable-word-amber"
        ),
        pytest.param(_dimming_result, "mixed/time", "+0.3%", "2", id="quiet-delta-on-bright-row"),
        pytest.param(
            multi_candidate_result, "decode/time", "-10.0%", "32", id="improved-delta-green"
        ),
        pytest.param(
            multi_candidate_result, "decode/time", "+4.0%", "31", id="regressed-delta-red"
        ),
    ],
)
def test_render_report_when_colored_does_style_each_cell_verdict_on_its_own(
    make_result: Callable[[], ComparisonResult], row: str, token: str, code: str
):
    line = line_containing(render_report(make_result(), ReportOptions(color=True)), row)

    assert styles_at(line, token) == [code]


def test_render_report_when_colored_does_leave_name_and_values_plain_on_a_quiet_row():
    cells = cells_of(
        line_containing(render_report(_dimming_result(), ReportOptions(color=True)), "flat/time")
    )

    assert "\x1b[" not in "│".join(cells[:2])
    assert "\x1b[" not in TRAILING_SGR_RUN.sub("", cells[2][: cells[2].index("~")])
    assert "\x1b[" not in TRAILING_SGR_RUN.sub("", cells[3][: cells[3].index("≈")])


# ---------------------------------------------------------------------------
# per-candidate summary
# ---------------------------------------------------------------------------


def _two_label_result(labels: tuple[str, str]) -> ComparisonResult:
    """One metric over two candidates labelled ``labels``, one improved and one regressed."""
    return create_comparison_result(
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


def test_render_report_when_ascii_labels_differ_in_length_does_pad_inside_the_bold_label():
    report = render_report(_two_label_result(("fast", "slower")), ReportOptions(color=True))

    assert styles_at(line_containing(report, "fast  "), "fast  ") == ["1"]


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
    report = strip_ansi(render_report(_two_label_result(labels)))

    summaries = [line_starting_with(report, label) for label in labels]
    assert [cell_len(line[: line.index("✓")]) for line in summaries] == [summary_column] * 2


# ---------------------------------------------------------------------------
# per-candidate highlights
# ---------------------------------------------------------------------------


def test_render_report_when_many_candidates_does_group_highlights_per_candidate():
    highlights = highlight_lines(render_report(multi_candidate_result()))

    assert highlights == [
        "  candidate-a",
        "    ✓ decode/time  -10.0%",
        "  candidate-b",
        "    ✗ decode/time   +4.0%",
        "  candidate-c",
        "    ≈ decode/time  unstable  noise ±30.0%",
        "  unstable metrics won't stabilize with more samples",
    ]
