"""Tests for the multi-candidate comparison text report.

These cover the candidate-per-column table: each candidate's figures paired with
its own verdict, the identical cell, how each cell is colored, bracketed names,
the sectioned and grouped layouts with several candidates, the per-candidate
summary lines, and the per-candidate highlights.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import pytest
from rich.cells import cell_len

from gymrat.report.text.render import render_report
from gymrat.report.types import CandidateMetric, MetricComparison, ReportOptions
from gymrat.verdict import GroupAggregate, KindAggregate
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
    table_rows,
)
from tests.report._comparisons import (
    NWayCandidate,
    create_candidate,
    create_comparison_result,
    metric_meta,
    multi_candidate_result,
    n_way_kind_metric,
    n_way_metric,
    other_kind,
)
from tests.report._verdicts import band_verdict, geomean_of, permutation_verdict

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
    ],
)
def test_render_report_when_colored_does_style_each_cell_verdict_on_its_own(
    make_result: Callable[[], ComparisonResult], row: str, token: str, code: str
):
    line = line_containing(render_report(make_result(), ReportOptions(color=True)), row)

    assert code in styles_at(line, token)


def test_render_report_when_colored_does_leave_name_and_values_plain_on_a_quiet_row():
    cells = cells_of(
        line_containing(render_report(_dimming_result(), ReportOptions(color=True)), "flat/time")
    )

    assert "\x1b[" not in "│".join(cells[:2])
    assert "\x1b[" not in TRAILING_SGR_RUN.sub("", cells[2][: cells[2].index("~")])
    assert "\x1b[" not in TRAILING_SGR_RUN.sub("", cells[3][: cells[3].index("≈")])


# ---------------------------------------------------------------------------
# sectioned layout
# ---------------------------------------------------------------------------


def _sectioned_bracketed_result() -> ComparisonResult:
    """Two kinds, one grouped, whose baseline, kinds, group and short names read as markup tags."""
    return create_comparison_result(
        baseline_label="[dim]main",
        metrics={
            "[bold]entity/spawn#other": n_way_kind_metric(
                kind="[underline]time",
                short_name="[bold]entity.spawn",
                candidates=[
                    NWayCandidate(verdict="improved", delta=-10, median=90),
                    NWayCandidate(verdict="regressed", delta=4, median=104),
                ],
            ),
            "encode#other": n_way_kind_metric(
                kind="[strike]memory",
                short_name="[italic]encode",
                candidates=[
                    NWayCandidate(verdict="improved", delta=-7, median=93),
                    NWayCandidate(verdict="improved", delta=-2, median=98),
                ],
            ),
        },
        candidates=[
            create_candidate(
                label=label,
                kinds=[
                    KindAggregate(
                        kind="[underline]time",
                        geomean=(time_geomean := geomean_of(time_delta, 1)),
                        groups=(GroupAggregate(group="[bold]entity", geomean=time_geomean),),
                        gated_geomean=time_geomean,
                    ),
                    KindAggregate(
                        kind="[strike]memory",
                        geomean=(memory_geomean := geomean_of(memory_delta, 1)),
                        groups=(),
                        gated_geomean=memory_geomean,
                    ),
                ],
            )
            for label, time_delta, memory_delta in (
                ("candidate-a", -10, -7),
                ("candidate-b", 4, -2),
            )
        ],
    )


def test_render_report_when_sectioned_names_carry_brackets_does_print_them_literally():
    report = render_report(_sectioned_bracketed_result())

    assert [stripped_cells(row)[:2] for row in table_rows(report)] == [
        ["[underline]time", "[dim]main"],
        ["[bold]entity", ""],
        ["spawn", "100ns ± 1%"],
        ["geomean · [bold]entity", ""],
        ["geomean · [underline]time", ""],
        ["[strike]memory", "[dim]main"],
        ["[italic]encode", "100ns ± 1%"],
        ["geomean · [strike]memory", ""],
    ]


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
    summaries = [line for line in report.split("\n") if re.search(r"✓ \d+ improved", line)]

    assert [line.split("  ✓")[0].rstrip() for line in summaries] == list(labels)
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


# ---------------------------------------------------------------------------
# single-kind grouping with multiple candidates
# ---------------------------------------------------------------------------


def _flat_grouped_bracketed_result() -> ComparisonResult:
    """Two candidates, single kind, whose one group reads as a markup tag."""
    return create_comparison_result(
        metrics={
            "[bold]entity/spawn#other": n_way_kind_metric(
                kind="other",
                short_name="[bold]entity.spawn",
                candidates=[
                    NWayCandidate(verdict="improved", delta=-10, median=90),
                    NWayCandidate(verdict="regressed", delta=4, median=104),
                ],
            ),
            "[bold]entity/alive#other": n_way_kind_metric(
                kind="other",
                short_name="[bold]entity.alive",
                candidates=[
                    NWayCandidate(verdict="no-signal", delta=0.3, median=100),
                    NWayCandidate(verdict="improved", delta=-5, median=95),
                ],
            ),
        },
        candidates=[
            create_candidate(
                label=label,
                kinds=[
                    KindAggregate(
                        kind="other",
                        geomean=(geomean := geomean_of(delta, 1)),
                        groups=(GroupAggregate(group="[bold]entity", geomean=geomean),),
                        gated_geomean=geomean,
                    )
                ],
            )
            for label, delta in (("candidate-a", -10), ("candidate-b", 4))
        ],
    )


def test_render_report_when_flat_grouped_many_candidates_does_print_group_brackets_literally():
    report = render_report(_flat_grouped_bracketed_result())

    assert [stripped_cells(row)[0] for row in table_rows(report)] == [
        "metric",
        "[bold]entity · other",
        "spawn",
        "alive",
        "geomean",
    ]
