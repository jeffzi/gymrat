"""Tests for the multi-candidate comparison text report.

These cover the candidate-per-column table: each candidate's figures paired with
its own verdict, how each cell is colored, bracketed metric names and candidate
labels, the per-candidate summary lines, and the per-candidate highlights.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import pytest
from rich.cells import cell_len

from gymrat.report.style import VERDICT_STYLES
from gymrat.report.text.render import render_report
from gymrat.report.types import ReportOptions
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
    sgr_for,
    stripped_cells,
    styles_at,
)
from tests.report._comparisons import (
    NWayCandidate,
    create_candidate,
    create_comparison_result,
    multi_candidate_result,
    n_way_metric,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from gymrat.report.types import CandidateMetric, ComparisonResult


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


def test_render_report_when_many_candidates_have_bracketed_labels_and_names_does_print_them_literally():
    result = multi_candidate_result(
        2, labels=("[bold]fast", "[dim]slow"), name="[italic]decode/time"
    )

    report = strip_ansi(render_report(result))

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
    ("make_result", "row", "token", "codes"),
    [
        pytest.param(_dimming_result, "flat/time", "~", sgr_for("dim"), id="quiet-glyph-dim"),
        pytest.param(
            multi_candidate_result,
            "decode/time",
            "unstable",
            sgr_for(VERDICT_STYLES["unstable"]),
            id="unstable-word-amber",
        ),
        pytest.param(
            _dimming_result, "mixed/time", "+0.3%", sgr_for("dim"), id="quiet-delta-on-bright-row"
        ),
        pytest.param(
            multi_candidate_result,
            "decode/time",
            "-10.0%",
            sgr_for(VERDICT_STYLES["improved"]),
            id="improved-delta-green",
        ),
    ],
)
def test_render_report_when_colored_does_style_each_cell_verdict_on_its_own(
    make_result: Callable[[], ComparisonResult], row: str, token: str, codes: list[str]
):
    line = line_containing(render_report(make_result(), ReportOptions(color=True)), row)

    assert styles_at(line, token) == codes


def _detecting(candidate: CandidateMetric) -> CandidateMetric:
    """``candidate`` with its verdict carrying a 2.1% detectable change."""
    if candidate.verdict is None or candidate.verdict.method == "exact":
        msg = f"no approximate verdict to carry a detectable change on {candidate!r}"
        raise AssertionError(msg)
    return replace(candidate, verdict=replace(candidate.verdict, detectable_pct=2.1))


def test_render_report_when_many_candidates_carry_detectable_changes_does_not_state_them():
    result = _dimming_result()
    metric = result.metrics["flat/time"]
    detecting = replace(metric, candidates=tuple(map(_detecting, metric.candidates)))

    report = render_report(replace(result, metrics={**result.metrics, "flat/time": detecting}))

    assert "~" in line_starting_with(report, "flat/time")
    assert "detects" not in report


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


def test_render_report_when_ascii_labels_differ_in_length_does_pad_inside_the_bold_label():
    report = render_report(
        multi_candidate_result(2, labels=("fast", "slower")), ReportOptions(color=True)
    )

    assert styles_at(line_containing(report, "improved"), "fast  ") == sgr_for("bold")


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
    report = strip_ansi(render_report(multi_candidate_result(2, labels=labels)))

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
