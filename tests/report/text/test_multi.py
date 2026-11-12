"""Tests for the multi-candidate comparison text report.

These cover the candidate-per-column table: how each cell is colored, bracketed
metric names and candidate labels, and the per-candidate summary lines.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from rich.cells import cell_len

from gymrat.model import Exclusion
from gymrat.report.style import VERDICT_STYLES
from gymrat.report.text.render import render_report
from gymrat.report.types import ReportOptions
from tests._ansi import (
    strip_ansi,
)
from tests.report._assertions import (
    line_containing,
    line_starting_with,
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
    other_kind,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from gymrat.report.types import ComparisonResult


# ---------------------------------------------------------------------------
# candidate columns
# ---------------------------------------------------------------------------


def test_render_report_when_candidate_labels_carry_brackets_does_print_them_literally():
    result = multi_candidate_result(2, labels=("[bold]fast", "[dim]slow"))

    report = strip_ansi(render_report(result))

    assert stripped_cells(line_starting_with(report, "metric")) == [
        "metric",
        "main",
        "[bold]fast",
        "[dim]slow",
    ]


def test_render_report_when_metric_name_carries_brackets_does_print_it_literally():
    result = multi_candidate_result(2, name="[italic]decode/time")

    report = strip_ansi(render_report(result))

    assert stripped_cells(line_starting_with(report, "[italic]decode/time"))[0] == (
        "[italic]decode/time"
    )


# ---------------------------------------------------------------------------
# candidate column color
# ---------------------------------------------------------------------------


def _dimming_result() -> ComparisonResult:
    """A two-candidate run whose ``flat/time`` row is quiet for its first candidate."""
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
        },
    )


@pytest.mark.parametrize(
    ("make_result", "row", "token", "codes"),
    [
        pytest.param(
            _dimming_result,
            "flat/time",
            "~",
            sgr_for(VERDICT_STYLES["within-noise"]),
            id="quiet-glyph-dim",
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


@pytest.mark.parametrize(
    ("labels", "summary_column"),
    [
        pytest.param(("fast", "测试指标"), 10, id="cjk-label-widest"),
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
# geomean row
# ---------------------------------------------------------------------------


def test_render_report_when_a_candidate_geomean_excludes_a_metric_does_count_kept_against_total():
    result = create_comparison_result(
        metrics=multi_candidate_result(2).metrics,
        candidates=[
            create_candidate(label="candidate-a", kinds=[other_kind(-10, 2)]),
            create_candidate(
                label="candidate-b",
                kinds=[
                    other_kind(4, 1, excluded=[Exclusion(metric="decode/time", reason="unstable")])
                ],
            ),
        ],
    )

    report = strip_ansi(render_report(result))

    provenance = stripped_cells(line_starting_with(report, "geomean"))[2:]
    assert [cell.rsplit(" ", 1)[-1] for cell in provenance] == ["(2/2)", "(1/2)"]


def test_render_report_when_a_candidate_geomean_covers_no_metric_does_show_the_no_stable_metrics_stand_in():
    excluded = [
        Exclusion(metric="decode/time", reason="unstable"),
        Exclusion(metric="encode/time", reason="unstable"),
    ]
    result = create_comparison_result(
        metrics=multi_candidate_result(2).metrics,
        candidates=[
            create_candidate(label="candidate-a", kinds=[other_kind(-10, 2)]),
            create_candidate(label="candidate-b", kinds=[other_kind(0, 0, excluded=excluded)]),
        ],
    )

    report = strip_ansi(render_report(result))

    assert "no stable metrics" in stripped_cells(line_starting_with(report, "geomean"))[-1]
