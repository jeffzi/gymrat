"""Tests for the verdict summary, highlights, gate trips and footers.

These cover the one-line verdict tally below the table,
the highlights block and its futility note, the ``--fail-on`` geomean gate-trip
lines, the verbose method footer, and the worktree-cleanup footer. The report
header and table are pinned by ``test_text`` and ``test_text_multi``; here the
report is driven end to end and only its assembled tail is asserted.

The colored comparison and measurement reports are covered too: how the
assembled report paints its verdict rows, run and column headers, the verdict
summary, the highlights block, and the verbose method footer and hint, and the
measure report's header and tables. Column alignment is checked by byte offset
only where that proves cross-section alignment; content is pinned by parsed
cell.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import pytest

from gymrat.model import Exclusion
from gymrat.report.text.render import render_measure_report, render_report
from gymrat.report.types import (
    CandidateMetric,
    GeomeanFailOn,
    MeasurementResult,
    MetricComparison,
    RegressedFailOn,
    ReportOptions,
)
from gymrat.targets import WorktreeRemovalFailure
from tests._ansi import strip_ansi
from tests.report._assertions import (
    cells_of,
    delta_cell,
    highlight_lines,
    line_containing,
    line_starting_with,
    styles_at,
    table_region,
    table_rows,
)
from tests.report._comparisons import (
    create_candidate,
    create_comparison_result,
    exact_metric,
    grouped_comparison,
    memory_kind,
    metric_meta,
    other_kind,
    permutation_metric,
    time_kind,
    two_kind_result,
    without_gated_geomean,
)
from tests.report._measurements import (
    create_measurement_result,
    measured_metric,
    two_kind_measurement,
)
from tests.report._verdicts import (
    band_metric,
    band_verdict,
    geomean_of,
    permutation_verdict,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from gymrat.model import ApproximateVerdict
    from gymrat.report.types import ComparisonResult


# ---------------------------------------------------------------------------
# ties starving the permutation test
# ---------------------------------------------------------------------------


def _identical_result() -> ComparisonResult:
    """A run whose ``tied/heap`` metric moved too little to break any pair apart."""
    return create_comparison_result(
        metrics={
            "faster/time": permutation_metric(verdict="improved", delta=-10, unit="ns"),
            "tied/heap": band_metric(verdict="no-signal", delta=-0.5, n=10, usable_n=0),
        }
    )


def test_render_report_when_ties_starve_the_test_does_mark_the_row_identical():
    row = line_starting_with(render_report(_identical_result()), "tied/heap")

    assert cells_of(row)[-1].strip() == "=   -0.5%  ±2.5%"


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

    assert [cell.strip() for cell in cells_of(row)] == [
        "tied/time",
        "100ns ± 1%",
        "100ns ± 1%  =  -0.5%",
        "90ns ± 1%  ✓  -10.0%",
    ]


# ---------------------------------------------------------------------------
# highlights block
# ---------------------------------------------------------------------------


def test_render_report_when_unstable_around_zero_candidate_median_does_state_noise_in_bytes():
    result = create_comparison_result(
        metrics={
            "jittery/heap": permutation_metric(
                verdict="unstable",
                delta=-100,
                baseline_median=100,
                noise_pct=0.5,
                noise_abs=6,
                unit="bytes",
            ),
        }
    )

    highlights = [line.strip() for line in highlight_lines(render_report(result))]

    assert highlights[0] == "≈ jittery/heap  unstable  ±6B noise on a 0B candidate median"


def test_render_report_when_metric_name_has_colon_word_does_align_deltas_with_other_highlights():
    result = create_comparison_result(
        metrics={
            "lat:100:p99/time": permutation_metric(verdict="regressed", delta=2.2),
            "slow/time": permutation_metric(verdict="regressed", delta=1.5),
        }
    )

    highlights = [strip_ansi(line) for line in highlight_lines(render_report(result))]

    assert highlights[0].index("+2.2%") == highlights[1].index("+1.5%")


# ---------------------------------------------------------------------------
# --fail-on geomean gate trips
# ---------------------------------------------------------------------------


def _tripping_result() -> ComparisonResult:
    """A two-kind run whose gating ``time`` kind regressed past a 2% threshold."""
    return replace(
        two_kind_result(),
        candidates=(
            create_candidate(
                kinds=[
                    replace(
                        time_kind(),
                        geomean=geomean_of(3.1, 3),
                        gated_geomean=geomean_of(3.1, 3),
                    ),
                    memory_kind(),
                ]
            ),
        ),
    )


def test_render_report_when_gate_trips_does_flag_the_trip_in_the_highlights():
    highlights = [
        line.strip()
        for line in highlight_lines(
            render_report(_tripping_result(), ReportOptions(fail_on=(GeomeanFailOn(pct=2),)))
        )
    ]

    assert highlights == [
        "✗ time · entity.spawn         +4.0%",
        "✓ time · entity.alive_check  -10.0%",
        "✓ memory · encode             -7.0%",
        "⚑ time gated geomean +3.1% exceeded --fail-on geomean:2",
    ]


def _gate_lines(report: str) -> list[str]:
    return [line.strip() for line in strip_ansi(report).split("\n") if line.strip().startswith("⚑")]


def _informational_kind_result() -> ComparisonResult:
    """A two-kind run whose second kind gates nothing, so it has no gated geomean."""
    return replace(
        two_kind_result(),
        candidates=(
            create_candidate(kinds=[time_kind(), without_gated_geomean(other_kind(9, 1))]),
        ),
    )


@pytest.mark.parametrize(
    ("result", "options"),
    [
        pytest.param(_tripping_result(), ReportOptions(), id="no-conditions"),
        pytest.param(
            _tripping_result(),
            ReportOptions(fail_on=(GeomeanFailOn(pct=10),)),
            id="threshold-beyond",
        ),
        pytest.param(
            _tripping_result(), ReportOptions(fail_on=(RegressedFailOn(),)), id="only-regressed"
        ),
        pytest.param(
            _informational_kind_result(),
            ReportOptions(fail_on=(GeomeanFailOn(pct=2),)),
            id="informational-kind",
        ),
        pytest.param(
            create_comparison_result(
                metrics={"slow/time": permutation_metric(verdict="regressed", delta=8)}
            ),
            ReportOptions(fail_on=(RegressedFailOn(),)),
            id="regression-already-shown",
        ),
        pytest.param(
            create_comparison_result(
                metrics={"slow/time": band_metric(verdict="regressed", delta=8, n=4)}
            ),
            ReportOptions(),
            id="no-regressed-condition",
        ),
    ],
)
def test_render_report_when_no_gate_trips_or_needs_explaining_does_say_nothing_about_a_gate(
    result: ComparisonResult, options: ReportOptions
):
    report = render_report(result, options)

    assert _gate_lines(report) == []


@pytest.mark.parametrize(
    ("geomean", "gated", "expected"),
    [
        pytest.param(5, 1, [], id="overall-trips-gated-does-not"),
        pytest.param(
            1,
            5,
            ["⚑ time gated geomean +5.0% exceeded --fail-on geomean:2"],
            id="gated-trips-overall-does-not",
        ),
    ],
)
def test_render_report_when_gating_does_judge_on_the_gated_geomean(
    geomean: float, gated: float, expected: list[str]
):
    result = replace(
        two_kind_result(),
        candidates=(
            create_candidate(
                kinds=[
                    replace(
                        time_kind(),
                        geomean=geomean_of(geomean, 3),
                        gated_geomean=geomean_of(gated, 3),
                    ),
                    memory_kind(),
                ]
            ),
        ),
    )

    highlights = [
        line.strip()
        for line in highlight_lines(
            render_report(result, ReportOptions(fail_on=(GeomeanFailOn(pct=2),)))
        )
    ]

    assert [line for line in highlights if line.startswith("⚑")] == expected


def test_render_report_when_gating_multi_candidate_does_flag_only_those_that_exceeded():
    highlights = highlight_lines(
        render_report(grouped_comparison(), ReportOptions(fail_on=(GeomeanFailOn(pct=2),)))
    )

    assert highlights == [
        "  candidate-a",
        "    ✓ time · entity.alive_check  -10.0%",
        "    ✓ memory · encode             -7.0%",
        "  candidate-b",
        "    ✗ time · entity.alive_check   +4.0%",
        "    ✓ memory · encode             -2.0%",
        "    ⚑ time gated geomean +4.0% exceeded --fail-on geomean:2",
    ]


# ---------------------------------------------------------------------------
# --fail-on regressed gate trips
# ---------------------------------------------------------------------------


def test_render_report_when_regressed_gate_trips_on_inconclusive_metric_does_name_it():
    result = create_comparison_result(
        metrics={"slow/time": band_metric(verdict="regressed", delta=8, n=4)}
    )

    report = render_report(result, ReportOptions(fail_on=(RegressedFailOn(),)))

    assert _gate_lines(report) == [
        "⚑ slow/time regressed +8.0% on 4 pairs tripped --fail-on regressed"
    ]


def test_render_report_when_gate_trips_with_color_does_paint_the_trip_red(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("FORCE_COLOR", "1")

    line = line_containing(
        render_report(_tripping_result(), ReportOptions(fail_on=(GeomeanFailOn(pct=2),))),
        "⚑",
    )

    assert "31" in styles_at(line, "⚑")
    assert "31" in styles_at(line, "+3.1%")


# ---------------------------------------------------------------------------
# verbose method footer
# ---------------------------------------------------------------------------


def test_render_report_when_verbose_and_all_exact_does_end_on_the_highlights_without_a_footer():
    result = create_comparison_result(metrics={"a/heap": exact_metric(delta=-7.9)})

    output = render_report(result, ReportOptions(verbose=True))

    assert output.split("\n\n")[-1].splitlines() == ["highlights", "  ✓ a/heap   -7.9%  (exact)"]


# ---------------------------------------------------------------------------
# mixed verdict methods
# ---------------------------------------------------------------------------


def _mixed_method_result() -> ComparisonResult:
    """A run whose metrics genuinely disagree on method.

    ``decode/time`` paired on 10 of the 12 rounds — enough for the permutation
    test — while ``encode/time`` paired on 4 and fell back to the noise band.
    """
    return create_comparison_result(
        samples=12,
        metrics={
            "decode/time": permutation_metric(verdict="improved", delta=-10, n=10),
            "encode/time": band_metric(verdict="no-signal", delta=1, n=4),
        },
    )


def test_render_report_when_methods_differ_does_name_each_with_its_pair_counts():
    report = render_report(_mixed_method_result(), ReportOptions(verbose=True))

    permutation_line = line_starting_with(report, "verdicts:")
    band_line = line_starting_with(report, "noise band")

    assert permutation_line == (
        "verdicts: sign-flip permutation test on pairs (n=10 ≥ 6) · ~ = no signal at α=0.05"
    )
    assert band_line == "noise band ±(half-range × K) — n=4 below permutation floor (6 pairs)"
    assert report.index(permutation_line) < report.index(band_line)


# ---------------------------------------------------------------------------
# worktree cleanup footer
# ---------------------------------------------------------------------------


def test_render_report_when_cleanup_removed_everything_cleanly_does_suppress_the_footer():
    result = create_comparison_result(worktrees_removed=3, worktrees_left_behind=[])

    output = render_report(result)

    assert "worktree" not in output
    assert "left behind" not in output


@pytest.mark.parametrize(
    ("result", "expected"),
    [
        pytest.param(
            create_comparison_result(
                worktrees_removed=2,
                worktrees_left_behind=[
                    WorktreeRemovalFailure(dir="/tmp/gymrat-abc", error="is locked")
                ],
            ),
            ["2 worktrees removed · 1 left behind", "left behind: /tmp/gymrat-abc (is locked)"],
            id="one-left-behind",
        ),
        pytest.param(
            create_comparison_result(
                worktrees_removed=1,
                worktrees_left_behind=[
                    WorktreeRemovalFailure(
                        dir="/tmp/gymrat-abc", error="contains modified or untracked files"
                    ),
                    WorktreeRemovalFailure(dir="/tmp/gymrat-def", error="is locked"),
                ],
            ),
            [
                "left behind: /tmp/gymrat-abc (contains modified or untracked files)",
                "left behind: /tmp/gymrat-def (is locked)",
            ],
            id="several-left-behind",
        ),
        pytest.param(
            create_comparison_result(worktree_prune_error="fatal: not a git repository"),
            ["worktree prune failed: fatal: not a git repository"],
            id="only-prune-failed",
        ),
    ],
)
def test_render_report_when_cleanup_left_worktrees_or_prune_failed_does_render_the_footer(
    result: ComparisonResult, expected: list[str]
):
    output = render_report(result)

    assert [line for line in expected if line not in output] == []


def _with_left_behind_reason(reason: str) -> ComparisonResult:
    return create_comparison_result(
        worktrees_removed=1,
        worktrees_left_behind=[WorktreeRemovalFailure(dir="/tmp/gymrat-abc", error=reason)],
    )


def _with_prune_error(reason: str) -> ComparisonResult:
    return create_comparison_result(worktree_prune_error=reason)


@pytest.mark.parametrize(
    "reason",
    [
        pytest.param("fatal:\tnot a git repository", id="tab"),
        pytest.param("fatal:    not a git repository", id="space-run"),
        pytest.param("  fatal: not a git repository \n", id="leading-and-trailing"),
        pytest.param("fatal:\r\nnot a git repository", id="carriage-return"),
        pytest.param(f"fatal:{chr(0xA0)}not a{chr(0x2003)}git repository", id="unicode-spaces"),
        pytest.param("\n\t fatal: \t\n  not   a git\n\nrepository\t ", id="mixed"),
    ],
)
@pytest.mark.parametrize(
    ("build_result", "expected"),
    [
        pytest.param(
            _with_left_behind_reason,
            "  left behind: /tmp/gymrat-abc (fatal: not a git repository)",
            id="left-behind",
        ),
        pytest.param(
            _with_prune_error,
            "  worktree prune failed: fatal: not a git repository",
            id="prune-error",
        ),
    ],
)
def test_render_report_when_a_cleanup_reason_has_whitespace_runs_does_collapse_each_to_one_space(
    build_result: Callable[[str], ComparisonResult], expected: str, reason: str
):
    result = build_result(reason)

    matching_lines = [
        line for line in render_report(result).split("\n") if "not a git repository" in line
    ]

    assert matching_lines == [expected]


def _summary_segment(summary: str, label: str) -> str:
    """The triple-space-delimited summary segment whose plain text contains *label*."""
    for seg in summary.split("   "):
        if label in strip_ansi(seg):
            return seg
    msg = f"no segment with {label!r} in summary: {strip_ansi(summary)!r}"
    raise AssertionError(msg)


def _colorful_result() -> ComparisonResult:
    """A run whose rows cover every verdict class, plus a geomean figure."""
    return create_comparison_result(
        metrics={
            "faster/time": permutation_metric(verdict="improved", delta=-17.5, unit="ns"),
            "slower/time": permutation_metric(verdict="regressed", delta=2.4, unit="ns"),
            "flat/time": permutation_metric(verdict="no-signal", delta=0.3, unit="ns"),
            "tied/heap": band_metric(verdict="no-signal", delta=-0.5, n=10, usable_n=0),
            "single-pair/time": band_metric(delta=-0.4, noise_pct=0.5, n=1, unit="ns"),
            "jittery/time": permutation_metric(verdict="unstable", delta=-50, noise_pct=30),
        },
        candidates=[
            create_candidate(
                kinds=[
                    other_kind(
                        -5.8,
                        3,
                        excluded=[Exclusion(metric="jittery/time", reason="unstable")],
                    )
                ]
            )
        ],
        worktrees_removed=1,
        worktrees_left_behind=[WorktreeRemovalFailure(dir="/tmp/gymrat-abc", error="is locked")],
        worktree_prune_error="fatal: not a git repository",
    )


def _flat_measurement() -> MeasurementResult:
    """A flat single-kind run of two metrics measured in nanoseconds."""
    return create_measurement_result(
        metrics={
            "decode/time": measured_metric(median=100, spread=1, unit="ns"),
            "encode/time": measured_metric(median=2048, spread=2, unit="ns"),
        }
    )


# ---------------------------------------------------------------------------
# flat single-kind table
# ---------------------------------------------------------------------------


def test_render_measure_report_when_one_kind_does_draw_one_flat_table_of_medians():
    report = render_measure_report(_flat_measurement())

    assert table_region(report) == [
        "gymrat measure · main · 10 samples · adapter: mitata",
        "metric",
        "<rule>",
        "decode/time",
        "encode/time",
    ]
    assert [cell.strip() for cell in cells_of(line_starting_with(report, "metric"))] == [
        "metric",
        "main",
    ]
    assert [[cell.strip() for cell in cells_of(row)] for row in table_rows(report)[1:]] == [
        ["decode/time", "100ns ± 1%"],
        ["encode/time", "2.0µs ± 2%"],
    ]


# ---------------------------------------------------------------------------
# metric row
# ---------------------------------------------------------------------------


def test_render_measure_report_when_metric_has_no_spread_does_state_the_bare_median():
    result = create_measurement_result(
        metrics={"decode/time": measured_metric(median=100, spread=None, unit="ns")}
    )

    row = line_starting_with(render_measure_report(result), "decode/time")

    assert cells_of(row)[-1].strip() == "100ns"


# ---------------------------------------------------------------------------
# multi-kind sections
# ---------------------------------------------------------------------------


def test_render_measure_report_when_kind_is_informational_by_metric_overrides_does_say_so():
    report = render_measure_report(replace(two_kind_measurement(), config_kinds=None))

    assert line_containing(report, "informational") == "informational — gating off"


# ---------------------------------------------------------------------------
# render_report — colored comparison report
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("metric", "glyph", "code"),
    [
        pytest.param("jittery/time", "≈", "33", id="unstable-yellow"),
        pytest.param("tied/heap", "=", "36", id="identical-cyan"),
        pytest.param("flat/time", "~", "2", id="within-noise-dim"),
        pytest.param("single-pair/time", "?", "2", id="inconclusive-dim"),
    ],
)
def test_render_report_when_colored_does_paint_each_verdict_on_its_row(
    metric: str, glyph: str, code: str
):

    row = line_containing(render_report(_colorful_result(), ReportOptions(color=True)), metric)

    assert code in styles_at(row, glyph)


@pytest.mark.parametrize(
    "metric",
    [
        pytest.param("flat/time", id="within-noise"),
        pytest.param("tied/heap", id="identical"),
        pytest.param("single-pair/time", id="inconclusive"),
        pytest.param("jittery/time", id="unstable"),
    ],
)
def test_render_report_when_colored_does_leave_name_and_value_cells_unstyled(metric: str):

    row = line_containing(render_report(_colorful_result(), ReportOptions(color=True)), metric)

    assert "\x1b[" not in "│".join(cells_of(row)[:-1])


@pytest.mark.parametrize(
    "baseline",
    [
        pytest.param("v", id="v"),
        pytest.param("s", id="s"),
        pytest.param("vs", id="vs"),
    ],
)
def test_render_report_when_colored_does_embolden_the_baseline_after_the_vs_prefix(baseline: str):
    result = create_comparison_result(
        baseline_label=baseline,
        metrics={"a/time": permutation_metric(verdict="improved", delta=-10, unit="ns")},
    )

    cell = delta_cell(line_containing(render_report(result, ReportOptions(color=True)), "metric  "))

    assert f"vs {baseline}" in strip_ansi(cell)
    assert styles_at(cell, baseline, last=True) == ["1", "4"]


def test_render_report_when_colored_does_leave_a_dotted_variant_name_out_of_dimming():
    result = create_comparison_result(
        baseline_label="main·1",  # cspell:disable-line
        candidates=[create_candidate(label="perf·2")],  # cspell:disable-line
    )

    header = line_containing(render_report(result, ReportOptions(color=True)), "gymrat compare")

    assert styles_at(header, "main·1") == ["1", "4"]  # cspell:disable-line
    assert styles_at(header, "perf·2") == ["1", "4"]  # cspell:disable-line


def test_render_report_when_colored_does_leave_a_dotted_adapter_name_out_of_dimming():
    result = create_comparison_result(adapter="metric·lines")  # cspell:disable-line

    header = line_containing(render_report(result, ReportOptions(color=True)), "gymrat compare")

    assert "adapter: metric·lines" in header  # cspell:disable-line


# ---------------------------------------------------------------------------
# highlights color
# ---------------------------------------------------------------------------


# The glyph and SGR color each highlight verdict class carries in a colored entry.
_HIGHLIGHT_GLYPH_COLOR: dict[ApproximateVerdict, tuple[str, str]] = {
    "improved": ("✓", "32"),
    "regressed": ("✗", "31"),
}


def test_render_report_when_colored_does_style_the_verdict_word_not_a_matching_name():
    result = create_comparison_result(
        metrics={
            "unstable-parse/time": band_metric(verdict="unstable", delta=5, noise_pct=30, n=10)
        }
    )
    entry = highlight_lines(render_report(result, ReportOptions(color=True)))[0]

    assert strip_ansi(entry).strip() == "≈ unstable-parse/time  unstable  noise ±30.0%"
    assert "33" in styles_at(entry, "≈")
    assert "33" in styles_at(entry, "unstable", last=True)


def test_render_report_when_colored_does_dim_the_evidence_suffixes():
    result = create_comparison_result(
        metrics={
            "cheaper#heap": exact_metric(delta=-7.9),
            "jittery#time": band_metric(verdict="unstable", delta=5, noise_pct=30, n=10),
        }
    )
    highlights = highlight_lines(render_report(result, ReportOptions(color=True)))
    exact_entry = next(line for line in highlights if "cheaper#heap" in strip_ansi(line))
    unstable_entry = next(line for line in highlights if "jittery#time" in strip_ansi(line))

    assert "2" in styles_at(exact_entry, "(exact)")
    assert "2" in styles_at(unstable_entry, "noise")


def test_render_report_when_colored_does_dim_the_futility_note():
    result = create_comparison_result(
        metrics={"jittery/time": band_metric(verdict="unstable", delta=5, noise_pct=30, n=10)}
    )
    note = line_containing(render_report(result, ReportOptions(color=True)), "won't stabilize")

    assert "2" in styles_at(note, "unstable metrics")


# ---------------------------------------------------------------------------
# method footer and hint color
# ---------------------------------------------------------------------------


def _band_fallback_result() -> ComparisonResult:
    """A single no-signal metric that fell back to the band, so the hint and footer render."""
    return create_comparison_result(metrics={"a/time": band_metric(verdict="no-signal", delta=-5)})


def test_render_report_when_color_option_false_does_override_force_color(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("FORCE_COLOR", "1")

    output = render_report(_colorful_result(), ReportOptions(color=False))

    assert "\x1b[" not in output
