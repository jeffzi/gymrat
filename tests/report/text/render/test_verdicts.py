"""Tests for the comparison report's verdicts, highlights, gate trips and footers.

These cover the identical-row verdict, highlight selection (:func:`select_highlights`)
and the highlights block it feeds, the ``--fail-on`` gate-trip lines, the
method footer (:func:`footer_lines`) and its hints, and the worktree-cleanup
footer. The report is driven end to end where the assembled output is the
behavior, and through the footer and selection functions where their own rules
are.

The colored report is covered too: how verdict cells, the run and column
headers, and the highlights block are painted, and how the color option
overrides ``FORCE_COLOR``.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import pytest

from gymrat.model import PERMUTATION_MIN_N, Exclusion
from gymrat.report.style import format_hint
from gymrat.report.text.render import footer_lines, render_report, select_highlights
from gymrat.report.types import GeomeanFailOn, RegressedFailOn, ReportOptions
from gymrat.targets import WorktreeRemovalFailure
from tests._ansi import sgr_codes, strip_ansi
from tests.report._assertions import (
    cells_of,
    delta_cell,
    highlight_lines,
    line_containing,
    line_starting_with,
    render_colored,
    render_plain,
    styles_at,
)
from tests.report._comparisons import (
    create_candidate,
    create_comparison_result,
    exact_metric,
    grouped_comparison,
    memory_kind,
    other_kind,
    permutation_metric,
    time_kind,
    two_kind_result,
    without_gated_geomean,
)
from tests.report._verdicts import (
    CandidateSpec,
    approximate_metric,
    band_metric,
    geomean_of,
    metric_for,
    one_sided_metric,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from gymrat.report.types import ComparisonResult, MetricComparisons


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


# ---------------------------------------------------------------------------
# highlights block
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
# highlights color
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("index", "marker", "expected"),
    [
        pytest.param(0, "✗", ["31"], id="regressed-glyph-red"),
        pytest.param(0, "+2.4%", ["31"], id="regressed-delta-red"),
        pytest.param(0, "slower/", ["2"], id="name-prefix-dim"),
        pytest.param(1, "✓", ["32"], id="improved-glyph-green"),
        pytest.param(1, "-17.5%", ["32"], id="improved-delta-green"),
    ],
)
def test_render_report_when_colored_does_paint_each_highlight_by_its_verdict(
    index: int, marker: str, expected: list[str]
):
    entry = highlight_lines(render_report(_colorful_result(), ReportOptions(color=True)))[index]

    assert styles_at(entry, marker) == expected


def test_render_report_when_colored_does_dim_the_kind_suffix_of_a_highlighted_name():
    result = create_comparison_result(
        metrics={"slow#time": permutation_metric(verdict="regressed", delta=4)}
    )

    entry = highlight_lines(render_report(result, ReportOptions(color=True)))[0]

    assert styles_at(entry, "#time") == ["2"]


def test_render_report_when_colored_does_embolden_each_candidate_label_in_the_highlights():
    entry = highlight_lines(render_report(grouped_comparison(), ReportOptions(color=True)))[0]

    assert styles_at(entry, "candidate-a") == ["1"]


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


def test_render_report_when_gate_trips_with_color_does_paint_the_trip_red():
    line = line_containing(
        render_report(
            _tripping_result(), ReportOptions(fail_on=(GeomeanFailOn(pct=2),), color=True)
        ),
        "⚑",
    )

    assert "31" in styles_at(line, "⚑")
    assert "31" in styles_at(line, "+3.1%")


# ---------------------------------------------------------------------------
# method footer
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
def test_footer_lines_when_verbose_does_name_the_band_cause_per_metric(
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
def test_footer_lines_when_not_verbose_does_pick_the_shortage_or_dropped_rounds_hint(
    metrics: MetricComparisons, samples: int, expected: list[str]
):
    lines = footer_lines(metrics, verbose=False, command="compare", samples=samples)

    assert [render_plain(line) for line in lines] == expected


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
    assert "n=4" in band_line
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
            ["2 worktrees removed · 1 left behind", "  left behind: /tmp/gymrat-abc (is locked)"],
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
                "1 worktree removed · 2 left behind",
                "  left behind: /tmp/gymrat-abc (contains modified or untracked files)",
                "  left behind: /tmp/gymrat-def (is locked)",
            ],
            id="several-left-behind",
        ),
        pytest.param(
            create_comparison_result(worktree_prune_error="fatal: not a git repository"),
            [
                "0 worktrees removed · 0 left behind",
                "  worktree prune failed: fatal: not a git repository",
            ],
            id="only-prune-failed",
        ),
    ],
)
def test_render_report_when_cleanup_left_worktrees_or_prune_failed_does_render_the_footer(
    result: ComparisonResult, expected: list[str]
):
    footer = render_report(result).split("\n\n")[-1].split("\n")

    assert footer == expected


def _with_left_behind_reason(reason: str) -> ComparisonResult:
    return create_comparison_result(
        worktrees_removed=1,
        worktrees_left_behind=[WorktreeRemovalFailure(dir="/tmp/gymrat-abc", error=reason)],
    )


def _with_prune_error(reason: str) -> ComparisonResult:
    return create_comparison_result(worktree_prune_error=reason)


#: A git reason carrying every whitespace class: tabs, space runs, CRLF, blank
#: lines, Unicode spaces, and leading and trailing whitespace.
_WHITESPACE_REASON = f"\n\t fatal:{chr(0xA0)}\r\n  not{chr(0x2003)}  a git\n\nrepository\t "


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
    build_result: Callable[[str], ComparisonResult], expected: str
):
    result = build_result(_WHITESPACE_REASON)

    matching_lines = [
        line for line in render_report(result).split("\n") if "not a git repository" in line
    ]

    assert matching_lines == [expected]


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
def test_render_report_when_colored_does_paint_only_the_verdict_cell(
    metric: str, glyph: str, code: str
):
    row = line_containing(render_report(_colorful_result(), ReportOptions(color=True)), metric)

    assert code in styles_at(row, glyph)
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
# color option precedence
# ---------------------------------------------------------------------------


def test_render_report_when_color_option_false_does_override_force_color(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("FORCE_COLOR", "1")

    output = render_report(_colorful_result(), ReportOptions(color=False))

    assert "\x1b[" not in output
