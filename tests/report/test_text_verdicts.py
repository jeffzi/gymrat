"""Tests for the verdict summary, highlights, gate trips and footers.

These cover the one-line verdict tally below the table,
the highlights block and its futility note, the ``--fail-on`` geomean gate-trip
lines, the verbose method footer, and the worktree-cleanup footer. The report
header and table are pinned by ``test_text`` and ``test_text_multi``; here the
report is driven end to end and only its assembled tail is asserted.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import TYPE_CHECKING

import pytest

from gymrat_py.report.text import render_report
from gymrat_py.report.types import (
    CandidateMetric,
    GeomeanFailOn,
    MetricComparison,
    RegressedFailOn,
    ReportOptions,
)
from gymrat_py.targets import WorktreeRemovalFailure
from tests.report._inputs import (
    band_metric,
    band_verdict,
    cells_of,
    create_candidate,
    create_comparison_result,
    exact_metric,
    exact_verdict,
    geomean_of,
    grouped_comparison,
    highlight_lines,
    line_containing,
    line_starting_with,
    memory_kind,
    metric_meta,
    other_kind,
    permutation_metric,
    permutation_verdict,
    single_sample_result,
    strip_ansi,
    styles_at,
    time_kind,
    two_kind_result,
    without_gated_geomean,
)

if TYPE_CHECKING:
    from gymrat_py.report.types import ComparisonResult


# ---------------------------------------------------------------------------
# verdict summary
# ---------------------------------------------------------------------------


def test_render_report_when_summarizing_does_count_every_verdict_class_on_one_line():
    result = create_comparison_result(
        metrics={
            "faster/time": permutation_metric(verdict="improved", delta=-10),
            "also-faster/time": permutation_metric(verdict="improved", delta=-5),
            "slower/time": permutation_metric(verdict="regressed", delta=8),
            "jittery/time": permutation_metric(verdict="unstable", delta=5, noise_pct=300),
            "flat/time": permutation_metric(verdict="no-signal", delta=0.2),
        }
    )

    report = render_report(result)

    assert line_starting_with(report, "✓ 2 improved") == (
        "✓ 2 improved   ✗ 1 regressed   ≈ 1 unstable   "
        "= 0 identical   ~ 1 within noise   ? 0 inconclusive"
    )


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


def test_render_report_when_ties_starve_the_test_does_tally_it_apart_from_within_noise():
    report = render_report(_identical_result())

    assert line_starting_with(report, "✓ 1 improved") == (
        "✓ 1 improved   ✗ 0 regressed   ≈ 0 unstable   "
        "= 1 identical   ~ 0 within noise   ? 0 inconclusive"
    )


def test_render_report_when_ties_starve_the_test_does_leave_it_out_of_the_highlights():
    highlights = [line.strip() for line in highlight_lines(render_report(_identical_result()))]

    assert highlights == ["✓ faster/time  -10.0%"]


def test_render_report_when_ties_starve_the_test_does_say_nothing_more_in_the_footer():
    assert "close-to-identical" not in render_report(_identical_result())


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
# every verdict on a single pair
# ---------------------------------------------------------------------------


def test_render_report_when_single_pair_does_mark_the_row_inconclusive():
    row = line_starting_with(render_report(single_sample_result()), "decode/time")

    assert cells_of(row)[-1].strip() == "?  -0.4%"


def test_render_report_when_single_pair_does_tally_the_metrics_in_their_own_bucket():
    report = render_report(single_sample_result())

    assert line_starting_with(report, "✓ 0 improved") == (
        "✓ 0 improved   ✗ 0 regressed   ≈ 0 unstable   "
        "= 0 identical   ~ 0 within noise   ? 2 inconclusive"
    )


def test_render_report_when_single_pair_does_hint_at_the_longer_run():
    assert (
        "re-run with gymrat compare --samples 6 or more for statistical verdicts"
        in render_report(single_sample_result())
    )


# ---------------------------------------------------------------------------
# highlights block
# ---------------------------------------------------------------------------


def test_render_report_when_highlighting_does_carry_glyph_delta_and_evidence():
    result = create_comparison_result(
        metrics={
            "slower/time": permutation_metric(verdict="regressed", delta=2.2, p=0.002),
            "cheaper/heap": exact_metric(delta=-7.9),
            "jittery/time": band_metric(verdict="unstable", delta=5, noise_pct=30),
        }
    )

    highlights = [line.strip() for line in highlight_lines(render_report(result))]

    assert highlights == [
        "✗ slower/time    +2.2%",
        "✓ cheaper/heap   -7.9%  (exact)",
        "≈ jittery/time  unstable  noise ±30.0%",
        "unstable metrics won't stabilize with more samples",
    ]


def test_render_report_when_highlighting_does_state_noise_in_absolute_units_when_large():
    result = create_comparison_result(
        metrics={
            "jittery/heap": permutation_metric(
                verdict="unstable",
                delta=5,
                baseline_median=5,
                noise_pct=7620,
                noise_abs=381,
                unit="bytes",
            ),
        }
    )

    highlights = [line.strip() for line in highlight_lines(render_report(result))]

    assert highlights[0] == "≈ jittery/heap  unstable  ±381B noise on a 5B median"


def test_render_report_when_nothing_moved_does_omit_the_highlights_block():
    result = create_comparison_result(
        metrics={"flat/time": permutation_metric(verdict="no-signal", delta=0.2)}
    )

    assert "highlights" not in render_report(result)


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


def test_render_report_when_gate_trips_does_echo_the_kind_geomean_and_condition():
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


@pytest.mark.parametrize(
    "options",
    [
        pytest.param(ReportOptions(), id="no-conditions"),
        pytest.param(ReportOptions(fail_on=(GeomeanFailOn(pct=10),)), id="threshold-beyond"),
        pytest.param(ReportOptions(fail_on=(RegressedFailOn(),)), id="only-regressed"),
    ],
)
def test_render_report_when_gate_would_not_trip_does_say_nothing_about_a_gate(
    options: ReportOptions,
):
    assert "⚑" not in render_report(_tripping_result(), options)


def test_render_report_when_kind_is_informational_does_say_nothing_about_a_gate():
    result = replace(
        two_kind_result(),
        candidates=(
            create_candidate(kinds=[time_kind(), without_gated_geomean(other_kind(9, 1))]),
        ),
    )

    report = render_report(result, ReportOptions(fail_on=(GeomeanFailOn(pct=2),)))

    assert "⚑" not in report


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


def test_render_report_when_gate_trips_with_color_does_paint_the_glyph_and_delta_red(
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
# closing the report
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "options",
    [
        pytest.param(ReportOptions(), id="plain"),
        pytest.param(ReportOptions(verbose=True), id="verbose"),
    ],
)
def test_render_report_when_closing_does_spell_out_no_legend(options: ReportOptions):
    result = create_comparison_result(
        metrics={"a/time": permutation_metric(verdict="improved", delta=-10)}
    )

    output = render_report(result, options)

    assert "legend:" not in output
    assert "candidates are judged against" not in output


# ---------------------------------------------------------------------------
# non-verbose footer
# ---------------------------------------------------------------------------


def test_render_report_when_not_verbose_does_stop_after_highlights():
    result = create_comparison_result(
        metrics={
            "a/time": permutation_metric(verdict="improved", delta=-10),
            "b/time": band_metric(verdict="improved", delta=-5, n=10, usable_n=3),
        }
    )

    output = render_report(result)

    assert "sign-flip permutation test" not in output
    assert "noise band" not in output


def test_render_report_when_not_verbose_does_name_dropped_rounds():
    result = create_comparison_result(
        metrics={"a/time": band_metric(verdict="no-signal", delta=-5, n=4)}
    )

    output = render_report(result)

    assert "noise band" not in output
    assert "some rounds were dropped" in output


def test_render_report_when_not_verbose_does_keep_the_worktree_footer_below_the_hint():
    result = create_comparison_result(
        metrics={"a/time": band_metric(verdict="no-signal", delta=-5, n=4)},
        worktrees_removed=1,
        worktrees_left_behind=[WorktreeRemovalFailure(dir="/tmp/gymrat-abc", error="is locked")],
    )

    lines = render_report(result).split("\n")

    assert "some rounds were dropped" in lines[-3]
    assert lines[-2] == "1 worktree removed · 1 left behind"
    assert lines[-1] == "  left behind: /tmp/gymrat-abc (is locked)"


# ---------------------------------------------------------------------------
# verbose method footer
# ---------------------------------------------------------------------------


def test_render_report_when_verbose_does_name_the_permutation_test():
    result = create_comparison_result(
        metrics={
            "a/time": permutation_metric(verdict="improved", delta=-10),
            "b/time": band_metric(verdict="improved", delta=-5),
        }
    )

    output = render_report(result, ReportOptions(verbose=True))

    assert "sign-flip permutation test" in output
    assert "n=10 ≥ 6" in output


def test_render_report_when_verbose_and_band_only_does_name_the_band_and_hint():
    result = create_comparison_result(
        metrics={"a/time": band_metric(verdict="no-signal", delta=-5)}
    )

    output = render_report(result, ReportOptions(verbose=True))

    assert "noise band ±(half-range × K)" in output
    assert "below permutation floor (6 pairs)" in output
    assert "sign-flip permutation test" not in output
    assert "some rounds were dropped" in output


def test_render_report_when_verbose_and_all_exact_does_name_no_method_or_hint():
    result = create_comparison_result(metrics={"a/heap": exact_metric(delta=-7.9)})

    output = render_report(result, ReportOptions(verbose=True))

    assert "sign-flip permutation test" not in output
    assert "noise band" not in output
    assert "some rounds were dropped" not in output


def test_render_report_when_verbose_and_permutation_carried_the_run_does_drop_the_hint():
    result = create_comparison_result(
        metrics={"a/time": permutation_metric(verdict="improved", delta=-10)}
    )

    output = render_report(result, ReportOptions(verbose=True))

    assert "some rounds were dropped" not in output


def test_render_report_when_verbose_does_phrase_each_band_fallback_by_its_cause():
    result = create_comparison_result(
        metrics={
            "short/time": band_metric(verdict="no-signal", delta=1, n=4),
            "tied/heap": band_metric(verdict="no-signal", delta=-0.5, n=10, usable_n=3),
        }
    )

    band_lines = [
        line
        for line in render_report(result, ReportOptions(verbose=True)).split("\n")
        if line.startswith("noise band")
    ]

    assert band_lines == [
        "noise band ±(half-range × K) — n=4 below permutation floor (6 pairs)",
        "noise band ±(half-range × K) — ties left n=3 usable pairs (6 needed)",
    ]


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


def test_render_report_when_methods_differ_does_hint_at_dropped_rounds():
    output = render_report(_mixed_method_result())

    assert "some rounds were dropped" in output


# ---------------------------------------------------------------------------
# worktree cleanup footer
# ---------------------------------------------------------------------------


def test_render_report_when_cleanup_left_nothing_behind_does_say_nothing():
    result = create_comparison_result(worktrees_removed=0, worktrees_left_behind=[])

    output = render_report(result)

    assert "worktree" not in output
    assert "left behind" not in output


def test_render_report_when_cleanup_removed_everything_cleanly_does_suppress_the_footer():
    result = create_comparison_result(worktrees_removed=3, worktrees_left_behind=[])

    output = render_report(result)

    assert "worktree" not in output
    assert "left behind" not in output


def test_render_report_when_worktrees_left_behind_does_render_the_footer():
    result = create_comparison_result(
        worktrees_removed=2,
        worktrees_left_behind=[WorktreeRemovalFailure(dir="/tmp/gymrat-abc", error="is locked")],
    )

    output = render_report(result)

    assert "2 worktrees removed · 1 left behind" in output
    assert "left behind: /tmp/gymrat-abc (is locked)" in output


def test_render_report_when_only_prune_failed_does_render_the_footer():
    result = create_comparison_result(
        worktrees_removed=3,
        worktrees_left_behind=[],
        worktree_prune_error="fatal: not a git repository",
    )

    output = render_report(result)

    assert "worktree prune failed: fatal: not a git repository" in output


def test_render_report_when_several_left_behind_does_name_each_with_its_reason():
    result = create_comparison_result(
        worktrees_removed=1,
        worktrees_left_behind=[
            WorktreeRemovalFailure(
                dir="/tmp/gymrat-abc", error="contains modified or untracked files"
            ),
            WorktreeRemovalFailure(dir="/tmp/gymrat-def", error="is locked"),
        ],
    )

    output = render_report(result)

    assert "left behind: /tmp/gymrat-abc (contains modified or untracked files)" in output
    assert "left behind: /tmp/gymrat-def (is locked)" in output


def test_render_report_when_only_prune_failed_and_nothing_else_does_report_the_reason():
    result = create_comparison_result(worktree_prune_error="fatal: not a git repository")

    output = render_report(result)

    assert "worktree prune failed: fatal: not a git repository" in output


def test_render_report_when_left_behind_reason_spans_lines_does_collapse_to_one_line():
    result = create_comparison_result(
        worktrees_removed=1,
        worktrees_left_behind=[
            WorktreeRemovalFailure(
                dir="/tmp/gymrat-abc",
                error="warning: could not open directory\n  fatal: '/tmp/gymrat-abc' is locked",
            ),
        ],
    )

    detail_lines = [line for line in render_report(result).split("\n") if "/tmp/gymrat-abc" in line]

    assert detail_lines == [
        (
            "  left behind: /tmp/gymrat-abc "
            "(warning: could not open directory fatal: '/tmp/gymrat-abc' is locked)"
        ),
    ]


def test_render_report_when_prune_reason_spans_lines_does_collapse_to_one_line():
    result = create_comparison_result(
        worktree_prune_error="warning: unable to unlink\n  fatal: not a git repository",
    )

    prune_lines = [line for line in render_report(result).split("\n") if "prune failed" in line]

    assert prune_lines == [
        "  worktree prune failed: warning: unable to unlink fatal: not a git repository",
    ]


def test_render_report_when_cleanup_is_dirty_does_close_with_left_behind_and_prune():
    result = create_comparison_result(
        worktrees_removed=0,
        worktrees_left_behind=[WorktreeRemovalFailure(dir="/tmp/gymrat-abc", error="is locked")],
        worktree_prune_error="fatal: not a git repository",
    )

    lines = render_report(result).split("\n")

    assert "0 worktrees removed · 1 left behind" in lines[-3]
    assert lines[-2] == "  left behind: /tmp/gymrat-abc (is locked)"
    assert lines[-1] == "  worktree prune failed: fatal: not a git repository"


# ---------------------------------------------------------------------------
# verdict cell alignment with mixed band/no-band/nan-delta rows (B21)
# ---------------------------------------------------------------------------


def _mixed_band_result() -> ComparisonResult:
    """A report mixing a banded row, a no-band exact row, and a NaN-delta row.

    The banded row carries a noise band, the exact row sits beside approximate
    ones (so the band column exists but the exact row has nothing to put there),
    and the NaN-delta row has an undefined ratio so its delta is absent. All
    three must share the same column layout in the verdict cell.
    """
    return create_comparison_result(
        metrics={
            "banded/time": permutation_metric(
                verdict="improved", delta=-10, noise_pct=2.5, unit="ns"
            ),
            "exact/heap": MetricComparison(
                baseline_median=49152,
                baseline_spread=None,
                candidates=(
                    CandidateMetric(
                        median=45261,
                        verdict=exact_verdict(verdict="improved", delta=-7.9),
                    ),
                ),
                meta=metric_meta("exact/heap", exact=True, unit="bytes"),
            ),
            "nan-delta/count": MetricComparison(
                baseline_median=0,
                baseline_spread=None,
                candidates=(CandidateMetric(median=120, verdict=exact_verdict(delta=math.nan)),),
                meta=metric_meta("nan-delta/count", exact=True),
            ),
        },
    )


def test_render_report_when_mixing_banded_exact_and_nan_delta_does_align_verdict_fields():
    """Mixed band/exact/NaN-delta rows align their verdict fields to the same columns."""
    report = render_report(_mixed_band_result())

    banded_cell = cells_of(line_starting_with(report, "banded/time"))[-1]
    exact_cell = cells_of(line_starting_with(report, "exact/heap"))[-1]
    nan_cell = cells_of(line_starting_with(report, "nan-delta/count"))[-1]

    assert banded_cell.strip() == "✓  -10.0%  ±2.5%"
    assert exact_cell.strip() == "✓   -7.9%"
    assert nan_cell.strip() == "~"

    # The delta field in the banded and exact rows must end at the same column
    # offset, proving the band gap is padded rather than collapsed.
    banded_delta_end = banded_cell.index("-10.0%") + len("-10.0%")
    exact_delta_end = exact_cell.index("-7.9%") + len("-7.9%")
    assert banded_delta_end == exact_delta_end


def test_render_report_when_mixing_banded_exact_and_nan_delta_does_match_styled_and_plain(
    monkeypatch: pytest.MonkeyPatch,
):
    """Styled and plain verdict cells have identical visible text for every row type."""
    plain_report = render_report(_mixed_band_result())

    monkeypatch.setenv("FORCE_COLOR", "1")
    monkeypatch.delenv("NO_COLOR", raising=False)
    styled_report = render_report(_mixed_band_result())

    for prefix in ("banded/time", "exact/heap", "nan-delta/count"):
        plain_cell = cells_of(line_starting_with(plain_report, prefix))[-1]
        styled_cell = cells_of(line_starting_with(styled_report, prefix))[-1]
        assert strip_ansi(styled_cell) == plain_cell
