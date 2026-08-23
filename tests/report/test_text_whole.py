"""Whole-report assembly and layout tests for the comparison report.

These port ``text-whole.test.ts``. Its structural ``tableRegion`` / ``stylesAt``
cases port directly. Its twelve ``toMatchFileSnapshot`` golden snapshots cannot
byte-match the rich output, so each is re-pinned as a content/shape assertion:
the table layout via :func:`table_region`, the assembled tail via the summary
line(s), the highlights block, and the footer/worktree lines, plus ``styles_at``
on the colored markers the snapshot fixed. Highlight entries are compared with their
internal padding collapsed — that padding is pinned exactly by
``test_text_verdicts`` — so these tests pin order and content without re-pinning
column widths a second time.
"""

from __future__ import annotations

import math
import re
from dataclasses import replace
from typing import TYPE_CHECKING

import pytest

from gymrat_py.config import KindEntry
from gymrat_py.model import Exclusion
from gymrat_py.report.text import render_report
from gymrat_py.report.types import CandidateMetric, MetricComparison, ReportOptions
from gymrat_py.targets import WorktreeRemovalFailure
from gymrat_py.verdict import GroupAggregate, KindAggregate
from tests.report._inputs import (
    NWayCandidate,
    band_verdict,
    cells_of,
    create_candidate,
    create_comparison_result,
    exact_verdict,
    geomean_of,
    grouped_comparison,
    highlight_lines,
    kind_metric,
    line_containing,
    line_starting_with,
    memory_kind,
    metric_meta,
    multi_candidate_result,
    n_way_kind_metric,
    other_kind,
    signed_rank_metric,
    signed_rank_verdict,
    single_sample_result,
    styles_at,
    table_region,
    two_kind_result,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from gymrat_py.report.types import ComparisonResult

_HEADER = (
    "gymrat compare · baseline main ↔ perf/faster-decode · 10 paired samples · adapter: mitata"
)


@pytest.fixture(autouse=True)
def _no_color(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.delenv("FORCE_COLOR", raising=False)


def _normalized_highlights(report: str) -> list[str]:
    """The highlight block's lines with runs of whitespace collapsed to one space.

    ``test_text_verdicts`` pins the exact padding; here the concern is order and
    content, so the alignment padding is folded away.
    """
    return [re.sub(r"\s+", " ", line.strip()) for line in highlight_lines(report)]


# ---------------------------------------------------------------------------
# flat single-kind layout
# ---------------------------------------------------------------------------


def _one_kind_result() -> ComparisonResult:
    """A single gating ``time`` kind whose two metrics share the ``entity`` group."""
    geomean = geomean_of(-3.2, 2)
    return create_comparison_result(
        metrics={
            "entity.alive_check/time": kind_metric(
                kind="time", short_name="entity.alive_check", verdict="improved", delta=-10
            ),
            "entity.spawn/time": kind_metric(
                kind="time", short_name="entity.spawn", verdict="regressed", delta=4
            ),
        },
        candidates=[
            create_candidate(
                kinds=[
                    KindAggregate(
                        kind="time",
                        geomean=geomean,
                        groups=(GroupAggregate(group="entity", geomean=geomean),),
                        gated_geomean=geomean,
                    )
                ]
            )
        ],
    )


def test_render_report_when_one_kind_does_keep_the_flat_layout_and_one_geomean_row():
    assert table_region(render_report(_one_kind_result())) == [
        _HEADER,
        "metric",
        "<rule>",
        "entity.alive_check/time",
        "entity.spawn/time",
        "<rule>",
        "geomean (2 stable metrics)",
    ]


def test_render_report_when_the_kind_does_not_gate_does_report_no_stable_metrics():
    result = create_comparison_result(
        metrics={
            "warmup/time": kind_metric(
                kind="time", short_name="warmup", verdict="improved", delta=-10, gating=False
            ),
        },
        candidates=[
            create_candidate(
                kinds=[KindAggregate(kind="time", geomean=geomean_of(-10, 1), groups=())]
            )
        ],
    )

    row = line_starting_with(render_report(result), "geomean")

    assert [cell.strip() for cell in cells_of(row)] == ["geomean", "", "", "—  no stable metrics"]


def test_render_report_when_flat_geomean_clears_its_band_does_paint_it_green(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("FORCE_COLOR", "1")
    result = replace(
        _one_kind_result(),
        candidates=(
            create_candidate(
                kinds=[
                    KindAggregate(
                        kind="time",
                        geomean=geomean_of(-3.2, 2),
                        groups=(),
                        gated_geomean=geomean_of(-3.2, 2, band=1),
                    )
                ]
            ),
        ),
    )

    line = line_containing(render_report(result), "geomean")

    assert styles_at(line, "-3.2%") == ["1", "32"]


# ---------------------------------------------------------------------------
# flat non-gating kind
# ---------------------------------------------------------------------------


def _flat_non_gating_result() -> ComparisonResult:
    """A single non-gating ``time`` kind whose informational tag carries the config source."""
    return create_comparison_result(
        metrics={
            "warmup/time": kind_metric(
                kind="time", short_name="warmup", verdict="improved", delta=-10, gating=False
            ),
            "cooldown/time": kind_metric(
                kind="time", short_name="cooldown", verdict="no-signal", delta=0.3, gating=False
            ),
        },
        candidates=[
            create_candidate(
                kinds=[KindAggregate(kind="time", geomean=geomean_of(-5, 2), groups=())]
            )
        ],
        config_kinds={"time": KindEntry(gating=False)},
    )


def test_render_report_when_the_sole_kind_gates_nothing_does_tag_before_the_header():
    assert table_region(render_report(_flat_non_gating_result())) == [
        _HEADER,
        "informational — gating off (config: kinds.time.gating = false)",
        "metric",
        "<rule>",
        "warmup/time",
        "cooldown/time",
        "<rule>",
        "geomean",
    ]


@pytest.mark.parametrize(
    ("make_result", "expected"),
    [
        pytest.param(
            _flat_non_gating_result,
            "informational — gating off (config: kinds.time.gating = false)",
            id="kind-level-config",
        ),
        pytest.param(
            lambda: replace(_flat_non_gating_result(), config_kinds=None),
            "informational — gating off",
            id="per-metric-overrides",
        ),
    ],
)
def test_render_report_when_flat_kind_is_informational_does_credit_the_config_source(
    make_result: Callable[[], ComparisonResult], expected: str
):
    report = render_report(make_result())

    assert line_containing(report, "informational") == expected


def test_render_report_when_flat_kind_is_informational_and_colored_does_dim_the_tag(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("FORCE_COLOR", "1")

    tag = line_containing(render_report(_flat_non_gating_result()), "informational")

    assert styles_at(tag, "informational") == ["2"]


# ---------------------------------------------------------------------------
# group and geomean labels, colored
# ---------------------------------------------------------------------------


def _flat_result() -> ComparisonResult:
    """A single-candidate run of one kind, whose table closes on a flat geomean row."""
    return create_comparison_result(
        metrics={"faster/time": signed_rank_metric(verdict="improved", delta=-17.5)}
    )


@pytest.mark.parametrize(
    "make_result",
    [
        pytest.param(two_kind_result, id="single-candidate"),
        pytest.param(grouped_comparison, id="multi-candidate"),
    ],
)
def test_render_report_when_colored_does_paint_the_group_sub_header_blue(
    monkeypatch: pytest.MonkeyPatch, make_result: Callable[[], ComparisonResult]
):
    monkeypatch.setenv("FORCE_COLOR", "1")

    row = line_containing(render_report(make_result()), "entity")

    assert styles_at(row, "entity") == ["34"]


@pytest.mark.parametrize(
    ("label", "make_result"),
    [
        pytest.param("geomean · entity", two_kind_result, id="group-single"),
        pytest.param("geomean · time", two_kind_result, id="kind-single"),
        pytest.param("geomean", _flat_result, id="flat-single"),
        pytest.param("geomean · entity", grouped_comparison, id="group-multi"),
        pytest.param("geomean · time", grouped_comparison, id="kind-multi"),
        pytest.param("geomean", multi_candidate_result, id="flat-multi"),
    ],
)
def test_render_report_when_colored_does_embolden_the_geomean_label(
    monkeypatch: pytest.MonkeyPatch, label: str, make_result: Callable[[], ComparisonResult]
):
    monkeypatch.setenv("FORCE_COLOR", "1")

    row = line_containing(render_report(make_result()), label)

    assert styles_at(row, label) == ["1"]


# ---------------------------------------------------------------------------
# geomean value stays quiet when its metrics landed within noise
# ---------------------------------------------------------------------------


def _quiet_two_kind_result() -> ComparisonResult:
    """A two-kind run whose every metric landed within noise.

    Each geomean figure sits far outside its own band, so a rule that reads the
    band alone would paint all of them green.
    """
    time_geomean = geomean_of(-8.5, 2)
    return create_comparison_result(
        metrics={
            "entity.alive_check/time": kind_metric(
                kind="time", short_name="entity.alive_check", verdict="no-signal", delta=-9
            ),
            "entity.spawn/time": kind_metric(
                kind="time", short_name="entity.spawn", verdict="no-signal", delta=-8
            ),
            "encode/heap": kind_metric(
                kind="memory",
                short_name="encode",
                verdict="no-signal",
                delta=-7,
                gating=False,
                unit="bytes",
            ),
        },
        candidates=[
            create_candidate(
                kinds=[
                    KindAggregate(
                        kind="time",
                        geomean=time_geomean,
                        groups=(GroupAggregate(group="entity", geomean=geomean_of(-8.6, 2)),),
                        gated_geomean=time_geomean,
                    ),
                    memory_kind(),
                ]
            )
        ],
        config_kinds={"memory": KindEntry(gating=False)},
    )


@pytest.mark.parametrize(
    ("label", "value"),
    [
        pytest.param("geomean · entity", "-8.6%", id="group"),
        pytest.param("geomean · time", "-8.5%", id="kind"),
    ],
)
def test_render_report_when_quiet_metrics_and_colored_does_leave_the_geomean_uncolored(
    monkeypatch: pytest.MonkeyPatch, label: str, value: str
):
    monkeypatch.setenv("FORCE_COLOR", "1")

    line = line_containing(render_report(_quiet_two_kind_result()), label)

    assert styles_at(line, value) == ["1"]


def test_render_report_when_quiet_flat_metric_and_colored_does_leave_the_geomean_uncolored(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("FORCE_COLOR", "1")
    result = create_comparison_result(
        metrics={"faster/time": signed_rank_metric(verdict="no-signal", delta=-0.5)}
    )

    line = line_containing(render_report(result), "geomean")

    assert styles_at(line, "-5.8%") == ["1"]


def test_render_report_when_colored_does_judge_each_candidate_column_by_its_own_verdicts(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("FORCE_COLOR", "1")
    result = create_comparison_result(
        metrics={
            "entity.alive_check/time": n_way_kind_metric(
                kind="time",
                short_name="entity.alive_check",
                candidates=[
                    NWayCandidate(verdict="no-signal", delta=-9, median=91),
                    NWayCandidate(verdict="improved", delta=-12, median=88),
                ],
            ),
            "encode/heap": n_way_kind_metric(
                kind="memory",
                short_name="encode",
                gating=False,
                candidates=[
                    NWayCandidate(verdict="no-signal", delta=-1, median=99),
                    NWayCandidate(verdict="improved", delta=-2, median=98),
                ],
            ),
        },
        candidates=[
            create_candidate(
                label="candidate-a",
                kinds=[
                    _time_kind_of(-9),
                    KindAggregate(kind="memory", geomean=geomean_of(-1, 1), groups=()),
                ],
            ),
            create_candidate(
                label="candidate-b",
                kinds=[
                    _time_kind_of(-12),
                    KindAggregate(kind="memory", geomean=geomean_of(-2, 1), groups=()),
                ],
            ),
        ],
        config_kinds={"memory": KindEntry(gating=False)},
    )

    line = line_containing(render_report(result), "geomean · time")

    assert styles_at(line, "-9.0%") == ["1"]
    assert styles_at(line, "-12.0%") == ["1", "32"]


def _time_kind_of(value: float) -> KindAggregate:
    geomean = geomean_of(value, 1)
    return KindAggregate(
        kind="time",
        geomean=geomean,
        groups=(GroupAggregate(group="entity", geomean=geomean),),
        gated_geomean=geomean,
    )


# ---------------------------------------------------------------------------
# whole-report assembly (golden conversions)
# ---------------------------------------------------------------------------


def _representative_result() -> ComparisonResult:
    return create_comparison_result(
        metrics={
            "decode/text=digits/time": MetricComparison(
                baseline_median=1735,
                baseline_spread=1,
                candidates=(
                    CandidateMetric(
                        median=1425,
                        spread=1,
                        verdict=signed_rank_verdict(verdict="improved", delta=-17.9, p=0.002),
                    ),
                ),
                meta=metric_meta("decode/text=digits/time", unit="ns"),
            ),
            "decode/text=words/time": MetricComparison(
                baseline_median=3065,
                baseline_spread=1,
                candidates=(
                    CandidateMetric(
                        median=3093,
                        spread=3,
                        verdict=signed_rank_verdict(verdict="no-signal", delta=0.9, p=0.49),
                    ),
                ),
                meta=metric_meta("decode/text=words/time", unit="ns"),
            ),
            "encode/time": MetricComparison(
                baseline_median=914,
                baseline_spread=1,
                candidates=(
                    CandidateMetric(
                        median=934,
                        spread=1,
                        verdict=signed_rank_verdict(verdict="regressed", delta=2.2, p=0.002),
                    ),
                ),
                meta=metric_meta("encode/time", unit="ns"),
            ),
            "encode/heap": MetricComparison(
                baseline_median=49152,
                baseline_spread=0,
                candidates=(
                    CandidateMetric(
                        median=45261,
                        spread=0,
                        verdict=exact_verdict(verdict="improved", delta=-7.9),
                    ),
                ),
                meta=metric_meta("encode/heap", exact=True, unit="bytes"),
            ),
        },
        candidates=[create_candidate(kinds=[other_kind(-6, 4)])],
    )


def test_render_report_when_representative_does_assemble_table_summary_and_highlights():
    report = render_report(_representative_result())

    assert table_region(report) == [
        _HEADER,
        "metric",
        "<rule>",
        "decode/text=digits/time",
        "decode/text=words/time",
        "encode/time",
        "encode/heap",
        "<rule>",
        "geomean (4 stable metrics)",
    ]
    assert line_starting_with(report, "✓ 2 improved") == (
        "✓ 2 improved   ✗ 1 regressed   ≈ 0 unstable   "
        "= 0 identical   ~ 1 within noise   ? 0 inconclusive"
    )
    assert _normalized_highlights(report) == [
        "✗ encode/time +2.2%",
        "✓ decode/text=digits/time -17.9%",
        "✓ encode/heap -7.9% (exact)",
    ]
    assert "Hint" not in report
    assert "worktree" not in report


def _degenerate_result() -> ComparisonResult:
    return create_comparison_result(
        samples=4,
        adapter="metric-lines",
        metrics={
            "zero-median/time": MetricComparison(
                baseline_median=0,
                baseline_spread=None,
                candidates=(CandidateMetric(median=0, verdict=exact_verdict(n=4)),),
                meta=metric_meta("zero-median/time", exact=True, unit="ns"),
            ),
            "nan-delta/count": MetricComparison(
                baseline_median=0,
                baseline_spread=None,
                candidates=(
                    CandidateMetric(median=120, verdict=exact_verdict(delta=math.nan, n=4)),
                ),
                meta=metric_meta("nan-delta/count", exact=True),
            ),
            "old-side-only/time": MetricComparison(
                baseline_median=2048,
                baseline_spread=2,
                candidates=(CandidateMetric(),),
                meta=metric_meta("old-side-only/time", unit="ns"),
            ),
            "throughput/ops": MetricComparison(
                baseline_median=1200,
                baseline_spread=5,
                candidates=(
                    CandidateMetric(
                        median=1560,
                        spread=4,
                        verdict=band_verdict(verdict="improved", delta=30, n=4, usable_n=4),
                    ),
                ),
                meta=metric_meta("throughput/ops", direction="higher", gating=False),
            ),
        },
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
        worktrees_removed=1,
        worktrees_left_behind=[
            WorktreeRemovalFailure(dir="/tmp/gymrat-abc123", error="contains modified files")
        ],
        worktree_prune_error="could not lock config file",
    )


def test_render_report_when_degenerate_and_dirty_cleanup_does_close_with_footer_and_worktrees():
    report = render_report(_degenerate_result())

    assert line_starting_with(report, "✓ 1 improved") == (
        "✓ 1 improved   ✗ 0 regressed   ≈ 0 unstable   "
        "= 0 identical   ~ 2 within noise   ? 0 inconclusive"
    )
    assert _normalized_highlights(report) == ["✓ throughput/ops +30.0%"]

    lines = report.split("\n")
    assert lines[-4:] == [
        "Hint: re-run with --samples 6 or more for statistical verdicts",
        "1 worktree removed · 1 left behind",
        "  left behind: /tmp/gymrat-abc123 (contains modified files)",
        "  worktree prune failed: could not lock config file",
    ]


def _two_candidate_result() -> ComparisonResult:
    return create_comparison_result(
        candidates=[
            create_candidate(label="perf/simd-decode", kinds=[other_kind(-12.4, 3, band=30)]),
            create_candidate(
                label="perf/lut-decode",
                kinds=[
                    other_kind(
                        1.2,
                        2,
                        band=30,
                        excluded=[Exclusion(metric="encode/time", reason="unstable")],
                    )
                ],
            ),
        ],
        metrics={
            "decode/text=digits/time": MetricComparison(
                baseline_median=1735,
                baseline_spread=1,
                candidates=(
                    CandidateMetric(
                        median=1425,
                        spread=1,
                        verdict=signed_rank_verdict(verdict="improved", delta=-17.9, p=0.002),
                    ),
                    CandidateMetric(
                        median=1698,
                        spread=2,
                        verdict=signed_rank_verdict(verdict="no-signal", delta=-2.1, p=0.32),
                    ),
                ),
                meta=metric_meta("decode/text=digits/time", unit="ns"),
            ),
            "encode/time": MetricComparison(
                baseline_median=914,
                baseline_spread=1,
                candidates=(
                    CandidateMetric(
                        median=934,
                        spread=1,
                        verdict=signed_rank_verdict(verdict="regressed", delta=2.2, p=0.002),
                    ),
                    CandidateMetric(
                        median=1200,
                        spread=12,
                        verdict=band_verdict(
                            verdict="unstable",
                            delta=31.3,
                            n=4,
                            usable_n=4,
                            noise_pct=30,
                            noise_abs=30,
                        ),
                    ),
                ),
                meta=metric_meta("encode/time", unit="ns"),
            ),
            "encode/heap": MetricComparison(
                baseline_median=49152,
                baseline_spread=0,
                candidates=(
                    CandidateMetric(
                        median=45261,
                        spread=0,
                        verdict=exact_verdict(verdict="improved", delta=-7.9),
                    ),
                    CandidateMetric(),
                ),
                meta=metric_meta("encode/heap", exact=True, unit="bytes"),
            ),
        },
    )


def test_render_report_when_verbose_two_candidate_does_summarize_group_and_footer_per_candidate():
    report = render_report(_two_candidate_result(), ReportOptions(verbose=True))

    summaries = [line for line in report.split("\n") if re.search(r"✓ \d+ improved", line)]
    assert [re.sub(r"\s+", " ", line) for line in summaries] == [
        (
            "perf/simd-decode ✓ 2 improved ✗ 1 regressed ≈ 0 unstable "
            "= 0 identical ~ 0 within noise ? 0 inconclusive"
        ),
        (
            "perf/lut-decode ✓ 0 improved ✗ 0 regressed ≈ 1 unstable "
            "= 0 identical ~ 1 within noise ? 0 inconclusive"
        ),
    ]
    assert _normalized_highlights(report) == [
        "perf/simd-decode",
        "✗ encode/time +2.2%",
        "✓ decode/text=digits/time -17.9%",
        "✓ encode/heap -7.9% (exact)",
        "perf/lut-decode",
        "≈ encode/time unstable noise ±30.0%",
        "unstable metrics won't stabilize with more samples",
    ]

    lines = report.split("\n")
    assert lines[-3:] == [
        "verdicts: Wilcoxon signed-rank on pairs (n=10 ≥ 6) · ~ = no signal at α=0.05",
        "noise band ±(half-range × K) — n=4 below signed-rank floor (6 pairs)",
        (
            "Hint: some rounds were dropped — not all samples produced paired measurements "
            "for every metric"
        ),
    ]


def test_render_report_when_sectioned_does_assemble_summary_and_highlights_below_the_table():
    report = render_report(two_kind_result())

    assert line_starting_with(report, "✓ 2 improved") == (
        "✓ 2 improved   ✗ 1 regressed   ≈ 0 unstable   "
        "= 0 identical   ~ 1 within noise   ? 0 inconclusive"
    )
    assert _normalized_highlights(report) == [
        "✗ time · entity.spawn +4.0%",
        "✓ time · entity.alive_check -10.0%",
        "✓ memory · encode -7.0%",
    ]


def test_render_report_when_single_sample_does_close_on_the_hint_with_no_highlights():
    report = render_report(single_sample_result())

    assert "highlights" not in report
    assert line_starting_with(report, "✓ 0 improved") == (
        "✓ 0 improved   ✗ 0 regressed   ≈ 0 unstable   "
        "= 0 identical   ~ 0 within noise   ? 2 inconclusive"
    )
    assert report.split("\n")[-1] == (
        "Hint: re-run with --samples 6 or more for statistical verdicts"
    )


def test_render_report_when_flat_non_gating_does_assemble_tag_summary_and_highlights():
    report = render_report(_flat_non_gating_result())

    assert line_containing(report, "informational") == (
        "informational — gating off (config: kinds.time.gating = false)"
    )
    assert line_starting_with(report, "✓ 1 improved") == (
        "✓ 1 improved   ✗ 0 regressed   ≈ 0 unstable   "
        "= 0 identical   ~ 1 within noise   ? 0 inconclusive"
    )
    assert _normalized_highlights(report) == ["✓ warmup/time -10.0%"]
