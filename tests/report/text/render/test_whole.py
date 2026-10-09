"""Whole-report assembly and layout tests for the comparison report.

These cover whole-report assembly and layout. The structural table-region and
styles-at cases are asserted directly. Most scenarios are pinned as a
content/shape assertion: the table layout via :func:`table_region`, the
assembled tail via the summary line(s), the highlights block, and the
footer/worktree lines, plus ``styles_at`` on the colored markers. Highlight
entries are compared with their internal padding collapsed — that padding is
pinned exactly by ``test_verdicts`` — so these tests pin order and content
without re-pinning column widths a second time.

A handful of representative layouts are also pinned byte for byte as plain
golden outputs, so a change to how the table is drawn cannot shift a padding
space or a rule dash unnoticed. Color is pinned per element with ``styles_at``
rather than as escape bytes, so a change in how rich encodes a style does not
read as a regression. The run header, column labels, section and group titles
and aggregate rows are styled here; verdict-cell colors live in
``test_verdicts`` and multi-candidate cell colors in ``test_multi``.
"""

from __future__ import annotations

import math
import re
from dataclasses import replace
from functools import partial
from typing import TYPE_CHECKING

import pytest

from gymrat.config import KindEntry
from gymrat.model import Exclusion
from gymrat.report.text.render import render_measure_report, render_report
from gymrat.report.types import CandidateMetric, MetricComparison, ReportOptions
from gymrat.targets import WorktreeRemovalFailure
from gymrat.verdict import KindAggregate
from tests.report._assertions import (
    cells_of,
    highlight_lines,
    line_containing,
    line_starting_with,
    styles_at,
    table_region,
)
from tests.report._comparisons import (
    NWayCandidate,
    create_candidate,
    create_comparison_result,
    exact_metric,
    gating_kind,
    grouped_comparison,
    kind_metric,
    memory_kind,
    mixed_methods_result,
    n_way_kind_metric,
    other_kind,
    permutation_metric,
    single_sample_result,
    two_kind_result,
)
from tests.report._measurements import two_kind_measurement
from tests.report._verdicts import (
    band_verdict,
    exact_verdict,
    geomean_of,
    metric_meta,
    permutation_verdict,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from syrupy.assertion import SnapshotAssertion

    from gymrat.report.types import ComparisonResult

_HEADER = (
    "gymrat compare · baseline main ↔ perf/faster-decode · 10 paired samples · adapter: mitata"
)


def _normalized_highlights(report: str) -> list[str]:
    """The highlight block's lines with runs of whitespace collapsed to one space.

    ``test_verdicts`` pins the exact padding; here the concern is order and
    content, so the alignment padding is folded away.

    Args:
        report: The rendered report, lines joined by newlines.

    Returns:
        The highlight block's lines, each stripped with whitespace runs folded
        to one space.
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
            "entity/alive_check#time": kind_metric(
                kind="time", short_name="entity.alive_check", verdict="improved", delta=-10
            ),
            "entity/spawn#time": kind_metric(
                kind="time", short_name="entity.spawn", verdict="regressed", delta=4
            ),
        },
        candidates=[create_candidate(kinds=[gating_kind("time", geomean, {"entity": geomean})])],
    )


def _non_gating_result() -> ComparisonResult:
    """One ``time`` kind that gates nothing, so its geomean has no stable metrics."""
    return create_comparison_result(
        metrics={
            "warmup#time": kind_metric(
                kind="time", short_name="warmup", verdict="improved", delta=-10, gating=False
            ),
        },
        candidates=[
            create_candidate(
                kinds=[KindAggregate(kind="time", geomean=geomean_of(-10, 1), groups=())]
            )
        ],
    )


def _non_gating_two_candidate_result() -> ComparisonResult:
    """Two candidates over one ``time`` kind that gates nothing."""
    return create_comparison_result(
        metrics={
            "warmup#time": n_way_kind_metric(
                kind="time",
                short_name="warmup",
                candidates=[
                    NWayCandidate(verdict="improved", delta=-10, median=90),
                    NWayCandidate(verdict="regressed", delta=4, median=104),
                ],
                gating=False,
            ),
        },
        candidates=[
            create_candidate(
                label=label, kinds=[KindAggregate(kind="time", geomean=geomean, groups=())]
            )
            for label, geomean in (
                ("candidate-a", geomean_of(-10, 1)),
                ("candidate-b", geomean_of(4, 1)),
            )
        ],
    )


# ---------------------------------------------------------------------------
# flat non-gating kind
# ---------------------------------------------------------------------------


def _flat_non_gating_result() -> ComparisonResult:
    """A single non-gating ``time`` kind whose informational tag carries the config source."""
    return create_comparison_result(
        metrics={
            "warmup#time": kind_metric(
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
    report = render_report(_flat_non_gating_result())

    assert table_region(report) == [
        _HEADER,
        "informational — gating off (config: kinds.time.gating = false)",
        "metric",
        "<rule>",
        "warmup#time",
        "cooldown/time",
        "<rule>",
        "geomean",
    ]


@pytest.mark.parametrize(
    "render",
    [
        pytest.param(
            lambda: render_report(replace(two_kind_result(), config_kinds=None)),
            id="compare",
        ),
        pytest.param(
            lambda: render_measure_report(replace(two_kind_measurement(), config_kinds=None)),
            id="measure",
        ),
    ],
)
def test_render_when_kind_is_informational_by_metric_overrides_does_tag_it_without_a_config_source(
    render: Callable[[], str],
):
    report = render()

    assert line_containing(report, "informational") == "informational — gating off"


# ---------------------------------------------------------------------------
# geomean color scenarios (styled in the element styling table)
# ---------------------------------------------------------------------------


def _quiet_two_kind_result() -> ComparisonResult:
    """A two-kind run whose every metric landed within noise.

    Each geomean figure sits far outside its own band, so a rule that reads the
    band alone would paint all of them green.

    Returns:
        The comparison result.
    """
    time_geomean = geomean_of(-8.5, 2)
    return create_comparison_result(
        metrics={
            "entity/alive_check#time": kind_metric(
                kind="time", short_name="entity.alive_check", verdict="no-signal", delta=-9
            ),
            "entity/spawn#time": kind_metric(
                kind="time", short_name="entity.spawn", verdict="no-signal", delta=-8
            ),
            "encode#memory": kind_metric(
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
                    gating_kind("time", time_geomean, {"entity": geomean_of(-8.6, 2)}),
                    memory_kind(),
                ]
            )
        ],
        config_kinds={"memory": KindEntry(gating=False)},
    )


def _quiet_flat_result() -> ComparisonResult:
    """A flat run whose one metric landed within noise, under the default geomean."""
    return create_comparison_result(
        metrics={"faster/time": permutation_metric(verdict="no-signal", delta=-0.5)}
    )


def _per_candidate_geomean_result() -> ComparisonResult:
    """Two candidates over a grouped ``time`` kind: one within noise, one improved."""
    return create_comparison_result(
        metrics={
            "entity/alive_check#time": n_way_kind_metric(
                kind="time",
                short_name="entity.alive_check",
                candidates=[
                    NWayCandidate(verdict="no-signal", delta=-9, median=91),
                    NWayCandidate(verdict="improved", delta=-12, median=88),
                ],
            ),
            "encode#memory": n_way_kind_metric(
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
                    gating_kind("time", geomean_of(-9, 1), {"entity": geomean_of(-9, 1)}),
                    KindAggregate(kind="memory", geomean=geomean_of(-1, 1), groups=()),
                ],
            ),
            create_candidate(
                label="candidate-b",
                kinds=[
                    gating_kind("time", geomean_of(-12, 1), {"entity": geomean_of(-12, 1)}),
                    KindAggregate(kind="memory", geomean=geomean_of(-2, 1), groups=()),
                ],
            ),
        ],
        config_kinds={"memory": KindEntry(gating=False)},
    )


# ---------------------------------------------------------------------------
# whole-report assembly
# ---------------------------------------------------------------------------


def _grouped_exact_mix_result() -> ComparisonResult:
    return create_comparison_result(
        metrics={
            "decode/text=digits#time": MetricComparison(
                baseline_median=1735,
                baseline_spread=1,
                candidates=(
                    CandidateMetric(
                        median=1425,
                        spread=1,
                        verdict=permutation_verdict(verdict="improved", delta=-17.9, p=0.002),
                    ),
                ),
                meta=metric_meta("decode/text=digits#time", unit="ns"),
            ),
            "decode/text=words#time": MetricComparison(
                baseline_median=3065,
                baseline_spread=1,
                candidates=(
                    CandidateMetric(
                        median=3093,
                        spread=3,
                        verdict=permutation_verdict(verdict="no-signal", delta=0.9, p=0.49),
                    ),
                ),
                meta=metric_meta("decode/text=words#time", unit="ns"),
            ),
            "encode#time": MetricComparison(
                baseline_median=914,
                baseline_spread=1,
                candidates=(
                    CandidateMetric(
                        median=934,
                        spread=1,
                        verdict=permutation_verdict(verdict="regressed", delta=2.2, p=0.002),
                    ),
                ),
                meta=metric_meta("encode#time", unit="ns"),
            ),
            "encode#heap": MetricComparison(
                baseline_median=49152,
                baseline_spread=0,
                candidates=(
                    CandidateMetric(
                        median=45261,
                        spread=0,
                        verdict=exact_verdict(verdict="improved", delta=-7.9),
                    ),
                ),
                meta=metric_meta("encode#heap", exact=True, unit="bytes"),
            ),
        },
        candidates=[create_candidate(kinds=[other_kind(-6, 4)])],
    )


def test_render_report_when_grouped_run_mixes_methods_does_assemble_the_whole_report():
    report = render_report(_grouped_exact_mix_result())

    assert table_region(report) == [
        _HEADER,
        "metric",
        "<rule>",
        "decode · other",
        "text=digits#time",
        "text=words#time",
        "",
        "encode#time",
        "encode#heap",
        "<rule>",
        "geomean (4 stable metrics)",
    ]
    assert line_starting_with(report, "✓ 2 improved") == (
        "✓ 2 improved   ✗ 1 regressed   ≈ 0 unstable   "
        "= 0 identical   ~ 1 within noise   ? 0 inconclusive"
    )
    assert _normalized_highlights(report) == [
        "✗ encode#time +2.2%",
        "✓ decode/text=digits#time -17.9%",
        "✓ encode#heap -7.9% (exact)",
    ]


def _degenerate_result() -> ComparisonResult:
    return create_comparison_result(
        samples=4,
        adapter="metric-lines",
        metrics={
            "zero-median/time": exact_metric(
                delta=0, n=4, unit="ns", baseline_median=0, short_name="zero-median/time"
            ),
            "nan-delta/count": exact_metric(
                delta=math.nan,
                n=4,
                unit=None,
                baseline_median=0,
                median=120,
                short_name="nan-delta/count",
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
                        excluded=[Exclusion(metric="encode#time", reason="unstable")],
                    )
                ],
            ),
        ],
        metrics={
            "decode/text=digits#time": MetricComparison(
                baseline_median=1735,
                baseline_spread=1,
                candidates=(
                    CandidateMetric(
                        median=1425,
                        spread=1,
                        verdict=permutation_verdict(verdict="improved", delta=-17.9, p=0.002),
                    ),
                    CandidateMetric(
                        median=1698,
                        spread=2,
                        verdict=permutation_verdict(verdict="no-signal", delta=-2.1, p=0.32),
                    ),
                ),
                meta=metric_meta("decode/text=digits#time", unit="ns"),
            ),
            "encode#time": MetricComparison(
                baseline_median=914,
                baseline_spread=1,
                candidates=(
                    CandidateMetric(
                        median=934,
                        spread=1,
                        verdict=permutation_verdict(verdict="regressed", delta=2.2, p=0.002),
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
                meta=metric_meta("encode#time", unit="ns"),
            ),
            "encode#heap": MetricComparison(
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
                meta=metric_meta("encode#heap", exact=True, unit="bytes"),
            ),
        },
    )


def test_render_report_when_single_sample_does_mark_verdicts_inconclusive():
    report = render_report(single_sample_result())

    assert cells_of(line_starting_with(report, "decode/time"))[-1].strip() == "?  -0.4%"
    assert report.split("\n\n")[1:] == [
        (
            "✓ 0 improved   ✗ 0 regressed   ≈ 0 unstable   "
            "= 0 identical   ~ 0 within noise   ? 2 inconclusive"
        ),
        "re-run with gymrat compare --samples 6 or more for statistical verdicts",
    ]


# ---------------------------------------------------------------------------
# byte-for-byte golden outputs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "make_result",
    [
        pytest.param(_one_kind_result, id="flat-grouped-plain"),
        pytest.param(_degenerate_result, id="flat-degenerate-plain"),
        pytest.param(_two_candidate_result, id="flat-two-candidates-plain"),
        pytest.param(two_kind_result, id="sectioned-single-candidate-plain"),
        pytest.param(grouped_comparison, id="sectioned-multi-candidate-plain"),
        pytest.param(_non_gating_result, id="flat-no-stable-metrics-plain"),
        pytest.param(
            _non_gating_two_candidate_result, id="flat-two-candidates-no-stable-metrics-plain"
        ),
        pytest.param(partial(mixed_methods_result, n=10), id="flat-mixed-methods-plain"),
    ],
)
def test_render_report_when_rendered_does_match_its_golden(
    make_result: Callable[[], ComparisonResult], snapshot: SnapshotAssertion
):
    report = render_report(make_result(), ReportOptions(color=False))

    assert report.split("\n") == snapshot


# ---------------------------------------------------------------------------
# element styling
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("make_result", "needle", "marker", "expected"),
    [
        pytest.param(_one_kind_result, "gymrat compare", "gymrat compare", ["1"], id="title-bold"),
        pytest.param(_one_kind_result, "gymrat compare", "·", ["2"], id="header-separator-dim"),
        pytest.param(_one_kind_result, "metric ", "main", ["1", "4"], id="column-label-underlined"),
        pytest.param(
            _one_kind_result, "entity · time", "entity · time", ["1", "34"], id="group-header-blue"
        ),
        pytest.param(_one_kind_result, "geomean", "geomean", ["1"], id="aggregate-label-bold"),
        pytest.param(two_kind_result, "geomean · time", "±2.0%", ["2"], id="aggregate-band-dim"),
        pytest.param(_one_kind_result, "highlights", "highlights", ["1"], id="highlights-bold"),
        pytest.param(grouped_comparison, "time", "time", ["1"], id="section-title-bold"),
        pytest.param(_quiet_flat_result, "geomean", "-5.8%", ["1"], id="quiet-flat-geomean"),
        pytest.param(
            _quiet_two_kind_result, "geomean · entity", "-8.6%", ["1"], id="quiet-group-geomean"
        ),
        pytest.param(
            _quiet_two_kind_result, "geomean · time", "-8.5%", ["1"], id="quiet-kind-geomean"
        ),
        pytest.param(
            two_kind_result, "geomean · entity", "-3.1%", ["1", "32"], id="improving-group-geomean"
        ),
        pytest.param(
            two_kind_result, "geomean · time", "-3.2%", ["1", "32"], id="improving-kind-geomean"
        ),
        pytest.param(
            grouped_comparison, "geomean · entity", "+4.0%", ["1", "31"], id="regressing-geomean"
        ),
        pytest.param(
            _per_candidate_geomean_result,
            "geomean · time",
            "-9.0%",
            ["1"],
            id="quiet-candidate-geomean",
        ),
        pytest.param(
            _per_candidate_geomean_result,
            "geomean · time",
            "-12.0%",
            ["1", "32"],
            id="improving-candidate-geomean",
        ),
        pytest.param(
            grouped_comparison, "geomean · entity", "1 stable metric", ["2"], id="stable-count-dim"
        ),
        pytest.param(
            grouped_comparison, "informational", "informational", ["2"], id="informational-dim"
        ),
    ],
)
def test_render_report_when_colored_does_style_each_element(
    make_result: Callable[[], ComparisonResult], needle: str, marker: str, expected: list[str]
):
    line = line_containing(render_report(make_result(), ReportOptions(color=True)), needle)

    assert styles_at(line, marker) == expected


@pytest.mark.parametrize(
    ("result", "marker", "expected"),
    [
        pytest.param(
            create_comparison_result(baseline_label="main·1"),  # cspell:disable-line
            "main·1",  # cspell:disable-line
            ["1", "4"],
            id="dotted-baseline-label",
        ),
        pytest.param(
            create_comparison_result(
                candidates=[create_candidate(label="perf·2")],  # cspell:disable-line
            ),
            "perf·2",  # cspell:disable-line
            ["1", "4"],
            id="dotted-candidate-label",
        ),
        pytest.param(
            create_comparison_result(adapter="metric·lines"),  # cspell:disable-line
            "metric·lines",  # cspell:disable-line
            [],
            id="dotted-adapter",
        ),
    ],
)
def test_render_report_when_header_part_holds_a_dot_does_leave_it_out_of_dimming(
    result: ComparisonResult, marker: str, expected: list[str]
):
    header = line_containing(render_report(result, ReportOptions(color=True)), "gymrat compare")

    assert styles_at(header, marker) == expected
