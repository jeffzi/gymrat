"""Whole-report assembly and layout tests for the comparison report.

Representative layouts are pinned byte for byte as plain golden outputs, so a
change to how the table is drawn cannot shift a padding space or a rule dash
unnoticed. Footer and worktree lines are pinned in ``test_verdicts``.

Color is pinned per element with ``styles_at`` rather than as escape bytes, so
a change in how rich encodes a style does not read as a regression. The run
header, column labels, section and group titles and aggregate rows are styled
here; verdict-cell colors live in ``test_verdicts`` and multi-candidate cell
colors in ``test_multi``.
"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import pytest

from gymrat.config import KindEntry
from gymrat.model import Exclusion
from gymrat.report.style import EMPHASIZED_DIRECTION_STYLES
from gymrat.report.text.render import render_report
from gymrat.report.types import CandidateMetric, MetricComparison, ReportOptions
from gymrat.sampling import CleanupResult, WorktreeRemovalFailure
from tests.report._assertions import (
    line_containing,
    sgr_for,
    styles_at,
)
from tests.report._comparisons import (
    NWayCandidate,
    create_candidate,
    create_comparison_result,
    entity_time_metrics,
    exact_metric,
    gating_kind,
    grouped_comparison,
    informational_kind,
    kind_metric,
    mixed_methods_result,
    n_way_kind_metric,
    other_kind,
    two_kind_result,
    undefined_ratio_metric,
)
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

# ---------------------------------------------------------------------------
# flat single-kind layout
# ---------------------------------------------------------------------------


def _one_kind_result() -> ComparisonResult:
    """A single gating ``time`` kind whose two metrics share the ``entity`` group."""
    geomean = geomean_of(-3.2, 2)
    return create_comparison_result(
        metrics=entity_time_metrics(),
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
        candidates=[create_candidate(kinds=[informational_kind("time", geomean_of(-10, 1))])],
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
            create_candidate(label=label, kinds=[informational_kind("time", geomean)])
            for label, geomean in (
                ("candidate-a", geomean_of(-10, 1)),
                ("candidate-b", geomean_of(4, 1)),
            )
        ],
    )


# ---------------------------------------------------------------------------
# geomean color scenarios (styled in the element styling table)
# ---------------------------------------------------------------------------


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
                    informational_kind("memory", geomean_of(-1, 1)),
                ],
            ),
            create_candidate(
                label="candidate-b",
                kinds=[
                    gating_kind("time", geomean_of(-12, 1), {"entity": geomean_of(-12, 1)}),
                    informational_kind("memory", geomean_of(-2, 1)),
                ],
            ),
        ],
        config_kinds={"memory": KindEntry(gating=False)},
    )


# ---------------------------------------------------------------------------
# byte-for-byte golden outputs
# ---------------------------------------------------------------------------


def _degenerate_result() -> ComparisonResult:
    return create_comparison_result(
        samples=4,
        adapter="metric-lines",
        metrics={
            "zero-median/time": exact_metric(
                delta=0, n=4, unit="ns", baseline_median=0, short_name="zero-median/time"
            ),
            "nan-delta/count": undefined_ratio_metric("nan-delta/count", n=4),
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
        cleanup=CleanupResult(
            removed=1,
            failures=(
                WorktreeRemovalFailure(dir="/tmp/gymrat-abc123", error="contains modified files"),
            ),
            prune_error="could not lock config file",
        ),
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
            "encode#memory": MetricComparison(
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
                meta=metric_meta("encode#memory", exact=True, unit="bytes"),
            ),
        },
    )


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
        pytest.param(
            two_kind_result,
            "geomean · entity",
            "-3.1%",
            sgr_for(EMPHASIZED_DIRECTION_STYLES["improved"]),
            id="improving-group-geomean",
        ),
        pytest.param(
            _per_candidate_geomean_result,
            "geomean · time",
            "-12.0%",
            sgr_for(EMPHASIZED_DIRECTION_STYLES["improved"]),
            id="improving-candidate-geomean",
        ),
    ],
)
def test_render_report_when_colored_does_style_each_element(
    make_result: Callable[[], ComparisonResult], needle: str, marker: str, expected: list[str]
):
    line = line_containing(render_report(make_result(), ReportOptions(color=True)), needle)

    assert styles_at(line, marker) == expected


def test_render_report_when_header_part_holds_a_dot_does_leave_it_out_of_dimming():
    result = create_comparison_result(adapter="metric·lines")  # cspell:disable-line

    header = line_containing(render_report(result, ReportOptions(color=True)), "gymrat compare")

    assert styles_at(header, "metric·lines") == []  # cspell:disable-line
