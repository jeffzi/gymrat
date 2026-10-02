"""Tests for selecting and labelling the metrics a report highlights.

These tests assert the *intent* of styling rather than exact escape bytes: they
render the markup string through :func:`gymrat.report.style.render_lines`
with color off to check the plain content, and with color on to check that the
expected SGR attribute code is present.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from gymrat.report.text.render import highlight_label, select_highlights
from tests.report._assertions import render_colored, styles_at
from tests.report._comparisons import kind_metric
from tests.report._verdicts import (
    CandidateSpec,
    approximate_metric,
    band_metric,
    metric_for,
    one_sided_metric,
)

if TYPE_CHECKING:
    from gymrat.report.types import MetricComparisons

# ---------------------------------------------------------------------------
# select_highlights
# ---------------------------------------------------------------------------


def test_select_highlights_when_mixed_verdicts_does_order_regressions_then_improvements_then_unstable():
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


def test_select_highlights_when_identical_does_leave_out():
    metrics: MetricComparisons = {
        "faster/time": approximate_metric(verdict="improved", delta=-10),
        "tied/heap": band_metric(n=10, usable_n=3),
    }

    highlights = select_highlights(metrics, 0)

    assert [highlight.name for highlight in highlights] == ["faster/time"]


def test_select_highlights_when_single_pair_does_leave_out():
    metrics: MetricComparisons = {
        "faster/time": approximate_metric(verdict="improved", delta=-10),
        "single-pair/time": band_metric(n=1, noise_pct=0.5),
    }

    highlights = select_highlights(metrics, 0)

    assert [highlight.name for highlight in highlights] == ["faster/time"]


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


def test_select_highlights_when_selected_does_carry_metric_and_candidate_slice():
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
    assert highlight.candidate is metrics["slower/time"].candidates[1]


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


def test_select_highlights_when_sub_minimum_band_does_exclude():
    metrics: MetricComparisons = {
        "short-improved/time": band_metric(verdict="improved", delta=-10, n=4),
        "adequate/time": approximate_metric(verdict="improved", delta=-5),
    }

    highlights = select_highlights(metrics, 0)

    assert [highlight.name for highlight in highlights] == ["adequate/time"]


# ---------------------------------------------------------------------------
# highlight_label — format_inline
# ---------------------------------------------------------------------------


def test_highlight_label_when_unqualified_does_dim_group_and_kind_in_colored_output():
    metrics: MetricComparisons = {
        "entity/alive_check#time": kind_metric(
            kind="time", short_name="entity.alive_check", verdict="improved", delta=-10
        ),
    }
    (highlight,) = select_highlights(metrics, 0)

    label = highlight_label(highlight, qualify=False)

    colored = render_colored(label)
    assert "2" in styles_at(colored, "entity/")
    assert "2" in styles_at(colored, "#time")
