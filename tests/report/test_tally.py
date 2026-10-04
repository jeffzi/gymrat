"""Tests for counting verdicts and rendering the verdict summary parts.

These tests assert the *intent* of styling rather than exact escape bytes: they
render the markup string through :func:`gymrat.report.style.render_lines`
with color off to check the plain content, and with color on to check that the
expected SGR attribute code is present.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from gymrat.report.tally import VerdictCounts, count_verdicts, verdict_summary_parts
from tests.report._assertions import render_colored, render_plain, sgr_codes
from tests.report._verdicts import (
    CandidateSpec,
    approximate_metric,
    band_metric,
    metric_for,
    one_sided_metric,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from gymrat.report.types import MetricComparisons


def _find_plain(parts: Sequence[str], needle: str) -> str:
    matches = [part for part in parts if needle in render_plain(part)]
    assert len(matches) == 1, f"expected exactly one part containing {needle!r}, got {matches}"
    return matches[0]


# ---------------------------------------------------------------------------
# count_verdicts
# ---------------------------------------------------------------------------


def test_count_verdicts_when_mixed_does_count_each_class_and_skip_no_verdict():
    metrics: MetricComparisons = {
        "faster/time": approximate_metric(verdict="improved", delta=-10),
        "also-faster/time": approximate_metric(verdict="improved", delta=-5),
        "slower/time": approximate_metric(verdict="regressed", delta=8),
        "jittery/time": approximate_metric(verdict="unstable", delta=5, noise_pct=300),
        "flat/time": approximate_metric(verdict="no-signal", delta=0.2),
        "one-sided/time": one_sided_metric(),
    }

    counts = count_verdicts(metrics, 0)

    assert counts == VerdictCounts(improved=2, regressed=1, unstable=1, no_signal=1)


def test_count_verdicts_when_no_metrics_does_report_zeros():
    counts = count_verdicts({}, 0)

    assert counts == VerdictCounts(improved=0, regressed=0, unstable=0, no_signal=0)


@pytest.mark.parametrize(
    ("candidate_index", "expected"),
    [
        pytest.param(0, VerdictCounts(improved=1, regressed=0, unstable=0, no_signal=0), id="c0"),
        pytest.param(1, VerdictCounts(improved=0, regressed=1, unstable=0, no_signal=0), id="c1"),
    ],
)
def test_count_verdicts_when_candidate_named_does_count_only_that_candidate(
    candidate_index: int, expected: VerdictCounts
):
    metrics: MetricComparisons = {
        "decode/time": metric_for([
            CandidateSpec(verdict="improved", delta=-10),
            CandidateSpec(verdict="regressed", delta=8),
        ]),
    }

    assert count_verdicts(metrics, candidate_index) == expected


# ---------------------------------------------------------------------------
# verdict_summary_parts
# ---------------------------------------------------------------------------

_MIXED: MetricComparisons = {
    "faster/time": approximate_metric(verdict="improved", delta=-10),
    "slower/time": approximate_metric(verdict="regressed", delta=8),
    "jittery/time": approximate_metric(verdict="unstable", delta=5, noise_pct=300),
    "flat/time": approximate_metric(verdict="no-signal", delta=0.2),
    "tied/heap": band_metric(n=10, usable_n=0),
    "single-pair/time": band_metric(n=1, noise_pct=0.5),
}


def test_verdict_summary_parts_when_plain_does_carry_no_ansi():
    parts = verdict_summary_parts(_MIXED, 0)

    assert "\x1b[" not in "".join(render_plain(part) for part in parts)


def test_verdict_summary_parts_when_mixed_does_tally_identical_and_single_pair_apart_from_noise():
    parts = verdict_summary_parts(_MIXED, 0)

    assert render_plain(_find_plain(parts, "identical")) == "= 1 identical"
    assert render_plain(_find_plain(parts, "inconclusive")) == "? 1 inconclusive"
    assert render_plain(_find_plain(parts, "within noise")) == "~ 1 within noise"


@pytest.mark.parametrize(
    ("label", "code"),
    [
        pytest.param("improved", "32", id="improved-green"),
        pytest.param("regressed", "31", id="regressed-red"),
        pytest.param("unstable", "33", id="unstable-yellow"),
        pytest.param("identical", "36", id="identical-cyan"),
    ],
)
def test_verdict_summary_parts_when_nonzero_does_color_the_part(label: str, code: str):
    parts = verdict_summary_parts(_MIXED, 0)
    part = _find_plain(parts, label)

    assert code in sgr_codes(render_colored(part))


@pytest.mark.parametrize("label", ["regressed", "identical"])
def test_verdict_summary_parts_when_zero_count_does_dim_the_part(label: str):
    only_improved: MetricComparisons = {
        "faster/time": approximate_metric(verdict="improved", delta=-10)
    }

    parts = verdict_summary_parts(only_improved, 0)
    part = _find_plain(parts, label)

    assert "2" in sgr_codes(render_colored(part))


def test_verdict_summary_parts_when_within_noise_nonzero_does_not_dim():
    parts = verdict_summary_parts(_MIXED, 0)
    part = _find_plain(parts, "within noise")

    assert "2" not in sgr_codes(render_colored(part))


def test_verdict_summary_parts_when_varying_counts_does_pad_to_widest_digit_width():
    metrics: MetricComparisons = {
        f"improved-{i}/time": approximate_metric(verdict="improved", delta=-(i + 1))
        for i in range(10)
    }
    metrics["regressed/time"] = approximate_metric(verdict="regressed", delta=5)
    metrics["noisy/time"] = approximate_metric(verdict="unstable", delta=3, noise_pct=300)

    parts = verdict_summary_parts(metrics, 0)

    assert render_plain(_find_plain(parts, "improved")) == "✓ 10 improved"
    assert render_plain(_find_plain(parts, "regressed")) == "✗  1 regressed"
    assert render_plain(_find_plain(parts, "unstable")) == "≈  1 unstable"
    assert render_plain(_find_plain(parts, "within noise")) == "~  0 within noise"


def test_verdict_summary_parts_when_sub_minimum_band_does_tally_as_inconclusive():
    metrics: MetricComparisons = {
        "short-improved/time": band_metric(verdict="improved", delta=-10, n=4),
        "adequate/time": approximate_metric(verdict="improved", delta=-5),
    }

    parts = verdict_summary_parts(metrics, 0)

    assert render_plain(_find_plain(parts, "improved")) == "✓ 1 improved"
    assert render_plain(_find_plain(parts, "inconclusive")) == "? 1 inconclusive"
