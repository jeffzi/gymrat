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
from tests._ansi import (
    sgr_codes,
)
from tests.report._assertions import (
    render_colored,
    render_plain,
)
from tests.report._comparisons import (
    NWayCandidate,
    every_class_metrics,
    n_way_metric,
    permutation_metric,
)
from tests.report._verdicts import one_sided_metric

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


_TWO_CANDIDATES: MetricComparisons = {
    "decode/time": n_way_metric([
        NWayCandidate(verdict="improved", delta=-10, median=90),
        NWayCandidate(verdict="regressed", delta=8, median=108),
    ]),
}


@pytest.mark.parametrize(
    ("metrics", "candidate_index", "expected"),
    [
        pytest.param(
            {
                "faster/time": permutation_metric(verdict="improved", delta=-10),
                "also-faster/time": permutation_metric(verdict="improved", delta=-5),
                "slower/time": permutation_metric(verdict="regressed", delta=8),
                "jittery/time": permutation_metric(verdict="unstable", delta=5, noise_pct=300),
                "flat/time": permutation_metric(verdict="no-signal", delta=0.2),
                "one-sided/time": one_sided_metric(),
            },
            0,
            VerdictCounts(improved=2, regressed=1, unstable=1, no_signal=1),
            id="mixed-skips-no-verdict",
        ),
        pytest.param(
            {},
            0,
            VerdictCounts(improved=0, regressed=0, unstable=0, no_signal=0),
            id="no-metrics",
        ),
        pytest.param(
            _TWO_CANDIDATES,
            0,
            VerdictCounts(improved=1, regressed=0, unstable=0, no_signal=0),
            id="first-candidate-counts-its-improvement",
        ),
        pytest.param(
            _TWO_CANDIDATES,
            1,
            VerdictCounts(improved=0, regressed=1, unstable=0, no_signal=0),
            id="second-candidate-counts-its-regression",
        ),
    ],
)
def test_count_verdicts_when_given_metrics_does_count_each_class_for_the_named_candidate(
    metrics: MetricComparisons, candidate_index: int, expected: VerdictCounts
):
    counts = count_verdicts(metrics, candidate_index)

    assert counts == expected


# ---------------------------------------------------------------------------
# verdict_summary_parts
# ---------------------------------------------------------------------------


def test_verdict_summary_parts_when_mixed_does_render_every_class_with_its_count_color():
    parts = verdict_summary_parts(every_class_metrics(), 0)

    assert [(render_plain(part), sorted(sgr_codes(render_colored(part)))) for part in parts] == [
        ("✓ 1 improved", ["32"]),
        ("✗ 1 regressed", ["31"]),
        ("≈ 1 unstable", ["33"]),
        ("= 1 identical", ["36"]),
        ("~ 1 within noise", []),
        ("? 1 inconclusive", []),
    ]


@pytest.mark.parametrize("label", ["regressed", "identical"])
def test_verdict_summary_parts_when_zero_count_does_dim_the_part(label: str):
    only_improved: MetricComparisons = {
        "faster/time": permutation_metric(verdict="improved", delta=-10)
    }

    parts = verdict_summary_parts(only_improved, 0)

    assert "2" in sgr_codes(render_colored(_find_plain(parts, label)))


def test_verdict_summary_parts_when_varying_counts_does_pad_to_widest_digit_width():
    metrics: MetricComparisons = {
        f"improved-{i}/time": permutation_metric(verdict="improved", delta=-(i + 1))
        for i in range(10)
    }
    metrics["regressed/time"] = permutation_metric(verdict="regressed", delta=5)
    metrics["noisy/time"] = permutation_metric(verdict="unstable", delta=3, noise_pct=300)

    parts = verdict_summary_parts(metrics, 0)

    assert render_plain(_find_plain(parts, "improved")) == "✓ 10 improved"
    assert render_plain(_find_plain(parts, "regressed")) == "✗  1 regressed"
    assert render_plain(_find_plain(parts, "unstable")) == "≈  1 unstable"
    assert render_plain(_find_plain(parts, "within noise")) == "~  0 within noise"
