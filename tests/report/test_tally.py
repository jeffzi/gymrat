"""Tests for counting verdicts and rendering the verdict summary parts.

These tests assert the *intent* of styling rather than exact escape bytes: they
render the markup string through :func:`gymrat.report.style.render_lines`
with color off to check the plain content, and with color on to check which
project style opens before each piece of a part.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from gymrat.report.style import VERDICT_STYLES, markup
from gymrat.report.tally import VerdictCounts, count_verdicts, verdict_summary_parts
from tests.report._assertions import (
    render_colored,
    render_plain,
    sgr_for,
    styles_at,
    word_styles,
)
from tests.report._comparisons import (
    every_class_metrics,
    permutation_candidate,
    permutation_metric,
    shared_baseline_metric,
)
from tests.report._verdicts import one_sided_metric

if TYPE_CHECKING:
    from gymrat.report.types import MetricComparisons


# ---------------------------------------------------------------------------
# count_verdicts
# ---------------------------------------------------------------------------


_TWO_CANDIDATES: MetricComparisons = {
    "decode/time": shared_baseline_metric([
        permutation_candidate(verdict="improved", delta=-10, median=90),
        permutation_candidate(verdict="regressed", delta=8, median=108),
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


_GLYPHS = ["✓", "✗", "≈", "=", "~", "?"]
_GLOSSES = ["improved", "regressed", "unstable", "identical", "within noise", "inconclusive"]


def test_verdict_summary_parts_when_mixed_does_color_only_each_count_by_its_class():
    parts = verdict_summary_parts(every_class_metrics(), 0)

    colored = [render_colored(part) for part in parts]
    assert [render_plain(part) for part in parts] == [
        "✓ 1 improved",
        "✗ 1 regressed",
        "≈ 1 unstable",
        "= 1 identical",
        "~ 1 within noise",
        "? 1 inconclusive",
    ]
    assert [
        next(params for word, params in word_styles(line) if word == "1") for line in colored
    ] == [
        sgr_for(VERDICT_STYLES["improved"]),
        sgr_for(VERDICT_STYLES["regressed"]),
        sgr_for(VERDICT_STYLES["unstable"]),
        sgr_for(VERDICT_STYLES["identical"]),
        [],
        [],
    ]
    assert [
        (styles_at(line, glyph), styles_at(line, gloss))
        for line, glyph, gloss in zip(colored, _GLYPHS, _GLOSSES, strict=True)
    ] == [([], [])] * len(_GLYPHS)


def test_verdict_summary_parts_when_zero_count_does_style_the_whole_part():
    only_improved: MetricComparisons = {
        "faster/time": permutation_metric(verdict="improved", delta=-10)
    }

    parts = verdict_summary_parts(only_improved, 0)

    assert [render_colored(part) for part in parts[1:]] == [
        render_colored(markup(render_plain(part), "dim")) for part in parts[1:]
    ]
    assert render_colored(parts[0]) != render_colored(markup(render_plain(parts[0]), "dim"))


def test_verdict_summary_parts_when_varying_counts_does_pad_to_widest_digit_width():
    metrics: MetricComparisons = {
        f"improved-{i}/time": permutation_metric(verdict="improved", delta=-(i + 1))
        for i in range(10)
    }
    metrics["regressed/time"] = permutation_metric(verdict="regressed", delta=5)
    metrics["noisy/time"] = permutation_metric(verdict="unstable", delta=3, noise_pct=300)

    parts = verdict_summary_parts(metrics, 0)

    assert [render_plain(part) for part in parts] == [
        "✓ 10 improved",
        "✗  1 regressed",
        "≈  1 unstable",
        "=  0 identical",
        "~  0 within noise",
        "?  0 inconclusive",
    ]
