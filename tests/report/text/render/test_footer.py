"""Tests for the footer lines that explain noise bands and suggest more samples.

These tests assert the *intent* of styling rather than exact escape bytes: they
render the markup string through :func:`gymrat.report.style.render_lines`
with color off to check the plain content, and with color on to check that the
expected SGR attribute code is present.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from gymrat.model import PERMUTATION_MIN_N
from gymrat.report.style import format_hint
from gymrat.report.text.render import footer_lines
from tests.report._assertions import render_colored, render_plain, sgr_codes
from tests.report._verdicts import approximate_metric, band_metric

if TYPE_CHECKING:
    from gymrat.report.types import MetricComparisons


# ---------------------------------------------------------------------------
# footer_lines
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


def test_footer_lines_when_plain_does_carry_no_ansi():
    metrics: MetricComparisons = {"a/time": approximate_metric(verdict="improved", delta=-10)}

    lines = _verbose_lines(metrics)

    assert len(lines) > 0
    assert all("\x1b[" not in render_plain(line) for line in lines)


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
                "iteration.soa_5field/heap": band_metric(n=8, usable_n=2),  # cspell:disable-line
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
def test_footer_lines_when_cause_varies_does_phrase_band_line_accordingly(
    metrics: MetricComparisons, expected: list[str]
):
    assert _band_lines_for(metrics) == expected


def test_footer_lines_when_hint_present_does_format_it():
    metrics: MetricComparisons = {"a/time": band_metric(n=4)}

    assert footer_lines(metrics, verbose=False, command="compare", samples=4) == [
        format_hint(SAMPLE_SHORTAGE_HINT)
    ]


@pytest.mark.parametrize(
    ("metrics", "expected"),
    [
        pytest.param(
            {
                "decode/time": band_metric(n=3),
                "encode/time": band_metric(n=5),
                "parse/time": approximate_metric(verdict="improved", delta=-10),
            },
            [SAMPLE_SHORTAGE_HINT_PLAIN],
            id="every-band-metric-short",
        ),
        pytest.param(
            {
                "entity.alive_check/heap": band_metric(n=10, usable_n=3),
                "iteration.soa_5field/heap": band_metric(n=8, usable_n=2),  # cspell:disable-line
                "parse/time": approximate_metric(verdict="improved", delta=-10),
            },
            [],
            id="ties-alone",
        ),
        pytest.param(
            {
                "decode/time": band_metric(n=3),
                "entity.alive_check/heap": band_metric(n=10, usable_n=3),
                "parse/time": approximate_metric(verdict="improved", delta=-10),
            },
            [SAMPLE_SHORTAGE_HINT_PLAIN],
            id="shortage-and-ties-different-metrics",
        ),
        pytest.param(
            {"parse/time": approximate_metric(verdict="improved", delta=-10)},
            [],
            id="permutation-carried-every-metric",
        ),
    ],
)
def test_footer_lines_when_cause_varies_does_hint_accordingly(
    metrics: MetricComparisons, expected: list[str]
):
    lines = footer_lines(metrics, verbose=False, command="compare", samples=4)

    assert [render_plain(line) for line in lines] == expected


@pytest.mark.parametrize(
    ("metrics", "samples"),
    [
        pytest.param(
            {"a/time": band_metric(n=PERMUTATION_MIN_N - 1)},
            PERMUTATION_MIN_N - 1,
            id="fewer-samples-than-floor",
        ),
        pytest.param({"a/time": band_metric(n=1)}, 1, id="single-sample"),
    ],
)
def test_footer_lines_when_samples_below_floor_does_suggest_more_samples(
    metrics: MetricComparisons, samples: int
):
    lines = footer_lines(metrics, verbose=False, command="compare", samples=samples)

    assert any("gymrat compare --samples" in line for line in lines)


@pytest.mark.parametrize(
    ("metrics", "samples"),
    [
        pytest.param(
            {
                "a/time": band_metric(n=3),
                "b/time": approximate_metric(verdict="improved", delta=-10),
            },
            10,
            id="enough-samples-but-rounds-dropped",
        ),
        pytest.param(
            {"a/time": band_metric(n=PERMUTATION_MIN_N - 1)},
            PERMUTATION_MIN_N,
            id="floor-reached-but-fewer-paired",
        ),
    ],
)
def test_footer_lines_when_samples_enough_does_name_dropped_rounds(
    metrics: MetricComparisons, samples: int
):
    lines = footer_lines(metrics, verbose=False, command="compare", samples=samples)

    assert not any("gymrat compare --samples" in line for line in lines)
    assert any("dropped" in line for line in lines)


def test_footer_lines_when_samples_enough_and_every_metric_tested_does_not_hint():
    metrics: MetricComparisons = {"a/time": approximate_metric(verdict="improved", delta=-10)}

    assert footer_lines(metrics, verbose=False, command="compare", samples=10) == []
