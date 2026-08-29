"""Tests for the report formatting and classification primitives.

These tests assert the *intent* of styling rather than exact escape bytes: they
render the markup string through :func:`gymrat.report.style.render_lines`
with color off to check the plain content, and with color on to check that the
expected SGR attribute code is present. ``format_delta`` takes an
:class:`~gymrat.model.Effect` rather than a bare number.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import pytest

from gymrat.model import (
    PERMUTATION_DESCRIPTOR,
    Effect,
    Exclusion,
    GeomeanResult,
    MetricUnit,
    MetricVerdict,
)
from gymrat.report.format import (
    DisplayClass,
    VerdictCounts,
    count_verdicts,
    display_class,
    footer_lines,
    format_delta,
    format_evidence,
    format_value,
    geomean_value_style,
    get_glyph,
    pluralize,
    scoped_geomean_label,
    select_highlights,
    verdict_summary_parts,
)
from gymrat.report.style import format_hint, render_lines
from gymrat.report.types import ReportOptions
from tests.report._inputs import (
    CandidateSpec,
    Metrics,
    approximate_metric,
    band_metric,
    band_verdict,
    exact_verdict,
    geomean_of,
    metric_for,
    one_sided_metric,
    permutation_verdict,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

MIN_PERMUTATION_N = PERMUTATION_DESCRIPTOR.min_n


def _percent(value: float) -> Effect:
    return Effect(value=value, unit="percent")


def _plain(part: str) -> str:
    """The color-free rendering of a markup ``part``."""
    return render_lines(part, color=False, width=200)


def _colored(part: str) -> str:
    """The colored rendering of a markup ``part``, ANSI escapes included."""
    return render_lines(part, color=True, width=200)


def _sgr_codes(text: str) -> list[str]:
    """Every SGR attribute code embedded in ``text`` (e.g. ``["2", "36"]``)."""
    return [code for run in re.findall(r"\x1b\[([0-9;]*)m", text) for code in run.split(";")]


def _find_plain(parts: Sequence[str], needle: str) -> str:
    """The one part whose plain rendering contains ``needle``."""
    matches = [part for part in parts if needle in _plain(part)]
    assert len(matches) == 1, f"expected exactly one part containing {needle!r}, got {matches}"
    return matches[0]


# ---------------------------------------------------------------------------
# format_value
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "unit", "expected"),
    [
        pytest.param(0, "ns", "0ns", id="ns-zero"),
        pytest.param(914, "ns", "914ns", id="ns-mid"),
        pytest.param(999, "ns", "999ns", id="ns-top"),
        pytest.param(1000, "ns", "1.0µs", id="us-floor"),
        pytest.param(1735, "ns", "1.7µs", id="us-mid"),
        pytest.param(26825, "ns", "26.8µs", id="us-high"),
        pytest.param(1_000_000, "ns", "1.0ms", id="ms-floor"),
        pytest.param(2_000_000_000, "ns", "2.0s", id="s-floor"),
        pytest.param(999.5, "ns", "1.0µs", id="ns-rounds-onto-us"),
        pytest.param(999_999.6, "ns", "1.0ms", id="us-rounds-onto-ms"),
        pytest.param(999_950_000, "ns", "1.0s", id="ms-rounds-onto-s"),
        pytest.param(512, "bytes", "512B", id="b-mid"),
        pytest.param(999, "bytes", "999B", id="b-top"),
        pytest.param(1000, "bytes", "1.0KB", id="kb-floor"),
        pytest.param(3600, "bytes", "3.6KB", id="kb-mid"),
        pytest.param(49152, "bytes", "49.2KB", id="kb-high"),
        pytest.param(1_000_000, "bytes", "1.0MB", id="mb-floor"),
        pytest.param(2_000_000_000, "bytes", "2.0GB", id="gb-floor"),
        pytest.param(999.5, "bytes", "1.0KB", id="b-rounds-onto-kb"),
        pytest.param(999_950, "bytes", "1.0MB", id="kb-rounds-onto-mb"),
        pytest.param(-512, "bytes", "-512B", id="neg-b"),
        pytest.param(-3600, "bytes", "-3.6KB", id="neg-kb"),
        pytest.param(-1_500_000, "bytes", "-1.5MB", id="neg-mb"),
        pytest.param(-1735, "ns", "-1.7µs", id="neg-us"),
        pytest.param(-2_000_000_000, "ns", "-2.0s", id="neg-s"),
        pytest.param(-999.5, "bytes", "-1.0KB", id="neg-rounds-onto-kb"),
    ],
)
def test_format_value_when_given_unit_does_scale_to_tier(
    value: float, unit: MetricUnit, expected: str
):
    assert format_value(value, unit) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        pytest.param(0, "0", id="zero"),
        pytest.param(1200, "1200", id="thousands"),
        pytest.param(1_100_000, "1100000", id="millions"),
    ],
)
def test_format_value_when_no_unit_does_round_to_int(value: float, expected: str):
    assert format_value(value) == expected


@pytest.mark.parametrize(
    ("value", "unit", "expected"),
    [
        pytest.param(float("inf"), "ns", "Infinity", id="positive-infinity"),
        pytest.param(float("-inf"), "bytes", "-Infinity", id="negative-infinity"),
        pytest.param(float("nan"), "ns", "NaN", id="not-a-number"),
    ],
)
def test_format_value_when_non_finite_does_use_sentinel_token(
    value: float, unit: MetricUnit, expected: str
):
    assert format_value(value, unit) == expected


# ---------------------------------------------------------------------------
# format_evidence
# ---------------------------------------------------------------------------


def test_format_evidence_when_counted_does_mark_exact():
    assert format_evidence(exact_verdict(verdict="improved", delta=-7.9)) == "(exact)"


def test_format_evidence_when_statistical_improvement_does_add_nothing():
    assert format_evidence(permutation_verdict(verdict="improved", delta=-10)) == ""


@pytest.mark.parametrize(
    ("noise_pct", "expected"),
    [
        pytest.param(30, "noise ±30.0%", id="well-below-cap"),
        pytest.param(100, "noise ±100.0%", id="at-cap"),
        pytest.param(2.5, "noise ±2.5%", id="fractional"),
        pytest.param(2, "noise ±2.0%", id="whole-number"),
        pytest.param(0.5, "noise ±0.5%", id="sub-percent"),
    ],
)
def test_format_evidence_when_unstable_within_cap_does_state_percentage(
    noise_pct: float, expected: str
):
    verdict = permutation_verdict(verdict="unstable", noise_pct=noise_pct, noise_abs=381)

    assert format_evidence(verdict, "bytes", 5) == expected


def test_format_evidence_when_unstable_past_cap_does_state_absolute_units():
    verdict = permutation_verdict(verdict="unstable", noise_pct=7620, noise_abs=381)

    assert format_evidence(verdict, "bytes", 5) == "±381B noise on a 5B median"


# ---------------------------------------------------------------------------
# format_delta
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("delta", "expected"),
    [
        pytest.param(2.2, "+2.2%", id="signs-a-regression"),
        pytest.param(-17.9, "-17.9%", id="signs-an-improvement"),
        pytest.param(0, "0.0%", id="exact-zero-unsigned"),
        pytest.param(30, "+30.0%", id="rounds-to-one-decimal"),
        pytest.param(0.04, "0.0%", id="positive-rounds-to-zero-unsigned"),
        pytest.param(-0.04, "0.0%", id="negative-rounds-to-zero-unsigned"),
        pytest.param(0.06, "+0.1%", id="just-above-rounding-floor"),
        pytest.param(-0.06, "-0.1%", id="just-below-rounding-floor"),
    ],
)
def test_format_delta_when_finite_does_sign_and_round(delta: float, expected: str):
    assert format_delta(_percent(delta)) == expected


def test_format_delta_when_undefined_arithmetic_does_render_nothing():
    assert format_delta(_percent(float("nan"))) == ""


# ---------------------------------------------------------------------------
# display_class
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("verdict", "expected"),
    [
        pytest.param(band_verdict(n=10, usable_n=0), "identical", id="every-pair-tied"),
        pytest.param(band_verdict(n=10, usable_n=3), "within-noise", id="ties-short-of-floor"),
        pytest.param(band_verdict(n=10, usable_n=6), "within-noise", id="ties-just-enough"),
        pytest.param(band_verdict(n=6, usable_n=5), "within-noise", id="one-below-floor"),
        pytest.param(band_verdict(n=5, usable_n=5), "within-noise", id="too-short-for-floor"),
        pytest.param(band_verdict(n=1, usable_n=1), "inconclusive", id="single-pair-only-floor"),
        pytest.param(band_verdict(n=1, usable_n=0), "inconclusive", id="single-pair-tie"),
        pytest.param(band_verdict(n=2, usable_n=2), "within-noise", id="two-pairs-give-spread"),
        pytest.param(
            band_verdict(verdict="improved", delta=-10, n=10, usable_n=3),
            "improved",
            id="band-found-improvement",
        ),
        pytest.param(
            band_verdict(verdict="unstable", n=10, usable_n=3),
            "unstable",
            id="noise-swamped-band",
        ),
        pytest.param(permutation_verdict(), "within-noise", id="permutation-no-signal"),
        pytest.param(exact_verdict(), "within-noise", id="counted-metric-unchanged"),
    ],
)
def test_display_class_maps_verdict_to_shown_class(verdict: MetricVerdict, expected: str):
    assert display_class(verdict) == expected


# ---------------------------------------------------------------------------
# get_glyph
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("shown", "expected"),
    [
        pytest.param("improved", "✓", id="improved"),
        pytest.param("regressed", "✗", id="regressed"),
        pytest.param("unstable", "≈", id="unstable"),
        pytest.param("identical", "=", id="identical"),
        pytest.param("within-noise", "~", id="within-noise"),
        pytest.param("inconclusive", "?", id="inconclusive"),
    ],
)
def test_get_glyph_marks_display_class(shown: DisplayClass, expected: str):
    assert get_glyph(shown) == expected


# ---------------------------------------------------------------------------
# count_verdicts
# ---------------------------------------------------------------------------


def test_count_verdicts_when_mixed_does_count_each_class_and_skip_no_verdict():
    metrics: Metrics = {
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
def test_count_verdicts_counts_only_the_named_candidate(
    candidate_index: int, expected: VerdictCounts
):
    metrics: Metrics = {
        "decode/time": metric_for(
            [
                CandidateSpec(verdict="improved", delta=-10),
                CandidateSpec(verdict="regressed", delta=8),
            ]
        ),
    }

    assert count_verdicts(metrics, candidate_index) == expected


# ---------------------------------------------------------------------------
# select_highlights
# ---------------------------------------------------------------------------


def test_select_highlights_orders_regressions_then_improvements_then_unstable():
    metrics: Metrics = {
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
    metrics: Metrics = {
        "faster/time": approximate_metric(verdict="improved", delta=-10),
        "tied/heap": band_metric(n=10, usable_n=3),
    }

    highlights = select_highlights(metrics, 0)

    assert [highlight.name for highlight in highlights] == ["faster/time"]


def test_select_highlights_when_single_pair_does_leave_out():
    metrics: Metrics = {
        "faster/time": approximate_metric(verdict="improved", delta=-10),
        "single-pair/time": band_metric(n=1, noise_pct=0.5),
    }

    highlights = select_highlights(metrics, 0)

    assert [highlight.name for highlight in highlights] == ["faster/time"]


def test_select_highlights_keeps_declaration_order_for_equal_magnitude():
    metrics: Metrics = {
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


def test_select_highlights_carries_metric_and_candidate_slice():
    metrics: Metrics = {
        "slower/time": metric_for(
            [
                CandidateSpec(verdict="improved", delta=-10),
                CandidateSpec(verdict="regressed", delta=8),
            ]
        ),
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
def test_select_highlights_ranks_each_candidate_by_its_own_verdicts(
    candidate_index: int, expected: list[str]
):
    metrics: Metrics = {
        "a/time": metric_for(
            [
                CandidateSpec(verdict="improved", delta=-4),
                CandidateSpec(verdict="regressed", delta=3),
            ]
        ),
        "b/time": metric_for(
            [
                CandidateSpec(verdict="regressed", delta=6),
                CandidateSpec(verdict="no-signal", delta=0.2),
            ]
        ),
    }

    highlights = select_highlights(metrics, candidate_index)

    assert [highlight.name for highlight in highlights] == expected


# ---------------------------------------------------------------------------
# scoped_geomean_label
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("scope", "geomean", "expected"),
    [
        pytest.param("entity", geomean_of(n=1), "geomean · entity (1)", id="nothing-excluded"),
        pytest.param(
            "entity",
            geomean_of(n=1, excluded=[Exclusion(metric="entity.spawn/time", reason="unstable")]),
            "geomean · entity (1/2)",
            id="one-exclusion",
        ),
        pytest.param(
            "memory",
            geomean_of(
                n=13,
                excluded=[
                    Exclusion(metric="a/heap", reason="unstable"),
                    Exclusion(metric="b/heap", reason="undefined-ratio"),
                ],
            ),
            "geomean · memory (13/15)",
            id="several-exclusions",
        ),
    ],
)
def test_scoped_geomean_label_counts_the_subset(scope: str, geomean: GeomeanResult, expected: str):
    assert scoped_geomean_label(scope, geomean) == expected


# ---------------------------------------------------------------------------
# geomean_value_style
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("geomean", "expected"),
    [
        pytest.param(geomean_of(value=-6, band=5), "bold green", id="improvement-past-band"),
        pytest.param(geomean_of(value=6, band=5), "bold red", id="regression-past-band"),
        pytest.param(geomean_of(value=-4, band=5), "bold", id="improvement-inside-band"),
        pytest.param(geomean_of(value=4, band=5), "bold", id="regression-inside-band"),
        pytest.param(geomean_of(value=-5, band=5), "bold", id="level-with-band"),
        pytest.param(geomean_of(value=-0.2, band=0), "bold green", id="improvement-no-band"),
        pytest.param(geomean_of(value=float("nan"), n=0), "bold", id="no-stable-metrics"),
    ],
)
def test_geomean_value_style_styles_by_value_against_band(geomean: GeomeanResult, expected: str):
    assert geomean_value_style(geomean, []) == expected


# ---------------------------------------------------------------------------
# verdict_summary_parts
# ---------------------------------------------------------------------------

_MIXED: Metrics = {
    "faster/time": approximate_metric(verdict="improved", delta=-10),
    "slower/time": approximate_metric(verdict="regressed", delta=8),
    "jittery/time": approximate_metric(verdict="unstable", delta=5, noise_pct=300),
    "flat/time": approximate_metric(verdict="no-signal", delta=0.2),
    "tied/heap": band_metric(n=10, usable_n=0),
    "single-pair/time": band_metric(n=1, noise_pct=0.5),
}


def test_verdict_summary_parts_when_plain_does_carry_no_ansi():
    parts = verdict_summary_parts(_MIXED, 0)

    assert "\x1b[" not in "".join(_plain(part) for part in parts)


def test_verdict_summary_parts_tallies_identical_and_single_pair_apart_from_noise():
    parts = verdict_summary_parts(_MIXED, 0)

    assert _plain(_find_plain(parts, "identical")) == "= 1 identical"
    assert _plain(_find_plain(parts, "inconclusive")) == "? 1 inconclusive"
    assert _plain(_find_plain(parts, "within noise")) == "~ 1 within noise"


@pytest.mark.parametrize(
    ("label", "code"),
    [
        pytest.param("improved", "32", id="improved-green"),
        pytest.param("regressed", "31", id="regressed-red"),
        pytest.param("unstable", "33", id="unstable-yellow"),
        pytest.param("identical", "36", id="identical-cyan"),
    ],
)
def test_verdict_summary_parts_colors_nonzero_part(label: str, code: str):
    parts = verdict_summary_parts(_MIXED, 0)
    part = _find_plain(parts, label)

    assert code in _sgr_codes(_colored(part))


@pytest.mark.parametrize("label", ["regressed", "identical"])
def test_verdict_summary_parts_dims_zero_count_part(label: str):
    only_improved: Metrics = {"faster/time": approximate_metric(verdict="improved", delta=-10)}

    parts = verdict_summary_parts(only_improved, 0)
    part = _find_plain(parts, label)

    assert "2" in _sgr_codes(_colored(part))


def test_verdict_summary_parts_dims_within_noise_regardless_of_count():
    parts = verdict_summary_parts(_MIXED, 0)
    part = _find_plain(parts, "within noise")

    assert "2" in _sgr_codes(_colored(part))


def test_verdict_summary_parts_pads_counts_to_widest_digit_width():
    metrics: Metrics = {
        f"improved-{i}/time": approximate_metric(verdict="improved", delta=-(i + 1))
        for i in range(10)
    }
    metrics["regressed/time"] = approximate_metric(verdict="regressed", delta=5)
    metrics["noisy/time"] = approximate_metric(verdict="unstable", delta=3, noise_pct=300)

    parts = verdict_summary_parts(metrics, 0)

    assert _plain(_find_plain(parts, "improved")) == "✓ 10 improved"
    assert _plain(_find_plain(parts, "regressed")) == "✗  1 regressed"
    assert _plain(_find_plain(parts, "unstable")) == "≈  1 unstable"
    assert _plain(_find_plain(parts, "within noise")) == "~  0 within noise"


# ---------------------------------------------------------------------------
# footer_lines
# ---------------------------------------------------------------------------


#: The one hint the footer offers, in the prose ``format_hint`` renders it from.
SAMPLE_SHORTAGE_HINT = "re-run with `gymrat compare --samples 6` or more for statistical verdicts"

#: The hint after ``format_hint`` → ``_plain`` round-trips (backticks stripped).
SAMPLE_SHORTAGE_HINT_PLAIN = (
    "re-run with gymrat compare --samples 6 or more for statistical verdicts"
)


def _verbose_lines(metrics: Metrics) -> list[str]:
    """The verbose method lines, with no hint contribution."""
    return [
        line
        for line in footer_lines(metrics, verbose=True, format_hint=format_hint, command="compare")
        if SAMPLE_SHORTAGE_HINT_PLAIN not in _plain(line)
    ]


def _band_lines_for(metrics: Metrics) -> list[str]:
    """The plain noise-band fallback lines, in the order they were emitted."""
    return [
        _plain(line) for line in _verbose_lines(metrics) if _plain(line).startswith("noise band")
    ]


def test_footer_lines_when_plain_does_carry_no_ansi():
    metrics: Metrics = {"a/time": approximate_metric(verdict="improved", delta=-10)}

    lines = _verbose_lines(metrics)

    assert len(lines) > 0
    assert all("\x1b[" not in _plain(line) for line in lines)


def test_footer_lines_dims_the_descriptive_verdict_line():
    metrics: Metrics = {"a/time": approximate_metric(verdict="improved", delta=-10)}

    verdict_line = next(line for line in _verbose_lines(metrics) if "permutation" in _plain(line))

    assert "2" in _sgr_codes(_colored(verdict_line))


def test_footer_lines_when_verbose_does_close_on_the_sample_shortage_hint():
    metrics: Metrics = {"a/time": band_metric(n=4)}

    lines = footer_lines(metrics, verbose=True, format_hint=format_hint, command="compare")

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
def test_footer_lines_phrases_band_line_by_cause(metrics: Metrics, expected: list[str]):
    assert _band_lines_for(metrics) == expected


def test_footer_lines_hands_the_hint_line_to_the_injected_formatter():
    metrics: Metrics = {"a/time": band_metric(n=4)}

    assert footer_lines(metrics, verbose=False, format_hint=format_hint, command="compare") == [
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
def test_footer_lines_hints_by_cause(metrics: Metrics, expected: list[str]):
    lines = footer_lines(metrics, verbose=False, format_hint=format_hint, command="compare")

    assert [_plain(line) for line in lines] == expected


@pytest.mark.parametrize(
    ("metrics", "samples"),
    [
        pytest.param(
            {"a/time": band_metric(n=MIN_PERMUTATION_N - 1)},
            MIN_PERMUTATION_N - 1,
            id="fewer-samples-than-floor",
        ),
        pytest.param({"a/time": band_metric(n=1)}, 1, id="single-sample"),
    ],
)
def test_footer_lines_when_samples_below_floor_does_suggest_more_samples(
    metrics: Metrics, samples: int
):
    lines = footer_lines(
        metrics, verbose=False, format_hint=format_hint, command="compare", samples=samples
    )

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
            {"a/time": band_metric(n=MIN_PERMUTATION_N - 1)},
            MIN_PERMUTATION_N,
            id="floor-reached-but-fewer-paired",
        ),
    ],
)
def test_footer_lines_when_samples_enough_does_name_dropped_rounds(metrics: Metrics, samples: int):
    lines = footer_lines(
        metrics, verbose=False, format_hint=format_hint, command="compare", samples=samples
    )

    assert not any("gymrat compare --samples" in line for line in lines)
    assert any("dropped" in line for line in lines)


def test_footer_lines_when_samples_enough_and_every_metric_tested_does_not_hint():
    metrics: Metrics = {"a/time": approximate_metric(verdict="improved", delta=-10)}

    assert (
        footer_lines(metrics, verbose=False, format_hint=format_hint, command="compare", samples=10)
        == []
    )


# ---------------------------------------------------------------------------
# pluralize
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("noun", "expected"),
    [
        pytest.param("pass", "2 passes", id="ends-in-s"),
        pytest.param("box", "2 boxes", id="ends-in-x"),
        pytest.param("buzz", "2 buzzes", id="ends-in-z"),
        pytest.param("branch", "2 branches", id="ends-in-ch"),
        pytest.param("dish", "2 dishes", id="ends-in-sh"),
        pytest.param("query", "2 queries", id="consonant-then-y"),
        pytest.param("key", "2 keys", id="vowel-then-y"),
        pytest.param("metric", "2 metrics", id="regular"),
        pytest.param("kept iteration", "2 kept iterations", id="multi-word-regular"),
        pytest.param("uncommitted file", "2 uncommitted files", id="multi-word-adjective"),
    ],
)
def test_pluralize_when_count_is_plural_does_apply_english_suffix_rules(noun: str, expected: str):
    assert pluralize(2, noun) == expected


@pytest.mark.parametrize("noun", ["pass", "query", "box", "metric", "kept iteration"])
def test_pluralize_when_count_is_one_does_leave_noun_unchanged(noun: str):
    assert pluralize(1, noun) == f"1 {noun}"


@pytest.mark.parametrize(
    ("count", "expected"),
    [
        pytest.param(0, "0 passes", id="zero"),
        pytest.param(2, "2 passes", id="many"),
        pytest.param(-1, "-1 passes", id="negative"),
    ],
)
def test_pluralize_when_count_is_not_one_does_use_the_plural_form(count: int, expected: str):
    assert pluralize(count, "pass") == expected


@pytest.mark.parametrize(
    ("count", "expected"),
    [
        pytest.param(1, "1 index", id="singular-keeps-noun"),
        pytest.param(2, "2 indices", id="plural-takes-override"),
        pytest.param(0, "0 indices", id="zero-takes-override"),
    ],
)
def test_pluralize_when_plural_given_does_override_the_suffix_rules(count: int, expected: str):
    assert pluralize(count, "index", "indices") == expected


# ---------------------------------------------------------------------------
# ReportOptions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("options", "expected"),
    [
        pytest.param(ReportOptions(), None, id="default-defers"),
        pytest.param(ReportOptions(color=True), True, id="forced-on"),
        pytest.param(ReportOptions(color=False), False, id="forced-off"),
    ],
)
def test_report_options_carries_an_optional_color_override(
    options: ReportOptions, expected: bool | None
):
    assert options.color is expected
