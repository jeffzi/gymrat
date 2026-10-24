"""Tests for the report value, evidence, and delta formatters."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from gymrat.report.format import format_evidence, format_percent_delta, format_value
from tests.report._verdicts import exact_verdict, permutation_verdict

if TYPE_CHECKING:
    from gymrat.model import MetricUnit

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
        pytest.param(1_000_000, "ns", "1.0ms", id="ms-floor"),
        pytest.param(2_000_000_000, "ns", "2.0s", id="s-floor"),
        pytest.param(999.5, "ns", "1.0µs", id="ns-rounds-onto-us"),
        pytest.param(999_999.6, "ns", "1.0ms", id="us-rounds-onto-ms"),
        pytest.param(999_950_000, "ns", "1.0s", id="ms-rounds-onto-s"),
        pytest.param(512, "bytes", "512B", id="b-mid"),
        pytest.param(999, "bytes", "999B", id="b-top"),
        pytest.param(1000, "bytes", "1.0KB", id="kb-floor"),
        pytest.param(3600, "bytes", "3.6KB", id="kb-mid"),
        pytest.param(1_000_000, "bytes", "1.0MB", id="mb-floor"),
        pytest.param(2_000_000_000, "bytes", "2.0GB", id="gb-floor"),
        pytest.param(999.5, "bytes", "1.0KB", id="b-rounds-onto-kb"),
        pytest.param(999_950, "bytes", "1.0MB", id="kb-rounds-onto-mb"),
        pytest.param(-3600, "bytes", "-3.6KB", id="neg-kb"),
        pytest.param(-999.5, "bytes", "-1.0KB", id="neg-rounds-onto-kb"),
    ],
)
def test_format_value_when_finite_does_scale_to_its_unit_tier(
    value: float, unit: MetricUnit, expected: str
):
    assert format_value(value, unit) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        pytest.param(0, "0", id="zero"),
        pytest.param(1200, "1200", id="thousands"),
        pytest.param(1199.6, "1200", id="rounds-up"),
        pytest.param(1199.4, "1199", id="rounds-down"),
    ],
)
def test_format_value_when_unitless_does_round_to_an_integer(value: float, expected: str):
    assert format_value(value, None) == expected


@pytest.mark.parametrize(
    ("value", "unit", "expected"),
    [
        pytest.param(float("inf"), "ns", "Infinity", id="positive-infinity"),
        pytest.param(float("-inf"), "bytes", "-Infinity", id="negative-infinity"),
        pytest.param(float("nan"), "ns", "NaN", id="not-a-number"),
    ],
)
def test_format_value_when_non_finite_does_render_the_sentinel(
    value: float, unit: MetricUnit | None, expected: str
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
    ],
)
def test_format_evidence_when_unstable_within_cap_does_state_the_noise_percentage(
    noise_pct: float, expected: str
):
    verdict = permutation_verdict(verdict="unstable", noise_pct=noise_pct, noise_abs=381)

    assert format_evidence(verdict, "bytes", 5) == expected


@pytest.mark.parametrize(
    ("noise_pct", "noise_abs", "baseline_median", "candidate_median", "expected"),
    [
        pytest.param(7620, 381, 5, None, "±381B noise on a 5B median", id="past-cap"),
        pytest.param(0.5, 6, 0, None, "±6B noise on a 0B median", id="zero-baseline-median"),
        pytest.param(
            0.5, 6, 100, 0, "±6B noise on a 0B candidate median", id="zero-candidate-median"
        ),
        pytest.param(
            0.5, 6, 1e-310, None, "±6B noise on a 0B median", id="overflowing-baseline-ratio"
        ),
        pytest.param(
            0.5,
            6,
            100,
            1e-310,
            "±6B noise on a 0B candidate median",
            id="overflowing-candidate-ratio",
        ),
    ],
)
def test_format_evidence_when_percentage_unusable_does_state_absolute_units(
    noise_pct: float,
    noise_abs: float,
    baseline_median: float,
    candidate_median: float | None,
    expected: str,
):
    verdict = permutation_verdict(verdict="unstable", noise_pct=noise_pct, noise_abs=noise_abs)

    assert format_evidence(verdict, "bytes", baseline_median, candidate_median) == expected


# ---------------------------------------------------------------------------
# format_percent_delta
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("delta", "expected"),
    [
        pytest.param(2.2, "+2.2%", id="signs-a-regression"),
        pytest.param(-17.9, "-17.9%", id="signs-an-improvement"),
        pytest.param(0, "0.0%", id="exact-zero-unsigned"),
        pytest.param(0.04, "0.0%", id="positive-rounds-to-zero-unsigned"),
        pytest.param(-0.04, "0.0%", id="negative-rounds-to-zero-unsigned"),
        pytest.param(0.06, "+0.1%", id="just-above-rounding-floor"),
        pytest.param(-0.06, "-0.1%", id="just-below-rounding-floor"),
    ],
)
def test_format_percent_delta_when_finite_does_round_to_a_signed_one_decimal_percentage(
    delta: float, expected: str
):
    assert format_percent_delta(delta) == expected


@pytest.mark.parametrize(
    "delta",
    [
        pytest.param(None, id="missing"),
        pytest.param(float("nan"), id="undefined-arithmetic"),
        pytest.param(float("inf"), id="positive-infinity"),
        pytest.param(float("-inf"), id="negative-infinity"),
    ],
)
def test_format_percent_delta_when_missing_or_non_finite_does_render_nothing(delta: float | None):
    assert format_percent_delta(delta) == ""
