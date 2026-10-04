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
# format_percent_delta
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
def test_format_percent_delta_when_finite_does_sign_and_round(delta: float, expected: str):
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
def test_format_percent_delta_when_no_finite_value_does_render_nothing(delta: float | None):
    assert format_percent_delta(delta) == ""
