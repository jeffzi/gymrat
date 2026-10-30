"""Tests for the report value, evidence, and delta formatters."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from gymrat.report.format import (
    MetricCellParts,
    format_evidence,
    format_metric_cell_parts,
    format_percent_delta,
    format_value,
)
from tests.report._verdicts import exact_verdict, permutation_verdict

if TYPE_CHECKING:
    from gymrat.model import MetricUnit

# ---------------------------------------------------------------------------
# format_value
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "unit", "expected"),
    [
        pytest.param(999, "ns", "999ns", id="ns-top"),
        pytest.param(999.5, "ns", "1.0µs", id="ns-rounds-onto-us"),
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
        pytest.param(1199.6, "1200", id="rounds-up"),
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


def test_format_evidence_when_unstable_within_cap_does_state_the_noise_percentage():
    verdict = permutation_verdict(verdict="unstable", noise_pct=100, noise_abs=381)

    assert format_evidence(verdict, "bytes", 5) == "noise ±100.0%"


@pytest.mark.parametrize(
    ("noise_pct", "noise_abs", "baseline_median", "candidate_median", "expected"),
    [
        pytest.param(7620, 381, 5, None, "±381B noise on a 5B median", id="past-cap"),
        pytest.param(0.5, 6, 0, None, "±6B noise on a 0B median", id="zero-baseline-median"),
        pytest.param(
            0.5, 6, 100, 0, "±6B noise on a 0B candidate median", id="zero-candidate-median"
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
# format_metric_cell_parts
# ---------------------------------------------------------------------------


def test_format_metric_cell_parts_when_spread_exceeds_the_cap_does_state_absolute_units():
    assert format_metric_cell_parts(5, 7620, "bytes") == MetricCellParts(
        magnitude="5B", spread="381B"
    )


# ---------------------------------------------------------------------------
# format_percent_delta
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("delta", "expected"),
    [
        pytest.param(2.2, "+2.2%", id="signs-a-regression"),
        pytest.param(-17.9, "-17.9%", id="signs-an-improvement"),
        pytest.param(-0.04, "0.0%", id="negative-rounds-to-zero-unsigned"),
        pytest.param(0.06, "+0.1%", id="just-above-rounding-floor"),
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
    ],
)
def test_format_percent_delta_when_missing_or_non_finite_does_render_nothing(delta: float | None):
    assert format_percent_delta(delta) == ""
