"""Behavioral tests for the verdict engine.

Covers pairing, delta computation, and per-metric method dispatch (exact,
permutation, band), driving that behavior through the public ``compute_verdicts``
API only.
"""

import math

import pytest

from gymrat_py.model import (
    BandVerdict,
    Effect,
    ExactVerdict,
    MetricMeta,
    MetricVerdict,
    Observations,
    PermutationVerdict,
)
from gymrat_py.verdict import compute_verdicts
from gymrat_py.warn import WarnSink
from tests.verdict._inputs import _noop_warn, create_samples

# ---------------------------------------------------------------------------
# Metric-meta fixtures shared across the verdict-engine cases
# ---------------------------------------------------------------------------

METRIC_EXACT_LOWER = {"metric": MetricMeta(direction="lower", gating=True, exact=True, unit=None)}
METRIC_EXACT_HIGHER = {"metric": MetricMeta(direction="higher", gating=True, exact=True, unit=None)}
METRIC_APPROX_LOWER = {"metric": MetricMeta(direction="lower", gating=True, exact=False, unit=None)}
METRIC_APPROX_HIGHER = {
    "metric": MetricMeta(direction="higher", gating=True, exact=False, unit=None)
}
METRIC_BYTES_LOWER = {
    "metric": MetricMeta(direction="lower", gating=True, exact=False, unit="bytes")
}
METRIC_BYTES_HIGHER = {
    "metric": MetricMeta(direction="higher", gating=True, exact=False, unit="bytes")
}
METRIC_NS_LOWER = {"metric": MetricMeta(direction="lower", gating=True, exact=False, unit="ns")}


def run(
    samples_a: list[dict[str, float]],
    samples_b: list[dict[str, float]],
    meta: dict[str, MetricMeta],
    *,
    unstable_noise_pct: float | None = None,
    warn: WarnSink | None = None,
) -> dict[str, MetricVerdict]:
    left = Observations.from_rounds(samples_a)
    right = Observations.from_rounds(samples_b)
    sink = _noop_warn if warn is None else warn
    if unstable_noise_pct is None:
        return compute_verdicts(left, right, meta, warn=sink)
    return compute_verdicts(left, right, meta, unstable_noise_pct=unstable_noise_pct, warn=sink)


def samples(*values: float) -> list[dict[str, float]]:
    return [{"metric": value} for value in values]


def get_permutation(result: dict[str, MetricVerdict], key: str = "metric") -> PermutationVerdict:
    verdict = result[key]
    assert isinstance(verdict, PermutationVerdict), f"expected permutation, got {verdict.method}"
    return verdict


def get_band(result: dict[str, MetricVerdict], key: str = "metric") -> BandVerdict:
    verdict = result[key]
    assert isinstance(verdict, BandVerdict), f"expected band, got {verdict.method}"
    return verdict


# Six paired windows noisy enough to reach noise_pct = 30 while staying on the
# permutation path: all six diffs non-zero and negative, and the two groups are
# separated enough that the sign-flip null makes the observed delta significant.
NOISY_PERMUTATION_A = samples(80.0, 90.0, 100.0, 100.0, 110.0, 120.0)
NOISY_PERMUTATION_B = samples(40.0, 45.0, 50.0, 50.0, 55.0, 60.0)

# Two paired windows noisy enough to reach noise_pct = 30 on the band path.
NOISY_BAND_A = samples(80.0, 120.0)
NOISY_BAND_B = samples(8.0, 12.0)


# ---------------------------------------------------------------------------
# Verdict record shape
# ---------------------------------------------------------------------------


def test_compute_verdicts_when_exact_does_carry_only_verdict_method_delta_and_n():
    result = run(samples(100.0), samples(95.0), METRIC_EXACT_LOWER)

    assert result["metric"] == ExactVerdict(
        method="exact",
        verdict="improved",
        delta=Effect(value=-5.0, unit="percent"),
        n=1,
    )


# ---------------------------------------------------------------------------
# Delta computation
# ---------------------------------------------------------------------------


def test_compute_verdicts_when_medians_differ_does_report_percentage_delta():
    result = run(samples(100.0), samples(110.0), METRIC_EXACT_LOWER)

    assert result["metric"].delta.value == pytest.approx(10.0, abs=1e-5)
    assert result["metric"].delta.unit == "percent"


def test_compute_verdicts_when_multiple_rounds_does_compute_delta_from_medians():
    result = run(
        samples(90.0, 100.0, 110.0),
        samples(85.0, 95.0, 105.0),
        METRIC_EXACT_LOWER,
    )

    assert result["metric"].delta.value == pytest.approx(-5.0, abs=1e-5)


def test_compute_verdicts_when_no_signal_does_still_report_delta():
    result = run(samples(100.0), samples(100.0), METRIC_EXACT_LOWER)

    verdict = result["metric"]
    assert verdict.verdict == "no-signal"
    assert verdict.delta.value == 0.0


# ---------------------------------------------------------------------------
# Exact path
# ---------------------------------------------------------------------------


def test_compute_verdicts_when_exact_and_tiny_difference_does_signal():
    result = run(samples(100.0), samples(100.01), METRIC_EXACT_LOWER)

    assert result["metric"].verdict != "no-signal"


@pytest.mark.parametrize(
    ("meta", "samples_b", "expected_delta", "expected_verdict"),
    [
        (METRIC_EXACT_LOWER, samples(95.0), -5.0, "improved"),
        (METRIC_EXACT_HIGHER, samples(105.0), 5.0, "improved"),
        (METRIC_EXACT_LOWER, samples(105.0), 5.0, "regressed"),
        (METRIC_EXACT_HIGHER, samples(95.0), -5.0, "regressed"),
    ],
)
def test_compute_verdicts_when_exact_does_classify_by_direction(
    meta: dict[str, MetricMeta],
    samples_b: list[dict[str, float]],
    expected_delta: float,
    expected_verdict: str,
):
    result = run(samples(100.0), samples_b, meta)

    verdict = result["metric"]
    assert verdict.verdict == expected_verdict
    assert verdict.delta.value == pytest.approx(expected_delta, abs=1e-5)


# ---------------------------------------------------------------------------
# Pairing and filtering
# ---------------------------------------------------------------------------


def test_compute_verdicts_when_paired_by_index_does_keep_all_windows():
    result = run(samples(90.0, 110.0), samples(85.0, 105.0), METRIC_EXACT_LOWER)

    assert result["metric"].n == 2


def test_compute_verdicts_when_metric_missing_from_left_does_drop_window():
    result = run(
        [{"metric": 100.0}, {}],
        samples(95.0, 90.0),
        METRIC_EXACT_LOWER,
    )

    assert result["metric"].n == 1


def test_compute_verdicts_when_metric_missing_from_right_does_drop_window():
    result = run(
        samples(100.0, 110.0),
        [{"metric": 95.0}, {}],
        METRIC_EXACT_LOWER,
    )

    assert result["metric"].n == 1


def test_compute_verdicts_when_metric_one_sided_across_all_windows_does_skip():
    result = run(
        [{"metricA": 100.0}],
        [{"metricB": 95.0}],
        {
            "metricA": MetricMeta(direction="lower", gating=True, exact=True, unit=None),
            "metricB": MetricMeta(direction="lower", gating=True, exact=True, unit=None),
        },
    )

    assert result == {}


def test_compute_verdicts_when_one_paired_window_survives_does_keep_metric():
    result = run(samples(100.0, 90.0), samples(95.0, 85.0), METRIC_EXACT_LOWER)

    assert "metric" in result


def test_compute_verdicts_when_metric_named_like_a_dunder_does_keep_verdict():
    proto = "__proto__"
    result = run(
        [{proto: 100.0}],
        [{proto: 95.0}],
        {proto: MetricMeta(direction="lower", gating=True, exact=True, unit=None)},
    )

    assert list(result.items()) == [
        (
            proto,
            ExactVerdict(
                method="exact",
                verdict="improved",
                delta=Effect(value=-5.0, unit="percent"),
                n=1,
            ),
        ),
    ]


# ---------------------------------------------------------------------------
# Multiple metrics
# ---------------------------------------------------------------------------


def test_compute_verdicts_when_multiple_paired_metrics_does_return_all():
    result = run(
        [{"a": 100.0, "b": 50.0}],
        [{"a": 95.0, "b": 45.0}],
        {
            "a": MetricMeta(direction="lower", gating=True, exact=True, unit=None),
            "b": MetricMeta(direction="lower", gating=True, exact=True, unit=None),
        },
    )

    assert "a" in result
    assert "b" in result


def test_compute_verdicts_when_metrics_differ_in_exactness_does_respect_per_metric_flag():
    result = run(
        [{"exactMetric": 100.0, "otherMetric": 50.0}],
        [{"exactMetric": 100.001, "otherMetric": 45.0}],
        {
            "exactMetric": MetricMeta(direction="lower", gating=True, exact=True, unit=None),
            "otherMetric": MetricMeta(direction="lower", gating=True, exact=False, unit=None),
        },
    )

    verdict = result["exactMetric"]
    assert verdict.verdict != "no-signal"
    assert verdict.method == "exact"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_compute_verdicts_when_zero_metric_value_does_report_zero_delta():
    result = run(samples(0.0), samples(0.0), METRIC_EXACT_LOWER)

    assert result["metric"].delta.value == 0.0


def test_compute_verdicts_when_baseline_median_zero_does_report_nan_delta():
    result = run(samples(0.0), samples(5.0), METRIC_EXACT_LOWER)

    assert math.isnan(result["metric"].delta.value)


@pytest.mark.parametrize(
    ("median_a", "median_b", "expected_delta", "expected_verdict"),
    [
        pytest.param(-100.0, -95.0, 5.0, "regressed", id="rises-toward-zero"),
        pytest.param(-1.0, -2.0, -100.0, "improved", id="falls-below-zero"),
    ],
)
def test_compute_verdicts_when_negative_median_moves_does_sign_delta_by_movement(
    median_a: float,
    median_b: float,
    expected_delta: float,
    expected_verdict: str,
):
    result = run(samples(median_a), samples(median_b), METRIC_EXACT_LOWER)

    verdict = result["metric"]
    assert verdict.delta.value == pytest.approx(expected_delta, abs=1e-5)
    assert verdict.verdict == expected_verdict


def test_compute_verdicts_when_many_windows_does_pair_all():
    result = run(create_samples(100, 100.0), create_samples(100, 95.0), METRIC_EXACT_LOWER)

    verdict = result["metric"]
    assert verdict.n == 100
    assert verdict.delta.value == pytest.approx(-5.0, abs=1e-5)


# ---------------------------------------------------------------------------
# Permutation method
# ---------------------------------------------------------------------------


def test_compute_verdicts_when_non_exact_and_six_pairs_does_use_permutation():
    result = run(create_samples(6, 100.0), create_samples(6, 95.0), METRIC_APPROX_LOWER)

    assert result["metric"].method == "permutation"


def test_compute_verdicts_when_permutation_p_not_significant_does_no_signal():
    samples_b = samples(99.0, 101.0, 98.0, 102.0, 97.0, 103.0)

    result = run(create_samples(6, 100.0), samples_b, METRIC_APPROX_LOWER)

    verdict = get_permutation(result)
    assert verdict.p >= 0.05
    assert verdict.verdict == "no-signal"


def test_compute_verdicts_when_permutation_delta_nan_does_no_signal():
    result = run(create_samples(6, 0.0), create_samples(6, 5.0), METRIC_APPROX_LOWER)

    verdict = get_permutation(result)
    assert math.isnan(verdict.delta.value)
    assert verdict.p == pytest.approx(1.0)
    assert verdict.verdict == "no-signal"


@pytest.mark.parametrize(
    "meta",
    [
        pytest.param(METRIC_APPROX_LOWER, id="lower"),
        pytest.param(METRIC_APPROX_HIGHER, id="higher"),
    ],
)
def test_compute_verdicts_when_permutation_delta_zero_does_no_signal(meta: dict[str, MetricMeta]):
    samples_a = samples(80.0, 90.0, 95.0, 100.0, 100.0, 105.0, 110.0, 120.0)
    samples_b = samples(81.0, 91.0, 96.0, 100.0, 100.0, 106.0, 111.0, 121.0)

    result = run(samples_a, samples_b, meta)

    verdict = get_permutation(result)
    assert verdict.delta.value == 0.0
    assert verdict.p == pytest.approx(1.0)
    assert verdict.verdict == "no-signal"


def test_compute_verdicts_when_permutation_delta_below_band_does_no_signal():
    # The permutation statistic *is* the delta functional, so a delta smaller
    # than the noise band never separates the two groups enough for the
    # sign-flip null to call it significant — unlike the retired signed-rank
    # test, which modelled scatter through a separate rank p-value and could
    # still signal here. This verdict flip is the deliberate divergence.
    samples_a = samples(80.0, 90.0, 100.0, 100.0, 110.0, 120.0)
    samples_b = samples(76.0, 85.5, 95.0, 95.0, 104.5, 114.0)

    result = run(samples_a, samples_b, METRIC_APPROX_LOWER)

    verdict = get_permutation(result)
    assert verdict.p == pytest.approx(0.5)
    assert verdict.delta.value == pytest.approx(-5.0, abs=1e-5)
    assert verdict.noise_pct == pytest.approx(30.0, abs=1e-5)
    assert verdict.verdict == "no-signal"


def test_compute_verdicts_when_zero_diffs_drop_usable_below_six_does_fall_back_to_band():
    samples_b = samples(100.0, 100.0, 100.0, 95.0, 90.0, 105.0)

    result = run(create_samples(6, 100.0), samples_b, METRIC_APPROX_LOWER)

    assert result["metric"].method == "band"


# ---------------------------------------------------------------------------
# Band method
# ---------------------------------------------------------------------------


def test_compute_verdicts_when_non_exact_and_few_pairs_does_use_band():
    result = run(create_samples(2, 100.0), create_samples(2, 95.0), METRIC_APPROX_LOWER)

    assert result["metric"].method == "band"


def test_compute_verdicts_when_band_delta_exceeds_band_does_signal():
    result = run(samples(100.0, 110.0), samples(30.0, 50.0), METRIC_APPROX_LOWER)

    verdict = result["metric"]
    assert verdict.method == "band"
    assert verdict.verdict == "improved"


def test_compute_verdicts_when_band_delta_within_band_does_no_signal():
    result = run(create_samples(2, 100.0), samples(101.0, 99.0), METRIC_APPROX_LOWER)

    verdict = result["metric"]
    assert verdict.method == "band"
    assert verdict.verdict == "no-signal"


def test_compute_verdicts_when_band_spread_present_does_scale_noise_by_spread():
    result = run(samples(80.0, 120.0), samples(90.0, 110.0), METRIC_APPROX_LOWER)

    assert get_band(result).noise_pct == pytest.approx(30.0, abs=1e-1)


def test_compute_verdicts_when_band_spread_tiny_does_apply_noise_floor():
    result = run(create_samples(2, 100.0), samples(100.0, 100.1), METRIC_APPROX_LOWER)

    assert get_band(result).noise_pct == pytest.approx(0.5, abs=1e-1)


@pytest.mark.parametrize(
    ("samples_b", "expected_n", "expected_usable_n"),
    [
        pytest.param(
            samples(100.0, 100.0, 100.0, 95.0, 90.0, 105.0),
            6,
            3,
            id="tied-pairs",
        ),
        pytest.param(
            samples(100.0, 95.0, 90.0),
            3,
            2,
            id="too-few-samples",
        ),
    ],
)
def test_compute_verdicts_when_band_fallback_does_report_usable_n(
    samples_b: list[dict[str, float]],
    expected_n: int,
    expected_usable_n: int,
):
    result = run(create_samples(len(samples_b), 100.0), samples_b, METRIC_APPROX_LOWER)

    verdict = get_band(result)
    assert verdict.n == expected_n
    assert verdict.usable_n == expected_usable_n


@pytest.mark.parametrize(
    ("pairs", "expected"),
    [
        pytest.param(1, "no-signal", id="single-window"),
        pytest.param(2, "improved", id="two-windows"),
    ],
)
def test_compute_verdicts_when_band_window_count_varies_does_gate_on_spread(
    pairs: int,
    expected: str,
):
    result = run(create_samples(pairs, 100.0), create_samples(pairs, 50.0), METRIC_APPROX_LOWER)

    verdict = result["metric"]
    assert verdict.method == "band"
    assert verdict.verdict == expected


def test_compute_verdicts_when_band_median_negative_does_use_magnitude():
    result = run(samples(-60.0, -40.0), samples(-50.0, -50.0), METRIC_APPROX_LOWER)

    assert get_band(result).noise_pct == pytest.approx(30.0, abs=1e-5)


def test_compute_verdicts_when_band_median_zero_does_treat_spread_as_zero():
    result = run(samples(-5.0, 5.0), samples(-10.0, 10.0), METRIC_APPROX_LOWER)

    assert get_band(result).noise_pct == pytest.approx(0.5, abs=1e-1)


# Two well-separated six-window groups: the sign-flip null makes the observed
# delta significant (p = 0.03125) in either pairing direction.
_SEPARATED_HIGH = samples(180.0, 190.0, 200.0, 200.0, 210.0, 220.0)
_SEPARATED_LOW = samples(90.0, 95.0, 100.0, 100.0, 105.0, 110.0)


@pytest.mark.parametrize(
    ("meta", "samples_a", "samples_b", "expected_verdict"),
    [
        (METRIC_APPROX_LOWER, _SEPARATED_HIGH, _SEPARATED_LOW, "improved"),
        (METRIC_APPROX_LOWER, _SEPARATED_LOW, _SEPARATED_HIGH, "regressed"),
        (METRIC_APPROX_HIGHER, _SEPARATED_LOW, _SEPARATED_HIGH, "improved"),
        (METRIC_APPROX_HIGHER, _SEPARATED_HIGH, _SEPARATED_LOW, "regressed"),
    ],
)
def test_compute_verdicts_when_permutation_direction_varies_does_classify(
    meta: dict[str, MetricMeta],
    samples_a: list[dict[str, float]],
    samples_b: list[dict[str, float]],
    expected_verdict: str,
):
    result = run(samples_a, samples_b, meta)

    verdict = result["metric"]
    assert verdict.method == "permutation"
    assert verdict.verdict == expected_verdict


@pytest.mark.parametrize(
    ("samples_a_value", "samples_b_value", "expected_verdict"),
    [
        (50.0, 100.0, "improved"),
        (100.0, 50.0, "regressed"),
    ],
)
def test_compute_verdicts_when_band_direction_higher_does_classify(
    samples_a_value: float,
    samples_b_value: float,
    expected_verdict: str,
):
    result = run(
        create_samples(2, samples_a_value),
        create_samples(2, samples_b_value),
        METRIC_APPROX_HIGHER,
    )

    verdict = result["metric"]
    assert verdict.method == "band"
    assert verdict.verdict == expected_verdict


def test_compute_verdicts_when_band_spread_high_does_report_wide_band_and_no_signal():
    result = run(samples(0.1, 0.1), samples(0.05, 0.15), METRIC_APPROX_LOWER)

    verdict = get_band(result)
    assert verdict.noise_pct == pytest.approx(75.0, abs=0.5)
    assert verdict.verdict == "no-signal"


def test_compute_verdicts_when_band_both_medians_zero_does_apply_floor():
    result = run(create_samples(2, 0.0), create_samples(2, 0.0), METRIC_APPROX_LOWER)

    assert get_band(result).noise_pct == pytest.approx(0.5, abs=1e-1)


def test_compute_verdicts_when_band_fewer_differing_than_min_n_does_no_signal():
    """D4: one tied + one doubled pair reads no-signal when only one differs."""
    result = run(samples(100.0, 100.0), samples(100.0, 210.0), METRIC_APPROX_LOWER)

    verdict = get_band(result)
    assert verdict.n == 2
    assert verdict.usable_n == 1
    assert verdict.verdict == "no-signal"


# ---------------------------------------------------------------------------
# Noise-band carrying on non-exact verdicts
# ---------------------------------------------------------------------------


def test_compute_verdicts_when_permutation_does_carry_noise_pct():
    samples_a = samples(80.0, 90.0, 100.0, 100.0, 110.0, 120.0)
    samples_b = samples(95.0, 95.0, 95.0, 105.0, 105.0, 105.0)

    result = run(samples_a, samples_b, METRIC_APPROX_LOWER)

    verdict = get_permutation(result)
    assert verdict.noise_pct == pytest.approx(30.0, abs=1e-5)


def test_compute_verdicts_when_permutation_median_zero_does_treat_spread_as_zero():
    samples_a = samples(-5.0, -3.0, -1.0, 1.0, 3.0, 5.0)
    samples_b = samples(-10.0, -6.0, -2.0, 2.0, 6.0, 10.0)

    result = run(samples_a, samples_b, METRIC_APPROX_LOWER)

    assert get_permutation(result).noise_pct == pytest.approx(0.5, abs=1e-5)


def test_compute_verdicts_when_permutation_median_negative_does_use_magnitude():
    samples_a = samples(-60.0, -55.0, -50.0, -50.0, -45.0, -40.0)
    samples_b = samples(-61.0, -56.0, -51.0, -51.0, -46.0, -41.0)

    result = run(samples_a, samples_b, METRIC_APPROX_LOWER)

    assert get_permutation(result).noise_pct == pytest.approx(30.0, abs=1e-5)


@pytest.mark.parametrize(
    ("method", "values_a", "values_b"),
    [
        pytest.param(
            "permutation",
            [180.0, 190.0, 200.0, 200.0, 210.0, 220.0],
            [90.0, 95.0, 100.0, 100.0, 105.0, 110.0],
            id="permutation",
        ),
        pytest.param("band", [180.0, 220.0], [90.0, 110.0], id="band"),
    ],
)
def test_compute_verdicts_when_non_exact_does_carry_noise_abs(
    method: str,
    values_a: list[float],
    values_b: list[float],
):
    result = run(samples(*values_a), samples(*values_b), METRIC_APPROX_LOWER)

    verdict = get_permutation(result) if method == "permutation" else get_band(result)
    assert verdict.noise_pct == pytest.approx(15.0, abs=1e-5)
    assert verdict.noise_abs == pytest.approx(30.0, abs=1e-5)


# ---------------------------------------------------------------------------
# Whole-byte resolution floor
# ---------------------------------------------------------------------------

# Six windows whose medians are 4 vs 3 bytes — a one-byte move (delta -25%) —
# spread enough that the sign-flip null makes the observed delta significant
# (p = 0.03125). The whole-byte floor, not the p-value, is what withholds the
# signal on a byte metric.
_BYTE_ONE_MOVE_A = samples(3.4, 3.7, 4.0, 4.0, 4.3, 4.6)
_BYTE_ONE_MOVE_B = samples(2.4, 2.7, 3.0, 3.0, 3.3, 3.6)

# Six windows whose medians are 100 vs 75 bytes — a 25-byte move that clears the
# whole-byte floor — with the same separation that keeps p significant.
_BYTE_CLEARS_A = samples(90.0, 95.0, 100.0, 100.0, 105.0, 110.0)
_BYTE_CLEARS_B = samples(67.0, 71.0, 75.0, 75.0, 79.0, 83.0)


@pytest.mark.parametrize("pairs", [2, 5])
def test_compute_verdicts_when_byte_move_is_one_byte_does_no_signal_on_band(pairs: int):
    result = run(create_samples(pairs, 4.0), create_samples(pairs, 3.0), METRIC_BYTES_LOWER)

    verdict = get_band(result)
    assert verdict.delta.value == pytest.approx(-25.0, abs=1e-5)
    assert verdict.noise_pct == pytest.approx(100 / 3, abs=1e-5)
    assert verdict.verdict == "no-signal"


@pytest.mark.parametrize(
    "meta",
    [
        pytest.param(METRIC_BYTES_LOWER, id="lower"),
        pytest.param(METRIC_BYTES_HIGHER, id="higher"),
    ],
)
def test_compute_verdicts_when_byte_move_is_one_byte_does_no_signal_on_permutation(
    meta: dict[str, MetricMeta],
):
    result = run(_BYTE_ONE_MOVE_A, _BYTE_ONE_MOVE_B, meta)

    verdict = get_permutation(result)
    assert verdict.p < 0.05
    assert verdict.delta.value == pytest.approx(-25.0, abs=1e-5)
    assert verdict.noise_pct == pytest.approx(100 / 3, abs=1e-5)
    assert verdict.verdict == "no-signal"


@pytest.mark.parametrize(
    ("meta", "expected"),
    [
        pytest.param(METRIC_BYTES_LOWER, "improved", id="lower"),
        pytest.param(METRIC_BYTES_HIGHER, "regressed", id="higher"),
    ],
)
def test_compute_verdicts_when_byte_move_clears_floor_does_signal(
    meta: dict[str, MetricMeta],
    expected: str,
):
    result = run(_BYTE_CLEARS_A, _BYTE_CLEARS_B, meta)

    verdict = get_permutation(result)
    assert verdict.delta.value == pytest.approx(-25.0, abs=1e-5)
    assert verdict.noise_pct == pytest.approx(16.0, abs=1e-5)
    assert verdict.verdict == expected


@pytest.mark.parametrize(
    "meta",
    [
        pytest.param(METRIC_NS_LOWER, id="ns"),
        pytest.param(METRIC_APPROX_LOWER, id="none"),
    ],
)
def test_compute_verdicts_when_unit_not_bytes_does_ignore_byte_floor(meta: dict[str, MetricMeta]):
    result = run(_BYTE_ONE_MOVE_A, _BYTE_ONE_MOVE_B, meta)

    verdict = get_permutation(result)
    assert verdict.noise_pct == pytest.approx(30.0, abs=1e-5)
    assert verdict.verdict == "improved"


def test_compute_verdicts_when_byte_floor_side_median_zero_does_exclude_that_side():
    result = run(create_samples(2, 4.0), create_samples(2, 0.0), METRIC_BYTES_LOWER)

    verdict = get_band(result)
    assert verdict.noise_pct == pytest.approx(25.0, abs=1e-5)
    assert verdict.verdict == "improved"


@pytest.mark.parametrize(
    ("values_b", "expected_band"),
    [
        pytest.param([1_000_000.0, 1_000_100.0], 0.5, id="stable"),
        pytest.param([800_000.0, 1_200_000.0], 30.0, id="wide"),
    ],
)
def test_compute_verdicts_when_byte_metric_is_megabyte_scale_does_keep_band(
    values_b: list[float],
    expected_band: float,
):
    result = run(create_samples(2, 1_000_000.0), samples(*values_b), METRIC_BYTES_LOWER)

    assert get_band(result).noise_pct == pytest.approx(expected_band, abs=1e-5)


@pytest.mark.parametrize(
    "meta",
    [
        pytest.param(METRIC_NS_LOWER, id="ns"),
        pytest.param(METRIC_APPROX_LOWER, id="none"),
    ],
)
def test_compute_verdicts_when_unit_not_bytes_does_keep_floor_for_small_move(
    meta: dict[str, MetricMeta],
):
    result = run(create_samples(2, 4.0), create_samples(2, 3.0), meta)

    verdict = get_band(result)
    assert verdict.noise_pct == pytest.approx(0.5, abs=1e-5)
    assert verdict.verdict == "improved"


# ---------------------------------------------------------------------------
# Unstable threshold
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("method", "unstable_noise_pct", "expected"),
    [
        ("permutation", 20.0, "unstable"),
        ("permutation", 30.0, "improved"),
        ("band", 20.0, "unstable"),
        ("band", 30.0, "improved"),
    ],
)
def test_compute_verdicts_when_noise_exceeds_threshold_does_mark_unstable(
    method: str,
    unstable_noise_pct: float,
    expected: str,
):
    samples_a, samples_b = (
        (NOISY_PERMUTATION_A, NOISY_PERMUTATION_B)
        if method == "permutation"
        else (NOISY_BAND_A, NOISY_BAND_B)
    )

    result = run(samples_a, samples_b, METRIC_APPROX_LOWER, unstable_noise_pct=unstable_noise_pct)

    verdict = result["metric"]
    assert verdict.method == method
    assert verdict.verdict == expected


@pytest.mark.parametrize(
    ("samples_b", "expected"),
    [
        pytest.param(
            samples(10.0, 150.0, 410.0),
            "no-signal",
            id="at-threshold",
        ),
        pytest.param(
            samples(10.0, 150.0, 412.0),
            "unstable",
            id="past-threshold",
        ),
    ],
)
def test_compute_verdicts_when_no_threshold_given_does_default_to_two_hundred(
    samples_b: list[dict[str, float]],
    expected: str,
):
    result = run(create_samples(3, 100.0), samples_b, METRIC_APPROX_LOWER)

    assert result["metric"].verdict == expected


def test_compute_verdicts_when_exact_metric_is_noisy_does_never_mark_unstable():
    result = run(
        samples(1.0, 100.0, 10_000.0),
        samples(1.0, 50.0, 10_000.0),
        METRIC_EXACT_LOWER,
        unstable_noise_pct=1.0,
    )

    verdict = result["metric"]
    assert verdict.method == "exact"
    assert verdict.verdict == "improved"


# ---------------------------------------------------------------------------
# D2: zero-median non-exact reports unstable
# ---------------------------------------------------------------------------

# When one side's median is 0 but that side has non-zero half-range, the noise
# fraction is undefined (division by zero). Rather than letting noise_pct reach
# inf, the engine caps it and forces the verdict to unstable on both paths.


@pytest.mark.parametrize(
    ("method", "values_a", "values_b"),
    [
        pytest.param("band", [-5.0, 5.0], [10.0, 10.0], id="band-baseline-zero"),
        pytest.param("band", [10.0, 10.0], [-5.0, 5.0], id="band-candidate-zero"),
        pytest.param(
            "permutation",
            [-5.0, -3.0, -1.0, 1.0, 3.0, 5.0],
            [10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
            id="permutation-baseline-zero",
        ),
        pytest.param(
            "permutation",
            [10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
            [-5.0, -3.0, -1.0, 1.0, 3.0, 5.0],
            id="permutation-candidate-zero",
        ),
    ],
)
def test_compute_verdicts_when_zero_median_with_spread_does_report_unstable(
    method: str,
    values_a: list[float],
    values_b: list[float],
):
    result = run(samples(*values_a), samples(*values_b), METRIC_APPROX_LOWER)

    verdict = get_permutation(result) if method == "permutation" else get_band(result)
    assert verdict.verdict == "unstable"
    assert not math.isinf(verdict.noise_pct)


def test_compute_verdicts_when_bytes_zero_median_and_zero_spread_does_not_report_unstable():
    """D2: byte-floor on a 0-byte median with zero spread does not force unstable."""
    result = run(create_samples(2, 0.0), create_samples(2, 0.0), METRIC_BYTES_LOWER)

    verdict = get_band(result)
    assert verdict.verdict != "unstable"
    assert not math.isinf(verdict.noise_pct)


# ---------------------------------------------------------------------------
# Divergence-1: warning when paired windows are dropped
# ---------------------------------------------------------------------------


def test_compute_verdicts_when_verdict_produced_and_windows_dropped_does_warn_once():
    collected: list[str] = []

    result = run(
        samples(100.0, 100.0, 90.0),
        [{"metric": 95.0}, {"other": 1.0}, {"metric": 85.0}],
        METRIC_EXACT_LOWER,
        warn=collected.append,
    )

    assert "metric" in result
    assert collected == [
        "metric: dropped 1 paired window(s) where the metric was measured on only one side",
    ]


def test_compute_verdicts_when_metric_fully_one_sided_does_not_warn():
    collected: list[str] = []

    result = run(
        [{"metricA": 100.0}],
        [{"metricB": 95.0}],
        {
            "metricA": MetricMeta(direction="lower", gating=True, exact=True, unit=None),
            "metricB": MetricMeta(direction="lower", gating=True, exact=True, unit=None),
        },
        warn=collected.append,
    )

    assert result == {}
    assert collected == []


def test_compute_verdicts_when_cleanly_paired_does_not_warn():
    collected: list[str] = []

    run(samples(100.0, 90.0), samples(95.0, 85.0), METRIC_EXACT_LOWER, warn=collected.append)

    assert collected == []


def test_compute_verdicts_when_windows_dropped_does_not_change_verdict_values():
    clean = run(samples(100.0, 90.0), samples(95.0, 85.0), METRIC_EXACT_LOWER)
    with_drops = run(
        samples(100.0, 100.0, 90.0),
        [{"metric": 95.0}, {"other": 1.0}, {"metric": 85.0}],
        METRIC_EXACT_LOWER,
    )

    assert with_drops == clean
