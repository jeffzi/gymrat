"""Behavioral tests for the verdict engine.

Covers pairing, delta computation, and per-metric method dispatch (exact,
permutation, band), driving that behavior through the public ``compute_verdicts``
API only.
"""

import math
import re
import statistics
from collections.abc import Callable

import pytest

from gymrat.model import (
    BandVerdict,
    MetricMeta,
    MetricVerdict,
    PermutationVerdict,
)
from gymrat.stats import detectable_shift
from tests.verdict._engine import (
    DIRECTION_MISMATCH_A,
    DIRECTION_MISMATCH_B,
    FLAT_PERMUTATION_ROUNDS,
    METRIC_APPROX_LOWER,
    METRIC_BYTES_LOWER,
    METRIC_EXACT_HIGHER,
    METRIC_EXACT_LOWER,
    NOISY_PERMUTATION_A,
    NOISY_PERMUTATION_B,
    OVERFLOWING_MEDIAN,
    SUBNORMAL_BASELINE,
    create_samples,
    get_band,
    get_permutation,
    meta_of,
    run,
    samples,
)

# ---------------------------------------------------------------------------
# Delta computation
# ---------------------------------------------------------------------------


def test_compute_verdicts_when_multiple_rounds_does_compute_delta_from_medians():
    result = run(
        samples(90.0, 100.0, 200.0),
        samples(85.0, 95.0, 105.0),
        METRIC_EXACT_LOWER,
    )

    assert result["metric"].delta == pytest.approx(-5.0, abs=1e-5)


# ---------------------------------------------------------------------------
# Exact path
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("meta", "samples_b", "expected_verdict"),
    [
        pytest.param(METRIC_EXACT_HIGHER, samples(105.0), "improved", id="higher-improved"),
        pytest.param(METRIC_EXACT_LOWER, samples(105.0), "regressed", id="lower-regressed"),
        pytest.param(METRIC_EXACT_LOWER, samples(100.0), "no-signal", id="unchanged"),
    ],
)
def test_compute_verdicts_when_exact_does_classify_by_direction(
    meta: dict[str, MetricMeta],
    samples_b: list[dict[str, float]],
    expected_verdict: str,
):
    result = run(samples(100.0), samples_b, meta)

    verdict = result["metric"]
    assert (verdict.method, verdict.verdict) == ("exact", expected_verdict)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_compute_verdicts_when_baseline_median_zero_does_report_nan_delta():
    result = run(samples(0.0), samples(5.0), METRIC_EXACT_LOWER)

    assert math.isnan(result["metric"].delta)


def test_compute_verdicts_when_baseline_median_zero_does_report_no_signal():
    result = run(samples(0.0), samples(5.0), METRIC_EXACT_LOWER)

    assert result["metric"].verdict == "no-signal"


# ---------------------------------------------------------------------------
# Permutation method
# ---------------------------------------------------------------------------


def test_compute_verdicts_when_permutation_delta_zero_does_report_no_signal():
    samples_a = samples(80.0, 90.0, 95.0, 100.0, 100.0, 105.0, 110.0, 120.0)
    samples_b = samples(81.0, 91.0, 96.0, 100.0, 100.0, 106.0, 111.0, 121.0)

    result = run(samples_a, samples_b, METRIC_APPROX_LOWER)

    verdict = get_permutation(result)
    assert verdict.p < 0.05
    assert (verdict.verdict, verdict.reason) == ("no-signal", None)


def test_compute_verdicts_when_permutation_delta_not_finite_does_report_no_signal():
    result = run(
        create_samples(FLAT_PERMUTATION_ROUNDS, SUBNORMAL_BASELINE),
        create_samples(FLAT_PERMUTATION_ROUNDS, 100.0),
        METRIC_APPROX_LOWER,
    )

    verdict = get_permutation(result)
    assert (verdict.verdict, verdict.reason) == ("no-signal", "non-finite-delta")


def test_compute_verdicts_when_band_delta_not_finite_does_report_no_signal():
    result = run(create_samples(2, SUBNORMAL_BASELINE), samples(100.0, 100.0), METRIC_APPROX_LOWER)

    verdict = get_band(result)
    assert (verdict.verdict, verdict.reason, verdict.usable_n) == (
        "no-signal",
        "non-finite-delta",
        2,
    )


# ---------------------------------------------------------------------------
# Why a no-signal verdict was withheld
# ---------------------------------------------------------------------------

# Ten windows whose paired differences cancel out (+-10, +-5, +-4, +-3, and +2
# against -1), so the sign-flip p-value is far from significant while the
# candidate median sits 0.5% above the baseline's.
_CANCELLING_A = create_samples(10, 100.0)
_CANCELLING_B = samples(90.0, 110.0, 95.0, 105.0, 96.0, 104.0, 102.0, 99.0, 97.0, 103.0)


@pytest.mark.parametrize(
    ("meta", "samples_a", "samples_b", "expected_reason"),
    [
        # The p-value is not significant either; the step floor outranks it.
        pytest.param(
            METRIC_BYTES_LOWER,
            _CANCELLING_A,
            _CANCELLING_B,
            "resolution",
            id="permutation-resolution-over-noise",
        ),
        pytest.param(
            METRIC_APPROX_LOWER,
            _CANCELLING_A,
            _CANCELLING_B,
            "noise",
            id="permutation-not-significant",
        ),
        # Nine of ten pairs rise yet the candidate median sits 2% lower: the
        # p-value is significant and nothing floors the delta.
        pytest.param(
            METRIC_APPROX_LOWER,
            DIRECTION_MISMATCH_A,
            DIRECTION_MISMATCH_B,
            "direction-mismatch",
            id="permutation-direction-mismatch",
        ),
    ],
)
def test_compute_verdicts_when_permutation_withheld_does_report_first_withheld_reason(
    meta: dict[str, MetricMeta],
    samples_a: list[dict[str, float]],
    samples_b: list[dict[str, float]],
    expected_reason: str,
):
    result = run(samples_a, samples_b, meta)

    verdict = get_permutation(result)
    assert (verdict.verdict, verdict.reason) == ("no-signal", expected_reason)


# ---------------------------------------------------------------------------
# Band method
# ---------------------------------------------------------------------------


def test_compute_verdicts_when_band_fallback_does_report_usable_n():
    samples_b = samples(100.0, 100.0, 100.0, 95.0, 90.0, 105.0)

    result = run(create_samples(6, 100.0), samples_b, METRIC_APPROX_LOWER)

    verdict = get_band(result)
    assert (verdict.n, verdict.usable_n) == (6, 3)


@pytest.mark.parametrize(
    ("samples_a", "samples_b", "expected"),
    [
        pytest.param(
            samples(100.0, 110.0), samples(30.0, 50.0), ("improved", None), id="exceeds-band"
        ),
        pytest.param(
            create_samples(2, 100.0),
            samples(99.8, 99.6),
            ("no-signal", "noise"),
            id="within-band",
        ),
    ],
)
def test_compute_verdicts_when_band_does_signal_only_beyond_noise_band(
    samples_a: list[dict[str, float]],
    samples_b: list[dict[str, float]],
    expected: tuple[str, str | None],
):
    result = run(samples_a, samples_b, METRIC_APPROX_LOWER)

    verdict = get_band(result)
    assert (verdict.verdict, verdict.reason) == expected


def test_compute_verdicts_when_band_fewer_differing_than_min_n_does_report_no_signal():
    result = run(samples(100.0, 100.0), samples(100.0, 210.0), METRIC_APPROX_LOWER)

    verdict = get_band(result)
    assert verdict.usable_n == 1
    assert (verdict.verdict, verdict.reason) == ("no-signal", "noise")


# ---------------------------------------------------------------------------
# Noise-band carrying on non-exact verdicts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("samples_a", "samples_b", "expected_noise_pct"),
    [
        pytest.param(samples(80.0, 120.0), samples(90.0, 110.0), 30.0, id="band-spread"),
        pytest.param(create_samples(2, 100.0), samples(100.0, 100.1), 0.5, id="band-floor"),
        # The candidate's relative spread sets the percentage although the
        # baseline has the larger half-range.
        pytest.param(
            samples(180.0, 220.0),
            samples(40.0, 60.0),
            30.0,
            id="band-pct-and-abs-from-different-sides",
        ),
    ],
)
def test_compute_verdicts_when_band_does_report_noise_pct_from_wider_relative_spread_above_floor(
    samples_a: list[dict[str, float]],
    samples_b: list[dict[str, float]],
    expected_noise_pct: float,
):
    result = run(samples_a, samples_b, METRIC_APPROX_LOWER)

    verdict = get_band(result)
    assert verdict.noise_pct == pytest.approx(expected_noise_pct, abs=1e-5)


@pytest.mark.parametrize(
    ("samples_a", "samples_b", "expected_noise_abs"),
    [
        pytest.param(samples(80.0, 120.0), samples(90.0, 110.0), 30.0, id="baseline-larger"),
        # The percentage floor does not floor the absolute band.
        pytest.param(
            create_samples(2, 100.0), samples(100.0, 100.1), 0.075, id="pct-floor-left-out"
        ),
        # The baseline's larger half-range sets the absolute band while the
        # candidate's relative spread sets the percentage.
        pytest.param(
            samples(180.0, 220.0),
            samples(40.0, 60.0),
            30.0,
            id="band-pct-and-abs-from-different-sides",
        ),
    ],
)
def test_compute_verdicts_when_band_does_report_noise_abs_from_larger_half_range(
    samples_a: list[dict[str, float]],
    samples_b: list[dict[str, float]],
    expected_noise_abs: float,
):
    result = run(samples_a, samples_b, METRIC_APPROX_LOWER)

    verdict = get_band(result)
    assert verdict.noise_abs == pytest.approx(expected_noise_abs, abs=1e-5)


# ---------------------------------------------------------------------------
# Unstable threshold
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("unstable_noise_pct", "expected"),
    [
        pytest.param(20.0, "unstable", id="above-threshold"),
        pytest.param(30.0, "improved", id="at-threshold"),
    ],
)
def test_compute_verdicts_when_threshold_given_does_mark_unstable_only_above_it(
    unstable_noise_pct: float,
    expected: str,
):
    result = run(
        NOISY_PERMUTATION_A,
        NOISY_PERMUTATION_B,
        METRIC_APPROX_LOWER,
        unstable_noise_pct=unstable_noise_pct,
    )

    verdict = get_permutation(result)
    assert verdict.verdict == expected


# ---------------------------------------------------------------------------
# Zero-median non-exact reports unstable
# ---------------------------------------------------------------------------

# When one side's median is 0 but that side has non-zero half-range, the noise
# fraction is undefined (division by zero). Rather than letting noise_pct reach
# inf, the engine caps it and forces the verdict to unstable on both paths.


_ZERO_MEDIAN_WITH_SPREAD = [
    pytest.param([-5.0, 5.0], [10.0, 10.0], id="baseline-zero"),
    pytest.param([10.0, 10.0], [-5.0, 5.0], id="candidate-zero"),
]


@pytest.mark.parametrize(("values_a", "values_b"), _ZERO_MEDIAN_WITH_SPREAD)
def test_compute_verdicts_when_zero_median_with_spread_does_report_unstable(
    values_a: list[float],
    values_b: list[float],
):
    result = run(samples(*values_a), samples(*values_b), METRIC_APPROX_LOWER)

    verdict = get_band(result)
    assert (verdict.verdict, verdict.reason) == ("unstable", None)


@pytest.mark.parametrize(("values_a", "values_b"), _ZERO_MEDIAN_WITH_SPREAD)
def test_compute_verdicts_when_zero_median_with_spread_does_cap_the_noise_band(
    values_a: list[float],
    values_b: list[float],
):
    result = run(samples(*values_a), samples(*values_b), METRIC_APPROX_LOWER)

    verdict = get_band(result)
    assert verdict.noise_pct == pytest.approx(0.5, abs=1e-5)


# ---------------------------------------------------------------------------
# Warning when paired windows are dropped
# ---------------------------------------------------------------------------


_ONE_SIDED_A = samples(100.0, 100.0, 100.0, 90.0)
_ONE_SIDED_B = [{"metric": 95.0}, {"other": 1.0}, {"other": 1.0}, {"metric": 85.0}]


def test_compute_verdicts_when_some_windows_one_sided_does_drop_them():
    result = run(_ONE_SIDED_A, _ONE_SIDED_B, METRIC_EXACT_LOWER)

    assert result["metric"].n == 2


def test_compute_verdicts_when_some_windows_one_sided_does_warn_once_naming_the_dropped_count():
    collected: list[str] = []

    run(_ONE_SIDED_A, _ONE_SIDED_B, METRIC_EXACT_LOWER, warn=collected.append)

    assert len(collected) == 1
    assert collected[0].startswith("warning: metric:")
    assert re.search(r"\b2\b", collected[0])


def test_compute_verdicts_when_metric_fully_one_sided_does_not_warn():
    collected: list[str] = []

    result = run(
        [{"metricA": 100.0}],
        [{"metricB": 95.0}],
        {
            "metricA": meta_of("lower", exact=True),
            "metricB": meta_of("lower", exact=True),
        },
        warn=collected.append,
    )

    assert result == {}
    assert collected == []


def test_compute_verdicts_when_cleanly_paired_does_not_warn():
    collected: list[str] = []

    run(samples(100.0, 90.0), samples(95.0, 85.0), METRIC_EXACT_LOWER, warn=collected.append)

    assert collected == []


# ---------------------------------------------------------------------------
# Detectable change
# ---------------------------------------------------------------------------


def _values(rounds: list[dict[str, float]]) -> list[float]:
    return [window["metric"] for window in rounds]


def test_compute_verdicts_when_permutation_does_carry_detectable_shift_over_baseline_median():
    baseline = _values(NOISY_PERMUTATION_A)
    shift = detectable_shift(baseline, _values(NOISY_PERMUTATION_B))

    result = run(NOISY_PERMUTATION_A, NOISY_PERMUTATION_B, METRIC_APPROX_LOWER)

    verdict = get_permutation(result)
    assert verdict.detectable_pct == pytest.approx(
        shift / abs(statistics.median(baseline)) * 100, abs=1e-9
    )


def test_compute_verdicts_when_band_does_carry_band_its_delta_had_to_clear():
    result = run(create_samples(3, 100.0), samples(100.0, 95.0, 90.0), METRIC_APPROX_LOWER)

    verdict = get_band(result)
    assert verdict.detectable_pct == pytest.approx(150 * 5 / 95, abs=1e-5)


@pytest.mark.parametrize(
    ("narrow", "meta", "samples_a", "samples_b"),
    [
        pytest.param(
            get_band,
            METRIC_APPROX_LOWER,
            samples(100.0, 100.0),
            samples(100.0, 210.0),
            id="band-one-pair-differs",
        ),
        pytest.param(
            get_band,
            METRIC_APPROX_LOWER,
            samples(0.0, 0.0),
            samples(1.0, 3.0),
            id="band-zero-baseline-median-without-spread",
        ),
        pytest.param(
            get_band,
            METRIC_APPROX_LOWER,
            create_samples(3, 10.0),
            samples(-5.0, OVERFLOWING_MEDIAN, 5.0),
            id="band-overflowing-band",
        ),
        pytest.param(
            get_permutation,
            METRIC_APPROX_LOWER,
            samples(-5.0, -3.0, -1.0, 1.0, 3.0, 5.0),
            create_samples(6, 10.0),
            id="permutation-zero-baseline-median",
        ),
        pytest.param(
            get_permutation,
            METRIC_BYTES_LOWER,
            create_samples(FLAT_PERMUTATION_ROUNDS, OVERFLOWING_MEDIAN),
            create_samples(FLAT_PERMUTATION_ROUNDS, 1e-300),
            id="permutation-overflowing-step-floor",
        ),
    ],
)
def test_compute_verdicts_when_no_finite_detectable_pct_does_carry_none(
    narrow: Callable[[dict[str, MetricVerdict]], BandVerdict | PermutationVerdict],
    meta: dict[str, MetricMeta],
    samples_a: list[dict[str, float]],
    samples_b: list[dict[str, float]],
):
    result = run(samples_a, samples_b, meta)

    verdict = narrow(result)
    assert verdict.detectable_pct is None
