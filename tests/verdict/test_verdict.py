"""Behavioral tests for the verdict engine.

Covers pairing, delta computation, and per-metric method dispatch (exact,
permutation, band), driving that behavior through the public ``compute_verdicts``
API only.
"""

import math
from collections.abc import Callable

import pytest

from gymrat.model import (
    BandVerdict,
    ExactVerdict,
    MetricMeta,
    MetricVerdict,
    PermutationVerdict,
)
from gymrat.verdict import compute_verdicts
from tests.verdict._engine import (
    BYTE_ONE_MOVE_A,
    BYTE_ONE_MOVE_B,
    FLAT_PERMUTATION_ROUNDS,
    METRIC_APPROX_HIGHER,
    METRIC_APPROX_LOWER,
    METRIC_BYTES_HIGHER,
    METRIC_BYTES_LOWER,
    METRIC_BYTES_NO_RESOLUTION_LOWER,
    METRIC_EXACT_BYTES_LOWER,
    METRIC_EXACT_HIGHER,
    METRIC_EXACT_LOWER,
    METRIC_EXACT_TIMER_LOWER,
    METRIC_NS_LOWER,
    METRIC_TIMER_LOWER,
    NOISY_BAND_A,
    NOISY_BAND_B,
    NOISY_PERMUTATION_A,
    NOISY_PERMUTATION_B,
    SEPARATED_HIGH,
    SEPARATED_LOW,
    TIMER_TICK_NS,
    create_samples,
    get_band,
    get_permutation,
    meta_of,
    noop_warn,
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
    ("meta", "samples_b", "expected_delta", "expected_verdict"),
    [
        pytest.param(METRIC_EXACT_HIGHER, samples(105.0), 5.0, "improved", id="higher-improved"),
        pytest.param(METRIC_EXACT_LOWER, samples(105.0), 5.0, "regressed", id="lower-regressed"),
        pytest.param(METRIC_EXACT_HIGHER, samples(95.0), -5.0, "regressed", id="higher-regressed"),
        pytest.param(METRIC_EXACT_LOWER, samples(100.01), 0.01, "regressed", id="tiny-difference"),
        pytest.param(METRIC_EXACT_LOWER, samples(95.0), -5.0, "improved", id="lower-improved"),
        pytest.param(METRIC_EXACT_LOWER, samples(100.0), 0.0, "no-signal", id="unchanged"),
        pytest.param(
            METRIC_EXACT_BYTES_LOWER, samples(101.0), 1.0, "regressed", id="ignores-byte-step"
        ),
        pytest.param(
            METRIC_EXACT_TIMER_LOWER, samples(105.0), 5.0, "regressed", id="ignores-declared-step"
        ),
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
    assert (verdict.method, verdict.verdict, verdict.n) == ("exact", expected_verdict, 1)
    assert verdict.delta == pytest.approx(expected_delta, abs=1e-5)


# ---------------------------------------------------------------------------
# Multiple metrics
# ---------------------------------------------------------------------------


def test_compute_verdicts_when_metrics_differ_in_exactness_does_respect_per_metric_flag():
    result = run(
        [{"exactMetric": 100.0, "otherMetric": 50.0}],
        [{"exactMetric": 100.001, "otherMetric": 45.0}],
        {
            "exactMetric": meta_of("lower", exact=True),
            "otherMetric": meta_of("lower"),
        },
    )

    exact = result["exactMetric"]
    assert exact.verdict == "regressed"
    assert exact.method == "exact"

    other = result["otherMetric"]
    assert other.method == "band"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_compute_verdicts_when_baseline_median_zero_does_report_nan_delta_with_no_signal():
    result = run(samples(0.0), samples(5.0), METRIC_EXACT_LOWER)

    assert math.isnan(result["metric"].delta)
    assert result["metric"].verdict == "no-signal"


# ---------------------------------------------------------------------------
# Permutation method
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "meta",
    [
        pytest.param(METRIC_APPROX_LOWER, id="lower"),
        pytest.param(METRIC_APPROX_HIGHER, id="higher"),
    ],
)
def test_compute_verdicts_when_permutation_delta_zero_does_report_no_signal(
    meta: dict[str, MetricMeta],
):
    samples_a = samples(80.0, 90.0, 95.0, 100.0, 100.0, 105.0, 110.0, 120.0)
    samples_b = samples(81.0, 91.0, 96.0, 100.0, 100.0, 106.0, 111.0, 121.0)

    result = run(samples_a, samples_b, meta)

    verdict = get_permutation(result)
    assert verdict.delta == 0.0
    assert verdict.p < 0.05
    assert verdict.verdict == "no-signal"


def test_compute_verdicts_when_paired_direction_opposes_delta_does_report_no_signal():
    # Nine of ten pairs rise, so the signed ranks favor an increase, yet the
    # candidate median sits 2% below the baseline median.
    samples_a = create_samples(4, 0.0) + create_samples(6, 10.0)
    samples_b = create_samples(4, 9.0) + samples(9.5) + create_samples(5, 10.1)

    result = run(samples_a, samples_b, METRIC_APPROX_LOWER)

    verdict = get_permutation(result)
    assert verdict.p < 0.05
    assert verdict.delta == pytest.approx(-2.0, abs=1e-5)
    assert verdict.verdict == "no-signal"


@pytest.mark.parametrize(
    ("baseline", "candidate"),
    [
        pytest.param(5e-324, 100.0, id="overflow-positive"),
        pytest.param(5e-324, -100.0, id="overflow-negative"),
        pytest.param(0.0, 100.0, id="zero-baseline"),
    ],
)
def test_compute_verdicts_when_permutation_delta_not_finite_does_report_no_signal(
    baseline: float,
    candidate: float,
):
    result = run(
        create_samples(FLAT_PERMUTATION_ROUNDS, baseline),
        create_samples(FLAT_PERMUTATION_ROUNDS, candidate),
        METRIC_APPROX_LOWER,
    )

    verdict = get_permutation(result)
    assert not math.isfinite(verdict.delta)
    assert verdict.verdict == "no-signal"


@pytest.mark.parametrize(
    ("meta", "samples_a", "samples_b", "expected_verdict"),
    [
        pytest.param(
            METRIC_APPROX_LOWER, SEPARATED_HIGH, SEPARATED_LOW, "improved", id="lower-improved"
        ),
        pytest.param(
            METRIC_APPROX_LOWER, SEPARATED_LOW, SEPARATED_HIGH, "regressed", id="lower-regressed"
        ),
        pytest.param(
            METRIC_APPROX_HIGHER, SEPARATED_LOW, SEPARATED_HIGH, "improved", id="higher-improved"
        ),
        pytest.param(
            METRIC_APPROX_HIGHER,
            SEPARATED_HIGH,
            SEPARATED_LOW,
            "regressed",
            id="higher-regressed",
        ),
        # Flat rounds (every rank tied) pin the all-tied-ranks path.
        pytest.param(
            METRIC_NS_LOWER,
            create_samples(FLAT_PERMUTATION_ROUNDS, 333.0),
            create_samples(FLAT_PERMUTATION_ROUNDS, 417.0),
            "regressed",
            id="flat-increase",
        ),
        pytest.param(
            METRIC_NS_LOWER,
            create_samples(FLAT_PERMUTATION_ROUNDS, 791.0),
            create_samples(FLAT_PERMUTATION_ROUNDS, 625.0),
            "improved",
            id="flat-decrease",
        ),
    ],
)
def test_compute_verdicts_when_permutation_direction_varies_does_classify(
    meta: dict[str, MetricMeta],
    samples_a: list[dict[str, float]],
    samples_b: list[dict[str, float]],
    expected_verdict: str,
):
    result = run(samples_a, samples_b, meta)

    assert get_permutation(result).verdict == expected_verdict


# ---------------------------------------------------------------------------
# Band method
# ---------------------------------------------------------------------------


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
    ("meta", "samples_a", "samples_b", "expected_delta", "expected"),
    [
        pytest.param(
            METRIC_APPROX_LOWER,
            samples(100.0, 110.0),
            samples(30.0, 50.0),
            (40 - 105) / 105 * 100,
            "improved",
            id="exceeds-band",
        ),
        pytest.param(
            METRIC_APPROX_LOWER,
            create_samples(2, 100.0),
            samples(99.8, 99.6),
            -0.3,
            "no-signal",
            id="within-band",
        ),
        # Candidate median 110 with half-range 50: a +10% delta inside a
        # 1.5 * 50 / 110 ~= 68.2% band.
        pytest.param(
            METRIC_APPROX_LOWER,
            create_samples(2, 100.0),
            samples(60.0, 160.0),
            10.0,
            "no-signal",
            id="within-spread-band",
        ),
        pytest.param(
            METRIC_APPROX_HIGHER,
            create_samples(2, 50.0),
            create_samples(2, 100.0),
            100.0,
            "improved",
            id="higher-improved",
        ),
        pytest.param(
            METRIC_APPROX_HIGHER,
            create_samples(2, 100.0),
            create_samples(2, 50.0),
            -50.0,
            "regressed",
            id="higher-regressed",
        ),
    ],
)
def test_compute_verdicts_when_band_does_classify(
    meta: dict[str, MetricMeta],
    samples_a: list[dict[str, float]],
    samples_b: list[dict[str, float]],
    expected_delta: float,
    expected: str,
):
    result = run(samples_a, samples_b, meta)

    verdict = result["metric"]
    assert verdict.method == "band"
    assert verdict.delta == pytest.approx(expected_delta, abs=1e-5)
    assert verdict.verdict == expected


def test_compute_verdicts_when_band_fewer_differing_than_min_n_does_report_no_signal():
    result = run(samples(100.0, 100.0), samples(100.0, 210.0), METRIC_APPROX_LOWER)

    verdict = get_band(result)
    assert verdict.n == 2
    assert verdict.usable_n == 1
    assert verdict.verdict == "no-signal"


# ---------------------------------------------------------------------------
# Noise-band carrying on non-exact verdicts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("narrow", "samples_a", "samples_b", "expected_noise_pct", "expected_noise_abs"),
    [
        pytest.param(
            get_band, samples(80.0, 120.0), samples(90.0, 110.0), 30.0, 30.0, id="band-spread"
        ),
        pytest.param(
            get_band, create_samples(2, 100.0), samples(100.0, 100.1), 0.5, 0.075, id="band-floor"
        ),
        # The candidate side's spread (half-range 50 over median 110) sets the band.
        pytest.param(
            get_band,
            create_samples(2, 100.0),
            samples(60.0, 160.0),
            150 * 50 / 110,
            75.0,
            id="band-candidate-spread",
        ),
        pytest.param(
            get_band,
            samples(-60.0, -40.0),
            samples(-50.0, -50.0),
            30.0,
            15.0,
            id="band-negative-median",
        ),
        pytest.param(
            get_band,
            samples(180.0, 220.0),
            samples(90.0, 110.0),
            15.0,
            30.0,
            id="band-abs-from-larger-half-range",
        ),
        pytest.param(
            get_permutation,
            samples(-60.0, -55.0, -50.0, -50.0, -45.0, -40.0),
            samples(-61.0, -56.0, -51.0, -51.0, -46.0, -41.0),
            30.0,
            15.0,
            id="permutation-negative-median",
        ),
        pytest.param(
            get_permutation,
            SEPARATED_HIGH,
            SEPARATED_LOW,
            15.0,
            30.0,
            id="permutation-abs-from-larger-half-range",
        ),
    ],
)
def test_compute_verdicts_when_non_exact_inputs_vary_does_report_noise_pct_and_abs(
    narrow: Callable[[dict[str, MetricVerdict]], BandVerdict | PermutationVerdict],
    samples_a: list[dict[str, float]],
    samples_b: list[dict[str, float]],
    expected_noise_pct: float,
    expected_noise_abs: float,
):
    result = run(samples_a, samples_b, METRIC_APPROX_LOWER)

    verdict = narrow(result)
    assert (verdict.noise_pct, verdict.noise_abs) == (
        pytest.approx(expected_noise_pct, abs=1e-5),
        pytest.approx(expected_noise_abs, abs=1e-5),
    )


# ---------------------------------------------------------------------------
# Measurement-step floor
# ---------------------------------------------------------------------------


# Six windows whose medians are 100 vs 75 bytes — a 25-byte move that clears the
# whole-byte floor — again shown by every pair, which keeps p significant.
_BYTE_CLEARS_A = samples(90.0, 95.0, 100.0, 100.0, 105.0, 110.0)
_BYTE_CLEARS_B = samples(67.0, 71.0, 75.0, 75.0, 79.0, 83.0)


def test_compute_verdicts_when_byte_move_is_one_byte_does_report_no_signal_on_band():
    result = run(create_samples(2, 4.0), create_samples(2, 3.0), METRIC_BYTES_LOWER)

    verdict = get_band(result)
    assert verdict.delta == pytest.approx(-25.0, abs=1e-5)
    assert verdict.noise_pct == pytest.approx(31.25, abs=1e-5)
    assert verdict.verdict == "no-signal"


@pytest.mark.parametrize(
    ("meta", "samples_a", "samples_b", "expected_noise_pct", "expected_delta", "expected_verdict"),
    [
        pytest.param(
            METRIC_BYTES_LOWER,
            BYTE_ONE_MOVE_A,
            BYTE_ONE_MOVE_B,
            31.25,
            -25.0,
            "no-signal",
            id="one-byte-lower",
        ),
        pytest.param(
            METRIC_BYTES_HIGHER,
            BYTE_ONE_MOVE_A,
            BYTE_ONE_MOVE_B,
            31.25,
            -25.0,
            "no-signal",
            id="one-byte-higher",
        ),
        pytest.param(
            METRIC_BYTES_LOWER,
            _BYTE_CLEARS_A,
            _BYTE_CLEARS_B,
            16.0,
            -25.0,
            "improved",
            id="clears-lower",
        ),
        pytest.param(
            METRIC_BYTES_HIGHER,
            _BYTE_CLEARS_A,
            _BYTE_CLEARS_B,
            16.0,
            -25.0,
            "regressed",
            id="clears-higher",
        ),
        pytest.param(
            METRIC_BYTES_LOWER,
            create_samples(FLAT_PERMUTATION_ROUNDS, 4.0),
            create_samples(FLAT_PERMUTATION_ROUNDS, 5.0),
            31.25,
            25.0,
            "no-signal",
            id="one-byte-increase",
        ),
        # The floor is 1.25 bytes over the baseline median (31.25% of 4 B) in
        # either direction, so a 4 -> 2 move (50%) clears it.
        pytest.param(
            METRIC_BYTES_LOWER,
            create_samples(FLAT_PERMUTATION_ROUNDS, 4.0),
            create_samples(FLAT_PERMUTATION_ROUNDS, 2.0),
            31.25,
            -50.0,
            "improved",
            id="two-byte-decrease",
        ),
    ],
)
def test_compute_verdicts_when_byte_permutation_does_give_signal_only_beyond_byte_floor(
    meta: dict[str, MetricMeta],
    samples_a: list[dict[str, float]],
    samples_b: list[dict[str, float]],
    *,
    expected_noise_pct: float,
    expected_delta: float,
    expected_verdict: str,
):
    result = run(samples_a, samples_b, meta)

    verdict = get_permutation(result)
    assert verdict.p < 0.05
    assert verdict.delta == pytest.approx(expected_delta, abs=1e-5)
    assert verdict.noise_pct == pytest.approx(expected_noise_pct, abs=1e-5)
    assert verdict.verdict == expected_verdict


@pytest.mark.parametrize(
    "meta",
    [
        pytest.param(METRIC_NS_LOWER, id="ns"),
        pytest.param(METRIC_APPROX_LOWER, id="none"),
        pytest.param(METRIC_BYTES_NO_RESOLUTION_LOWER, id="bytes"),
    ],
)
@pytest.mark.parametrize(
    ("narrow", "samples_a", "samples_b", "expected_noise_pct"),
    [
        # The -25% delta sits inside the 30% band, so this row also pins that the
        # permutation path never compares the delta to the noise band.
        pytest.param(get_permutation, BYTE_ONE_MOVE_A, BYTE_ONE_MOVE_B, 30.0, id="permutation"),
        pytest.param(get_band, create_samples(2, 4.0), create_samples(2, 3.0), 0.5, id="band"),
    ],
)
def test_compute_verdicts_when_resolution_is_none_does_take_no_step_floor(
    meta: dict[str, MetricMeta],
    narrow: Callable[[dict[str, MetricVerdict]], BandVerdict | PermutationVerdict],
    samples_a: list[dict[str, float]],
    samples_b: list[dict[str, float]],
    expected_noise_pct: float,
):
    result = run(samples_a, samples_b, meta)

    verdict = narrow(result)
    assert verdict.noise_pct == pytest.approx(expected_noise_pct, abs=1e-5)
    assert verdict.verdict == "improved"


def test_compute_verdicts_when_byte_floor_side_median_zero_does_exclude_that_side():
    result = run(create_samples(2, 4.0), create_samples(2, 0.0), METRIC_BYTES_LOWER)

    verdict = get_band(result)
    assert verdict.noise_pct == pytest.approx(31.25, abs=1e-5)
    assert verdict.verdict == "improved"


@pytest.mark.parametrize(
    ("values_b", "expected_band", "expected_verdict"),
    [
        pytest.param([1_000_000.0, 1_000_100.0], 0.5, "no-signal", id="stable"),
        pytest.param([800_000.0, 1_200_000.0], 30.0, "no-signal", id="wide"),
        pytest.param([999_000.0, 999_000.0], 0.5, "no-signal", id="stable-move-inside-band"),
        pytest.param([990_000.0, 990_000.0], 0.5, "improved", id="stable-move-past-band"),
        pytest.param([700_000.0, 1_100_000.0], 100 / 3, "no-signal", id="wide-move-inside-band"),
    ],
)
def test_compute_verdicts_when_byte_metric_is_megabyte_scale_does_keep_band(
    values_b: list[float],
    expected_band: float,
    expected_verdict: str,
):
    result = run(create_samples(2, 1_000_000.0), samples(*values_b), METRIC_BYTES_LOWER)

    verdict = get_band(result)
    assert verdict.noise_pct == pytest.approx(expected_band, abs=1e-5)
    assert verdict.verdict == expected_verdict


# Medians one tick apart (333 -> 375 ns) sit inside the 1.25-step floor; two
# ticks apart (333 -> 417 ns) clear it.
_TIMER_STEP_FLOOR_PCT = 1.25 * TIMER_TICK_NS / 333 * 100


@pytest.mark.parametrize(
    ("narrow", "meta", "samples_a", "samples_b", "expected_noise_pct", "expected_verdict"),
    [
        pytest.param(
            get_permutation,
            METRIC_TIMER_LOWER,
            create_samples(FLAT_PERMUTATION_ROUNDS, 332.9999999550637),
            create_samples(FLAT_PERMUTATION_ROUNDS, 374.99999998),
            _TIMER_STEP_FLOOR_PCT,
            "no-signal",
            id="timer-one-tick-permutation",
        ),
        pytest.param(
            get_band,
            METRIC_TIMER_LOWER,
            create_samples(3, 333.0),
            create_samples(3, 375.0),
            _TIMER_STEP_FLOOR_PCT,
            "no-signal",
            id="timer-one-tick-band",
        ),
        pytest.param(
            get_permutation,
            METRIC_TIMER_LOWER,
            create_samples(FLAT_PERMUTATION_ROUNDS, 333.0),
            create_samples(FLAT_PERMUTATION_ROUNDS, 417.0),
            _TIMER_STEP_FLOOR_PCT,
            "regressed",
            id="timer-two-ticks-permutation",
        ),
        pytest.param(
            get_band,
            METRIC_TIMER_LOWER,
            create_samples(3, 333.0),
            create_samples(3, 417.0),
            _TIMER_STEP_FLOOR_PCT,
            "regressed",
            id="timer-two-ticks-band",
        ),
        pytest.param(
            get_band,
            METRIC_BYTES_LOWER,
            create_samples(2, 100.0),
            create_samples(2, 75.0),
            1.25,
            "improved",
            id="bytes-without-scatter",
        ),
    ],
)
def test_compute_verdicts_when_resolution_set_does_floor_band_at_one_and_a_quarter_steps(
    narrow: Callable[[dict[str, MetricVerdict]], BandVerdict | PermutationVerdict],
    meta: dict[str, MetricMeta],
    *,
    samples_a: list[dict[str, float]],
    samples_b: list[dict[str, float]],
    expected_noise_pct: float,
    expected_verdict: str,
):
    result = run(samples_a, samples_b, meta)

    verdict = narrow(result)
    assert verdict.noise_pct == pytest.approx(expected_noise_pct, abs=1e-5)
    assert verdict.verdict == expected_verdict


# ---------------------------------------------------------------------------
# Unstable threshold
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("samples_a", "samples_b", "method", "unstable_noise_pct", "expected"),
    [
        pytest.param(
            NOISY_PERMUTATION_A,
            NOISY_PERMUTATION_B,
            "permutation",
            20.0,
            "unstable",
            id="permutation-unstable",
        ),
        pytest.param(
            NOISY_PERMUTATION_A,
            NOISY_PERMUTATION_B,
            "permutation",
            30.0,
            "improved",
            id="permutation-improved",
        ),
        pytest.param(NOISY_BAND_A, NOISY_BAND_B, "band", 20.0, "unstable", id="band-unstable"),
        pytest.param(NOISY_BAND_A, NOISY_BAND_B, "band", 30.0, "improved", id="band-improved"),
    ],
)
def test_compute_verdicts_when_threshold_given_does_mark_unstable_only_above_it(
    samples_a: list[dict[str, float]],
    samples_b: list[dict[str, float]],
    method: str,
    unstable_noise_pct: float,
    expected: str,
):
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
    result = compute_verdicts(
        create_samples(3, 100.0), samples_b, METRIC_APPROX_LOWER, warn=noop_warn
    )

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


# Below 0.625 bytes the baseline's 1.25-byte step floor exceeds the default 200%
# unstable band, on both paths, even when the floor is too large to express as a
# percentage; the candidate's median plays no part in the floor.
@pytest.mark.parametrize(
    ("rounds", "baseline", "candidate", "expected"),
    [
        pytest.param(2, 0.6, 0.7, "unstable", id="small-baseline-band"),
        pytest.param(
            FLAT_PERMUTATION_ROUNDS, 0.6, 0.7, "unstable", id="small-baseline-permutation"
        ),
        pytest.param(2, 1e-310, 1e-300, "unstable", id="overflowing-baseline-band"),
        pytest.param(
            FLAT_PERMUTATION_ROUNDS,
            1e-310,
            1e-300,
            "unstable",
            id="overflowing-baseline-permutation",
        ),
        pytest.param(2, 4.0, 0.4, "improved", id="small-candidate"),
    ],
)
def test_compute_verdicts_when_byte_median_is_small_does_mark_unstable_only_from_baseline(
    rounds: int,
    baseline: float,
    candidate: float,
    expected: str,
):
    result = run(
        create_samples(rounds, baseline), create_samples(rounds, candidate), METRIC_BYTES_LOWER
    )

    assert result["metric"].verdict == expected


# ---------------------------------------------------------------------------
# Zero-median non-exact reports unstable
# ---------------------------------------------------------------------------

# When one side's median is 0 but that side has non-zero half-range, the noise
# fraction is undefined (division by zero). Rather than letting noise_pct reach
# inf, the engine caps it and forces the verdict to unstable on both paths.


@pytest.mark.parametrize(
    ("narrow", "values_a", "values_b"),
    [
        pytest.param(get_band, [-5.0, 5.0], [10.0, 10.0], id="band-baseline-zero"),
        pytest.param(get_band, [10.0, 10.0], [-5.0, 5.0], id="band-candidate-zero"),
        pytest.param(
            get_permutation,
            [-5.0, -3.0, -1.0, 1.0, 3.0, 5.0],
            [10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
            id="permutation-baseline-zero",
        ),
        pytest.param(
            get_permutation,
            [10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
            [-5.0, -3.0, -1.0, 1.0, 3.0, 5.0],
            id="permutation-candidate-zero",
        ),
        pytest.param(get_band, [-5.0, 5.0], [-10.0, 10.0], id="band-both-zero"),
        pytest.param(
            get_permutation,
            [-5.0, -3.0, -1.0, 1.0, 3.0, 5.0],
            [-10.0, -6.0, -2.0, 2.0, 6.0, 10.0],
            id="permutation-both-zero",
        ),
    ],
)
def test_compute_verdicts_when_zero_median_with_spread_does_report_unstable(
    narrow: Callable[[dict[str, MetricVerdict]], BandVerdict | PermutationVerdict],
    values_a: list[float],
    values_b: list[float],
):
    result = run(samples(*values_a), samples(*values_b), METRIC_APPROX_LOWER)

    verdict = narrow(result)
    assert verdict.verdict == "unstable"
    assert verdict.noise_pct == pytest.approx(0.5, abs=1e-5)


# A subnormal median: any spread (or the one-byte floor) divided by it overflows
# to infinity.
_OVERFLOWING_MEDIAN = 1e-310

# When the noise ratio overflows, noise_pct stays finite and the noise band falls
# back to the floor or step floor. Where a side has a non-zero half-range the
# verdict is forced unstable with the floor as its noise_pct; the byte-floor row
# has no spread and a tiny candidate, so the baseline's 1.25-byte step floor
# (31.25%) remains. The permutation case keeps both sides' medians equal and
# tiny, so the delta stays finite (zero) and only the noise ratio overflows.


@pytest.mark.parametrize(
    ("narrow", "meta", "tiny_median_pair", "expected_verdict", "expected_noise_pct"),
    [
        pytest.param(
            get_band,
            METRIC_APPROX_LOWER,
            (create_samples(3, 10.0), samples(-5.0, _OVERFLOWING_MEDIAN, 5.0)),
            "unstable",
            0.5,
            id="band-spread",
        ),
        pytest.param(
            get_permutation,
            METRIC_APPROX_LOWER,
            (
                samples(-5.0, -3.0, 0.0, 4e-310, 3.0, 5.0),
                samples(-6.0, -4.0, 1e-310, 3e-310, 4.0, 6.0),
            ),
            "unstable",
            0.5,
            id="permutation-spread",
        ),
        pytest.param(
            get_band,
            METRIC_BYTES_LOWER,
            (create_samples(2, 4.0), create_samples(2, _OVERFLOWING_MEDIAN)),
            "improved",
            31.25,
            id="byte-floor",
        ),
    ],
)
def test_compute_verdicts_when_noise_ratio_overflows_does_fall_back_to_floor(
    narrow: Callable[[dict[str, MetricVerdict]], BandVerdict | PermutationVerdict],
    meta: dict[str, MetricMeta],
    tiny_median_pair: tuple[list[dict[str, float]], list[dict[str, float]]],
    expected_verdict: str,
    expected_noise_pct: float,
):
    result = run(*tiny_median_pair, meta)

    verdict = narrow(result)
    assert verdict.noise_pct == pytest.approx(expected_noise_pct, abs=1e-5)
    assert verdict.verdict == expected_verdict


@pytest.mark.parametrize(
    "meta",
    [
        pytest.param(METRIC_APPROX_LOWER, id="no-unit"),
        pytest.param(METRIC_BYTES_LOWER, id="bytes"),
    ],
)
def test_compute_verdicts_when_both_medians_zero_without_spread_does_report_no_signal_at_floor(
    meta: dict[str, MetricMeta],
):
    result = run(create_samples(2, 0.0), create_samples(2, 0.0), meta)

    verdict = get_band(result)
    assert verdict.verdict == "no-signal"
    assert verdict.noise_pct == pytest.approx(0.5, abs=1e-5)


# ---------------------------------------------------------------------------
# Warning when paired windows are dropped
# ---------------------------------------------------------------------------


def test_compute_verdicts_when_some_windows_one_sided_does_drop_them_with_one_warning():
    collected: list[str] = []

    result = run(
        samples(100.0, 100.0, 100.0, 90.0),
        [{"metric": 95.0}, {"other": 1.0}, {"other": 1.0}, {"metric": 85.0}],
        METRIC_EXACT_LOWER,
        warn=collected.append,
    )

    delta = result["metric"].delta
    assert result == {
        "metric": ExactVerdict(method="exact", verdict="improved", delta=delta, n=2),
    }
    assert delta == pytest.approx(-5 / 95 * 100, abs=1e-5)
    assert collected == [
        "metric: dropped 2 paired windows where the metric was measured on only one side"
    ]


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
