"""Behavioral tests for the verdict engine.

Covers pairing, delta computation, and per-metric method dispatch (exact,
permutation, band), driving that behavior through the public ``compute_verdicts``
API only.
"""

import math
import statistics
from collections.abc import Callable

import pytest

from gymrat.model import (
    BandVerdict,
    ExactVerdict,
    MetricMeta,
    MetricVerdict,
    PermutationVerdict,
)
from gymrat.stats import detectable_shift
from gymrat.verdict import compute_verdicts
from tests.verdict._engine import (
    BYTE_ONE_MOVE_A,
    BYTE_ONE_MOVE_B,
    DIRECTION_MISMATCH_A,
    DIRECTION_MISMATCH_B,
    FLAT_PERMUTATION_ROUNDS,
    METRIC_APPROX_HIGHER,
    METRIC_APPROX_LOWER,
    METRIC_BYTES_LOWER,
    METRIC_EXACT_BYTES_LOWER,
    METRIC_EXACT_HIGHER,
    METRIC_EXACT_LOWER,
    METRIC_EXACT_TIMER_LOWER,
    METRIC_NS_LOWER,
    NOISY_BAND_A,
    NOISY_BAND_B,
    NOISY_PERMUTATION_A,
    NOISY_PERMUTATION_B,
    SEPARATED_HIGH,
    SEPARATED_LOW,
    SUBNORMAL_BASELINE,
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
    assert (verdict.verdict, verdict.reason) == ("no-signal", None)


def test_compute_verdicts_when_paired_direction_opposes_delta_does_report_no_signal():
    result = run(DIRECTION_MISMATCH_A, DIRECTION_MISMATCH_B, METRIC_APPROX_LOWER)

    verdict = get_permutation(result)
    assert verdict.p < 0.05
    assert verdict.delta == pytest.approx(-2.0, abs=1e-5)
    assert (verdict.verdict, verdict.reason) == ("no-signal", "direction-mismatch")


@pytest.mark.parametrize(
    ("baseline", "candidate"),
    [
        pytest.param(SUBNORMAL_BASELINE, 100.0, id="overflow-positive"),
        pytest.param(SUBNORMAL_BASELINE, -100.0, id="overflow-negative"),
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
    assert (verdict.verdict, verdict.reason) == ("no-signal", "non-finite-delta")


@pytest.mark.parametrize(
    ("meta", "values_b", "expected_usable_n", "expected_noise_pct"),
    [
        pytest.param(METRIC_APPROX_LOWER, [100.0, 100.0], 2, 0.5, id="lower-overflow-positive"),
        pytest.param(METRIC_APPROX_LOWER, [-100.0, -100.0], 2, 0.5, id="lower-overflow-negative"),
        pytest.param(METRIC_APPROX_HIGHER, [100.0, 100.0], 2, 0.5, id="higher-overflow-positive"),
        # One pair differs, too few for the band, and the delta still has no size.
        pytest.param(
            METRIC_APPROX_LOWER, [SUBNORMAL_BASELINE, 100.0], 1, 150.0, id="too-few-differ"
        ),
    ],
)
def test_compute_verdicts_when_band_delta_not_finite_does_report_no_signal(
    meta: dict[str, MetricMeta],
    values_b: list[float],
    expected_usable_n: int,
    expected_noise_pct: float,
):
    result = run(create_samples(2, SUBNORMAL_BASELINE), samples(*values_b), meta)

    verdict = get_band(result)
    assert not math.isfinite(verdict.delta)
    assert (verdict.verdict, verdict.reason, verdict.usable_n) == (
        "no-signal",
        "non-finite-delta",
        expected_usable_n,
    )
    assert verdict.noise_pct == pytest.approx(expected_noise_pct)


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

    verdict = get_permutation(result)
    assert (verdict.verdict, verdict.reason) == (expected_verdict, None)


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
            ("improved", None),
            id="exceeds-band",
        ),
        pytest.param(
            METRIC_APPROX_LOWER,
            create_samples(2, 100.0),
            samples(99.8, 99.6),
            -0.3,
            ("no-signal", "noise"),
            id="within-band",
        ),
        # Candidate median 110 with half-range 50: a +10% delta inside a
        # 1.5 * 50 / 110 ~= 68.2% band.
        pytest.param(
            METRIC_APPROX_LOWER,
            create_samples(2, 100.0),
            samples(60.0, 160.0),
            10.0,
            ("no-signal", "noise"),
            id="within-spread-band",
        ),
        pytest.param(
            METRIC_APPROX_HIGHER,
            create_samples(2, 50.0),
            create_samples(2, 100.0),
            100.0,
            ("improved", None),
            id="higher-improved",
        ),
        pytest.param(
            METRIC_APPROX_HIGHER,
            create_samples(2, 100.0),
            create_samples(2, 50.0),
            -50.0,
            ("regressed", None),
            id="higher-regressed",
        ),
    ],
)
def test_compute_verdicts_when_band_does_classify(
    meta: dict[str, MetricMeta],
    samples_a: list[dict[str, float]],
    samples_b: list[dict[str, float]],
    expected_delta: float,
    expected: tuple[str, str | None],
):
    result = run(samples_a, samples_b, meta)

    verdict = get_band(result)
    assert verdict.delta == pytest.approx(expected_delta, abs=1e-5)
    assert (verdict.verdict, verdict.reason) == expected


def test_compute_verdicts_when_band_fewer_differing_than_min_n_does_report_no_signal():
    result = run(samples(100.0, 100.0), samples(100.0, 210.0), METRIC_APPROX_LOWER)

    verdict = get_band(result)
    assert verdict.n == 2
    assert verdict.usable_n == 1
    assert (verdict.verdict, verdict.reason) == ("no-signal", "noise")


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


# A subnormal median: any spread (or the one-byte floor) divided by it overflows
# to infinity.
_OVERFLOWING_MEDIAN = 1e-310


# Below 0.625 bytes the baseline's 1.25-byte step floor exceeds the default 200%
# unstable band, on both paths, even when the floor is too large to express as a
# percentage; the candidate's median plays no part in the floor.
@pytest.mark.parametrize(
    ("narrow", "rounds", "baseline", "candidate", "expected"),
    [
        pytest.param(get_band, 2, 0.6, 0.7, "unstable", id="small-baseline-band"),
        pytest.param(
            get_permutation,
            FLAT_PERMUTATION_ROUNDS,
            0.6,
            0.7,
            "unstable",
            id="small-baseline-permutation",
        ),
        pytest.param(
            get_band, 2, _OVERFLOWING_MEDIAN, 1e-300, "unstable", id="overflowing-baseline-band"
        ),
        pytest.param(
            get_permutation,
            FLAT_PERMUTATION_ROUNDS,
            _OVERFLOWING_MEDIAN,
            1e-300,
            "unstable",
            id="overflowing-baseline-permutation",
        ),
        pytest.param(get_band, 2, 4.0, 0.4, "improved", id="small-candidate"),
    ],
)
def test_compute_verdicts_when_byte_median_is_small_does_mark_unstable_only_from_baseline(
    narrow: Callable[[dict[str, MetricVerdict]], BandVerdict | PermutationVerdict],
    rounds: int,
    baseline: float,
    candidate: float,
    expected: str,
):
    result = run(
        create_samples(rounds, baseline), create_samples(rounds, candidate), METRIC_BYTES_LOWER
    )

    verdict = narrow(result)
    assert (verdict.verdict, verdict.reason) == (expected, None)


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
    assert (verdict.verdict, verdict.reason) == ("unstable", None)
    assert verdict.noise_pct == pytest.approx(0.5, abs=1e-5)


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


# ---------------------------------------------------------------------------
# Detectable change
# ---------------------------------------------------------------------------


def _values(rounds: list[dict[str, float]]) -> list[float]:
    return [window["metric"] for window in rounds]


@pytest.mark.parametrize(
    ("samples_a", "samples_b"),
    [
        pytest.param(NOISY_PERMUTATION_A, NOISY_PERMUTATION_B, id="decrease"),
        pytest.param(SEPARATED_LOW, SEPARATED_HIGH, id="increase"),
        pytest.param(
            samples(-60.0, -55.0, -50.0, -50.0, -45.0, -40.0),
            samples(-62.0, -54.0, -53.0, -49.0, -47.0, -38.0),
            id="negative-median",
        ),
    ],
)
def test_compute_verdicts_when_permutation_does_carry_detectable_shift_over_baseline_median(
    samples_a: list[dict[str, float]],
    samples_b: list[dict[str, float]],
):
    baseline = _values(samples_a)
    shift = detectable_shift(baseline, _values(samples_b))

    result = run(samples_a, samples_b, METRIC_APPROX_LOWER)

    verdict = get_permutation(result)
    assert verdict.detectable_pct == pytest.approx(
        shift / abs(statistics.median(baseline)) * 100, abs=1e-9
    )


@pytest.mark.parametrize(
    ("meta", "samples_a", "samples_b", "expected_pct"),
    [
        pytest.param(
            METRIC_APPROX_LOWER, samples(80.0, 120.0), samples(90.0, 110.0), 30.0, id="spread"
        ),
        pytest.param(
            METRIC_APPROX_LOWER,
            create_samples(2, 100.0),
            samples(60.0, 160.0),
            150 * 50 / 110,
            id="candidate-spread",
        ),
        pytest.param(
            METRIC_APPROX_LOWER,
            samples(-60.0, -40.0),
            samples(-50.0, -50.0),
            30.0,
            id="negative-median",
        ),
        pytest.param(
            METRIC_APPROX_LOWER,
            create_samples(2, 100.0),
            samples(100.1, 99.9),
            0.5,
            id="noise-floor",
        ),
        pytest.param(
            METRIC_APPROX_LOWER,
            create_samples(3, 100.0),
            samples(100.0, 95.0, 90.0),
            150 * 5 / 95,
            id="two-of-three-differ",
        ),
        pytest.param(
            METRIC_BYTES_LOWER, create_samples(2, 4.0), create_samples(2, 3.0), 31.25, id="step"
        ),
    ],
)
def test_compute_verdicts_when_band_does_carry_band_its_delta_had_to_clear(
    meta: dict[str, MetricMeta],
    samples_a: list[dict[str, float]],
    samples_b: list[dict[str, float]],
    expected_pct: float,
):
    result = run(samples_a, samples_b, meta)

    verdict = get_band(result)
    assert verdict.detectable_pct == pytest.approx(expected_pct, abs=1e-5)


# The baseline median is 4 bytes and every pair moves by exactly one byte, so the
# centered differences are all zero and the shift alone would be near zero.
def test_compute_verdicts_when_permutation_shift_below_step_floor_does_carry_step_floor():
    result = run(BYTE_ONE_MOVE_A, BYTE_ONE_MOVE_B, METRIC_BYTES_LOWER)

    verdict = get_permutation(result)
    assert verdict.detectable_pct == pytest.approx(31.25, abs=1e-5)


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
            create_samples(2, 0.0),
            create_samples(2, 0.0),
            id="band-no-pair-differs",
        ),
        pytest.param(
            get_band,
            METRIC_APPROX_LOWER,
            samples(-5.0, 5.0),
            samples(10.0, 10.0),
            id="band-zero-baseline-median",
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
            samples(-5.0, _OVERFLOWING_MEDIAN, 5.0),
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
            METRIC_APPROX_LOWER,
            samples(-5.0, -3.0, 0.0, 4e-310, 3.0, 5.0),
            samples(-6.0, -4.0, 1e-310, 3e-310, 4.0, 6.0),
            id="permutation-overflowing-shift",
        ),
        pytest.param(
            get_permutation,
            METRIC_BYTES_LOWER,
            create_samples(FLAT_PERMUTATION_ROUNDS, _OVERFLOWING_MEDIAN),
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


@pytest.mark.parametrize(
    ("narrow", "meta", "samples_a", "samples_b", "unstable_noise_pct", "expected_pct"),
    [
        pytest.param(
            get_permutation,
            METRIC_APPROX_LOWER,
            NOISY_PERMUTATION_A,
            NOISY_PERMUTATION_B,
            20.0,
            10.0,
            id="permutation-noisy",
        ),
        pytest.param(
            get_band, METRIC_APPROX_LOWER, NOISY_BAND_A, NOISY_BAND_B, 20.0, 30.0, id="band-noisy"
        ),
        pytest.param(
            get_permutation,
            METRIC_BYTES_LOWER,
            create_samples(FLAT_PERMUTATION_ROUNDS, 0.6),
            create_samples(FLAT_PERMUTATION_ROUNDS, 0.7),
            200.0,
            1.25 / 0.6 * 100,
            id="permutation-step-floor",
        ),
        pytest.param(
            get_band,
            METRIC_BYTES_LOWER,
            create_samples(2, 0.6),
            create_samples(2, 0.7),
            200.0,
            1.25 / 0.6 * 100,
            id="band-step-floor",
        ),
    ],
)
def test_compute_verdicts_when_unstable_does_keep_detectable_pct(
    *,
    narrow: Callable[[dict[str, MetricVerdict]], BandVerdict | PermutationVerdict],
    meta: dict[str, MetricMeta],
    samples_a: list[dict[str, float]],
    samples_b: list[dict[str, float]],
    unstable_noise_pct: float,
    expected_pct: float,
):
    result = run(samples_a, samples_b, meta, unstable_noise_pct=unstable_noise_pct)

    verdict = narrow(result)
    assert verdict.verdict == "unstable"
    assert verdict.detectable_pct == pytest.approx(expected_pct, abs=1e-5)
