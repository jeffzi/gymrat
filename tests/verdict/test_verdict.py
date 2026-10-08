"""Behavioral tests for the verdict engine.

Covers pairing, delta computation, and per-metric method dispatch (exact,
permutation, band), driving that behavior through the public ``compute_verdicts``
API only, and hierarchical kind/group aggregation through the public
``compute_kind_aggregates`` API: bucketing order, per-kind grouping, and the
per-subset exclusion taxonomy.
"""

import math
from collections.abc import Callable, Sequence

import pytest

from gymrat.model import (
    BandVerdict,
    ExactVerdict,
    Exclusion,
    GeomeanResult,
    MetricMeta,
    MetricVerdict,
    PermutationVerdict,
)
from gymrat.utils import WarnSink
from gymrat.verdict import KindAggregate, compute_kind_aggregates, compute_verdicts
from tests.verdict._inputs import (
    METRIC_BYTES_LOWER,
    MetricSpec,
    build_inputs,
    create_samples,
    noop_warn,
    samples,
    unstable_band_verdict,
)

# ---------------------------------------------------------------------------
# Metric-meta fixtures shared across the verdict-engine cases
# ---------------------------------------------------------------------------

METRIC_EXACT_LOWER = {"metric": MetricMeta(direction="lower", gating=True, exact=True, unit=None)}
METRIC_EXACT_HIGHER = {"metric": MetricMeta(direction="higher", gating=True, exact=True, unit=None)}
METRIC_APPROX_LOWER = {"metric": MetricMeta(direction="lower", gating=True, exact=False, unit=None)}
METRIC_APPROX_HIGHER = {
    "metric": MetricMeta(direction="higher", gating=True, exact=False, unit=None)
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
    """Pair two round lists and compute verdicts, defaulting warn and noise to test-friendly values."""
    sink = noop_warn if warn is None else warn
    if unstable_noise_pct is None:
        return compute_verdicts(samples_a, samples_b, meta, warn=sink)
    return compute_verdicts(
        samples_a, samples_b, meta, unstable_noise_pct=unstable_noise_pct, warn=sink
    )


def get_permutation(result: dict[str, MetricVerdict], key: str = "metric") -> PermutationVerdict:
    """Narrow the verdict at *key* to ``PermutationVerdict``, failing if the method differs."""
    verdict = result[key]
    assert isinstance(verdict, PermutationVerdict), f"expected permutation, got {verdict.method}"
    return verdict


def get_band(result: dict[str, MetricVerdict], key: str = "metric") -> BandVerdict:
    """Narrow the verdict at *key* to ``BandVerdict``, failing if the method differs."""
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
        delta=-5.0,
        n=1,
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
        pytest.param(METRIC_EXACT_LOWER, samples(100.0), 0.0, "no-signal", id="unchanged"),
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
    assert verdict.delta == pytest.approx(expected_delta, abs=1e-5)


# ---------------------------------------------------------------------------
# Multiple metrics
# ---------------------------------------------------------------------------


def test_compute_verdicts_when_metrics_differ_in_exactness_does_respect_per_metric_flag():
    result = run(
        [{"exactMetric": 100.0, "otherMetric": 50.0}],
        [{"exactMetric": 100.001, "otherMetric": 45.0}],
        {
            "exactMetric": MetricMeta(direction="lower", gating=True, exact=True, unit=None),
            "otherMetric": MetricMeta(direction="lower", gating=True, exact=False, unit=None),
        },
    )

    exact = result["exactMetric"]
    assert exact.verdict != "no-signal"
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
    assert verdict.delta == pytest.approx(expected_delta, abs=1e-5)
    assert verdict.verdict == expected_verdict


# ---------------------------------------------------------------------------
# Permutation method
# ---------------------------------------------------------------------------


def test_compute_verdicts_when_permutation_delta_nan_does_no_signal():
    result = run(create_samples(6, 0.0), create_samples(6, 5.0), METRIC_APPROX_LOWER)

    verdict = get_permutation(result)
    assert math.isnan(verdict.delta)
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
    assert verdict.delta == 0.0
    assert verdict.p == pytest.approx(1.0)
    assert verdict.verdict == "no-signal"


def test_compute_verdicts_when_permutation_delta_below_band_does_no_signal():
    # The permutation statistic *is* the delta functional, so a delta smaller
    # than the noise band never separates the two groups enough for the
    # sign-flip null to call it significant.
    samples_a = samples(80.0, 90.0, 100.0, 100.0, 110.0, 120.0)
    samples_b = samples(76.0, 85.5, 95.0, 95.0, 104.5, 114.0)

    result = run(samples_a, samples_b, METRIC_APPROX_LOWER)

    verdict = get_permutation(result)
    assert verdict.p == pytest.approx(0.5)
    assert verdict.delta == pytest.approx(-5.0, abs=1e-5)
    assert verdict.noise_pct == pytest.approx(30.0, abs=1e-5)
    assert verdict.verdict == "no-signal"


# Two well-separated six-window groups: the sign-flip null makes the observed
# delta significant (p = 0.03125) in either pairing direction.
_SEPARATED_HIGH = samples(180.0, 190.0, 200.0, 200.0, 210.0, 220.0)
_SEPARATED_LOW = samples(90.0, 95.0, 100.0, 100.0, 105.0, 110.0)


@pytest.mark.parametrize(
    ("meta", "samples_a", "samples_b", "expected_verdict"),
    [
        pytest.param(
            METRIC_APPROX_LOWER, _SEPARATED_HIGH, _SEPARATED_LOW, "improved", id="lower-improved"
        ),
        pytest.param(
            METRIC_APPROX_LOWER, _SEPARATED_LOW, _SEPARATED_HIGH, "regressed", id="lower-regressed"
        ),
        pytest.param(
            METRIC_APPROX_HIGHER, _SEPARATED_LOW, _SEPARATED_HIGH, "improved", id="higher-improved"
        ),
        pytest.param(
            METRIC_APPROX_HIGHER,
            _SEPARATED_HIGH,
            _SEPARATED_LOW,
            "regressed",
            id="higher-regressed",
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

    verdict = result["metric"]
    assert verdict.method == "permutation"
    assert verdict.verdict == expected_verdict


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
    ("meta", "samples_a", "samples_b", "expected"),
    [
        pytest.param(
            METRIC_APPROX_LOWER,
            samples(100.0, 110.0),
            samples(30.0, 50.0),
            "improved",
            id="exceeds-band",
        ),
        pytest.param(
            METRIC_APPROX_LOWER,
            create_samples(2, 100.0),
            samples(99.8, 99.6),
            "no-signal",
            id="within-band",
        ),
        pytest.param(
            METRIC_APPROX_LOWER,
            create_samples(2, 100.0),
            create_samples(2, 50.0),
            "improved",
            id="two-windows",
        ),
        pytest.param(
            METRIC_APPROX_HIGHER,
            create_samples(2, 50.0),
            create_samples(2, 100.0),
            "improved",
            id="higher-improved",
        ),
        pytest.param(
            METRIC_APPROX_HIGHER,
            create_samples(2, 100.0),
            create_samples(2, 50.0),
            "regressed",
            id="higher-regressed",
        ),
    ],
)
def test_compute_verdicts_when_band_does_classify(
    meta: dict[str, MetricMeta],
    samples_a: list[dict[str, float]],
    samples_b: list[dict[str, float]],
    expected: str,
):
    result = run(samples_a, samples_b, meta)

    verdict = result["metric"]
    assert verdict.method == "band"
    assert verdict.verdict == expected


def test_compute_verdicts_when_band_spread_high_does_report_wide_band_and_no_signal():
    # Candidate median 110 with half-range 50: a +10% delta inside a
    # 1.5 * 50 / 110 ~= 68.2% band.
    result = run(create_samples(2, 100.0), samples(60.0, 160.0), METRIC_APPROX_LOWER)

    verdict = get_band(result)
    assert verdict.delta == pytest.approx(10.0, abs=1e-5)
    assert verdict.noise_pct == pytest.approx(150 * 50 / 110, abs=1e-5)
    assert verdict.verdict == "no-signal"


def test_compute_verdicts_when_band_fewer_differing_than_min_n_does_no_signal():
    result = run(samples(100.0, 100.0), samples(100.0, 210.0), METRIC_APPROX_LOWER)

    verdict = get_band(result)
    assert verdict.n == 2
    assert verdict.usable_n == 1
    assert verdict.verdict == "no-signal"


# ---------------------------------------------------------------------------
# Noise-band carrying on non-exact verdicts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("narrow", "samples_a", "samples_b", "expected_noise_pct"),
    [
        pytest.param(get_band, samples(80.0, 120.0), samples(90.0, 110.0), 30.0, id="band-spread"),
        pytest.param(
            get_band, create_samples(2, 100.0), samples(100.0, 100.1), 0.5, id="band-floor"
        ),
        pytest.param(
            get_band,
            samples(-60.0, -40.0),
            samples(-50.0, -50.0),
            30.0,
            id="band-negative-median",
        ),
        pytest.param(
            get_permutation,
            samples(-60.0, -55.0, -50.0, -50.0, -45.0, -40.0),
            samples(-61.0, -56.0, -51.0, -51.0, -46.0, -41.0),
            30.0,
            id="permutation-negative-median",
        ),
    ],
)
def test_compute_verdicts_when_non_exact_inputs_vary_does_report_noise_pct(
    narrow: Callable[[dict[str, MetricVerdict]], BandVerdict | PermutationVerdict],
    samples_a: list[dict[str, float]],
    samples_b: list[dict[str, float]],
    expected_noise_pct: float,
):
    result = run(samples_a, samples_b, METRIC_APPROX_LOWER)

    assert narrow(result).noise_pct == pytest.approx(expected_noise_pct, abs=1e-5)


@pytest.mark.parametrize(
    ("narrow", "values_a", "values_b"),
    [
        pytest.param(
            get_permutation,
            [180.0, 190.0, 200.0, 200.0, 210.0, 220.0],
            [90.0, 95.0, 100.0, 100.0, 105.0, 110.0],
            id="permutation",
        ),
        pytest.param(get_band, [180.0, 220.0], [90.0, 110.0], id="band"),
    ],
)
def test_compute_verdicts_when_non_exact_does_carry_noise_abs(
    narrow: Callable[[dict[str, MetricVerdict]], BandVerdict | PermutationVerdict],
    values_a: list[float],
    values_b: list[float],
):
    result = run(samples(*values_a), samples(*values_b), METRIC_APPROX_LOWER)

    verdict = narrow(result)
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
    assert verdict.delta == pytest.approx(-25.0, abs=1e-5)
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
    assert verdict.delta == pytest.approx(-25.0, abs=1e-5)
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
    assert verdict.delta == pytest.approx(-25.0, abs=1e-5)
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
def test_compute_verdicts_when_noise_exceeds_threshold_does_mark_unstable(
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
# back to the floor or byte floor. Where a side has a non-zero half-range the
# verdict is forced unstable with the floor as its noise_pct; the byte-floor row
# has no spread, so the tiny side only loses its byte-floor term and the other
# side's one-byte floor (25%) remains. The permutation case keeps both sides'
# medians tiny so every sign-flipped delta stays finite and only the noise ratio
# overflows.


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
            25.0,
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
    assert math.isfinite(verdict.noise_pct)
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


@pytest.mark.parametrize(
    ("dropped", "expected"),
    [
        pytest.param(
            1,
            "metric: dropped 1 paired window where the metric was measured on only one side",
            id="one-dropped",
        ),
        pytest.param(
            2,
            "metric: dropped 2 paired windows where the metric was measured on only one side",
            id="two-dropped",
        ),
    ],
)
def test_compute_verdicts_when_verdict_produced_and_windows_dropped_does_warn_once(
    dropped: int,
    expected: str,
):
    one_sided: list[dict[str, float]] = [{"other": 1.0}] * dropped
    collected: list[str] = []

    result = run(
        samples(100.0, *[100.0] * dropped, 90.0),
        [{"metric": 95.0}, *one_sided, {"metric": 85.0}],
        METRIC_EXACT_LOWER,
        warn=collected.append,
    )

    assert result == {
        "metric": ExactVerdict(
            method="exact",
            verdict="improved",
            delta=-5 / 95 * 100,
            n=2,
        ),
    }
    assert collected == [expected]


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


def kinds_of(specs: Sequence[MetricSpec]) -> list[KindAggregate]:
    """The kind aggregates for a spec list — ``build_inputs`` fed into ``compute_kind_aggregates``."""
    verdicts, metric_meta = build_inputs(specs)
    return compute_kind_aggregates(verdicts, metric_meta)


def kind_named(aggregates: Sequence[KindAggregate], kind: str) -> KindAggregate:
    """The aggregate for ``kind``, or a failure naming the kinds produced."""
    for aggregate in aggregates:
        if aggregate.kind == kind:
            return aggregate
    names = ", ".join(aggregate.kind for aggregate in aggregates)
    pytest.fail(f'no aggregate for kind "{kind}", only: {names}')


# ---------------------------------------------------------------------------
# Kind aggregate shape and empty inputs
# ---------------------------------------------------------------------------


def test_compute_kind_aggregates_when_single_non_gating_metric_does_carry_kind_geomean_no_gate():
    result = kinds_of(
        [MetricSpec(name="warmup", gating=False, delta=0.0)],
    )

    assert result == [
        KindAggregate(
            kind="time",
            geomean=GeomeanResult(value=0.0, n=1, band=0.0, excluded=()),
            groups=(),
            gated_geomean=None,
        ),
    ]


def test_compute_kind_aggregates_when_nothing_measured_does_return_no_aggregates():
    assert kinds_of([]) == []


# ---------------------------------------------------------------------------
# Grouping by metric name contract (path minus last segment)
# ---------------------------------------------------------------------------


def test_compute_kind_aggregates_when_multi_segment_names_does_group_by_path_prefix():
    [kind] = kinds_of(
        [
            MetricSpec(name="decode/time#time", delta=-10.0),
            MetricSpec(name="decode/alloc#time", delta=-5.0),
            MetricSpec(name="encode/time#time", delta=-10.0),
        ],
    )

    assert [group.group for group in kind.groups] == ["decode", "encode"]
    assert kind.groups[0].geomean.n == 2
    assert kind.groups[1].geomean.n == 1


def test_compute_kind_aggregates_when_single_segment_name_does_count_in_kind_not_group():
    [kind] = kinds_of(
        [
            MetricSpec(name="decode/time#time", delta=-10.0),
            MetricSpec(name="warmup#time", delta=-10.0),
        ],
    )

    assert [group.group for group in kind.groups] == ["decode"]
    assert kind.groups[0].geomean.n == 1
    assert kind.geomean.n == 2


def test_compute_kind_aggregates_when_all_single_segment_does_give_kind_no_groups():
    [kind] = kinds_of(
        [
            MetricSpec(name="alpha#time", delta=-10.0),
            MetricSpec(name="beta#time", delta=-5.0),
        ],
    )

    assert kind.groups == ()


def test_compute_kind_aggregates_when_grouped_name_in_one_kind_does_leave_other_kind_flat():
    result = kinds_of(
        [
            MetricSpec(name="decode/time#time", kind="time", delta=-10.0),
            MetricSpec(name="heap#memory", kind="memory", delta=-10.0),
        ],
    )

    assert [group.group for group in kind_named(result, "time").groups] == ["decode"]
    assert kind_named(result, "memory").groups == ()


# ---------------------------------------------------------------------------
# Ordering by first mention
# ---------------------------------------------------------------------------


def test_compute_kind_aggregates_when_many_kinds_and_groups_does_order_by_first_mention():
    result = kinds_of(
        [
            MetricSpec(name="encode/time#time", kind="time", delta=-10.0),
            MetricSpec(name="encode/heap#memory", kind="memory", delta=-10.0),
            MetricSpec(name="decode/time#time", kind="time", delta=-10.0),
        ],
    )

    assert [aggregate.kind for aggregate in result] == ["time", "memory"]
    assert [group.group for group in kind_named(result, "time").groups] == ["encode", "decode"]


# ---------------------------------------------------------------------------
# Geomean scope: kind, gated and group
# ---------------------------------------------------------------------------


def test_compute_kind_aggregates_when_metrics_differ_only_in_gating_does_gate_over_gating_alone():
    [kind] = kinds_of(
        [
            MetricSpec(name="decode/time#time", gating=True, delta=-10.0),
            MetricSpec(name="decode/alloc#time", gating=False, delta=-5.0),
        ],
    )

    assert kind.geomean.n == 2
    assert kind.gated_geomean is not None
    assert (kind.gated_geomean.n, kind.gated_geomean.value) == (1, pytest.approx(-10.0, abs=1e-5))
    assert kind.groups[0].geomean.n == 2


def test_compute_kind_aggregates_when_metric_unstable_does_exclude_it_only_where_it_belongs():
    [kind] = kinds_of(
        [
            MetricSpec(name="decode/bad#time", verdict=unstable_band_verdict()),
            MetricSpec(name="decode/good#time", delta=-5.0),
            MetricSpec(name="encode/fine#time", delta=-5.0),
        ],
    )

    excluded = (Exclusion(metric="decode/bad#time", reason="unstable"),)
    assert (kind.geomean.n, kind.geomean.excluded) == (2, excluded)
    assert [(group.group, group.geomean.n, group.geomean.excluded) for group in kind.groups] == [
        ("decode", 1, excluded),
        ("encode", 1, ()),
    ]
