"""Behavioral tests for the measurement-step floor of the verdict engine.

Covers how a metric's declared resolution floors the noise band and gates a
signal on both the permutation and band paths, through the public
``compute_verdicts`` API only.
"""

from collections.abc import Callable

import pytest

from gymrat.model import BandVerdict, MetricMeta, MetricVerdict, PermutationVerdict
from tests.verdict._engine import (
    BYTE_ONE_MOVE_A,
    BYTE_ONE_MOVE_B,
    FLAT_PERMUTATION_ROUNDS,
    METRIC_APPROX_LOWER,
    METRIC_BYTES_HIGHER,
    METRIC_BYTES_LOWER,
    METRIC_BYTES_NO_RESOLUTION_LOWER,
    METRIC_NS_LOWER,
    METRIC_TIMER_LOWER,
    TIMER_TICK_NS,
    create_samples,
    get_band,
    get_permutation,
    run,
    samples,
)

# ---------------------------------------------------------------------------
# Measurement-step floor
# ---------------------------------------------------------------------------


# Six windows whose medians are 100 vs 75 bytes — a 25-byte move that clears the
# whole-byte floor — again shown by every pair, which keeps p significant.
_BYTE_CLEARS_A = samples(90.0, 95.0, 100.0, 100.0, 105.0, 110.0)
_BYTE_CLEARS_B = samples(67.0, 71.0, 75.0, 75.0, 79.0, 83.0)


@pytest.mark.parametrize(
    (
        "meta",
        "samples_a",
        "samples_b",
        "expected_noise_pct",
        "expected_delta",
        "expected_verdict",
        "expected_reason",
    ),
    [
        pytest.param(
            METRIC_BYTES_LOWER,
            BYTE_ONE_MOVE_A,
            BYTE_ONE_MOVE_B,
            31.25,
            -25.0,
            "no-signal",
            "resolution",
            id="one-byte-lower",
        ),
        pytest.param(
            METRIC_BYTES_HIGHER,
            BYTE_ONE_MOVE_A,
            BYTE_ONE_MOVE_B,
            31.25,
            -25.0,
            "no-signal",
            "resolution",
            id="one-byte-higher",
        ),
        pytest.param(
            METRIC_BYTES_LOWER,
            _BYTE_CLEARS_A,
            _BYTE_CLEARS_B,
            16.0,
            -25.0,
            "improved",
            None,
            id="clears-lower",
        ),
        pytest.param(
            METRIC_BYTES_HIGHER,
            _BYTE_CLEARS_A,
            _BYTE_CLEARS_B,
            16.0,
            -25.0,
            "regressed",
            None,
            id="clears-higher",
        ),
        pytest.param(
            METRIC_BYTES_LOWER,
            create_samples(FLAT_PERMUTATION_ROUNDS, 4.0),
            create_samples(FLAT_PERMUTATION_ROUNDS, 5.0),
            31.25,
            25.0,
            "no-signal",
            "resolution",
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
            None,
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
    expected_reason: str | None,
):
    result = run(samples_a, samples_b, meta)

    verdict = get_permutation(result)
    assert verdict.p < 0.05
    assert verdict.delta == pytest.approx(expected_delta, abs=1e-5)
    assert verdict.noise_pct == pytest.approx(expected_noise_pct, abs=1e-5)
    assert (verdict.verdict, verdict.reason) == (expected_verdict, expected_reason)


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


@pytest.mark.parametrize(
    ("baseline", "candidate", "expected_noise_pct", "expected_verdict", "expected_reason"),
    [
        pytest.param(4.0, 0.0, 31.25, "improved", None, id="zero-candidate"),
        pytest.param(0.0, 5.0, 0.5, "no-signal", "non-finite-delta", id="zero-baseline"),
    ],
)
def test_compute_verdicts_when_byte_side_median_zero_does_floor_from_baseline_only(
    baseline: float,
    candidate: float,
    expected_noise_pct: float,
    expected_verdict: str,
    expected_reason: str | None,
):
    result = run(create_samples(2, baseline), create_samples(2, candidate), METRIC_BYTES_LOWER)

    verdict = get_band(result)
    assert verdict.noise_pct == pytest.approx(expected_noise_pct, abs=1e-5)
    assert (verdict.verdict, verdict.reason) == (expected_verdict, expected_reason)


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
    (
        "narrow",
        "meta",
        "samples_a",
        "samples_b",
        "expected_noise_pct",
        "expected_verdict",
        "expected_reason",
    ),
    [
        pytest.param(
            get_permutation,
            METRIC_TIMER_LOWER,
            create_samples(FLAT_PERMUTATION_ROUNDS, 333.0),
            create_samples(FLAT_PERMUTATION_ROUNDS, 375.0),
            _TIMER_STEP_FLOOR_PCT,
            "no-signal",
            "resolution",
            id="timer-one-tick-permutation",
        ),
        pytest.param(
            get_band,
            METRIC_TIMER_LOWER,
            create_samples(3, 333.0),
            create_samples(3, 375.0),
            _TIMER_STEP_FLOOR_PCT,
            "no-signal",
            "resolution",
            id="timer-one-tick-band",
        ),
        pytest.param(
            get_permutation,
            METRIC_TIMER_LOWER,
            create_samples(FLAT_PERMUTATION_ROUNDS, 333.0),
            create_samples(FLAT_PERMUTATION_ROUNDS, 417.0),
            _TIMER_STEP_FLOOR_PCT,
            "regressed",
            None,
            id="timer-two-ticks-permutation",
        ),
        pytest.param(
            get_band,
            METRIC_TIMER_LOWER,
            create_samples(3, 333.0),
            create_samples(3, 417.0),
            _TIMER_STEP_FLOOR_PCT,
            "regressed",
            None,
            id="timer-two-ticks-band",
        ),
        pytest.param(
            get_band,
            METRIC_BYTES_LOWER,
            create_samples(2, 100.0),
            create_samples(2, 75.0),
            1.25,
            "improved",
            None,
            id="bytes-without-scatter",
        ),
        pytest.param(
            get_band,
            METRIC_BYTES_LOWER,
            create_samples(2, 4.0),
            create_samples(2, 3.0),
            31.25,
            "no-signal",
            "resolution",
            id="bytes-one-step-band",
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
    expected_reason: str | None,
):
    result = run(samples_a, samples_b, meta)

    verdict = narrow(result)
    assert verdict.noise_pct == pytest.approx(expected_noise_pct, abs=1e-5)
    assert (verdict.verdict, verdict.reason) == (expected_verdict, expected_reason)
