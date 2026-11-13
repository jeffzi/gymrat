"""Behavioral tests for the measurement-step floor of the verdict engine.

Covers how a metric's declared resolution floors the noise band and gates a
signal on both the permutation and band paths, through the public
``compute_verdicts`` API only.
"""

import pytest

from tests.verdict._engine import (
    BYTE_ONE_MOVE_A,
    BYTE_ONE_MOVE_B,
    METRIC_APPROX_LOWER,
    METRIC_BYTES_LOWER,
    METRIC_TIMER_LOWER,
    OVERFLOWING_MEDIAN,
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
    ("samples_a", "samples_b", "expected_verdict", "expected_reason"),
    [
        pytest.param(
            BYTE_ONE_MOVE_A, BYTE_ONE_MOVE_B, "no-signal", "resolution", id="one-byte-lower"
        ),
        pytest.param(_BYTE_CLEARS_A, _BYTE_CLEARS_B, "improved", None, id="clears-lower"),
    ],
)
def test_compute_verdicts_when_byte_permutation_does_give_signal_only_beyond_byte_floor(
    samples_a: list[dict[str, float]],
    samples_b: list[dict[str, float]],
    expected_verdict: str,
    expected_reason: str | None,
):
    result = run(samples_a, samples_b, METRIC_BYTES_LOWER)

    verdict = get_permutation(result)
    assert verdict.p < 0.05
    assert (verdict.verdict, verdict.reason) == (expected_verdict, expected_reason)


def test_compute_verdicts_when_resolution_is_none_does_take_no_step_floor():
    result = run(BYTE_ONE_MOVE_A, BYTE_ONE_MOVE_B, METRIC_APPROX_LOWER)

    verdict = get_permutation(result)
    assert verdict.noise_pct == pytest.approx(30.0, abs=1e-5)


# The -25% delta sits inside the 30% noise band: the permutation path never
# compares the delta to the band.
def test_compute_verdicts_when_permutation_delta_inside_noise_band_does_still_signal():
    result = run(BYTE_ONE_MOVE_A, BYTE_ONE_MOVE_B, METRIC_APPROX_LOWER)

    verdict = get_permutation(result)
    assert verdict.verdict == "improved"


@pytest.mark.parametrize(
    ("baseline", "candidate", "expected_noise_pct"),
    [
        pytest.param(4.0, 0.0, 31.25, id="zero-candidate"),
        pytest.param(0.0, 5.0, 0.5, id="zero-baseline"),
    ],
)
def test_compute_verdicts_when_byte_side_median_zero_does_floor_from_baseline_only(
    baseline: float,
    candidate: float,
    expected_noise_pct: float,
):
    result = run(create_samples(2, baseline), create_samples(2, candidate), METRIC_BYTES_LOWER)

    verdict = get_band(result)
    assert verdict.noise_pct == pytest.approx(expected_noise_pct, abs=1e-5)


@pytest.mark.parametrize(
    ("baseline", "candidate", "expected_verdict", "expected_reason"),
    [
        pytest.param(4.0, 0.0, "improved", None, id="zero-candidate"),
        pytest.param(0.0, 5.0, "no-signal", "non-finite-delta", id="zero-baseline"),
    ],
)
def test_compute_verdicts_when_byte_side_median_zero_does_judge_against_the_baseline_floor(
    baseline: float,
    candidate: float,
    expected_verdict: str,
    expected_reason: str | None,
):
    result = run(create_samples(2, baseline), create_samples(2, candidate), METRIC_BYTES_LOWER)

    verdict = get_band(result)
    assert (verdict.verdict, verdict.reason) == (expected_verdict, expected_reason)


# Medians one tick apart (333 -> 375 ns) sit inside the 1.25-step floor.
_TIMER_STEP_FLOOR_PCT = 1.25 * TIMER_TICK_NS / 333 * 100


def test_compute_verdicts_when_resolution_set_does_floor_band_at_one_and_a_quarter_steps():
    result = run(create_samples(3, 333.0), create_samples(3, 375.0), METRIC_TIMER_LOWER)

    verdict = get_band(result)
    assert verdict.noise_pct == pytest.approx(_TIMER_STEP_FLOOR_PCT, abs=1e-5)
    assert (verdict.verdict, verdict.reason) == ("no-signal", "resolution")


# The baseline's 1.25-byte step floor exceeds the default 200% unstable band
# even when the floor is too large to express as a percentage.
def test_compute_verdicts_when_baseline_byte_median_overflows_does_mark_unstable():
    result = run(
        create_samples(2, OVERFLOWING_MEDIAN), create_samples(2, 1e-300), METRIC_BYTES_LOWER
    )

    verdict = get_band(result)
    assert (verdict.verdict, verdict.reason) == ("unstable", None)


# The baseline median is 4 bytes and every pair moves by exactly one byte, so the
# centered differences are all zero and the shift alone would be near zero.
def test_compute_verdicts_when_permutation_shift_below_step_floor_does_carry_step_floor():
    result = run(BYTE_ONE_MOVE_A, BYTE_ONE_MOVE_B, METRIC_BYTES_LOWER)

    verdict = get_permutation(result)
    assert verdict.detectable_pct == pytest.approx(31.25, abs=1e-5)


# Below 0.625 bytes the baseline's 1.25-byte step floor exceeds the default 200%
# unstable band.
def test_compute_verdicts_when_unstable_does_keep_detectable_pct():
    result = run(create_samples(2, 0.6), create_samples(2, 0.7), METRIC_BYTES_LOWER)

    verdict = get_band(result)
    assert verdict.verdict == "unstable"
    assert verdict.detectable_pct == pytest.approx(1.25 / 0.6 * 100, abs=1e-5)
