"""Behavioral tests for the measurement-step floor of the verdict engine.

Covers how a metric's declared resolution floors the noise band and gates a
signal on both the permutation and band paths, through the public
``compute_verdicts`` API only.
"""

from collections.abc import Callable

import pytest

from gymrat.model import BandVerdict, MetricVerdict, PermutationVerdict
from tests.verdict._engine import (
    BYTE_ONE_MOVE_A,
    BYTE_ONE_MOVE_B,
    FLAT_PERMUTATION_ROUNDS,
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
    (
        "samples_a",
        "samples_b",
        "expected_noise_pct",
        "expected_delta",
        "expected_verdict",
        "expected_reason",
    ),
    [
        pytest.param(
            BYTE_ONE_MOVE_A,
            BYTE_ONE_MOVE_B,
            31.25,
            -25.0,
            "no-signal",
            "resolution",
            id="one-byte-lower",
        ),
        pytest.param(
            _BYTE_CLEARS_A,
            _BYTE_CLEARS_B,
            16.0,
            -25.0,
            "improved",
            None,
            id="clears-lower",
        ),
    ],
)
def test_compute_verdicts_when_byte_permutation_does_give_signal_only_beyond_byte_floor(
    samples_a: list[dict[str, float]],
    samples_b: list[dict[str, float]],
    *,
    expected_noise_pct: float,
    expected_delta: float,
    expected_verdict: str,
    expected_reason: str | None,
):
    result = run(samples_a, samples_b, METRIC_BYTES_LOWER)

    verdict = get_permutation(result)
    assert verdict.p < 0.05
    assert verdict.delta == pytest.approx(expected_delta, abs=1e-5)
    assert verdict.noise_pct == pytest.approx(expected_noise_pct, abs=1e-5)
    assert (verdict.verdict, verdict.reason) == (expected_verdict, expected_reason)


# The -25% delta sits inside the 30% band, so this also pins that the permutation
# path never compares the delta to the noise band.
def test_compute_verdicts_when_resolution_is_none_does_take_no_step_floor():
    result = run(BYTE_ONE_MOVE_A, BYTE_ONE_MOVE_B, METRIC_APPROX_LOWER)

    verdict = get_permutation(result)
    assert verdict.noise_pct == pytest.approx(30.0, abs=1e-5)
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


# Medians one tick apart (333 -> 375 ns) sit inside the 1.25-step floor; two
# ticks apart (333 -> 417 ns) clear it.
_TIMER_STEP_FLOOR_PCT = 1.25 * TIMER_TICK_NS / 333 * 100


@pytest.mark.parametrize(
    (
        "narrow",
        "samples_a",
        "samples_b",
        "expected_noise_pct",
        "expected_verdict",
        "expected_reason",
    ),
    [
        pytest.param(
            get_band,
            create_samples(3, 333.0),
            create_samples(3, 375.0),
            _TIMER_STEP_FLOOR_PCT,
            "no-signal",
            "resolution",
            id="timer-one-tick-band",
        ),
        pytest.param(
            get_permutation,
            create_samples(FLAT_PERMUTATION_ROUNDS, 333.0),
            create_samples(FLAT_PERMUTATION_ROUNDS, 417.0),
            _TIMER_STEP_FLOOR_PCT,
            "regressed",
            None,
            id="timer-two-ticks-permutation",
        ),
    ],
)
def test_compute_verdicts_when_resolution_set_does_floor_band_at_one_and_a_quarter_steps(
    narrow: Callable[[dict[str, MetricVerdict]], BandVerdict | PermutationVerdict],
    *,
    samples_a: list[dict[str, float]],
    samples_b: list[dict[str, float]],
    expected_noise_pct: float,
    expected_verdict: str,
    expected_reason: str | None,
):
    result = run(samples_a, samples_b, METRIC_TIMER_LOWER)

    verdict = narrow(result)
    assert verdict.noise_pct == pytest.approx(expected_noise_pct, abs=1e-5)
    assert (verdict.verdict, verdict.reason) == (expected_verdict, expected_reason)


# The baseline's 1.25-byte step floor exceeds the default 200% unstable band
# even when the floor is too large to express as a percentage.
def test_compute_verdicts_when_baseline_byte_median_overflows_does_mark_unstable():
    result = run(
        create_samples(2, OVERFLOWING_MEDIAN), create_samples(2, 1e-300), METRIC_BYTES_LOWER
    )

    verdict = get_band(result)
    assert (verdict.verdict, verdict.reason) == ("unstable", None)
