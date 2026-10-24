"""Behavioral tests for the pure descriptive-statistics helpers and the sign-flip permutation test.

The permutation test pairs ``x`` and ``y`` index-wise over the shorter input
and derives its two-sided p-value from an
exact sign-flip enumeration (small samples) or a fixed-seed Monte Carlo
resample (large samples). scipy is the authority for the pinned p-values below;
they were captured by running the statistic through
``scipy.stats.permutation_test`` directly.
"""

import math
import statistics
from itertools import combinations_with_replacement
from typing import cast
from unittest.mock import create_autospec

import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st

from gymrat.model import Direction
from gymrat.stats import (
    SignFlipResult,
    combine_geomean,
    compute_half_range,
    count_nonzero_pairs,
    detectable_shift,
    normalize_ratio,
    percent_delta,
    sign_flip_permutation_test,
)

# ---------------------------------------------------------------------------
# percent_delta
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("reference", "value", "expected"),
    [
        pytest.param(100.0, 90.0, -10.0, id="positive-reference-decrease"),
        pytest.param(100.0, 125.0, 25.0, id="positive-reference-increase"),
        pytest.param(-10.0, -5.0, 50.0, id="negative-reference-toward-zero-is-positive"),
        pytest.param(-10.0, -15.0, -50.0, id="negative-reference-away-from-zero-is-negative"),
        pytest.param(0.0, 0.0, 0.0, id="both-zero"),
        pytest.param(0.0, 5.0, math.nan, id="only-reference-zero-value-positive"),
        pytest.param(0.0, -5.0, math.nan, id="only-reference-zero-value-negative"),
    ],
)
def test_percent_delta_when_reference_given_does_scale_by_magnitude_or_return_nan_when_undefined(
    reference: float, value: float, expected: float
):
    assert percent_delta(reference, value) == pytest.approx(expected, nan_ok=True)


# ---------------------------------------------------------------------------
# compute_half_range
# ---------------------------------------------------------------------------


def test_compute_half_range_when_finite_samples_does_return_half_of_span():
    assert compute_half_range([2, 8, 4]) == 3.0


def test_compute_half_range_when_empty_does_raise_valueerror():
    with pytest.raises(ValueError, match="empty"):
        compute_half_range([])


_bounded_floats = st.floats(
    min_value=-1e6,
    max_value=1e6,
    allow_nan=False,
    allow_infinity=False,
)


@given(values=st.lists(_bounded_floats, min_size=1))
def test_compute_half_range_when_finite_samples_does_return_non_negative(values: list[float]):
    assert compute_half_range(values) >= 0.0


@given(values=st.lists(_bounded_floats, min_size=1), shift=_bounded_floats)
def test_compute_half_range_when_shifted_does_return_same_value(values: list[float], shift: float):
    shifted = [value + shift for value in values]

    assert math.isclose(
        compute_half_range(shifted),
        compute_half_range(values),
        rel_tol=1e-9,
        abs_tol=1e-6,
    )


@given(
    values=st.lists(
        st.floats(allow_nan=True, allow_infinity=True),
        min_size=1,
    ),
)
@example(values=[1.0, float("nan"), 3.0])
@example(values=[1.0, float("inf"), 3.0])
def test_compute_half_range_when_any_floats_does_return_nan_exactly_when_non_finite_present(
    values: list[float],
):
    has_non_finite = any(not math.isfinite(value) for value in values)

    assert math.isnan(compute_half_range(values)) == has_non_finite


# ---------------------------------------------------------------------------
# normalize_ratio
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("delta", "direction", "expected"),
    [
        pytest.param(50.0, "lower", pytest.approx(1.5), id="lower-formula"),
        pytest.param(50.0, "higher", pytest.approx(1.0 / 1.5), id="higher-formula"),
        pytest.param(float("nan"), "lower", "undefined-ratio", id="delta-nan"),
        pytest.param(-100.0, "lower", "infinite-rho", id="lower-rho-zero"),
        pytest.param(-200.0, "lower", "infinite-rho", id="lower-rho-negative"),
        pytest.param(-100.0, "higher", "infinite-rho", id="higher-divide-by-zero"),
        pytest.param(math.inf, "lower", "infinite-rho", id="lower-rho-infinite"),
    ],
)
def test_normalize_ratio_when_delta_and_direction_given_does_return_ratio_or_reason(
    delta: float,
    direction: Direction,
    expected: object,
):
    assert normalize_ratio(delta, direction) == expected


_positive_factor_deltas = st.floats(
    min_value=-99.0,
    max_value=1e6,
    allow_nan=False,
    allow_infinity=False,
)


@given(delta=_positive_factor_deltas)
def test_normalize_ratio_when_higher_does_reciprocate_lower(delta: float):
    lower_rho = normalize_ratio(delta, "lower")
    higher_rho = normalize_ratio(delta, "higher")

    # delta >= -99 keeps the factor strictly positive, so both are always usable.
    assert isinstance(lower_rho, float)
    assert isinstance(higher_rho, float)
    assert math.isclose(higher_rho, 1.0 / lower_rho, rel_tol=1e-9)


@given(delta=_positive_factor_deltas, direction=st.sampled_from(["lower", "higher"]))
def test_normalize_ratio_when_round_tripped_does_preserve_percent_delta(
    delta: float,
    direction: Direction,
):
    # delta >= -99 keeps the factor strictly positive, so rho is always usable.
    rho = cast("float", normalize_ratio(delta, direction))
    recovered_delta = (rho - 1.0) * 100.0 if direction == "lower" else (1.0 / rho - 1.0) * 100.0

    renormalized_rho = normalize_ratio(recovered_delta, direction)

    assert isinstance(renormalized_rho, float)
    assert math.isclose(renormalized_rho, rho, rel_tol=1e-9, abs_tol=1e-12)


# ---------------------------------------------------------------------------
# combine_geomean
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("entries", "expected"),
    [
        pytest.param([(1.5, 4.0)], (50.0, 4.0), id="single-entry"),
        pytest.param(
            [(1.5, 2.0), (2.0, 3.0)],
            ((math.sqrt(3.0) - 1.0) * 100.0, math.hypot(2.0, 3.0) / 2.0),
            id="multiple-entries",
        ),
        pytest.param([], (0.0, 0.0), id="empty"),
    ],
)
def test_combine_geomean_when_entries_given_does_return_percent_and_band(
    entries: list[tuple[float, float]],
    expected: tuple[float, float],
):
    assert combine_geomean(entries) == pytest.approx(expected)


_positive_rho = st.floats(
    min_value=1e-3,
    max_value=1e3,
    allow_nan=False,
    allow_infinity=False,
)
_noise = st.floats(
    min_value=0.0,
    max_value=1e6,
    allow_nan=False,
    allow_infinity=False,
)
_entries = st.lists(st.tuples(_positive_rho, _noise), min_size=1, max_size=50)


@given(entries=_entries)
def test_combine_geomean_when_any_inputs_does_stay_above_negative_100(
    entries: list[tuple[float, float]],
):
    value, _ = combine_geomean(entries)

    assert value > -100.0


@given(entries=_entries, data=st.data())
def test_combine_geomean_when_shuffled_does_return_same_value(
    entries: list[tuple[float, float]],
    data: st.DataObject,
):
    shuffled = data.draw(st.permutations(entries))
    original = combine_geomean(entries)

    permuted = combine_geomean(list(shuffled))

    assert permuted == pytest.approx(original, rel=1e-9, abs=1e-9)


# ---------------------------------------------------------------------------
# count_nonzero_pairs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("x", "y", "expected"),
    [
        pytest.param([], [], 0, id="empty"),
        pytest.param([3.0, 5.0, 7.0], [3.0, 5.0, 7.0], 0, id="all-tied"),
        pytest.param([1, 2, 3, 4, 9], [1, 5, 3, 7], 2, id="ties-and-unequal-lengths"),
        pytest.param([4.0, 1.0], [1.0, 4.0], 2, id="no-ties"),
        pytest.param(
            [math.inf, -math.inf, 1.0],
            [math.inf, -math.inf, 2.0],
            1,
            id="equal-infinities-tied",
        ),
        pytest.param([math.nan, 1.0], [math.nan, 2.0], 2, id="nan-pair-differs"),
    ],
)
def test_count_nonzero_pairs_when_paired_positionally_does_count_differing_pairs(
    x: list[float],
    y: list[float],
    expected: int,
):
    assert count_nonzero_pairs(x, y) == expected


# ---------------------------------------------------------------------------
# sign_flip_permutation_test — signed-rank statistic
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("x", "y", "expected"),
    [
        # Differences [+1, -0.5, +3] rank [2, 1, 3]: 2 - 1 + 3.
        pytest.param([10.0, 10.0, 10.0], [11.0, 9.5, 13.0], 4.0, id="ranks-favor-increases"),
        pytest.param([11.0, 9.5, 13.0], [10.0, 10.0, 10.0], -4.0, id="ranks-favor-decreases"),
        # Differences [+1, -1] share the average rank 1.5, so they cancel.
        pytest.param([10.0, 10.0], [11.0, 9.0], 0.0, id="ranks-balance"),
    ],
)
def test_sign_flip_permutation_test_when_paired_samples_does_return_signed_rank_sum_statistic(
    x: list[float],
    y: list[float],
    expected: float,
):
    assert sign_flip_permutation_test(x, y).statistic == pytest.approx(expected, nan_ok=True)


# ---------------------------------------------------------------------------
# sign_flip_permutation_test — pinned empirical fixtures
# ---------------------------------------------------------------------------

_SIX_PAIR_X = [10, 12, 14, 16, 18, 20]
_SIX_PAIR_Y = [9, 10, 13, 14, 15, 17]
_TIED_PAIRS = [1, 2, 3, 4, 96, 97, 98, 99]


@pytest.mark.parametrize(
    ("x", "y", "expected_p"),
    [
        pytest.param(
            [10, 11, 12, 13, 14, 15],
            [10, 11, 12, 13, 14, 15.5],
            1.0,
            id="near-identical-no-separation",
        ),
        pytest.param(
            [10, 20, 30, 40],
            [10, 21, 30, 41],
            0.5,
            id="interleaved-ties-two-differing-pairs",
        ),
        pytest.param([1, 3], [3, 1], 1.0, id="symmetric-swap-clipped-to-one"),
        pytest.param(
            [10, 11, 12, 13, 14, 15, 16, 17],
            [12, 11, 15, 13, 18, 15, 21, 23],
            0.0625,
            id="interleaved-ties-five-differing-pairs",
        ),
        pytest.param(
            [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
            [12, 13, 15, 13, 18, 15, 21, 23, 24, 19],
            0.015625,
            id="interleaved-ties-seven-differing-pairs",
        ),
        # Fourteen same-direction pairs leave exact enumeration: the seeded Monte
        # Carlo draw hits the all-negative sign pattern once, so p is 4 / 10_000
        # rather than the 2 / 10_000 floor.
        pytest.param(
            [100.0 + 2.0 * i for i in range(14)],
            [95.0 + 2.0 * i for i in range(14)],
            4 / 10_000,
            id="fourteen-pair-seeded-monte-carlo",
        ),
        # Unequal lengths pair index-wise over the shorter input.
        pytest.param(
            [*_SIX_PAIR_X, 99],
            _SIX_PAIR_Y,
            0.03125,
            id="unequal-lengths-pair-over-shorter",
        ),
        # Differences [-12, -12, -12, -12, +48, +48]: the two increases hold the
        # two largest ranks, so the signed-rank sum is only +1 (-10 + 11) and
        # many sign patterns reach it.
        pytest.param(
            [12] * 6,
            [0, 0, 0, 0, 60, 60],
            0.8125,
            id="few-large-increases-against-many-small-decreases",
        ),
        # Against five finite increases, the infinite decrease takes rank 6, so
        # the signed-rank sum is 15 - 6 = 9 and 28 of 64 sign patterns reach it.
        pytest.param(
            [0.0] * 6,
            [1.0, 2.0, 3.0, 4.0, 5.0, -math.inf],
            0.4375,
            id="infinite-difference-ranks-largest",
        ),
        pytest.param([333.0] * 10, [417.0] * 10, 2 / 2**10, id="flat-increase"),
        pytest.param([791.0] * 10, [625.0] * 10, 2 / 2**10, id="flat-decrease"),
        pytest.param(
            [100.0 + i for i in range(10)],
            [100.5 + i for i in range(10)],
            2 / 2**10,
            id="spread-half-unit-increase",
        ),
        pytest.param(_SIX_PAIR_X, _SIX_PAIR_Y, 2 / 2**6, id="six-pair-all-negative-diff"),
        pytest.param(
            [100.0 + 2.0 * i for i in range(13)],
            [95.0 + 2.0 * i for i in range(13)],
            2 / 2**13,
            id="thirteen-pair-largest-exact-enumeration",
        ),
        # 8 ties + 6 differing pairs = 14 total pairs stay on exact enumeration
        # because ties drop out before the 2**n budget applies.
        pytest.param(
            [*_TIED_PAIRS, *_SIX_PAIR_X],
            [*_TIED_PAIRS, *_SIX_PAIR_Y],
            2 / 2**6,
            id="tied-pairs-added-to-six-pair",
        ),
    ],
)
def test_sign_flip_permutation_test_when_paired_samples_does_return_pinned_p(
    x: list[float],
    y: list[float],
    expected_p: float,
):
    assert sign_flip_permutation_test(x, y).p == pytest.approx(expected_p)


# ---------------------------------------------------------------------------
# sign_flip_permutation_test — Monte Carlo regime (more than 13 differing pairs)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", [20, 30, 50])
@pytest.mark.parametrize(
    ("x_start", "y_start", "step"),
    [
        pytest.param(333.0, 417.0, 0.0, id="flat-increase"),
        pytest.param(100.0, 100.5, 1.0, id="spread-half-unit-increase"),
    ],
)
def test_sign_flip_permutation_test_when_every_pair_moves_same_way_above_exact_does_return_monte_carlo_floor(
    n: int,
    x_start: float,
    y_start: float,
    step: float,
):
    x = [x_start + step * i for i in range(n)]
    y = [y_start + step * i for i in range(n)]

    p = sign_flip_permutation_test(x, y).p

    # No seeded resample reaches the observed extreme: p is the Monte Carlo
    # floor 2 / (budget + 1).
    assert p == pytest.approx(2 / 10_000)


def test_sign_flip_permutation_test_when_called_twice_above_exact_does_return_same_p():
    x = [100.0] * 20
    y = [100.0 + (-1) ** i * (i + 1) for i in range(20)]
    first = sign_flip_permutation_test(x, y).p

    second = sign_flip_permutation_test(x, y).p

    assert second == first


# ---------------------------------------------------------------------------
# sign_flip_permutation_test — order invariance (exact regime)
# ---------------------------------------------------------------------------


def _is_nonzero(value: int) -> bool:
    return value != 0


# Small non-zero half-integer differences make equal magnitudes with opposite
# signs common: those are the pairs whose shared average rank keeps the p-value
# independent of order. Integer baselines keep each half-integer difference
# exact once added back. At most 13 pairs keeps the test on exact enumeration,
# where every sign pattern is counted once.
_reordered_differences = st.lists(
    st.integers(min_value=-4, max_value=4).filter(_is_nonzero),
    min_size=2,
    max_size=13,
).flatmap(
    lambda differences: st.tuples(
        st.just(differences),
        st.permutations(differences),
        st.lists(
            st.integers(min_value=1, max_value=1000),
            min_size=len(differences),
            max_size=len(differences),
        ),
    )
)

# `sign_flip_permutation_test` imports scipy lazily on its first call, so the
# first example in a worker pays a one-time ~400ms import cost that trips
# hypothesis's per-example deadline. That deadline is a wall-clock constraint
# orthogonal to the invariants under test.
_no_deadline = settings(deadline=None)


@_no_deadline
@given(case=_reordered_differences)
@example(case=([1, -1, 1, 1, 2, 3], [-1, 1, 1, 3, 1, 2], [100] * 6))
@example(case=([1, -1, 1, 1, 2, 3], [-1, 1, 1, 3, 1, 2], [1, 7, 950, 3, 64, 500]))
def test_sign_flip_permutation_test_when_differences_match_in_any_order_does_return_same_p(
    case: tuple[list[int], list[int], list[int]],
):
    differences, reordered, baselines = case
    constant = [100.0] * len(differences)
    expected = sign_flip_permutation_test(constant, [100.0 + d / 2 for d in differences]).p

    p = sign_flip_permutation_test(
        [float(b) for b in baselines],
        [b + d / 2 for b, d in zip(baselines, reordered, strict=True)],
    ).p

    assert p == pytest.approx(expected)


# ---------------------------------------------------------------------------
# sign_flip_permutation_test — degenerate guards (no permutation run)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("x", "y", "expected_statistic"),
    [
        pytest.param([], [], 0.0, id="empty"),
        pytest.param([3.0, 5.0, 7.0], [3.0, 5.0, 7.0], 0.0, id="all-diffs-zero"),
        pytest.param([4.0], [1.0], -1.0, id="single-pair"),
        pytest.param([1.0, 2.0, 3.0], [1.0, 2.0, 4.0], 1.0, id="one-differing-among-tied"),
        pytest.param([1.0, math.nan, 3.0], [2.0, 5.0, 4.0], math.nan, id="nan-in-baseline"),
        pytest.param([1.0, 2.0, 3.0], [2.0, math.nan, 4.0], math.nan, id="nan-in-candidate"),
    ],
)
def test_sign_flip_permutation_test_when_nothing_to_test_does_return_p_one_with_its_statistic(
    x: list[float],
    y: list[float],
    expected_statistic: float,
):
    result = sign_flip_permutation_test(x, y)

    assert (result.p, result.statistic) == (1.0, pytest.approx(expected_statistic, nan_ok=True))


# ---------------------------------------------------------------------------
# sign_flip_permutation_test — p-value range
# ---------------------------------------------------------------------------

# Sizes stay within exact-enumeration range (2..8) to keep each example fast.
_paired_samples = st.lists(
    st.tuples(_bounded_floats, _bounded_floats),
    min_size=2,
    max_size=8,
)


@_no_deadline
@given(pairs=_paired_samples)
def test_sign_flip_permutation_test_when_any_finite_pairs_does_return_p_in_unit_interval(
    pairs: list[tuple[float, float]],
):
    x = [pair[0] for pair in pairs]
    y = [pair[1] for pair in pairs]

    p = sign_flip_permutation_test(x, y).p

    assert 0.0 < p <= 1.0


# ---------------------------------------------------------------------------
# detectable_shift
# ---------------------------------------------------------------------------

_SIGNIFICANCE = 0.05
_BASELINE = 100.0


def _centered(differences: list[int]) -> list[float]:
    walsh_averages = [(a + b) / 2 for a, b in combinations_with_replacement(differences, 2)]
    center = statistics.median(walsh_averages)
    return [d - center for d in differences]


def _both_directions_significant(centered: list[float], shift: float) -> bool:
    zeros = [0.0] * len(centered)
    raised = sign_flip_permutation_test(zeros, [c + shift for c in centered]).p
    lowered = sign_flip_permutation_test(zeros, [c - shift for c in centered]).p
    return raised < _SIGNIFICANCE and lowered < _SIGNIFICANCE


def _detectable_shift_of(differences: list[int]) -> float:
    return detectable_shift(
        [_BASELINE] * len(differences),
        [_BASELINE + d for d in differences],
    )


# Small integer differences make tied magnitudes and zero differences common.
# Six pairs is the fewest whose all-same-sign pattern reaches p < 0.05 under
# exact enumeration; above 13 pairs the test switches to its seeded Monte Carlo
# resample.
_integer_differences = st.lists(st.integers(min_value=-6, max_value=6), min_size=6, max_size=20)

# Distinct non-zero magnitudes keep the differences tie-free. At most 13 pairs
# keeps the test on exact enumeration.
_tie_free_differences = st.lists(
    st.tuples(st.integers(min_value=1, max_value=30), st.booleans()),
    min_size=6,
    max_size=13,
    unique_by=lambda pair: pair[0],
).map(lambda pairs: [magnitude if positive else -magnitude for magnitude, positive in pairs])

_slow_permutations = settings(deadline=None, max_examples=30)


@_slow_permutations
@given(differences=_integer_differences)
@example(differences=[2] * 8)
@example(differences=[0, 0, 0, 1, 0, 0, 0, 0])
@example(differences=[1, -1, 1, -1, 2, -2, 1, 1, 0, 3, -1, 2, 1, -2, 0, 1])
@example(differences=[3] * 18)
def test_detectable_shift_when_integer_differences_does_make_both_directions_significant(
    differences: list[int],
):
    shift = _detectable_shift_of(differences)

    assert _both_directions_significant(_centered(differences), shift)


# Integer differences put the Hodges-Lehmann estimate on a 1/4 grid, so every
# shift where a direction's significance can change (a pairwise average of the
# centered differences) sits on a 1/8 grid. Between two such points the
# significance is constant, so probing 1/16 below the returned shift covers
# every smaller shift down to the previous grid point.
_HALF_GRID_STEP = 1 / 16


@_slow_permutations
@given(differences=_tie_free_differences)
@example(differences=[1, 2, 3, 4, 5, 6])
@example(differences=[-7, 3, 1, 12, -2, 5, 9, -4, 6, 11, 8, -10, 13])
def test_detectable_shift_when_tie_free_exact_regime_does_leave_smaller_shifts_insignificant(
    differences: list[int],
):
    shift = _detectable_shift_of(differences)

    assert math.isfinite(shift)
    assert not _both_directions_significant(
        _centered(differences),
        max(shift - _HALF_GRID_STEP, 0.0),
    )


@_slow_permutations
@given(differences=_integer_differences)
@example(differences=[2] * 8)
@example(differences=[1, 2, 3, 4, 5, 6])
def test_detectable_shift_when_integer_differences_does_not_exceed_shift_where_all_share_sign(
    differences: list[int],
):
    sign_sharing_shift = max(abs(c) for c in _centered(differences))

    shift = _detectable_shift_of(differences)

    assert shift <= sign_sharing_shift or math.isclose(shift, sign_sharing_shift, abs_tol=1e-9)


@pytest.mark.parametrize(
    ("x", "y", "expected"),
    [
        pytest.param([], [], math.inf, id="no-pairs"),
        pytest.param([10.0] * 5, [11.0, 12.0, 13.0, 14.0, 15.0], math.inf, id="five-pairs"),
        pytest.param([10.0] * 5, [10.0] * 5, math.inf, id="five-tied-pairs"),
        pytest.param(
            [10.0] * 8,
            [11.0, math.nan, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0],
            math.nan,
            id="nan-in-candidate",
        ),
    ],
)
def test_detectable_shift_when_no_shift_can_reach_significance_does_return_inf_or_nan(
    x: list[float],
    y: list[float],
    expected: float,
):
    assert detectable_shift(x, y) == pytest.approx(expected, nan_ok=True)


def test_detectable_shift_when_no_candidate_registers_does_run_the_test_at_most_six_times(
    monkeypatch: pytest.MonkeyPatch,
):
    never_significant = create_autospec(
        sign_flip_permutation_test, return_value=SignFlipResult(p=1.0, statistic=0.0)
    )
    monkeypatch.setattr("gymrat.stats.sign_flip_permutation_test", never_significant)
    differences = [-7, 3, 1, 12, -2, 5, 9, -4]

    shift = detectable_shift([0.0] * len(differences), differences)

    assert never_significant.call_count <= 6
    centered = _centered(differences)
    assert shift >= max(abs(value) for value in centered)
