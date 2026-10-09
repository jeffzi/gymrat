"""Behavioral tests for the pure descriptive-statistics helpers and the sign-flip permutation test.

The permutation test pairs ``x`` and ``y`` index-wise over the shorter input
and derives its two-sided p-value from an
exact sign-flip enumeration (small samples) or a fixed-seed Monte Carlo
resample (large samples). scipy is the authority for the pinned p-values below;
they were captured by running the statistic through
``scipy.stats.permutation_test`` directly.
"""

import math
from typing import cast

import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st

from gymrat.model import Direction
from gymrat.stats import (
    combine_geomean,
    compute_half_range,
    count_nonzero_pairs,
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
# sign_flip_permutation_test — pinned empirical fixtures
# ---------------------------------------------------------------------------

_SIX_PAIR_X = [10, 12, 14, 16, 18, 20]
_SIX_PAIR_Y = [9, 10, 13, 14, 15, 17]


@pytest.mark.parametrize(
    ("x", "y", "expected_p"),
    [
        pytest.param(
            _SIX_PAIR_X,
            _SIX_PAIR_Y,
            0.25,
            id="six-pair-all-negative-diff",
        ),
        pytest.param(
            [10, 11, 12, 13, 14, 15],
            [10, 11, 12, 13, 14, 15.5],
            1.0,
            id="near-identical-no-separation",
        ),
        pytest.param(
            [10, 20, 30, 40],
            [10, 21, 30, 41],
            1.0,
            id="interleaved-ties-two-differing-pairs",
        ),
        pytest.param([1, 3], [3, 1], 1.0, id="symmetric-swap-clipped-to-one"),
        pytest.param(
            [10, 11, 12, 13, 14, 15, 16, 17],
            [12, 11, 15, 13, 18, 15, 21, 23],
            0.5,
            id="interleaved-ties-five-differing-pairs",
        ),
        pytest.param(
            [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
            [12, 13, 15, 13, 18, 15, 21, 23, 24, 19],
            0.125,
            id="interleaved-ties-seven-differing-pairs",
        ),
        pytest.param(
            [100.0 + 2.0 * i for i in range(13)],
            [95.0 + 2.0 * i for i in range(13)],
            0.0625,
            id="thirteen-pair-largest-exact-enumeration",
        ),
        pytest.param(
            [100.0 + 2.0 * i for i in range(14)],
            [95.0 + 2.0 * i for i in range(14)],
            0.028,
            id="fourteen-pair-seeded-monte-carlo",
        ),
        # Unequal lengths pair index-wise over the shorter input.
        pytest.param(
            [*_SIX_PAIR_X, 99],
            _SIX_PAIR_Y,
            0.25,
            id="unequal-lengths-pair-over-shorter",
        ),
        # Baseline [12]*6 vs candidate [0,0,0,0,60,60]: some rearrangements flip
        # enough low candidate values into the baseline to make its median zero,
        # which makes the delta undefined (division by zero). Those rearrangements
        # must count on the baseline side (against the observed delta), not be
        # discarded or treated as evidence for the delta. The correct exact p is
        # 0.25, not 0.125.
        pytest.param(
            [12] * 6,
            [0, 0, 0, 0, 60, 60],
            0.25,
            id="zero-median-rearrangements-count-against",
        ),
        # A single flip leaves both medians at zero: that rearrangement is a 0%
        # delta, not an undefined one, so the null is {-200, 0, 0, 200} and the
        # exact p is 0.5.
        pytest.param([0, 5, 5], [0, -5, -5], 0.5, id="both-medians-zero-counts-no-change"),
    ],
)
def test_sign_flip_permutation_test_when_paired_samples_does_return_pinned_p(
    x: list[float],
    y: list[float],
    expected_p: float,
):
    assert sign_flip_permutation_test(x, y) == pytest.approx(expected_p)


def test_sign_flip_permutation_test_when_tied_pairs_reduce_exact_budget_does_report_exact_p():
    # Tied pairs reduce the effective budget, keeping the path exact.
    #
    # Eight extreme tied pairs plus the standard six differing pairs give 14
    # total.  2**14 > RESAMPLE_BUDGET would push scipy onto the Monte Carlo
    # path, but tied pairs contribute the same value to both sides under every
    # flip, so the effective space is 2**6 = 64 <= RESAMPLE_BUDGET — exact
    # enumeration.
    #
    # Extreme tied values sit outside the differing-pair range and do not shift
    # medians, so the exact p equals the ties-free six-pair p of 0.25.  An MC
    # path over all 14 pairs would produce a close but not byte-identical
    # estimate.
    x = [1, 2, 3, 4, 96, 97, 98, 99, *_SIX_PAIR_X]
    y = [1, 2, 3, 4, 96, 97, 98, 99, *_SIX_PAIR_Y]

    p = sign_flip_permutation_test(x, y)

    assert p == 0.25


# ---------------------------------------------------------------------------
# sign_flip_permutation_test — degenerate guards (no scipy invocation)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("x", "y"),
    [
        pytest.param([], [], id="empty"),
        pytest.param([3.0, 5.0, 7.0], [3.0, 5.0, 7.0], id="all-diffs-zero"),
        pytest.param([4.0], [1.0], id="single-pair"),
        # median(x) == 0 with a non-zero candidate median makes the delta functional
        # divide by zero: the observed statistic is non-finite, so there is no
        # direction to test.
        pytest.param([-1.0, 1.0], [5.0, 6.0], id="baseline-median-zero"),
        pytest.param([1.0, 2.0], [math.inf, math.inf], id="infinite-delta"),
    ],
)
def test_sign_flip_permutation_test_when_nothing_to_test_does_return_p_one(
    x: list[float], y: list[float]
):
    assert sign_flip_permutation_test(x, y) == 1.0


# ---------------------------------------------------------------------------
# sign_flip_permutation_test — p-value range
# ---------------------------------------------------------------------------

# Positive-only samples keep both baseline medians strictly positive, so the
# delta functional is always finite and the two-sided p-value is well defined in
# either pairing direction. Sizes stay within exact-enumeration range (2..8) to
# keep each example fast.
_positive_floats = st.floats(
    min_value=1.0,
    max_value=1e6,
    allow_nan=False,
    allow_infinity=False,
)
_paired_samples = st.lists(
    st.tuples(_positive_floats, _positive_floats),
    min_size=2,
    max_size=8,
)

# The test imports scipy lazily on its first non-degenerate call, so the first
# example in a worker pays a one-time ~400ms import cost that trips hypothesis's
# per-example deadline. That deadline is a wall-clock constraint orthogonal to
# the invariants under test.
_no_deadline = settings(deadline=None)


@_no_deadline
@given(pairs=_paired_samples)
def test_sign_flip_permutation_test_when_any_positive_pairs_does_return_p_in_unit_interval(
    pairs: list[tuple[float, float]],
):
    x = [pair[0] for pair in pairs]
    y = [pair[1] for pair in pairs]

    p = sign_flip_permutation_test(x, y)

    assert 0.0 < p <= 1.0
