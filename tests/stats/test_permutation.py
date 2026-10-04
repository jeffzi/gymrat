"""Behavioral tests for the sign-flip permutation test.

The permutation test pairs ``x`` and ``y`` index-wise over the shorter input
and derives its two-sided p-value from an
exact sign-flip enumeration (small samples) or a fixed-seed Monte Carlo
resample (large samples). scipy is the authority for the pinned p-values below;
they were captured by running the statistic through
``scipy.stats.permutation_test`` directly.
"""

import math

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from gymrat.model import PERMUTATION_P_THRESHOLD
from gymrat.stats import RESAMPLE_BUDGET, count_nonzero_pairs, sign_flip_permutation_test

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
    ],
)
def test_sign_flip_permutation_test_when_paired_samples_does_return_pinned_p(
    x: list[float],
    y: list[float],
    expected_p: float,
):
    assert sign_flip_permutation_test(x, y) == pytest.approx(expected_p)


def test_sign_flip_permutation_test_when_inputs_differ_in_length_does_pair_over_shorter():
    x = [*_SIX_PAIR_X, 99]

    p = sign_flip_permutation_test(x, _SIX_PAIR_Y)

    assert p == pytest.approx(0.25)


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
# Determinism policy constants
# ---------------------------------------------------------------------------


def test_resample_budget_when_inspected_does_bracket_the_exact_enumeration_boundary():
    assert 2**13 <= RESAMPLE_BUDGET < 2**14


# ---------------------------------------------------------------------------
# Property-based invariants
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


# ---------------------------------------------------------------------------
# Spec-pinned regression cases
# ---------------------------------------------------------------------------


def test_sign_flip_permutation_test_when_zero_median_rearrangements_does_count_against():
    # Zero-median rearrangements count against the observed delta.
    #
    # Baseline [12]*6 vs candidate [0,0,0,0,60,60]: some rearrangements flip
    # enough low candidate values into the baseline to make its median zero,
    # which makes the delta undefined (division by zero).  Those rearrangements
    # must count on the baseline side (against the observed delta), not be
    # discarded or treated as evidence for the delta.  The correct exact p is
    # 0.25, not 0.125.
    p = sign_flip_permutation_test([12] * 6, [0, 0, 0, 0, 60, 60])

    assert p == pytest.approx(0.25)


def test_sign_flip_permutation_test_when_rearrangement_zeroes_both_medians_does_count_no_change():
    # A single flip leaves both medians at zero: that rearrangement is a 0% delta,
    # not an undefined one, so the null is {-200, 0, 0, 200} and the exact p is 0.5.
    p = sign_flip_permutation_test([0, 5, 5], [0, -5, -5])

    assert p == pytest.approx(0.5)


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
    # medians, so the exact p equals the ties-free six-pair p.  An MC path
    # over all 14 pairs would produce a close but not byte-identical estimate.
    x = [1, 2, 3, 4, 96, 97, 98, 99, *_SIX_PAIR_X]
    y = [1, 2, 3, 4, 96, 97, 98, 99, *_SIX_PAIR_Y]

    no_ties = sign_flip_permutation_test(_SIX_PAIR_X, _SIX_PAIR_Y)
    with_ties = sign_flip_permutation_test(x, y)

    assert with_ties == no_ties


def test_permutation_p_threshold_when_inspected_does_match_strict_default():
    assert PERMUTATION_P_THRESHOLD == 0.05
