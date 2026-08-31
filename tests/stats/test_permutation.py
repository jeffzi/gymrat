"""Behavioral tests for the sign-flip permutation test.

The permutation test pairs ``x`` and ``y`` index-wise over the shorter input,
reports a ``SignificanceResult``, and
derives its two-sided p-value from an exact sign-flip enumeration (small
samples) or a fixed-seed Monte Carlo resample (large samples). scipy is the
authority for the pinned p-values below; they were captured by running the
statistic through ``scipy.stats.permutation_test`` directly.
"""

import math

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from gymrat.stats import SignificanceResult, sign_flip_permutation_test
from gymrat.stats.permutation import RESAMPLE_BUDGET

# ---------------------------------------------------------------------------
# sign_flip_permutation_test — pinned empirical fixtures
# ---------------------------------------------------------------------------

_SIX_PAIR_X = [10, 12, 14, 16, 18, 20]
_SIX_PAIR_Y = [9, 10, 13, 14, 15, 17]


@pytest.mark.parametrize(
    ("x", "y", "expected_p", "expected_n"),
    [
        pytest.param(
            _SIX_PAIR_X,
            _SIX_PAIR_Y,
            0.25,
            6,
            id="six-pair-all-negative-diff",
        ),
        pytest.param(
            [10, 11, 12, 13, 14, 15],
            [10, 11, 12, 13, 14, 15.5],
            1.0,
            1,
            id="near-identical-no-separation",
        ),
    ],
)
def test_sign_flip_permutation_test_when_paired_samples_does_return_pinned_p_and_n(
    x: list[float],
    y: list[float],
    expected_p: float,
    expected_n: int,
):
    result = sign_flip_permutation_test(x, y)

    assert result.p == pytest.approx(expected_p)
    assert result.n == expected_n


def test_sign_flip_permutation_test_when_inputs_differ_in_length_does_pair_over_shorter():
    x = [*_SIX_PAIR_X, 99]

    result = sign_flip_permutation_test(x, _SIX_PAIR_Y)

    assert result.n == 6
    assert result.p == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# sign_flip_permutation_test — degenerate guards (no scipy invocation)
# ---------------------------------------------------------------------------


def test_sign_flip_permutation_test_when_inputs_empty_does_return_p_one_and_n_zero():
    assert sign_flip_permutation_test([], []) == SignificanceResult(p=1.0, n=0)


def test_sign_flip_permutation_test_when_all_diffs_zero_does_return_p_one_and_n_zero():
    assert sign_flip_permutation_test([3.0, 5.0, 7.0], [3.0, 5.0, 7.0]) == SignificanceResult(
        p=1.0, n=0
    )


def test_sign_flip_permutation_test_when_single_pair_does_return_p_one_and_counted_n():
    assert sign_flip_permutation_test([4.0], [1.0]) == SignificanceResult(p=1.0, n=1)


def test_sign_flip_permutation_test_when_baseline_median_zero_does_return_p_one_and_counted_n():
    # median(x) == 0 with a non-zero candidate median makes the delta functional
    # divide by zero: the observed statistic is non-finite, so there is no
    # direction to test and the guard reports a non-significant p == 1.0.
    x = [-1.0, 1.0]
    y = [5.0, 6.0]

    result = sign_flip_permutation_test(x, y)

    assert result == SignificanceResult(p=1.0, n=2)


# ---------------------------------------------------------------------------
# Determinism by explicit policy — both sides of the exact / Monte Carlo seam
# ---------------------------------------------------------------------------


def _arithmetic_pairs(n: int) -> tuple[list[float], list[float]]:
    """Build ``n`` (x, y) pairs on parallel arithmetic sequences, x offset +5 above y."""
    return (
        [100.0 + 2.0 * i for i in range(n)],
        [95.0 + 2.0 * i for i in range(n)],
    )


_EXACT_PAIRS_N13 = _arithmetic_pairs(13)
_MONTE_CARLO_PAIRS_N14 = _arithmetic_pairs(14)


def test_sign_flip_permutation_test_when_exact_path_does_return_same_p_across_calls():
    x, y = _EXACT_PAIRS_N13

    first = sign_flip_permutation_test(x, y)
    second = sign_flip_permutation_test(x, y)

    assert first.n == 13
    assert 2**first.n <= RESAMPLE_BUDGET
    assert first.p == second.p


def test_sign_flip_permutation_test_when_monte_carlo_path_does_return_same_p_across_calls():
    x, y = _MONTE_CARLO_PAIRS_N14

    first = sign_flip_permutation_test(x, y)
    second = sign_flip_permutation_test(x, y)

    assert first.n == 14
    assert 2**first.n > RESAMPLE_BUDGET
    assert first.p == second.p


# ---------------------------------------------------------------------------
# Determinism policy constants
# ---------------------------------------------------------------------------


def test_resample_budget_when_inspected_does_bracket_the_exact_enumeration_boundary():
    assert 2**13 <= RESAMPLE_BUDGET < 2**14


# ---------------------------------------------------------------------------
# Dormancy — exported from the package, wired into nothing else
# ---------------------------------------------------------------------------


def test_sign_flip_permutation_test_when_imported_from_stats_does_match_direct_import():
    from gymrat import stats

    assert stats.sign_flip_permutation_test is sign_flip_permutation_test
    assert "sign_flip_permutation_test" in stats.__all__


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


# ---------------------------------------------------------------------------
# Spec-pinned regression cases
# ---------------------------------------------------------------------------


def test_sign_flip_permutation_test_when_zero_median_rearrangements_does_count_against():
    """Sign-flip rearrangements that produce a zero baseline median count against the delta.

    Baseline ``[12]*6`` vs candidate ``[0,0,0,0,60,60]``: some rearrangements
    flip enough low candidate values into the baseline to make its median zero,
    which makes the delta undefined (division by zero).  Those rearrangements must
    count on the baseline side (against the observed delta), not be discarded or
    treated as evidence for the delta.  The correct exact p is 0.25, not 0.125.
    """
    result = sign_flip_permutation_test([12] * 6, [0, 0, 0, 0, 60, 60])

    assert result.n == 6
    assert result.p == pytest.approx(0.25)
    assert math.isfinite(result.p)


def test_sign_flip_permutation_test_when_tied_pairs_reduce_exact_budget_does_report_exact_p():
    """Only differing pairs count toward the exact/MC budget decision.

    Eight extreme tied pairs plus the standard six differing pairs give 14 total.
    ``2**14 > RESAMPLE_BUDGET`` would push scipy onto the Monte Carlo path, but
    tied pairs contribute the same value to both sides under every flip, so the
    effective space is ``2**6 = 64 <= RESAMPLE_BUDGET`` — exact enumeration.

    Extreme tied values sit outside the differing-pair range and do not shift
    medians, so the exact p equals the ties-free six-pair p.  An MC path over
    all 14 pairs would produce a close but not byte-identical estimate.
    """
    x = [1, 2, 3, 4, 96, 97, 98, 99, *_SIX_PAIR_X]
    y = [1, 2, 3, 4, 96, 97, 98, 99, *_SIX_PAIR_Y]

    no_ties = sign_flip_permutation_test(_SIX_PAIR_X, _SIX_PAIR_Y)
    with_ties = sign_flip_permutation_test(x, y)

    assert with_ties.n == 6
    assert 2**with_ties.n <= RESAMPLE_BUDGET
    assert with_ties.p == no_ties.p


def test_permutation_descriptor_when_inspected_does_expose_strict_p_threshold():
    from gymrat.model.verdict_method import PERMUTATION_FLOORS

    assert PERMUTATION_FLOORS.p_threshold == 0.05


@_no_deadline
@given(pairs=_paired_samples)
def test_sign_flip_permutation_test_when_any_positive_pairs_does_return_p_in_unit_interval(
    pairs: list[tuple[float, float]],
):
    x = [pair[0] for pair in pairs]
    y = [pair[1] for pair in pairs]

    result = sign_flip_permutation_test(x, y)

    assert 0.0 < result.p <= 1.0


@_no_deadline
@given(pairs=_paired_samples)
def test_sign_flip_permutation_test_when_swapped_does_return_same_p(
    pairs: list[tuple[float, float]],
):
    x = [pair[0] for pair in pairs]
    y = [pair[1] for pair in pairs]

    forward = sign_flip_permutation_test(x, y)
    swapped = sign_flip_permutation_test(y, x)

    assert swapped.n == forward.n
    assert swapped.p == pytest.approx(forward.p)
    assert math.isfinite(forward.p)
