"""Behavioral tests for the Wilcoxon signed-rank wrapper and its result shape."""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from gymrat_py.stats import SignificanceResult, wilcoxon_signed_rank

# ---------------------------------------------------------------------------
# wilcoxon_signed_rank — pinned fixtures
# ---------------------------------------------------------------------------

_SIX_PAIR_X = [10, 12, 14, 16, 18, 20]
_SIX_PAIR_Y = [9, 10, 13, 14, 15, 17]


@pytest.mark.parametrize(
    ("x", "y", "expected_p", "expected_n"),
    [
        pytest.param(
            _SIX_PAIR_X,
            _SIX_PAIR_Y,
            0.03125,
            6,
            id="six-pair",
        ),
        pytest.param(
            [11, 12, 13, 10],
            [12, 10, 10, 14],
            1.0,
            4,
            id="even-split",
        ),
        pytest.param(
            [5, 7, 10, 12, 14, 16, 18, 20],
            [5, 7, 9, 10, 13, 14, 15, 17],
            0.03125,
            6,
            id="prepended-zero-diff-pairs",
        ),
    ],
)
def test_wilcoxon_signed_rank_when_paired_samples_does_return_pinned_p_and_n(
    x: list[float],
    y: list[float],
    expected_p: float,
    expected_n: int,
):
    result = wilcoxon_signed_rank(x, y)

    assert result.p == pytest.approx(expected_p)
    assert result.n == expected_n


def test_wilcoxon_signed_rank_when_inputs_differ_in_length_does_pair_over_shorter():
    x = [*_SIX_PAIR_X, 99]

    result = wilcoxon_signed_rank(x, _SIX_PAIR_Y)

    assert result.n == 6
    assert result.p == pytest.approx(0.03125)


def test_wilcoxon_signed_rank_when_pairing_leaves_single_pair_does_return_p_one():
    assert wilcoxon_signed_rank([1.0, 2.0], [5.0]) == SignificanceResult(p=1.0, n=1)


# ---------------------------------------------------------------------------
# wilcoxon_signed_rank — degenerate guards (no scipy invocation)
# ---------------------------------------------------------------------------


def test_wilcoxon_signed_rank_when_inputs_empty_does_return_p_one_and_n_zero():
    assert wilcoxon_signed_rank([], []) == SignificanceResult(p=1.0, n=0)


def test_wilcoxon_signed_rank_when_all_diffs_zero_does_return_p_one_and_n_zero():
    assert wilcoxon_signed_rank([3.0, 5.0, 7.0], [3.0, 5.0, 7.0]) == SignificanceResult(p=1.0, n=0)


def test_wilcoxon_signed_rank_when_single_pair_does_return_p_one_and_counted_n():
    assert wilcoxon_signed_rank([4.0], [1.0]) == SignificanceResult(p=1.0, n=1)


# ---------------------------------------------------------------------------
# SignificanceResult — shared result shape
# ---------------------------------------------------------------------------


def test_significance_result_carries_p_and_n():
    result = SignificanceResult(p=0.25, n=4)

    assert result.p == 0.25
    assert result.n == 4


def test_significance_result_is_frozen():
    result = SignificanceResult(p=0.5, n=3)

    with pytest.raises((AttributeError, TypeError)):
        result.p = 0.1  # type: ignore[misc]


def test_significance_result_lives_in_shared_results_module():
    from gymrat_py.stats.results import SignificanceResult as SharedResult

    assert SharedResult is SignificanceResult


# ---------------------------------------------------------------------------
# Property-based invariants
# ---------------------------------------------------------------------------

_bounded_floats = st.floats(
    min_value=-1e6,
    max_value=1e6,
    allow_nan=False,
    allow_infinity=False,
)
_paired_samples = st.lists(st.tuples(_bounded_floats, _bounded_floats), max_size=30)

# The wrapper imports scipy lazily on its first non-degenerate call, so the first
# example in a worker pays a one-time ~400ms import cost that trips hypothesis's
# per-example deadline. The deadline is orthogonal to the invariants under test.
_no_deadline = settings(deadline=None)


@_no_deadline
@given(pairs=_paired_samples)
def test_wilcoxon_signed_rank_property_p_stays_within_unit_interval(
    pairs: list[tuple[float, float]],
):
    x = [pair[0] for pair in pairs]
    y = [pair[1] for pair in pairs]

    result = wilcoxon_signed_rank(x, y)

    assert 0.0 <= result.p <= 1.0


@_no_deadline
@given(pairs=_paired_samples)
def test_wilcoxon_signed_rank_property_is_invariant_under_swap(
    pairs: list[tuple[float, float]],
):
    x = [pair[0] for pair in pairs]
    y = [pair[1] for pair in pairs]

    forward = wilcoxon_signed_rank(x, y)
    swapped = wilcoxon_signed_rank(y, x)

    assert swapped.n == forward.n
    assert swapped.p == pytest.approx(forward.p)


@_no_deadline
@given(pairs=_paired_samples, extras=st.lists(_bounded_floats, max_size=10))
def test_wilcoxon_signed_rank_property_zero_diff_pairs_do_not_change_n(
    pairs: list[tuple[float, float]],
    extras: list[float],
):
    x = [pair[0] for pair in pairs]
    y = [pair[1] for pair in pairs]

    base = wilcoxon_signed_rank(x, y)
    augmented = wilcoxon_signed_rank(x + extras, y + extras)

    assert augmented.n == base.n
