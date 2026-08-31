"""Behavioral tests for the pure descriptive-statistics helpers."""

import math

import pytest
from hypothesis import given
from hypothesis import strategies as st

from gymrat.stats import (
    Direction,
    GeomeanCombination,
    combine_geomean,
    compute_half_range,
    compute_median,
    normalize_ratio,
)

# ---------------------------------------------------------------------------
# compute_median
# ---------------------------------------------------------------------------


def test_compute_median_when_odd_length_does_return_middle_element():
    assert compute_median([3, 1, 2]) == 2.0


def test_compute_median_when_even_length_does_return_mean_of_middle_two():
    assert compute_median([1, 2, 3, 4]) == 2.5


def test_compute_median_when_called_does_not_mutate_input():
    values = [3, 1, 2]

    compute_median(values)

    assert values == [3, 1, 2]


def test_compute_median_when_empty_does_raise_valueerror():
    with pytest.raises(ValueError, match="empty"):
        compute_median([])


# ---------------------------------------------------------------------------
# compute_half_range
# ---------------------------------------------------------------------------


def test_compute_half_range_when_finite_samples_does_return_half_of_span():
    assert compute_half_range([2, 8, 4]) == 3.0


@pytest.mark.parametrize(
    "values",
    [
        pytest.param([1.0, float("nan"), 3.0], id="nan-sample"),
        pytest.param([1.0, float("inf"), 3.0], id="pos-inf-sample"),
        pytest.param([float("nan"), float("inf"), float("-inf")], id="all-non-finite"),
    ],
)
def test_compute_half_range_when_any_sample_non_finite_does_return_nan(values: list[float]):
    assert math.isnan(compute_half_range(values))


def test_compute_half_range_when_empty_does_raise_valueerror():
    with pytest.raises(ValueError, match="empty"):
        compute_half_range([])


# ---------------------------------------------------------------------------
# normalize_ratio
# ---------------------------------------------------------------------------


def test_normalize_ratio_when_direction_lower_does_apply_lower_formula():
    outcome = normalize_ratio(50.0, "lower")

    assert outcome.reason is None
    assert outcome.rho == pytest.approx(1.5)


def test_normalize_ratio_when_direction_higher_does_apply_higher_formula():
    outcome = normalize_ratio(50.0, "higher")

    assert outcome.reason is None
    assert outcome.rho == pytest.approx(1.0 / 1.5)


def test_normalize_ratio_when_delta_nan_does_return_undefined_ratio():
    outcome = normalize_ratio(float("nan"), "lower")

    assert outcome.rho is None
    assert outcome.reason == "undefined-ratio"


@pytest.mark.parametrize(
    ("delta", "direction"),
    [
        pytest.param(-100.0, "lower", id="lower-rho-zero"),
        pytest.param(-200.0, "lower", id="lower-rho-negative"),
        pytest.param(-100.0, "higher", id="higher-divide-by-zero"),
    ],
)
def test_normalize_ratio_when_rho_not_positive_finite_does_return_infinite_rho(
    delta: float,
    direction: Direction,
):
    outcome = normalize_ratio(delta, direction)

    assert outcome.rho is None
    assert outcome.reason == "infinite-rho"


# ---------------------------------------------------------------------------
# combine_geomean
# ---------------------------------------------------------------------------


def test_combine_geomean_when_multiple_entries_does_return_value_band_and_count():
    result = combine_geomean([(1.5, 2.0), (2.0, 3.0)])

    assert result.n == 2
    assert result.value == pytest.approx((math.sqrt(3.0) - 1.0) * 100.0)
    assert result.band == pytest.approx(math.sqrt(13.0) / 2.0)


def test_combine_geomean_when_empty_does_return_zeros():
    result = combine_geomean([])

    assert result == GeomeanCombination(value=0.0, n=0, band=0.0)


def test_combine_geomean_when_single_entry_does_return_percent_and_own_band():
    result = combine_geomean([(1.5, 4.0)])

    assert result.n == 1
    assert result.value == pytest.approx(50.0)
    assert result.band == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# Property-based invariants
# ---------------------------------------------------------------------------

_bounded_floats = st.floats(
    min_value=-1e6,
    max_value=1e6,
    allow_nan=False,
    allow_infinity=False,
)


@given(values=st.lists(_bounded_floats, min_size=1))
def test_compute_median_when_any_samples_does_lie_within_min_and_max(values: list[float]):
    median = compute_median(values)

    assert min(values) <= median <= max(values)


@given(values=st.lists(_bounded_floats, min_size=1), data=st.data())
def test_compute_median_when_shuffled_does_return_same_value(
    values: list[float],
    data: st.DataObject,
):
    shuffled = data.draw(st.permutations(values))

    assert compute_median(list(shuffled)) == compute_median(values)


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
def test_compute_half_range_when_non_finite_present_does_return_nan(values: list[float]):
    has_non_finite = any(not math.isfinite(value) for value in values)

    assert math.isnan(compute_half_range(values)) == has_non_finite


@given(
    delta=st.floats(
        min_value=-99.0,
        max_value=1e6,
        allow_nan=False,
        allow_infinity=False,
    ),
)
def test_normalize_ratio_when_higher_does_reciprocate_lower(delta: float):
    lower_rho = normalize_ratio(delta, "lower").rho
    higher_rho = normalize_ratio(delta, "higher").rho

    # delta >= -99 keeps the factor strictly positive, so both are always usable.
    assert lower_rho is not None
    assert higher_rho is not None
    assert math.isclose(higher_rho, 1.0 / lower_rho, rel_tol=1e-9)


@given(
    delta=st.floats(
        min_value=-99.0,
        max_value=1e6,
        allow_nan=False,
        allow_infinity=False,
    ),
    direction=st.sampled_from(["lower", "higher"]),
)
def test_normalize_ratio_when_round_tripped_does_preserve_percent_delta(
    delta: float,
    direction: Direction,
):
    rho = normalize_ratio(delta, direction).rho

    # delta >= -99 keeps the factor strictly positive, so rho is always usable.
    assert rho is not None
    recovered_delta = (rho - 1.0) * 100.0 if direction == "lower" else (1.0 / rho - 1.0) * 100.0
    renormalized_rho = normalize_ratio(recovered_delta, direction).rho

    assert renormalized_rho is not None
    assert math.isclose(renormalized_rho, rho, rel_tol=1e-9, abs_tol=1e-12)


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
    assert combine_geomean(entries).value > -100.0


@given(entries=_entries, data=st.data())
def test_combine_geomean_when_shuffled_does_return_same_value(
    entries: list[tuple[float, float]],
    data: st.DataObject,
):
    shuffled = data.draw(st.permutations(entries))

    original = combine_geomean(entries)
    permuted = combine_geomean(list(shuffled))

    assert permuted.n == original.n
    assert math.isclose(permuted.value, original.value, rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(permuted.band, original.band, rel_tol=1e-9, abs_tol=1e-9)


# ---------------------------------------------------------------------------
# Result-type shape
# ---------------------------------------------------------------------------


def test_ratio_outcome_when_field_assigned_does_raise_frozen():
    outcome = normalize_ratio(50.0, "lower")

    with pytest.raises((AttributeError, TypeError)):
        outcome.rho = 2.0  # type: ignore[misc]


def test_geomean_combination_when_field_assigned_does_raise_frozen():
    result = combine_geomean([(1.5, 2.0)])

    with pytest.raises((AttributeError, TypeError)):
        result.value = 0.0  # type: ignore[misc]
