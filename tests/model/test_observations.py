import inspect

import pytest
from hypothesis import given
from hypothesis import strategies as st

from gymrat_py.model import (
    DROP_UNPAIRED,
    Observations,
    pair_metric,
)

_METRIC_NAMES = ["time", "mem", "cpu"]

_finite_floats = st.floats(allow_nan=False, allow_infinity=False)


@st.composite
def _round_lists(draw: st.DrawFn) -> list[dict[str, float]]:
    """Draw a run of rounds, each a metric-mapping over a random subset of ``_METRIC_NAMES``."""
    count = draw(st.integers(min_value=0, max_value=6))
    rounds: list[dict[str, float]] = []
    for _ in range(count):
        names = draw(
            st.lists(
                st.sampled_from(_METRIC_NAMES),
                unique=True,
                max_size=len(_METRIC_NAMES),
            ),
        )
        rounds.append({name: draw(_finite_floats) for name in names})
    return rounds


# ---------------------------------------------------------------------------
# Observations.from_rounds
# ---------------------------------------------------------------------------


def test_from_rounds_when_given_samples_does_build_indexed_single_repeat_keys():
    obs = Observations.from_rounds([{"time": 1.0}, {"time": 2.0}])

    assert list(obs.by_key.keys()) == [0, 1]
    assert obs.by_key[0] == ({"time": 1.0},)
    assert obs.by_key[1] == ({"time": 2.0},)


def test_from_rounds_when_given_empty_samples_does_build_no_keys():
    obs = Observations.from_rounds([])

    assert list(obs.by_key.keys()) == []


# ---------------------------------------------------------------------------
# Multi-repeat construction
# ---------------------------------------------------------------------------


def test_observations_when_key_has_multiple_repeats_does_construct_without_error():
    obs = Observations(by_key={0: ({"time": 1.0}, {"time": 2.0})})

    assert len(obs.by_key[0]) == 2


# ---------------------------------------------------------------------------
# pair_metric — happy path and truncation
# ---------------------------------------------------------------------------


def test_pair_metric_when_shared_metric_across_equal_runs_does_align_values():
    left = Observations.from_rounds([{"t": 1.0}, {"t": 2.0}])
    right = Observations.from_rounds([{"t": 10.0}, {"t": 20.0}])

    result = pair_metric(left, right, "t")

    assert result.left == [1.0, 2.0]
    assert result.right == [10.0, 20.0]
    assert result.dropped == 0


def test_pair_metric_when_lengths_differ_does_truncate_to_shorter_run():
    left = Observations.from_rounds([{"t": 1.0}, {"t": 2.0}, {"t": 3.0}])
    right = Observations.from_rounds([{"t": 10.0}, {"t": 20.0}])

    result = pair_metric(left, right, "t")

    assert result.left == [1.0, 2.0]
    assert result.right == [10.0, 20.0]


# ---------------------------------------------------------------------------
# pair_metric — dropping unpaired keys
# ---------------------------------------------------------------------------


def test_pair_metric_when_metric_missing_on_one_side_does_drop_from_both():
    left = Observations.from_rounds([{"t": 1.0}, {"t": 2.0}, {"t": 3.0}])
    right = Observations.from_rounds([{"t": 10.0}, {"other": 99.0}, {"t": 30.0}])

    result = pair_metric(left, right, "t")

    assert result.left == [1.0, 3.0]
    assert result.right == [10.0, 30.0]
    assert len(result.left) == len(result.right)
    assert result.dropped == 1


def test_pair_metric_when_metric_missing_on_both_sides_does_not_increment_dropped():
    left = Observations.from_rounds([{"t": 1.0}, {"other": 5.0}])
    right = Observations.from_rounds([{"t": 10.0}, {"other": 50.0}])

    result = pair_metric(left, right, "t")

    assert result.left == [1.0]
    assert result.right == [10.0]
    assert result.dropped == 0


def test_pair_metric_when_metric_absent_everywhere_does_return_empty_sequences():
    left = Observations.from_rounds([{"t": 1.0}])
    right = Observations.from_rounds([{"t": 2.0}])

    result = pair_metric(left, right, "missing")

    assert result.left == []
    assert result.right == []


# ---------------------------------------------------------------------------
# pair_metric — single-repeat requirement
# ---------------------------------------------------------------------------


def test_pair_metric_when_container_has_multiple_repeats_does_raise_value_error():
    multi = Observations(by_key={0: ({"t": 1.0}, {"t": 2.0})})
    single = Observations.from_rounds([{"t": 1.0}])

    with pytest.raises(ValueError, match="single-repeat"):
        pair_metric(multi, single, "t")


# ---------------------------------------------------------------------------
# pair_metric — named drop policy seam
# ---------------------------------------------------------------------------


def test_pair_metric_default_policy_is_drop_unpaired():
    default = inspect.signature(pair_metric).parameters["policy"].default

    assert default == DROP_UNPAIRED
    assert DROP_UNPAIRED == "drop-unpaired"


# ---------------------------------------------------------------------------
# pair_metric — property-based invariants
# ---------------------------------------------------------------------------

_any_containers = given(
    left_rounds=_round_lists(),
    right_rounds=_round_lists(),
    metric=st.sampled_from(_METRIC_NAMES),
)


@_any_containers
def test_pair_metric_when_given_any_containers_does_return_bounded_equal_length_sequences(
    left_rounds: list[dict[str, float]],
    right_rounds: list[dict[str, float]],
    metric: str,
):
    left = Observations.from_rounds(left_rounds)
    right = Observations.from_rounds(right_rounds)

    result = pair_metric(left, right, metric)

    assert len(result.left) == len(result.right)
    assert len(result.left) <= min(len(left_rounds), len(right_rounds))


@_any_containers
def test_pair_metric_when_given_any_containers_does_drop_unpaired_keys_preserving_order(
    left_rounds: list[dict[str, float]],
    right_rounds: list[dict[str, float]],
    metric: str,
):
    left = Observations.from_rounds(left_rounds)
    right = Observations.from_rounds(right_rounds)

    result = pair_metric(left, right, metric)

    shared = [key for key in left.by_key if key in right.by_key]
    kept = [
        key for key in shared if metric in left.by_key[key][0] and metric in right.by_key[key][0]
    ]
    not_kept = [key for key in shared if key not in kept]
    exactly_one = [
        key for key in shared if (metric in left.by_key[key][0]) != (metric in right.by_key[key][0])
    ]
    assert set(kept) | set(not_kept) == set(shared)
    assert all(
        metric not in left.by_key[key][0] or metric not in right.by_key[key][0] for key in not_kept
    )
    assert kept == sorted(kept)
    assert result.left == [left.by_key[key][0][metric] for key in kept]
    assert result.right == [right.by_key[key][0][metric] for key in kept]
    assert result.dropped == len(exactly_one)


@given(values=st.lists(_finite_floats, max_size=6))
def test_pair_metric_when_paired_against_itself_does_return_metric_values_unchanged(
    values: list[float],
):
    obs = Observations.from_rounds([{"time": value} for value in values])

    result = pair_metric(obs, obs, "time")

    assert result.left == values
    assert result.right == values
