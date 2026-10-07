import pytest
from hypothesis import given
from hypothesis import strategies as st

from gymrat.model import pair_metric

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
# pair_metric — examples
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("left", "right", "metric", "expected"),
    [
        pytest.param(
            [{"t": 1.0}, {"t": 2.0}],
            [{"t": 10.0}, {"t": 20.0}],
            "t",
            ((1.0, 2.0), (10.0, 20.0), 0),
            id="equal-runs",
        ),
        pytest.param(
            [{"t": 1.0}, {"t": 2.0}, {"t": 3.0}],
            [{"t": 10.0}, {"t": 20.0}],
            "t",
            ((1.0, 2.0), (10.0, 20.0), 0),
            id="truncates-to-shorter",
        ),
        pytest.param(
            [{"t": 1.0}, {"t": 2.0}, {"t": 3.0}],
            [{"t": 10.0}, {"other": 99.0}, {"t": 30.0}],
            "t",
            ((1.0, 3.0), (10.0, 30.0), 1),
            id="missing-one-side",
        ),
        pytest.param(
            [{"t": 1.0}, {"other": 5.0}],
            [{"t": 10.0}, {"other": 50.0}],
            "t",
            ((1.0,), (10.0,), 0),
            id="missing-both-sides",
        ),
        pytest.param([{"t": 1.0}], [{"t": 2.0}], "missing", ((), (), 0), id="absent-everywhere"),
    ],
)
def test_pair_metric_when_rounds_given_does_pair_the_rounds_both_sides_measured(
    left: list[dict[str, float]],
    right: list[dict[str, float]],
    metric: str,
    expected: tuple[tuple[float, ...], tuple[float, ...], int],
):
    result = pair_metric(left, right, metric)

    assert (result.left, result.right, result.dropped) == expected


# ---------------------------------------------------------------------------
# pair_metric — property-based invariants
# ---------------------------------------------------------------------------

_any_rounds = given(
    left=_round_lists(),
    right=_round_lists(),
    metric=st.sampled_from(_METRIC_NAMES),
)


@_any_rounds
def test_pair_metric_when_given_any_rounds_does_drop_unpaired_rounds_preserving_order(
    left: list[dict[str, float]],
    right: list[dict[str, float]],
    metric: str,
):
    result = pair_metric(left, right, metric)

    shared = range(min(len(left), len(right)))
    kept = [index for index in shared if metric in left[index] and metric in right[index]]
    exactly_one = [index for index in shared if (metric in left[index]) != (metric in right[index])]
    assert result.left == tuple(left[index][metric] for index in kept)
    assert result.right == tuple(right[index][metric] for index in kept)
    assert result.dropped == len(exactly_one)


@given(values=st.lists(_finite_floats, max_size=6))
def test_pair_metric_when_paired_against_itself_does_return_metric_values_unchanged(
    values: list[float],
):
    rounds = [{"time": value} for value in values]

    result = pair_metric(rounds, rounds, "time")

    assert result.left == tuple(values)
    assert result.right == tuple(values)
