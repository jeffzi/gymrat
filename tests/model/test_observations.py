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
# pair_metric — happy path and truncation
# ---------------------------------------------------------------------------


def test_pair_metric_when_shared_metric_across_equal_runs_does_align_values():
    left = [{"t": 1.0}, {"t": 2.0}]
    right = [{"t": 10.0}, {"t": 20.0}]

    result = pair_metric(left, right, "t")

    assert result.left == (1.0, 2.0)
    assert result.right == (10.0, 20.0)
    assert result.dropped == 0


def test_pair_metric_when_lengths_differ_does_truncate_to_shorter_run():
    left = [{"t": 1.0}, {"t": 2.0}, {"t": 3.0}]
    right = [{"t": 10.0}, {"t": 20.0}]

    result = pair_metric(left, right, "t")

    assert result.left == (1.0, 2.0)
    assert result.right == (10.0, 20.0)
    assert result.dropped == 0


# ---------------------------------------------------------------------------
# pair_metric — dropping unpaired rounds
# ---------------------------------------------------------------------------


def test_pair_metric_when_metric_missing_on_one_side_does_drop_from_both():
    left = [{"t": 1.0}, {"t": 2.0}, {"t": 3.0}]
    right = [{"t": 10.0}, {"other": 99.0}, {"t": 30.0}]

    result = pair_metric(left, right, "t")

    assert result.left == (1.0, 3.0)
    assert result.right == (10.0, 30.0)
    assert len(result.left) == len(result.right)
    assert result.dropped == 1


def test_pair_metric_when_metric_missing_on_both_sides_does_not_increment_dropped():
    left = [{"t": 1.0}, {"other": 5.0}]
    right = [{"t": 10.0}, {"other": 50.0}]

    result = pair_metric(left, right, "t")

    assert result.left == (1.0,)
    assert result.right == (10.0,)
    assert result.dropped == 0


def test_pair_metric_when_metric_absent_everywhere_does_return_empty_sequences():
    left = [{"t": 1.0}]
    right = [{"t": 2.0}]

    result = pair_metric(left, right, "missing")

    assert result.left == ()
    assert result.right == ()


# ---------------------------------------------------------------------------
# pair_metric — property-based invariants
# ---------------------------------------------------------------------------

_any_rounds = given(
    left=_round_lists(),
    right=_round_lists(),
    metric=st.sampled_from(_METRIC_NAMES),
)


@_any_rounds
def test_pair_metric_when_given_any_rounds_does_return_bounded_equal_length_sequences(
    left: list[dict[str, float]],
    right: list[dict[str, float]],
    metric: str,
):
    result = pair_metric(left, right, metric)

    assert len(result.left) == len(result.right)
    assert len(result.left) <= min(len(left), len(right))


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
