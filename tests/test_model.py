import pytest
from hypothesis import example, given
from hypothesis import strategies as st

from gymrat.model import (
    BandVerdict,
    Direction,
    ExactVerdict,
    MetricVerdict,
    PermutationVerdict,
    is_improvement,
    pair_metric,
    verdict_stats,
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
# pair_metric
# ---------------------------------------------------------------------------

_any_rounds = given(
    left=_round_lists(),
    right=_round_lists(),
    metric=st.sampled_from(_METRIC_NAMES),
)


@_any_rounds
@example(
    left=[{"t": 1.0}, {"t": 2.0}, {"t": 3.0}], right=[{"t": 10.0}, {"t": 20.0}], metric="t"
).via("truncates to the shorter run")
@example(
    left=[{"t": 1.0}, {"t": 2.0}, {"t": 3.0}],
    right=[{"t": 10.0}, {"other": 99.0}, {"t": 30.0}],
    metric="t",
).via("missing on one side")
@example(left=[{"t": 1.0}, {"other": 5.0}], right=[{"t": 10.0}, {"other": 50.0}], metric="t").via(
    "missing on both sides"
)
def test_pair_metric_when_given_any_rounds_does_drop_unpaired_rounds_preserving_order(
    left: list[dict[str, float]],
    right: list[dict[str, float]],
    metric: str,
):
    shared = range(min(len(left), len(right)))
    kept = [index for index in shared if metric in left[index] and metric in right[index]]
    exactly_one = [index for index in shared if (metric in left[index]) != (metric in right[index])]

    result = pair_metric(left, right, metric)

    assert result.left == tuple(left[index][metric] for index in kept)
    assert result.right == tuple(right[index][metric] for index in kept)
    assert result.dropped == len(exactly_one)


# ---------------------------------------------------------------------------
# is_improvement
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("direction", "delta", "expected"),
    [
        pytest.param("lower", -1.0, True, id="lower-negative"),
        pytest.param("lower", 0.0, False, id="lower-zero"),
        pytest.param("higher", 1.0, True, id="higher-positive"),
        pytest.param("higher", 0.0, False, id="higher-zero"),
        pytest.param("lower", float("nan"), False, id="lower-nan"),
        pytest.param("higher", float("nan"), False, id="higher-nan"),
    ],
)
def test_is_improvement_when_delta_and_direction_vary_does_improve_only_on_a_strict_good_move(
    direction: Direction,
    delta: float,
    expected: bool,
):
    result = is_improvement(delta, direction)

    assert result is expected


@pytest.mark.parametrize(
    ("verdict", "expected"),
    [
        pytest.param(
            PermutationVerdict(
                method="permutation",
                verdict="no-signal",
                p=0.2,
                noise_pct=3.0,
                noise_abs=3.0,
                delta=1.0,
                n=8,
                reason="noise",
                detectable_pct=2.5,
            ),
            (0.2, 3.0, "noise", 2.5),
            id="permutation",
        ),
        pytest.param(
            BandVerdict(
                method="band",
                verdict="no-signal",
                usable_n=2,
                noise_pct=3.0,
                noise_abs=3.0,
                delta=1.0,
                n=2,
                reason="noise",
                detectable_pct=4.0,
            ),
            (None, 3.0, "noise", 4.0),
            id="band",
        ),
        pytest.param(
            ExactVerdict(method="exact", verdict="improved", delta=1.0, n=1),
            (None, None, None, None),
            id="exact",
        ),
    ],
)
def test_verdict_stats_when_method_varies_does_carry_fields_the_method_computes(
    verdict: MetricVerdict,
    expected: tuple[float | None, float | None, str | None, float | None],
):
    result = verdict_stats(verdict)

    assert result == expected
