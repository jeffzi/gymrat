import pytest

from gymrat.errors import GymratError
from gymrat.sample_balance import require_balanced_samples
from gymrat.utils import balanced_counts_around, balancing_cycle, is_balanced


@pytest.mark.parametrize(
    ("targets", "expected"),
    [
        pytest.param(1, 1, id="one-target"),
        pytest.param(2, 2, id="two-targets"),
        pytest.param(4, 4, id="even"),
        pytest.param(6, 6, id="even-larger"),
        pytest.param(3, 6, id="odd"),
        pytest.param(5, 10, id="odd-larger"),
    ],
)
def test_balancing_cycle_when_given_a_target_count_does_return_its_cycle_length(
    targets: int, expected: int
):
    result = balancing_cycle(targets)

    assert result == expected


@pytest.mark.parametrize(
    ("samples", "targets", "expected"),
    [
        pytest.param(7, 1, True, id="one-target-any-count"),
        pytest.param(12, 2, True, id="multiple-of-two"),
        pytest.param(12, 3, True, id="multiple-of-six"),
        pytest.param(9, 3, False, id="multiple-of-n-but-not-2n"),
        pytest.param(10, 4, False, id="not-a-multiple"),
        pytest.param(0, 2, False, id="zero"),
    ],
)
def test_is_balanced_when_given_a_sample_count_does_require_a_positive_multiple_of_the_cycle(
    samples: int, targets: int, expected: bool
):
    result = is_balanced(samples, targets)

    assert result is expected


@pytest.mark.parametrize(
    ("samples", "targets", "expected"),
    [
        pytest.param(10, 3, (6, 12), id="both-sides"),
        pytest.param(11, 2, (10, 12), id="two-targets"),
        pytest.param(4, 3, (6,), id="below-would-be-zero"),
        pytest.param(1, 2, (2,), id="one-sample-two-targets"),
    ],
)
def test_balanced_counts_around_when_count_is_unbalanced_does_return_the_nearest_valid_counts(
    samples: int, targets: int, expected: tuple[int, ...]
):
    result = balanced_counts_around(samples, targets)

    assert result == expected


@pytest.mark.parametrize(
    ("samples", "targets"),
    [
        pytest.param(5, 1, id="one-target"),
        pytest.param(12, 3, id="odd-target-count"),
        pytest.param(8, 4, id="even-target-count"),
    ],
)
def test_require_balanced_samples_when_count_balances_does_not_raise(samples: int, targets: int):
    require_balanced_samples(samples, targets)


@pytest.mark.parametrize(
    ("samples", "targets", "message"),
    [
        pytest.param(
            10,
            3,
            "samples = 10 cannot balance 3 targets: use a multiple of 6 (6 or 12)",
            id="both-sides",
        ),
        pytest.param(
            4,
            3,
            "samples = 4 cannot balance 3 targets: use a multiple of 6 (6)",
            id="only-the-one-above",
        ),
        pytest.param(
            3,
            2,
            "samples = 3 cannot balance 2 targets: use a multiple of 2 (2 or 4)",
            id="two-targets",
        ),
        pytest.param(
            6,
            4,
            "samples = 6 cannot balance 4 targets: use a multiple of 4 (4 or 8)",
            id="even-target-count",
        ),
    ],
)
def test_require_balanced_samples_when_count_is_unbalanced_does_refuse_with_the_nearest_counts(
    samples: int, targets: int, message: str
):
    with pytest.raises(GymratError) as caught:
        require_balanced_samples(samples, targets)

    assert (str(caught.value), caught.value.reason) == (message, "unbalanced-samples")
