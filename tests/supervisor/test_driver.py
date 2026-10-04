"""Behavioral tests for the cost rule every driver applies to a reported cost."""

import math

import pytest

from gymrat.supervisor.claude import usable_cost


@pytest.mark.parametrize(
    ("reported", "expected"),
    [
        pytest.param(0.42, 0.42, id="positive-float"),
        pytest.param(3, 3.0, id="positive-int"),
        pytest.param(None, None, id="missing"),
        pytest.param("0.42", None, id="string"),
        pytest.param(True, None, id="true"),
        pytest.param(False, None, id="false"),
        pytest.param(math.nan, None, id="nan"),
        pytest.param(math.inf, None, id="positive-infinity"),
        pytest.param(-math.inf, None, id="negative-infinity"),
        pytest.param(0.0, None, id="zero"),
        pytest.param(-0.3, None, id="negative"),
    ],
)
def test_usable_cost_when_reported_does_accept_only_finite_positive_numbers(
    reported: object,
    expected: float | None,
):
    cost = usable_cost(reported)

    assert cost == expected
    assert type(cost) is type(expected)
