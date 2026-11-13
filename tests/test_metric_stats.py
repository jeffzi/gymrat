"""Tests for the per-metric summary statistics in :mod:`gymrat.metric_stats`."""

import pytest

from gymrat.errors import GymratError
from gymrat.metric_stats import compute_metric_stats


@pytest.mark.parametrize(
    ("values", "expected_median", "expected_spread"),
    [
        pytest.param([5.0], 5.0, None, id="single-value"),
        pytest.param([10.0, 20.0, 30.0], 20.0, pytest.approx(50.0), id="odd-length-sorted"),
    ],
)
def test_compute_metric_stats_when_given_values_does_return_the_summary_stats(
    values: list[float],
    expected_median: float,
    expected_spread: object,
):
    stats = compute_metric_stats("x", values)

    assert (stats.median, stats.spread) == (expected_median, expected_spread)


def test_compute_metric_stats_when_no_values_does_raise_naming_the_metric():
    with pytest.raises(GymratError, match=r"\bdecode/time\b"):
        compute_metric_stats("decode/time", [])
