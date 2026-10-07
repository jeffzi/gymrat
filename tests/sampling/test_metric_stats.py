"""Tests for the per-metric median and spread computed from collected samples."""

import dataclasses

import pytest

from gymrat.sampling import MetricStats, compute_metric_stats, own_values


def test_compute_metric_stats_when_empty_does_return_absent_median_and_spread():
    stats = compute_metric_stats([])

    assert stats == MetricStats(median=None, spread=None)


def test_compute_metric_stats_when_single_value_does_return_median_without_spread():
    stats = compute_metric_stats([5.0])

    assert stats.median == 5.0
    assert stats.spread is None


@pytest.mark.parametrize(
    ("values", "expected_median", "expected_spread"),
    [
        pytest.param([10.0, 20.0, 30.0], 20.0, 50.0, id="odd-length-sorted"),
        pytest.param([30.0, 10.0, 40.0, 20.0], 25.0, 60.0, id="even-length-unsorted"),
    ],
)
def test_compute_metric_stats_when_multiple_values_does_return_median_and_percent_spread(
    values: list[float],
    expected_median: float,
    expected_spread: float,
):
    stats = compute_metric_stats(values)

    assert stats.median == expected_median
    assert stats.spread == pytest.approx(expected_spread)


def test_compute_metric_stats_when_median_zero_does_omit_spread():
    stats = compute_metric_stats([-1.0, 0.0, 1.0])

    assert stats.median == 0.0
    assert stats.spread is None


def test_compute_metric_stats_when_ratio_overflows_to_infinity_does_omit_spread():
    stats = compute_metric_stats([0.0, 5e-324, 1.0])

    assert stats.median == 5e-324
    assert stats.spread is None


def test_metric_stats_when_field_assigned_does_raise_frozen():
    stats = compute_metric_stats([1.0, 2.0])

    with pytest.raises(dataclasses.FrozenInstanceError):
        stats.median = 9.0  # type: ignore[misc]


def test_own_values_when_rounds_missing_metric_does_skip_them():
    samples = [{"x": 1.0}, {"y": 2.0}, {"x": 3.0}]

    assert own_values(samples, "x") == [1.0, 3.0]
