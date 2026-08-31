"""Tests for the fail-on gate evaluation and empty-geomean warning."""

import io

import pytest

from gymrat.cli.gating import should_fail_gate, warn_empty_geomean_gates
from gymrat.report.types import GeomeanFailOn, RegressedFailOn
from tests.report._inputs import (
    create_candidate,
    create_comparison_result,
    other_kind,
    permutation_metric,
    without_gated_geomean,
)

# ---------------------------------------------------------------------------
# should_fail_gate
# ---------------------------------------------------------------------------


def test_should_fail_gate_when_no_conditions_does_not_trip():
    result = create_comparison_result(
        metrics={"m/time": permutation_metric(verdict="regressed", delta=4, gating=True)},
        candidates=[create_candidate()],
    )

    assert should_fail_gate((), result) is False


def test_should_fail_gate_when_regressed_gating_metric_present_does_trip():
    result = create_comparison_result(
        metrics={"m/time": permutation_metric(verdict="regressed", delta=4, gating=True)},
        candidates=[create_candidate()],
    )

    assert should_fail_gate((RegressedFailOn(),), result) is True


def test_should_fail_gate_when_regression_is_non_gating_does_not_trip():
    result = create_comparison_result(
        metrics={"m/time": permutation_metric(verdict="regressed", delta=4, gating=False)},
        candidates=[create_candidate()],
    )

    assert should_fail_gate((RegressedFailOn(),), result) is False


@pytest.mark.parametrize(
    ("geomean_value", "expected"),
    [
        pytest.param(5.0, True, id="above-threshold-trips"),
        pytest.param(2.0, True, id="exactly-on-trips"),
        pytest.param(1.0, False, id="below-threshold-no-trip"),
    ],
)
def test_should_fail_gate_when_geomean_at_or_above_threshold_does_trip(
    geomean_value: float, expected: bool
):
    result = create_comparison_result(
        candidates=[create_candidate(kinds=[other_kind(geomean_value, 3)])],
    )

    assert should_fail_gate((GeomeanFailOn(pct=2.0),), result) is expected


def test_should_fail_gate_when_gated_geomean_has_no_samples_does_not_trip():
    result = create_comparison_result(
        candidates=[create_candidate(kinds=[other_kind(5.0, 0)])],
    )

    assert should_fail_gate((GeomeanFailOn(pct=2.0),), result) is False


def test_should_fail_gate_when_kind_is_non_gating_does_not_trip_on_geomean():
    result = create_comparison_result(
        candidates=[create_candidate(kinds=[without_gated_geomean(other_kind(5.0, 3))])],
    )

    assert should_fail_gate((GeomeanFailOn(pct=2.0),), result) is False


def test_should_fail_gate_when_multiple_conditions_does_trip_if_any_matches():
    result = create_comparison_result(
        metrics={"m/time": permutation_metric(verdict="regressed", delta=4, gating=True)},
        candidates=[create_candidate(kinds=[other_kind(1.0, 3)])],
    )

    assert should_fail_gate((RegressedFailOn(), GeomeanFailOn(pct=99.0)), result) is True


# ---------------------------------------------------------------------------
# warn_empty_geomean_gates
# ---------------------------------------------------------------------------


def test_warn_empty_geomean_gates_when_candidate_has_no_samples_does_warn(
    monkeypatch: pytest.MonkeyPatch,
):
    captured = io.StringIO()
    monkeypatch.setattr("sys.stderr", captured)
    result = create_comparison_result(
        candidates=[
            create_candidate(label="cand-empty", kinds=[other_kind(5.0, 0)]),
            create_candidate(label="cand-ok", kinds=[other_kind(5.0, 3)]),
        ],
    )

    warn_empty_geomean_gates((GeomeanFailOn(pct=2.0),), result)

    assert (
        captured.getvalue()
        == 'warning: geomean gate for "cand-empty" had no stable gating metrics to measure\n'
    )


def test_warn_empty_geomean_gates_when_no_geomean_condition_does_stay_silent(
    monkeypatch: pytest.MonkeyPatch,
):
    captured = io.StringIO()
    monkeypatch.setattr("sys.stderr", captured)
    result = create_comparison_result(
        candidates=[create_candidate(label="cand-empty", kinds=[other_kind(5.0, 0)])],
    )

    warn_empty_geomean_gates((RegressedFailOn(),), result)

    assert captured.getvalue() == ""
