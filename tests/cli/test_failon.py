"""Tests for fail-on parsing and gate evaluation.

These cover the ``parse_fail_on`` grammar, the ``should_fail_gate`` trip/no-trip
matrix (gating-only, exactly-on, OR semantics), and the empty-geomean warning.
"""

import io

import pytest
import typer

from gymrat_py.cli.gating import should_fail_gate, warn_empty_geomean_gates
from gymrat_py.cli.shared import parse_fail_on
from gymrat_py.report.types import GeomeanFailOn, RegressedFailOn
from tests.report._inputs import (
    create_candidate,
    create_comparison_result,
    other_kind,
    signed_rank_metric,
    without_gated_geomean,
)

# ---------------------------------------------------------------------------
# parse_fail_on
# ---------------------------------------------------------------------------


def test_parse_fail_on_accepts_regressed():
    assert parse_fail_on("regressed") == RegressedFailOn()


@pytest.mark.parametrize(
    ("value", "expected_pct"),
    [
        pytest.param("geomean:2", 2.0, id="integer"),
        pytest.param("geomean:-1.5", -1.5, id="negative-decimal"),
    ],
)
def test_parse_fail_on_accepts_geomean_percentage(value: str, expected_pct: float):
    condition = parse_fail_on(value)

    assert condition == GeomeanFailOn(pct=expected_pct)


@pytest.mark.parametrize(
    "value",
    ["geomean:", "geomean:0x10", "unknown", "", " geomean:2"],
)
def test_parse_fail_on_rejects_everything_else(value: str):
    with pytest.raises(typer.BadParameter) as exc:
        parse_fail_on(value)

    assert (
        exc.value.message
        == 'allowed values are "regressed" or "geomean:<number>" (e.g. geomean:2).'
    )


# ---------------------------------------------------------------------------
# should_fail_gate
# ---------------------------------------------------------------------------


def test_should_fail_gate_when_no_conditions_does_not_trip():
    result = create_comparison_result(
        metrics={"m/time": signed_rank_metric(verdict="regressed", delta=4, gating=True)},
        candidates=[create_candidate()],
    )

    assert should_fail_gate((), result) is False


def test_should_fail_gate_when_regressed_gating_metric_present_trips():
    result = create_comparison_result(
        metrics={"m/time": signed_rank_metric(verdict="regressed", delta=4, gating=True)},
        candidates=[create_candidate()],
    )

    assert should_fail_gate((RegressedFailOn(),), result) is True


def test_should_fail_gate_when_regression_is_non_gating_does_not_trip():
    result = create_comparison_result(
        metrics={"m/time": signed_rank_metric(verdict="regressed", delta=4, gating=False)},
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
def test_should_fail_gate_geomean_trips_at_or_above_threshold(geomean_value: float, expected: bool):
    result = create_comparison_result(
        candidates=[create_candidate(kinds=[other_kind(geomean_value, 3)])],
    )

    assert should_fail_gate((GeomeanFailOn(pct=2.0),), result) is expected


def test_should_fail_gate_when_gated_geomean_has_no_samples_does_not_trip():
    result = create_comparison_result(
        candidates=[create_candidate(kinds=[other_kind(5.0, 0)])],
    )

    assert should_fail_gate((GeomeanFailOn(pct=2.0),), result) is False


def test_should_fail_gate_when_kind_is_non_gating_never_trips_on_geomean():
    result = create_comparison_result(
        candidates=[create_candidate(kinds=[without_gated_geomean(other_kind(5.0, 3))])],
    )

    assert should_fail_gate((GeomeanFailOn(pct=2.0),), result) is False


def test_should_fail_gate_conditions_or_together():
    result = create_comparison_result(
        metrics={"m/time": signed_rank_metric(verdict="regressed", delta=4, gating=True)},
        candidates=[create_candidate(kinds=[other_kind(1.0, 3)])],
    )

    assert should_fail_gate((RegressedFailOn(), GeomeanFailOn(pct=99.0)), result) is True


# ---------------------------------------------------------------------------
# warn_empty_geomean_gates
# ---------------------------------------------------------------------------


def test_warn_empty_geomean_gates_warns_once_per_empty_candidate(
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


def test_warn_empty_geomean_gates_stays_silent_without_a_geomean_condition(
    monkeypatch: pytest.MonkeyPatch,
):
    captured = io.StringIO()
    monkeypatch.setattr("sys.stderr", captured)
    result = create_comparison_result(
        candidates=[create_candidate(label="cand-empty", kinds=[other_kind(5.0, 0)])],
    )

    warn_empty_geomean_gates((RegressedFailOn(),), result)

    assert captured.getvalue() == ""
