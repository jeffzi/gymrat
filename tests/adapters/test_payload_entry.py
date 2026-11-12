"""Behavioral tests for the payload helpers in :mod:`gymrat.adapters.payload_entry`."""

import math

import pytest
from pydantic import BaseModel

from gymrat.adapters.payload_entry import (
    describe_error,
    first_problem,
    is_finite_or_warn,
    named_subject,
    record_metric,
)
from gymrat.utils import expected_got
from tests._validation import validation_error

# ---------------------------------------------------------------------------
# describe_error
# ---------------------------------------------------------------------------


def test_describe_error_when_string_holds_a_line_terminator_does_escape_only_the_terminator():
    described = describe_error('naïve "x"\nboom')

    assert described == r'naïve "x"\nboom'


def test_describe_error_when_value_is_not_a_string_does_render_it_as_json():
    described = describe_error({"code": 7})

    assert described == '{"code": 7}'


# ---------------------------------------------------------------------------
# record_metric
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("metrics", "warned"),
    [
        pytest.param({"total_ms": 1.0}, True, id="name-already-recorded"),
        pytest.param({"alloc_bytes": 1.0}, False, id="name-new"),
    ],
)
def test_record_metric_when_storing_a_reading_does_keep_it_warning_only_on_a_collision(
    metrics: dict[str, float], warned: bool
):
    warnings: list[str] = []

    record_metric(metrics, "total_ms", 2.0, warnings.append, "rename the runs")

    assert metrics["total_ms"] == 2.0
    assert [("total_ms" in w, "rename the runs" in w) for w in warnings] == (
        [(True, True)] if warned else []
    )


# ---------------------------------------------------------------------------
# named_subject
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("entry", "expected"),
    [
        pytest.param({"alias": "line\u2028break"}, 'benchmark "line\\u2028break"', id="string"),
        pytest.param({"alias": 3}, "benchmark", id="not-a-string"),
        pytest.param(None, "benchmark", id="not-an-object"),
    ],
)
def test_named_subject_when_entry_varies_does_name_it_only_by_a_string_field(
    entry: object, expected: str
):
    assert named_subject("benchmark", entry, "alias") == expected


# ---------------------------------------------------------------------------
# is_finite_or_warn
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "finite"),
    [
        pytest.param(1.5, True, id="finite"),
        pytest.param(math.inf, False, id="infinity"),
        pytest.param(10**400, False, id="integer-overflows-float"),
    ],
)
def test_is_finite_or_warn_when_checking_a_reading_does_warn_only_when_not_finite(
    value: float, finite: bool
):
    warnings: list[str] = []

    result = is_finite_or_warn(value, warnings.append, "run", "stats.p50")

    assert result is finite
    assert ["stats.p50" in w for w in warnings] == ([] if finite else [True])


# ---------------------------------------------------------------------------
# first_problem
# ---------------------------------------------------------------------------


class _Task(BaseModel):
    name: str


@pytest.mark.parametrize(
    ("value", "prefix", "expected"),
    [
        pytest.param({}, (), " with missing name", id="missing-key"),
        pytest.param(
            {"name": 1},
            (),
            f" with invalid name: {expected_got('a string', 1)}",
            id="invalid-key",
        ),
        pytest.param(None, (), f": {expected_got('an object', None)}", id="invalid-root"),
        pytest.param({}, ("result",), " with missing result.name", id="prefixed"),
    ],
)
def test_first_problem_when_validation_fails_does_phrase_the_first_error_as_a_tail(
    value: object, prefix: tuple[str, ...], expected: str
):
    error = validation_error(_Task, value)

    tail = first_problem(error, prefix)

    assert tail == expected
