from typing import Annotated, Any

import pytest
from pydantic import BaseModel, Field, Strict, TypeAdapter, ValidationError
from pydantic_core import ErrorDetails

from gymrat.config import MAX_SAFE_INTEGER
from gymrat.utils import (
    NON_BLANK_PATTERN,
    describe_key,
    drop_prefix_errors,
    expected_got,
    first_problem,
    phrase_for_error,
)


def _validation_error(annotation: Any, value: object) -> ValidationError:
    try:
        TypeAdapter(annotation).validate_python(value)
    except ValidationError as error:
        return error
    pytest.fail(f"{value!r} unexpectedly validated against {annotation!r}")


@pytest.mark.parametrize(
    ("annotation", "value", "expected"),
    [
        pytest.param(Annotated[int, Strict()], "banana", "an integer", id="int-type"),
        pytest.param(
            Annotated[int, Field(le=MAX_SAFE_INTEGER)],
            MAX_SAFE_INTEGER + 1,
            "a number at or below 9007199254740991",
            id="less-than-equal-large-bound",
        ),
        pytest.param(
            Annotated[float, Field(gt=0)], -1.0, "a number above 0", id="greater-than-whole"
        ),
        pytest.param(
            Annotated[float, Field(gt=0.5)], 0.5, "a number above 0.5", id="greater-than-fraction"
        ),
        pytest.param(
            Annotated[str, Field(min_length=1)], "", "a non-empty string", id="empty-string"
        ),
        pytest.param(
            Annotated[str, Field(pattern=NON_BLANK_PATTERN)],
            " ",
            "a non-empty string",
            id="blank-string",
        ),
        pytest.param(
            Annotated[str, Field(min_length=2)], "a", "a valid value", id="longer-min-length"
        ),
    ],
)
def test_phrase_for_error_when_validation_fails_does_phrase_expected_shape(
    annotation: object, value: object, expected: str
):
    error = _validation_error(annotation, value).errors()[0]

    phrase = phrase_for_error(error)

    assert phrase == expected


# ---------------------------------------------------------------------------
# drop_prefix_errors
# ---------------------------------------------------------------------------


def _error_at(*loc: str) -> ErrorDetails:
    return ErrorDetails(type="missing", loc=loc, msg="Field required", input=None)


@pytest.mark.parametrize(
    ("locations", "kept"),
    [
        pytest.param(
            [("metrics",), ("metrics", "total_ms"), ("bench",)],
            [("metrics", "total_ms"), ("bench",)],
            id="parent-of-a-failing-child-dropped",
        ),
        pytest.param(
            [("metrics",), ("metrics",)],
            [("metrics",), ("metrics",)],
            id="identical-locations-kept",
        ),
    ],
)
def test_drop_prefix_errors_when_locations_nest_does_keep_only_the_deepest(
    locations: list[tuple[str, ...]], kept: list[tuple[str, ...]]
):
    errors = [_error_at(*loc) for loc in locations]

    remaining = drop_prefix_errors(errors)

    assert [error["loc"] for error in remaining] == kept


# ---------------------------------------------------------------------------
# describe_key
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("loc", "expected"),
    [
        pytest.param(("metrics", "total_ms", 0), "metrics.total_ms.0", id="plain-parts"),
        pytest.param(("metrics", ""), 'metrics.""', id="empty-part"),
        pytest.param(("metrics", "a.b"), 'metrics."a.b"', id="part-with-a-dot"),
        pytest.param(("metrics", "a b"), 'metrics."a b"', id="part-with-a-space"),
        pytest.param(("metrics", "a\u2028b"), 'metrics."a\\u2028b"', id="line-separator"),
    ],
)
def test_describe_key_when_parts_vary_does_quote_only_parts_misread_bare(
    loc: tuple[str | int, ...], expected: str
):
    assert describe_key(loc) == expected


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
    error = _validation_error(_Task, value)

    tail = first_problem(error, prefix)

    assert tail == expected
