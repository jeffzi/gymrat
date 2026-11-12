from typing import Annotated

import pytest
from pydantic import Field
from pydantic_core import ErrorDetails

from gymrat.config import MAX_SAFE_INTEGER
from gymrat.utils import (
    NON_BLANK_PATTERN,
    describe_key,
    drop_prefix_errors,
    phrase_for_error,
)
from tests._validation import validation_error


@pytest.mark.parametrize(
    ("annotation", "value", "bound"),
    [
        pytest.param(
            Annotated[int, Field(le=MAX_SAFE_INTEGER)],
            MAX_SAFE_INTEGER + 1,
            "9007199254740991",
            id="less-than-equal-large-bound",
        ),
        pytest.param(Annotated[float, Field(gt=0)], -1.0, "0", id="greater-than-whole"),
        pytest.param(Annotated[float, Field(gt=0.5)], 0.5, "0.5", id="greater-than-fraction"),
    ],
)
def test_phrase_for_error_when_constraint_fails_does_insert_the_bound(
    annotation: object, value: object, bound: str
):
    error = validation_error(annotation, value).errors()[0]

    phrase = phrase_for_error(error)

    assert phrase.endswith(f" {bound}")


@pytest.mark.parametrize(
    ("annotation", "value", "expected"),
    [
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
            Annotated[str, Field(pattern=NON_BLANK_PATTERN)],
            "\u00a0",
            "a non-empty string",
            id="no-break-space",
        ),
        pytest.param(
            Annotated[str, Field(min_length=2)], "a", "a valid value", id="longer-min-length"
        ),
    ],
)
def test_phrase_for_error_when_validation_fails_does_phrase_expected_shape(
    annotation: object, value: object, expected: str
):
    error = validation_error(annotation, value).errors()[0]

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
