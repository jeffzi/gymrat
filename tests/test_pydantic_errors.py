from typing import Literal

import pytest

from gymrat.pydantic_errors import alternatives


@pytest.mark.parametrize(
    ("literal", "expected"),
    [
        pytest.param(Literal["a", "b", "c"], '"a", "b" or "c"', id="three-strings"),
        pytest.param(Literal["a", "b"], '"a" or "b"', id="two-strings"),
        pytest.param(Literal["a"], '"a"', id="single-string"),
        pytest.param(Literal[0, 1, 2], "0, 1 or 2", id="integers"),
    ],
)
def test_alternatives_when_given_literal_does_join_json_values_with_or_before_last(
    literal: object, expected: str
):
    phrase = alternatives(literal)

    assert phrase == expected
