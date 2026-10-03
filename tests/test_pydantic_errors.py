from dataclasses import dataclass
from typing import Annotated, Literal

import pytest
from pydantic import BaseModel, Field, Strict, TypeAdapter, ValidationError

from gymrat.pydantic_errors import NON_BLANK_PATTERN, phrase_for_error


class _Point(BaseModel):
    x: int


@dataclass(frozen=True, slots=True)
class _Pair:
    left: int


@pytest.mark.parametrize(
    ("annotation", "value", "expected"),
    [
        pytest.param(Annotated[int, Strict()], "banana", "an integer", id="int-type"),
        pytest.param(Annotated[float, Strict()], "banana", "a number", id="float-type"),
        pytest.param(
            Annotated[float, Field(allow_inf_nan=False)], float("nan"), "a number", id="nan"
        ),
        pytest.param(Annotated[str, Strict()], 1, "a string", id="string-type"),
        pytest.param(Annotated[bool, Strict()], 1, "a boolean", id="bool-type"),
        pytest.param(dict[str, int], 1, "an object", id="dict-type"),
        pytest.param(_Point, 1, "an object", id="model-type"),
        pytest.param(_Pair, 1, "an object", id="dataclass-type"),
        pytest.param(list[int], "banana", "an array", id="list-type"),
        pytest.param(tuple[int, ...], "banana", "an array", id="tuple-type"),
        pytest.param(Annotated[int, Field(gt=0)], 0, "a number greater than 0", id="greater-than"),
        pytest.param(
            Annotated[int, Field(ge=1)], 0, "a number at or above 1", id="greater-than-equal"
        ),
        pytest.param(
            Annotated[int, Field(le=5)], 6, "a number at or below 5", id="less-than-equal"
        ),
        pytest.param(Literal["a", "b"], "c", "'a' or 'b'", id="literal"),
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
        pytest.param(
            Annotated[str, Field(max_length=1)], "ab", "a valid value", id="unmapped-type"
        ),
    ],
)
def test_phrase_for_error_when_validation_fails_does_phrase_expected_shape(
    annotation: object, value: object, expected: str
):
    with pytest.raises(ValidationError) as exc:
        TypeAdapter(annotation).validate_python(value)
    error = exc.value.errors()[0]

    phrase = phrase_for_error(error)

    assert phrase == expected
