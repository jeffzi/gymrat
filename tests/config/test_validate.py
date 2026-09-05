"""Tests for ``flag_problem``."""

import pytest

from gymrat.config import flag_problem

# ---------------------------------------------------------------------------
# flag_problem
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value",
    [
        pytest.param("", id="empty"),
        pytest.param("   ", id="whitespace-spaces"),
        pytest.param("\t", id="whitespace-tab"),
    ],
)
def test_flag_problem_when_value_blank_does_return_problem_naming_flag(value: str):
    result = flag_problem("bench", value)

    assert result is not None
    assert "--bench" in result
    assert "non-empty" in result


def test_flag_problem_when_value_none_does_return_none():
    assert flag_problem("bench", None) is None


def test_flag_problem_when_value_non_empty_does_return_none():
    assert flag_problem("bench", "real-command") is None
