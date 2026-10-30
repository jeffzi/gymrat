"""Tests for the SGR-grouping helper the rendering tests assert through."""

import pytest

from tests._ansi import sgr_attributes


@pytest.mark.parametrize(
    ("params", "expected"),
    [
        pytest.param(["2", "34"], ["2", "34"], id="plain-attributes"),
        pytest.param(["38", "2", "1", "2", "3"], ["38;2;1;2;3"], id="truecolor-foreground"),
        pytest.param(["1", "48", "5", "2"], ["1", "48;5;2"], id="256-color-background"),
    ],
)
def test_sgr_attributes_when_params_carry_styles_does_keep_each_extended_color_run_whole(
    params: list[str], expected: list[str]
):
    attributes = sgr_attributes(params)

    assert attributes == expected
