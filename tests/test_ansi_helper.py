"""Tests for the SGR-inspection helper the rendering tests assert through."""

import pytest

from tests._ansi import sgr_codes


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        pytest.param("\x1b[2;34mx\x1b[0m", {"2", "34"}, id="plain-attributes"),
        pytest.param("\x1b[38;2;1;2;3mx", {"38;2;1;2;3"}, id="truecolor-foreground"),
        pytest.param("\x1b[1;48;5;2mx", {"1", "48;5;2"}, id="256-color-background"),
    ],
)
def test_sgr_codes_when_text_carries_styles_does_keep_each_extended_color_run_whole(
    text: str, expected: set[str]
):
    assert sgr_codes(text) == expected
