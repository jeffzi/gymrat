"""Tests for CLI theme wiring.

Running-state elements (spinners, in-flight timers) and alert surfaces (idle
warnings, caps) take their colors from the style constants. The theme entries
that Rich progress columns hard-code (``progress.spinner``,
``progress.elapsed``) follow those constants so a color change in one place
propagates everywhere.
"""

import pytest
from rich.style import Style

from gymrat.cli.style import CLI_THEME, STYLE_RUNNING, STYLE_TIMER_RUNNING


@pytest.mark.parametrize(
    ("theme_key", "style"),
    [
        pytest.param("progress.spinner", STYLE_RUNNING, id="spinner-running"),
        pytest.param("progress.elapsed", STYLE_TIMER_RUNNING, id="elapsed-timer-running"),
    ],
)
def test_cli_theme_when_progress_entry_resolved_does_match_its_style_constant(
    theme_key: str, style: str
):
    assert CLI_THEME.styles[theme_key] == Style.parse(style)
