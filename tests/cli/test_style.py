"""Tests for CLI style vocabulary constants and theme wiring.

Running-state elements (spinners, in-flight timers) render cyan.  Alert
surfaces (idle warnings, caps) render yellow.  The theme entries that Rich
progress columns hard-code (``progress.spinner``, ``progress.elapsed``) follow
the style constants so a colour change in one place propagates everywhere.
"""

from rich.style import Style

from gymrat.cli.style import (
    CLI_THEME,
    STYLE_ALERT,
    STYLE_RUNNING,
    STYLE_TIMER_RUNNING,
)

# ---------------------------------------------------------------------------
# running vs alert colour split
# ---------------------------------------------------------------------------


def test_style_running_when_referenced_does_equal_cyan():
    assert STYLE_RUNNING == "cyan"


def test_style_timer_running_when_referenced_does_equal_cyan():
    assert STYLE_TIMER_RUNNING == "cyan"


def test_style_alert_when_referenced_does_equal_yellow():
    assert STYLE_ALERT == "yellow"


# ---------------------------------------------------------------------------
# CLI_THEME wiring
# ---------------------------------------------------------------------------


def test_cli_theme_when_spinner_resolved_does_match_running_style():
    assert CLI_THEME.styles["progress.spinner"] == Style.parse(STYLE_RUNNING)


def test_cli_theme_when_elapsed_resolved_does_match_timer_running_style():
    assert CLI_THEME.styles["progress.elapsed"] == Style.parse(STYLE_TIMER_RUNNING)
