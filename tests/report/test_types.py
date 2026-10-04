"""Tests for the report option and result types."""

import pytest

from gymrat.report.types import ReportOptions

# ---------------------------------------------------------------------------
# ReportOptions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("options", "expected"),
    [
        pytest.param(ReportOptions(), None, id="default-defers"),
        pytest.param(ReportOptions(color=True), True, id="forced-on"),
        pytest.param(ReportOptions(color=False), False, id="forced-off"),
    ],
)
def test_report_options_when_color_override_given_does_carry_it(
    options: ReportOptions, expected: bool | None
):
    assert options.color is expected
