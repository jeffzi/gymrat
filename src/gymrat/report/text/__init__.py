"""Text subpackage: human-readable compare, measure, and probe reports."""

from gymrat.report.text.probe import render_probe_report
from gymrat.report.text.render import (
    format_cleanup_failures,
    paired_samples,
    render_measure_report,
    render_report,
    with_display_labels,
)

__all__ = [
    "format_cleanup_failures",
    "paired_samples",
    "render_measure_report",
    "render_probe_report",
    "render_report",
    "with_display_labels",
]
