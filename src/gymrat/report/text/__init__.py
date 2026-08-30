"""Text subpackage: human-readable compare and measure reports."""

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
    "render_report",
    "with_display_labels",
]
