"""Report rendering package.

Public style and color primitives are re-exported here so callers can import
them from :mod:`gymrat_py.report` without reaching into submodules.
"""

from gymrat_py.report.style import (
    AGGREGATE_LABEL_STYLE,
    GROUP_LABEL_STYLE,
    LABEL_DISPLAY_WIDTH,
    VARIANT_NAME_STYLE,
    VERDICT_STYLES,
    format_hint_label,
    highlight_inline_code,
    make_capture_console,
    render_lines,
    shorten_label,
    truncate_labels,
)

__all__ = [
    "AGGREGATE_LABEL_STYLE",
    "GROUP_LABEL_STYLE",
    "LABEL_DISPLAY_WIDTH",
    "VARIANT_NAME_STYLE",
    "VERDICT_STYLES",
    "format_hint_label",
    "highlight_inline_code",
    "make_capture_console",
    "render_lines",
    "shorten_label",
    "truncate_labels",
]
