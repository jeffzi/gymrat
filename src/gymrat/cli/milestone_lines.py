"""Plain-mode milestone lines that more than one progress display prints.

The measure/compare display and the iterate display each announce a finished
prepare step; building that line here keeps the two commands' wording identical.
"""

from __future__ import annotations

from gymrat.utils import format_duration


def prepare_milestone(label: str, elapsed_ms: float) -> str:
    """Return the milestone line announcing a finished prepare step.

    Args:
        label: The target whose prepare step finished.
        elapsed_ms: How long the prepare step ran, in milliseconds.

    Returns:
        The milestone line without its timestamp prefix.
    """
    return f"prepared {label} ({format_duration(elapsed_ms)})"
