"""Shared Markdown-slicing helper for tests that read generated or bundled docs."""

import re


def md_section(md: str, start_marker: str | None, stop_pattern: str | None) -> str:
    """Slice one section out of a Markdown document.

    Args:
        md: The Markdown text to slice.
        start_marker: Text the slice starts at; ``None`` starts at the top.
        stop_pattern: Regular expression whose first match after the start ends
            the slice; ``None`` runs to the end.

    Returns:
        The text from the start marker up to, not including, the stop match.

    Raises:
        ValueError: When ``start_marker`` is absent, instead of returning a wrong slice.
    """
    start = md.index(start_marker) if start_marker is not None else 0
    stop = re.compile(stop_pattern).search(md, start + 1) if stop_pattern is not None else None
    return md[start:] if stop is None else md[start : stop.start()]
