"""Shared ANSI-stripping helpers for test modules."""

import re

ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
"""Matches any ANSI escape sequence (CSI + final byte), not just SGR."""

SGR_RE = re.compile(r"\x1b\[([0-9;]*)m")
"""Matches SGR (Select Graphic Rendition) sequences only."""


def strip_ansi(text: str) -> str:
    """Remove all ANSI escape sequences from ``text``."""
    return ANSI_RE.sub("", text)


def strip_sgr(text: str) -> str:
    """Remove SGR sequences only, preserving cursor-control escapes."""
    return SGR_RE.sub("", text)
