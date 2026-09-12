"""Shared ANSI-stripping and SGR-inspection helpers for test modules."""

import re

ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
"""Matches any ANSI escape sequence (CSI + final byte), not just SGR."""

SGR_RE = re.compile(r"\x1b\[([0-9;]*)m")
"""Matches SGR (Select Graphic Rendition) sequences only."""

SGR_BOLD = 1
SGR_DIM = 2
SGR_RED = 31
SGR_GREEN = 32
SGR_YELLOW = 33
SGR_BLUE = 34
SGR_CYAN = 36


def strip_ansi(text: str) -> str:
    """Remove all ANSI escape sequences from ``text``."""
    return ANSI_RE.sub("", text)


def strip_sgr(text: str) -> str:
    """Remove SGR sequences only, preserving cursor-control escapes."""
    return SGR_RE.sub("", text)


def has_sgr(text: str, code: int) -> bool:
    """True when ``text`` contains an ANSI SGR sequence with parameter ``code``."""
    target = str(code)
    return any(target in match.group(1).split(";") for match in SGR_RE.finditer(text))


def assert_has_sgr(lines: list[str], code: int) -> None:
    """Assert ``lines`` is non-empty and at least one line carries SGR ``code``."""
    assert lines
    assert any(has_sgr(line, code) for line in lines)
