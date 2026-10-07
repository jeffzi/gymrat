"""Shared ANSI-stripping and SGR-inspection helpers for test modules."""

import re

ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
"""Matches any ANSI escape sequence (CSI + final byte), not just SGR."""

SGR_RE = re.compile(r"\x1b\[([0-9;]*)m")
"""Matches SGR (Select Graphic Rendition) sequences only."""

TRAILING_SGR_RUN = re.compile(r"(?:\x1b\[[0-9;]*m)*$")
"""Matches the run of SGR escapes a string ends on, with nothing between them."""

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


def stripped_lines(text: str, *, keep_blank: bool) -> list[str]:
    """Split ``text`` into lines with SGR sequences and surrounding whitespace removed.

    Args:
        text: The rendered text, one line per newline.
        keep_blank: Whether whitespace-only lines stay in the result, as empty strings.

    Returns:
        The stripped lines, in order.
    """
    return [strip_sgr(line).strip() for line in text.split("\n") if keep_blank or line.strip()]


def normalize(text: str) -> str:
    """Strip ANSI codes then collapse whitespace, so a reflowed block matches."""
    return " ".join(strip_ansi(text).split())


def sgr_params(text: str) -> str:
    """The SGR parameter list of the last SGR escape in ``text``.

    Asserts which attributes a styled span carries without pinning the exact
    escape bytes rich emits.

    Args:
        text: Rendered output holding at least one SGR escape.

    Returns:
        The escape's raw parameter list, such as ``"2;34"``.
    """
    match = list(SGR_RE.finditer(text))[-1]
    return match.group(1)


def sgr_codes(text: str) -> set[str]:
    """Every SGR parameter code present in ``text``, resets left out."""
    codes: set[str] = set()
    for escape in SGR_RE.finditer(text):
        codes.update(param for param in escape.group(1).split(";") if param not in {"", "0"})
    return codes


def has_sgr(text: str, code: int) -> bool:
    """True when ``text`` opens the SGR style ``code``; a reset (``0``) is never a style."""
    return str(code) in sgr_codes(text)


def assert_has_sgr(lines: list[str], code: int) -> None:
    """Assert ``lines`` is non-empty and at least one line carries SGR ``code``."""
    assert lines
    assert any(has_sgr(line, code) for line in lines)
