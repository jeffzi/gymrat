"""Shared ANSI-stripping and SGR-inspection helpers for test modules."""

import re

ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
"""Matches any ANSI escape sequence (CSI + final byte), not just SGR."""

SGR_RE = re.compile(r"\x1b\[([0-9;]*)m")
"""Matches SGR (Select Graphic Rendition) sequences only."""

TRAILING_SGR_RUN = re.compile(r"(?:\x1b\[[0-9;]*m)*$")
"""Matches the run of SGR escapes a string ends on, with nothing between them."""


def strip_ansi(text: str) -> str:
    """Remove every CSI escape sequence from ``text``, cursor controls included, not only SGR."""
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


def warning_lines(stderr: str) -> list[str]:
    """The lines of ``stderr`` that open with ``warning: ``, stripped of color."""
    return [line for line in strip_ansi(stderr).splitlines() if line.startswith("warning: ")]


def normalize(text: str) -> str:
    """Strip ANSI codes then collapse whitespace, so a reflowed block matches."""
    return " ".join(strip_ansi(text).split())


_EXTENDED_COLOR_INTRODUCERS = frozenset({"38", "48", "58"})
"""SGR codes that open a 256-color or truecolor run instead of naming an attribute."""

_EXTENDED_COLOR_LENGTHS = {"5": 3, "2": 5}
"""Parameters an extended color run spans, by its mode: ``38;5;n`` or ``38;2;r;g;b``."""


def sgr_attributes(params: list[str]) -> list[str]:
    """Group one escape's SGR parameters into attributes, keeping color runs whole.

    A 256-color (``38;5;n``) or truecolor (``38;2;r;g;b``) run stays one entry, so
    its numeric components never read as attributes such as ``2`` (dim) or ``1``
    (bold).

    Args:
        params: The escape's parameter list, already split on ``;``.

    Returns:
        One entry per attribute, in order; resets and empty parameters are kept.
    """
    attributes: list[str] = []
    index = 0
    while index < len(params):
        param = params[index]
        mode = params[index + 1] if index + 1 < len(params) else ""
        if param in _EXTENDED_COLOR_INTRODUCERS and mode in _EXTENDED_COLOR_LENGTHS:
            span = _EXTENDED_COLOR_LENGTHS[mode]
            attributes.append(";".join(params[index : index + span]))
            index += span
            continue
        attributes.append(param)
        index += 1
    return attributes
