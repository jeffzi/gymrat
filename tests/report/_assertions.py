"""Rendered-text assertion helpers for report formatting tests."""

from __future__ import annotations

import re

from gymrat.report.style import render_lines
from tests._ansi import (
    SGR_RE,
    TRAILING_SGR_RUN,
    strip_ansi,
)

# The column separator every rendered table row is split on.
_SEPARATOR = "│"
# A table rule: dashes meeting the first column separator at a crossing junction.
_RULE = re.compile(r"^─+┼")
# A section border: only dashes and top-T junctions, edge to edge.
_BORDER = re.compile(r"^[─┬]+$")


def render_plain(*markup: str) -> str:
    """The rich ``markup`` rendered with color off, one line per argument."""
    return render_lines(*markup, color=False)


def render_colored(*markup: str) -> str:
    """The rich ``markup`` rendered with color on, one line per argument."""
    return render_lines(*markup, color=True)


def table_rows(report: str) -> list[str]:
    """Every rendered table row of ``report``, styling stripped, in report order."""
    return [strip_ansi(line) for line in report.split("\n") if _SEPARATOR in line]


def cells_of(line: str) -> list[str]:
    """The cells of a rendered table line, padding included."""
    return line.split(_SEPARATOR)


def stripped_cells(line: str) -> list[str]:
    """The cells of a rendered table line, padding stripped."""
    return [cell.strip() for cell in cells_of(line)]


def rule_lines(report: str) -> list[str]:
    """Every table rule of ``report``, styling stripped, in report order."""
    return [bare for line in report.split("\n") if _RULE.match(bare := strip_ansi(line))]


def delta_cell(line: str) -> str:
    """The last cell of a rendered table line — the delta/verdict column."""
    return cells_of(line)[-1]


def line_starting_with(report: str, prefix: str) -> str:
    """The single rendered line starting with ``prefix``, or a failure naming the report."""
    for candidate in report.split("\n"):
        if candidate.startswith(prefix):
            return candidate
    msg = f"no line starting with {prefix!r} in report:\n{report}"
    raise AssertionError(msg)


def line_containing(report: str, needle: str) -> str:
    """The first rendered line containing ``needle``, or a failure naming the report.

    A colored line starts with escape codes rather than its text, so the color
    tests match on content instead of a prefix.

    Args:
        report: The rendered report, lines joined by newlines.
        needle: The text the line must contain.

    Returns:
        The first line of ``report`` that contains ``needle``.

    Raises:
        AssertionError: No line of ``report`` contains ``needle``.
    """
    for candidate in report.split("\n"):
        if needle in candidate:
            return candidate
    msg = f"no line containing {needle!r} in report:\n{report}"
    raise AssertionError(msg)


def styles_at(line: str, marker: str, *, last: bool = False) -> list[str]:
    r"""The SGR parameters opened immediately before ``marker`` in ``line``.

    Only the unbroken run of escape sequences touching the marker counts, so a
    style opened at the start of the line does not leak into the result. A reset
    (``0`` or an empty parameter list) is dropped: it closes styles rather than
    opening one.

    Rich packs several parameters into one escape (``\\x1b[1;4m``), so each run is
    split on both the escape boundaries and the ``;`` inside them.

    Args:
        line: One rendered line, escape codes included.
        marker: The text whose opening styles are read.
        last: Read the trailing occurrence of a repeated marker instead of the
            leading one.

    Returns:
        The SGR parameters, in the order they were opened.

    Raises:
        AssertionError: ``marker`` does not occur in ``line``.
    """
    index = line.rfind(marker) if last else line.find(marker)
    if index == -1:
        msg = f"no {marker!r} in line: {line!r}"
        raise AssertionError(msg)
    run = TRAILING_SGR_RUN.search(line[:index])
    opened = run.group(0) if run is not None else ""
    params: list[str] = []
    for escape in SGR_RE.finditer(opened):
        params.extend(param for param in escape.group(1).split(";") if param not in {"", "0"})
    return params


def offsets_of(line: str, glyph: str) -> list[int]:
    """Character offsets of every occurrence of ``glyph`` in a rendered line.

    Two table lines whose ``│`` separators sit at the same offsets have aligned
    columns.

    Args:
        line: One rendered line.
        glyph: The text to locate.

    Returns:
        The start offset of each occurrence, left to right.
    """
    return [match.start() for match in re.finditer(re.escape(glyph), line)]


def table_region(report: str) -> list[str]:
    """The table region of a report: its shape down to the last table row."""
    lines = report.split("\n")
    last = -1
    for index, line in enumerate(lines):
        if _SEPARATOR in strip_ansi(line):
            last = index
    if last == -1:
        msg = f"no table rows in report:\n{report}"
        raise AssertionError(msg)
    shape: list[str] = []
    for line in lines[: last + 1]:
        bare = strip_ansi(line)
        if _RULE.match(bare):
            shape.append("<rule>")
        elif _BORDER.match(bare):
            shape.append("<border>")
        elif _SEPARATOR not in bare:
            shape.append(bare.rstrip())
        else:
            shape.append(cells_of(bare)[0].strip())
    return shape


def highlight_lines(report: str) -> list[str]:
    """The lines of the ``highlights`` block, its heading excluded.

    The block runs from the line after the ``highlights`` heading down to the
    next blank line (or the end of the report). Lines keep their styling, so the
    color tests can read the SGR parameters off a highlight entry.

    Args:
        report: The rendered report, lines joined by newlines.

    Returns:
        The block's lines, or an empty list when the report has no block.
    """
    lines = report.split("\n")
    start = next(
        (index for index, line in enumerate(lines) if strip_ansi(line) == "highlights"), -1
    )
    if start == -1:
        return []
    rest = lines[start + 1 :]
    try:
        end = rest.index("")
    except ValueError:
        return rest
    return rest[:end]
