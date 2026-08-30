"""Rendered-text assertion helpers for report formatting tests."""

from __future__ import annotations

import re

from tests._ansi import SGR_RE, strip_ansi

# The column separator every rendered table row is split on.
_SEPARATOR = "│"
# A trailing run of SGR escapes with nothing but escapes between them and the end.
_TRAILING_SGR_RUN = re.compile(r"(?:\x1b\[[0-9;]*m)*$")
# A table rule: dashes meeting the first column separator at a crossing junction.
_RULE = re.compile(r"^─+┼")
# A section border: only dashes and top-T junctions, edge to edge.
_BORDER = re.compile(r"^[─┬]+$")
# A line dimmed end to end: opens with SGR 2 and closes with a reset. Rich closes
# a dim span with a full reset (SGR 0) rather than the incremental dim-off SGR 22.
DIMMED_LINE = re.compile(r"^\x1b\[2m.*\x1b\[0m$")


def table_rows(report: str) -> list[str]:
    """Every rendered table row of ``report``, styling stripped, in report order."""
    return [strip_ansi(line) for line in report.split("\n") if _SEPARATOR in line]


def cells_of(line: str) -> list[str]:
    """The cells of a rendered table line, padding included."""
    return line.split(_SEPARATOR)


def delta_cell(line: str) -> str:
    """The last cell of a rendered table line — the delta/verdict column."""
    return cells_of(line)[-1]


def last_table_row(report: str) -> str:
    """The last rendered table row of a report — the row the table closes on."""
    return table_rows(report)[-1]


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
    opening one. Pass ``last`` to read the trailing occurrence of a repeated
    marker instead of the leading one.

    Rich packs several parameters into one escape (``\\x1b[1;4m``), so each run is
    split on both the escape boundaries and the ``;`` inside them.
    """
    index = line.rfind(marker) if last else line.find(marker)
    if index == -1:
        msg = f"no {marker!r} in line: {line!r}"
        raise AssertionError(msg)
    run = _TRAILING_SGR_RUN.search(line[:index])
    opened = run.group(0) if run is not None else ""
    params: list[str] = []
    for escape in SGR_RE.finditer(opened):
        params.extend(param for param in escape.group(1).split(";") if param not in {"", "0"})
    return params


def offsets_of(line: str, glyph: str) -> list[int]:
    """Character offsets of every occurrence of ``glyph`` in a rendered line."""
    offsets: list[int] = []
    start = line.find(glyph)
    while start != -1:
        offsets.append(start)
        start = line.find(glyph, start + 1)
    return offsets


def separator_offsets(line: str) -> list[int]:
    """Character offsets of every column separator in a rendered table line.

    Two lines whose separators sit at the same offsets have aligned columns.
    """
    return [index for index, char in enumerate(line) if char == _SEPARATOR]


def separator_styles(line: str) -> list[list[str]]:
    """The SGR parameters still open at each column separator of ``line``.

    A separator that inherits its row's style reports that style here; one left in
    the terminal's default color reports nothing.
    """
    closers: dict[str, re.Pattern[str]] = {
        "0": re.compile(r"^\d+$"),
        "22": re.compile(r"^[12]$"),
        "23": re.compile(r"^3$"),
        "24": re.compile(r"^4$"),
        "39": re.compile(r"^(?:3[0-7]|9[0-7])$"),
        "49": re.compile(r"^(?:4[0-7]|10[0-7])$"),
    }
    open_params: list[str] = []
    styles: list[list[str]] = []
    for token in re.finditer(r"\x1b\[([0-9;]*)m|│", line):
        if token.group(0) == _SEPARATOR:
            styles.append(list(open_params))
            continue
        for param in token.group(1).split(";"):
            if param == "":
                continue
            closes = closers.get(param)
            if closes is None:
                open_params.append(param)
            else:
                open_params = [p for p in open_params if not closes.match(p)]
    return styles


def table_shape(report: str) -> list[str]:
    """One entry per report line, coarse enough to read as a layout.

    A table row collapses to its first cell, a header rule collapses to
    ``"<rule>"``, a section's top border to ``"<border>"``, and every other line
    stays as its plain text.
    """
    shape: list[str] = []
    for line in report.split("\n"):
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
    return table_shape(report)[: last + 1]


def highlight_lines(report: str) -> list[str]:
    """The lines of the ``highlights`` block, its heading excluded.

    The block runs from the line after the ``highlights`` heading down to the
    next blank line (or the end of the report). Lines keep their styling, so the
    color tests can read the SGR parameters off a highlight entry. An absent
    block yields an empty list.
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
