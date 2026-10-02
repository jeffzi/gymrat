"""Tests for the styled cell a verdict column is built from.

Its plain text is the column's width source, and its styles sit on the glyph,
the delta (or the word standing in for it) and the band, never on padding.
"""

from __future__ import annotations

import pytest
from rich.text import Text

from gymrat.report.table.markup import VerdictParts, VerdictWidths, verdict_cell

# ---------------------------------------------------------------------------
# styled verdict cell
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("parts", "widths", "delta_style", "band_style", "expected"),
    [
        pytest.param(
            VerdictParts(glyph="~", delta="", word="", band="2.5%", pairs=""),
            VerdictWidths(delta=7, band=4),
            None,
            "dim",
            ("~           ±2.5%", [("~", "green"), ("±2.5%", "dim")]),
            id="delta-empty-band-present",
        ),
        pytest.param(
            VerdictParts(glyph="✓", delta="-10.0%", word="", band="2.5%", pairs="n=8"),
            VerdictWidths(delta=7, band=4),
            "red",
            "dim",
            ("✓   -10.0%  ±2.5%  n=8", [("✓", "green"), ("-10.0%", "red"), ("±2.5%", "dim")]),
            id="all-fields-present",
        ),
        pytest.param(
            VerdictParts(glyph="✓", delta="-10.0%", word="", band="", pairs="n=8"),
            VerdictWidths(delta=7, band=4),
            "red",
            "dim",
            ("✓   -10.0%         n=8", [("✓", "green"), ("-10.0%", "red")]),
            id="band-absent-with-pairs-reserves-band-slot",
        ),
        pytest.param(
            VerdictParts(glyph="≈", delta="", word="unstable", band="", pairs=""),
            VerdictWidths(delta=6, band=0),
            "red",
            None,
            ("≈  unstable", [("≈", "green"), ("unstable", "red")]),
            id="word-stands-in-for-delta",
        ),
        pytest.param(
            VerdictParts(glyph="~", delta="+4.0%", word="", band="", pairs=""),
            VerdictWidths(delta=6, band=0),
            None,
            None,
            ("~   +4.0%", [("~", "green")]),
            id="delta-unstyled-band-absent",
        ),
    ],
)
def test_verdict_cell_when_fields_padded_to_widths_does_style_only_field_text(
    parts: VerdictParts,
    widths: VerdictWidths,
    delta_style: str | None,
    band_style: str | None,
    expected: tuple[str, list[tuple[str, str]]],
):
    plain, styled = expected

    cell = verdict_cell(
        parts, widths, glyph_style="green", delta_style=delta_style, band_style=band_style
    )

    assert isinstance(cell, Text)
    assert cell.plain == plain
    assert [(cell.plain[span.start : span.end], span.style) for span in cell.spans] == styled
