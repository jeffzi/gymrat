"""Tests for a table's verdict cells and its geomean styles.

A verdict cell's plain text is the column's width source, and its styles sit on
the glyph, the delta (or the word standing in for it) and the band, never on
padding.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from gymrat.report.format import PLUS_MINUS
from gymrat.report.style import EMPHASIZED_DIRECTION_STYLES
from gymrat.report.table.cells import (
    DETECTS_PREFIX,
    VerdictParts,
    VerdictWidths,
    geomean_value_style,
    verdict_cell,
    verdict_widths,
)
from tests.report._verdicts import geomean_of

if TYPE_CHECKING:
    from gymrat.model import GeomeanResult
    from gymrat.report.display import DisplayClass

# ---------------------------------------------------------------------------
# styled verdict cell
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("parts", "widths", "delta_style", "band_style", "expected"),
    [
        pytest.param(
            VerdictParts(glyph="✓", delta="-10.0%", word="", band="2.5%", pairs="n=8"),
            VerdictWidths(delta=7, band=4),
            "red",
            "dim",
            (
                f"✓   -10.0%  {PLUS_MINUS}2.5%  n=8",
                [("✓", "green"), ("-10.0%", "red"), (f"{PLUS_MINUS}2.5%", "dim")],
            ),
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
            VerdictParts(glyph="✓", delta="-10.0%", word="", band="2.5%", pairs="n=8"),
            VerdictWidths(delta=7, band=4, detects=4),
            "red",
            "dim",
            (
                f"✓   -10.0%  {PLUS_MINUS}2.5%  n=8",
                [("✓", "green"), ("-10.0%", "red"), (f"{PLUS_MINUS}2.5%", "dim")],
            ),
            id="detects-absent-with-pairs-reserves-nothing",
        ),
        pytest.param(
            VerdictParts(
                glyph="~", delta="+1.0%", word="", band="2.5%", pairs="n=8", detects="3.0%"
            ),
            VerdictWidths(delta=5, band=4, detects=4, pairs=4),
            "red",
            "dim",
            (
                f"~  +1.0%  {PLUS_MINUS}2.5%  n=8   {DETECTS_PREFIX}3.0%",
                [
                    ("~", "green"),
                    ("+1.0%", "red"),
                    (f"{PLUS_MINUS}2.5%", "dim"),
                    (f"{DETECTS_PREFIX}3.0%", "dim"),
                ],
            ),
            id="detects-present-pads-the-pair-count",
        ),
        pytest.param(
            VerdictParts(glyph="≈", delta="", word="unstable", band="", pairs="n=8"),
            VerdictWidths(delta=9, band=4),
            "red",
            "dim",
            ("≈  unstable          n=8", [("≈", "green"), ("unstable", "red")]),
            id="word-padded-to-a-wider-delta-column",
        ),
        pytest.param(
            VerdictParts(glyph="~", delta="+1.0%", word="", band="2.5%", pairs=""),
            VerdictWidths(delta=5, band=6),
            "red",
            "dim",
            (
                f"~  +1.0%  {PLUS_MINUS}  2.5%",
                [("~", "green"), ("+1.0%", "red"), (f"{PLUS_MINUS}  2.5%", "dim")],
            ),
            id="band-right-aligned-behind-its-sign",
        ),
    ],
)
def test_verdict_cell_when_parts_vary_does_pad_each_field_to_its_width_with_styles_on_text_only(
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

    assert cell.plain == plain
    assert [(cell.plain[span.start : span.end], span.style) for span in cell.spans] == styled


# ---------------------------------------------------------------------------
# verdict_widths
# ---------------------------------------------------------------------------


def test_verdict_widths_when_some_cells_state_a_detectable_change_does_measure_pairs_only_on_those():
    cells = [
        VerdictParts(glyph="~", delta="-1.0%", word="", band="2.5%", pairs="n=12345"),
        VerdictParts(glyph="~", delta="+10.0%", word="", band="12.5%", pairs="n=8", detects="3.0%"),
    ]

    widths = verdict_widths(cells)

    assert widths == VerdictWidths(delta=6, band=5, detects=4, pairs=3)


# ---------------------------------------------------------------------------
# geomean_value_style
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("geomean", "expected"),
    [
        pytest.param(
            geomean_of(value=-6, band=5),
            EMPHASIZED_DIRECTION_STYLES["improved"],
            id="improvement-past-band",
        ),
        pytest.param(
            geomean_of(value=6, band=5),
            EMPHASIZED_DIRECTION_STYLES["regressed"],
            id="regression-past-band",
        ),
        pytest.param(geomean_of(value=-5, band=5), "bold", id="improvement-level-with-band"),
        pytest.param(geomean_of(value=5, band=5), "bold", id="regression-level-with-band"),
    ],
)
def test_geomean_value_style_when_value_inside_or_beyond_band_does_color_only_beyond_it(
    geomean: GeomeanResult, expected: str
):
    assert geomean_value_style(geomean, []) == expected


@pytest.mark.parametrize(
    ("outcomes", "expected"),
    [
        pytest.param(["within-noise"], "bold", id="every-outcome-quiet-vetoes-color"),
        pytest.param([None, "within-noise"], "bold", id="missing-outcomes-ignored-by-the-veto"),
        pytest.param([None], EMPHASIZED_DIRECTION_STYLES["improved"], id="no-outcome-leaves-band"),
        pytest.param(
            ["within-noise", "improved"],
            EMPHASIZED_DIRECTION_STYLES["improved"],
            id="one-moving-outcome-lifts-veto",
        ),
    ],
)
def test_geomean_value_style_when_outcomes_vary_does_veto_color_only_when_every_one_is_quiet(
    outcomes: list[DisplayClass | None], expected: str
):
    geomean = geomean_of(value=-6, band=5)

    style = geomean_value_style(geomean, outcomes)

    assert style == expected
