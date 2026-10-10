"""Tests for a table's verdict cells and its geomean styles and lookups.

A verdict cell's plain text is the column's width source, and its styles sit on
the glyph, the delta (or the word standing in for it) and the band, never on
padding.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from rich.text import Text

from gymrat.report.table.cells import (
    NO_AGGREGATE,
    VerdictParts,
    VerdictWidths,
    geomean_value_style,
    group_geomean_of,
    kind_geomean_of,
    verdict_cell,
)
from tests.report._comparisons import create_candidate, memory_kind, time_kind
from tests.report._verdicts import geomean_of

if TYPE_CHECKING:
    from gymrat.model import GeomeanResult

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

    assert isinstance(cell, Text)
    assert cell.plain == plain
    assert [(cell.plain[span.start : span.end], span.style) for span in cell.spans] == styled


# ---------------------------------------------------------------------------
# geomean_value_style
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("geomean", "expected"),
    [
        pytest.param(geomean_of(value=-6, band=5), "bold green", id="improvement-past-band"),
        pytest.param(geomean_of(value=6, band=5), "bold red", id="regression-past-band"),
        pytest.param(geomean_of(value=-5, band=5), "bold", id="improvement-level-with-band"),
        pytest.param(geomean_of(value=5, band=5), "bold", id="regression-level-with-band"),
        pytest.param(geomean_of(value=-0.2, band=0), "bold green", id="improvement-no-band"),
        pytest.param(geomean_of(value=float("nan"), n=0), "bold", id="no-stable-metrics"),
    ],
)
def test_geomean_value_style_when_value_inside_or_beyond_band_does_color_only_beyond_it(
    geomean: GeomeanResult, expected: str
):
    assert geomean_value_style(geomean, []) == expected


# ---------------------------------------------------------------------------
# kind_geomean_of / group_geomean_of — reading a candidate's aggregate back out
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kind", "expected"),
    [
        pytest.param("memory", memory_kind().geomean, id="reported-kind"),
        pytest.param("cpu", NO_AGGREGATE, id="unreported-kind"),
    ],
)
def test_kind_geomean_of_when_looking_up_a_kind_does_return_its_geomean_or_no_aggregate(
    kind: str, expected: GeomeanResult
):
    candidate = create_candidate(kinds=[time_kind(), memory_kind()])

    geomean = kind_geomean_of(candidate, kind)

    assert geomean == expected


@pytest.mark.parametrize(
    ("kind", "group", "expected"),
    [
        pytest.param("time", "entity", time_kind().groups[0].geomean, id="reported-group"),
        pytest.param("time", "render", NO_AGGREGATE, id="unreported-group"),
        pytest.param("cpu", "entity", NO_AGGREGATE, id="unreported-kind"),
    ],
)
def test_group_geomean_of_when_looking_up_a_group_does_return_its_geomean_or_no_aggregate(
    kind: str, group: str, expected: GeomeanResult
):
    candidate = create_candidate(kinds=[time_kind(), memory_kind()])

    geomean = group_geomean_of(candidate, kind, group)

    assert geomean == expected
