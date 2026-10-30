"""Tests for drawing a planned table body.

The body snapshot pins the rule placement only a planned body reaches: two rules
in a row with no row between them, whose flush also draws a rule closing a batch.
The ordinary rules between a header, its rows and an aggregate row are pinned by
the whole-report snapshots. The value-cell tests pin how a magnitude and its
spread are aligned in their fields.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from rich.text import Text

from gymrat.report.format import SPREAD_SEPARATOR, MetricCellParts
from gymrat.report.table.grid import (
    AggregateLine,
    HeaderLine,
    RuleLine,
    ValueWidths,
    join_value_cell,
    render_body,
)

if TYPE_CHECKING:
    from syrupy.assertion import SnapshotAssertion

    from gymrat.report.table.grid import BodyLine

# ---------------------------------------------------------------------------
# body rules
# ---------------------------------------------------------------------------


def _text_cells(line: BodyLine[str, str]) -> tuple[Text, ...]:
    """Two plain ``Text`` cells for a header or aggregate line."""
    match line:
        case HeaderLine():
            return (Text("metric"), Text("value"))
        case AggregateLine(label=label, cell=cell):
            return (Text(label), Text(cell))
        case _:
            msg = f"no cells for {line!r}"
            raise AssertionError(msg)


def test_render_body_when_two_rules_have_no_row_between_does_draw_one_rule_line_per_rule(
    snapshot: SnapshotAssertion,
):
    body: list[BodyLine[str, str]] = [
        HeaderLine(),
        RuleLine(),
        RuleLine(),
        AggregateLine("geomean", "-3.2%"),
    ]

    lines = render_body(body, [8, 6], _text_cells, color=False)

    assert lines == snapshot


# ---------------------------------------------------------------------------
# join_value_cell
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("parts", "widths", "expected"),
    [
        pytest.param(
            MetricCellParts(magnitude="1.2ms", spread="3%"),
            ValueWidths(magnitude=6, spread=4),
            f" 1.2ms{SPREAD_SEPARATOR}  3%",
            id="spread-present",
        ),
        pytest.param(
            MetricCellParts(magnitude="1.2ms", spread=""),
            ValueWidths(magnitude=6, spread=4),
            " 1.2ms       ",
            id="spread-absent-in-a-spread-column",
        ),
        pytest.param(
            MetricCellParts(magnitude="1.2ms", spread=""),
            ValueWidths(magnitude=6, spread=0),
            " 1.2ms",
            id="column-without-spreads",
        ),
    ],
)
def test_join_value_cell_when_spreads_vary_does_right_align_magnitude_and_spread(
    parts: MetricCellParts, widths: ValueWidths, expected: str
):
    assert join_value_cell(parts, widths) == expected
