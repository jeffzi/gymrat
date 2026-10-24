"""Tests for drawing a planned table body.

These pin the two rule placements only a planned body reaches: two rules in a row
with no row between them, and a rule closing the body. The ordinary rules between
a header, its rows and an aggregate row are pinned by the whole-report snapshots.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from rich.text import Text

from gymrat.report.table.grid import (
    AggregateLine,
    HeaderLine,
    MetricLine,
    RuleLine,
    render_body,
)

if TYPE_CHECKING:
    from syrupy.assertion import SnapshotAssertion

    from gymrat.report.table.grid import BodyLine

# ---------------------------------------------------------------------------
# body rules
# ---------------------------------------------------------------------------


def _text_cells(line: BodyLine[str, str]) -> tuple[Text, ...]:
    """Two plain ``Text`` cells for a header, metric, or aggregate line."""
    match line:
        case HeaderLine():
            return (Text("metric"), Text("value"))
        case MetricLine(row=row):
            return (Text(row), Text("1"))
        case AggregateLine(label=label, cell=cell):
            return (Text(label), Text(cell))
        case _:
            msg = f"no cells for {line!r}"
            raise AssertionError(msg)


@pytest.mark.parametrize(
    "body",
    [
        pytest.param(
            [HeaderLine(), RuleLine(), RuleLine(), AggregateLine("geomean", "-3.2%")],
            id="two-rules-with-no-row-between",
        ),
        pytest.param([HeaderLine(), RuleLine()], id="rule-closing-the-body"),
    ],
)
def test_render_body_when_rules_placed_does_draw_one_rule_line_per_rule(
    body: list[BodyLine[str, str]], snapshot: SnapshotAssertion
):
    lines = render_body(body, [8, 6], _text_cells, color=False)

    assert lines == snapshot
