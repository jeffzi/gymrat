"""Tests for drawing a planned table body.

These pin the rules a planned body draws between a header and its rows and ahead
of an aggregate row, including two rules in a row and a rule closing the body.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from rich.text import Text

from gymrat.report.table.render import (
    AggregateLine,
    HeaderLine,
    MetricLine,
    RuleLine,
    render_body,
)

if TYPE_CHECKING:
    from gymrat.report.table.render import BodyLine

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
    ("body", "expected"),
    [
        pytest.param(
            [
                HeaderLine(),
                RuleLine(),
                MetricLine("decode"),
                RuleLine(),
                AggregateLine("geomean", "-3.2%"),
            ],
            [
                "metric   │ value",
                "─────────┼───────",
                "decode   │ 1",
                "─────────┼───────",
                "geomean  │ -3.2%",
            ],
            id="rule-after-header-and-before-aggregate",
        ),
        pytest.param(
            [HeaderLine(), RuleLine(), RuleLine(), AggregateLine("geomean", "-3.2%")],
            [
                "metric   │ value",
                "─────────┼───────",
                "─────────┼───────",
                "geomean  │ -3.2%",
            ],
            id="two-rules-with-no-row-between",
        ),
        pytest.param(
            [HeaderLine(), RuleLine()],
            ["metric   │ value", "─────────┼───────"],
            id="rule-closing-the-body",
        ),
    ],
)
def test_render_body_when_rules_placed_does_draw_one_rule_line_per_rule(
    body: list[BodyLine[str, str]], expected: list[str]
):
    lines = render_body(body, [8, 6], _text_cells, color=False)

    assert lines == expected
