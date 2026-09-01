"""The shared machinery both text tables draw through.

The data half builds the string content of each cell: the builders that pad a
value cell's magnitude and spread, and a verdict cell's glyph, delta and band,
into fields of their own, plus the body planner that lays a
:class:`~gymrat.report.sections.SectionLayout` out as titles, borders, rules
and rows.

The rendering half draws the grid. The box chrome — column padding, the ``│``
separators, the ``─`` rules and their ``┼``/``┬`` junctions — is delegated to a
:class:`rich.table.Table`, and styling is carried as rich markup that
:func:`~gymrat.report.style.render_lines` resolves to color once. In-cell
sub-field alignment stays in the string builders, because that is the behavior
the tests pin; only the grid around the cells is rich's.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, assert_never

from rich import box
from rich.cells import cell_len
from rich.table import Table
from rich.text import Text

from gymrat.report.sections import GroupBlock, MetricBlock, informational_tag
from gymrat.report.style import RENDER_WIDTH, markup, render_lines
from gymrat.report.table.markup import (
    CELL_GUTTER,
    METRIC_COLUMN_HEADER,
    METRIC_COLUMN_MIN,
    VALUE_COLUMN_MIN,
    VERDICT_COLUMN_MIN,
    AggregateColumnCell,
    ValueWidths,
    VerdictParts,
    VerdictWidths,
    geomean_column_cell,
    group_metric_cell,
    header_metric_cell,
    indented_section_label,
    join_value_cell,
    join_verdict_cell,
    style_verdict_cell,
    value_widths,
    verdict_parts,
    verdict_widths,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from gymrat.config import KindEntry
    from gymrat.report.sections import SectionLayout, SectionPlan


# ---------------------------------------------------------------------------
# Body planning
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AggregateRow[Cell]:
    """An aggregate row: the scope it covers, and the cell it states for the column."""

    label: str
    cell: Cell


@dataclass(frozen=True, slots=True)
class TitleLine:
    """A full-width markup line above a section — its informational tag."""

    text: str


@dataclass(frozen=True, slots=True)
class BlankLine:
    """An empty line separating blocks or sections."""


@dataclass(frozen=True, slots=True)
class HeaderLine:
    """The column-header row, carrying a section title when the table is sectioned."""

    title: str | None = None


@dataclass(frozen=True, slots=True)
class RuleLine:
    """The rule separating a header (or the body) from what follows."""


@dataclass(frozen=True, slots=True)
class BorderLine:
    """The top border opening a section's box."""


@dataclass(frozen=True, slots=True)
class GroupLine:
    """A group sub-header naming the block beneath it."""

    label: str


@dataclass(frozen=True, slots=True)
class MetricLine[Metric]:
    """A metric row, holding the row it draws."""

    row: Metric


@dataclass(frozen=True, slots=True)
class AggregateLine[Cell]:
    """An aggregate row, holding the scope label and the cell it states."""

    label: str
    cell: Cell


type BodyLine[Metric, Cell] = (
    TitleLine
    | BlankLine
    | HeaderLine
    | RuleLine
    | BorderLine
    | GroupLine
    | MetricLine[Metric]
    | AggregateLine[Cell]
)


@dataclass(frozen=True, slots=True)
class AggregateRows[Metric, Cell]:
    """The aggregate-row builders a table supplies — the half that differs per table.

    Attributes:
        group: Builds the aggregate closing one group of a kind.
        kind: Builds the aggregate closing one whole kind section.
        flat: Builds the single aggregate closing a flat table.
    """

    group: Callable[[str, str, Sequence[Metric]], AggregateRow[Cell]]
    kind: Callable[[str, Sequence[Metric]], AggregateRow[Cell]]
    flat: Callable[[Sequence[Metric]], AggregateRow[Cell]]


def _section_metrics[Metric](section: SectionPlan[Metric]) -> list[Metric]:
    """Every metric row a section holds, its group blocks flattened back into row order."""
    rows: list[Metric] = []
    for block in section.blocks:
        if isinstance(block, GroupBlock):
            rows.extend(block.metrics)
        else:
            rows.append(block.metric)
    return rows


def _plan_blocks[Metric, Cell](
    section: SectionPlan[Metric],
    rows: AggregateRows[Metric, Cell] | None,
) -> list[BodyLine[Metric, Cell]]:
    """The lines one section's blocks produce: groups, standalone metrics, sub-geomeans."""
    lines: list[BodyLine[Metric, Cell]] = []
    for index, block in enumerate(section.blocks):
        previous = section.blocks[index - 1] if index > 0 else None
        if previous is not None and (
            isinstance(previous, GroupBlock) or isinstance(block, GroupBlock)
        ):
            lines.append(BlankLine())

        if isinstance(block, MetricBlock):
            lines.append(MetricLine(row=block.metric))
            continue

        lines.append(GroupLine(label=block.group))
        lines.extend(MetricLine(row=metric) for metric in block.metrics)
        if rows is not None:
            aggregate = rows.group(section.kind, block.group, block.metrics)
            lines.append(AggregateLine(label=aggregate.label, cell=aggregate.cell))
    return lines


def plan_body[Metric, Cell](
    layout: SectionLayout[Metric],
    rows: AggregateRows[Metric, Cell] | None,
    annotation: Callable[[SectionPlan[Metric]], str | None],
) -> list[BodyLine[Metric, Cell]]:
    """The body of a table: its header, its metric rows, and the aggregate closing each scope.

    A run of one kind draws flat — a single header and one closing geomean. A run
    of several draws one boxed section per kind, each closed by its own geomean.

    Args:
        layout: The sectioned layout to lay out.
        rows: The aggregate-row builders, or ``None`` for a table with no geomeans.
        annotation: The informational tag for a section, or ``None`` when it gates.

    Returns:
        The body lines, in draw order.
    """
    if len(layout.sections) <= 1:
        return _plan_flat_body(layout, rows, annotation)

    lines: list[BodyLine[Metric, Cell]] = []
    for section in layout.sections:
        lines.append(BlankLine())
        tag = annotation(section)
        if tag is not None:
            lines.append(TitleLine(text=tag))
        lines.append(BorderLine())
        lines.append(HeaderLine(title=section.kind))
        lines.append(RuleLine())
        lines.extend(_plan_blocks(section, rows))
        if rows is not None:
            lines.append(RuleLine())
            aggregate = rows.kind(section.kind, _section_metrics(section))
            lines.append(AggregateLine(label=aggregate.label, cell=aggregate.cell))
    return lines


def _plan_flat_body[Metric, Cell](
    layout: SectionLayout[Metric],
    rows: AggregateRows[Metric, Cell] | None,
    annotation: Callable[[SectionPlan[Metric]], str | None],
) -> list[BodyLine[Metric, Cell]]:
    """The body a single-kind run draws: no box, one header, and one closing geomean."""
    body: list[BodyLine[Metric, Cell]] = []
    section = layout.sections[0] if layout.sections else None
    if section is not None:
        tag = annotation(section)
        if tag is not None:
            body.append(TitleLine(text=tag))
    body.append(HeaderLine())
    body.append(RuleLine())
    if section is not None and any(
        isinstance(block, GroupBlock) and len(block.metrics) > 1 for block in section.blocks
    ):
        # Flat layout shows one closing aggregate; suppress per-group aggregates.
        block_lines = _plan_blocks(section, None)
        kind = section.kind
        for i, line in enumerate(block_lines):
            if isinstance(line, GroupLine):
                block_lines[i] = GroupLine(label=f"{line.label} · {kind}")
        body.extend(block_lines)
    else:
        body.extend(MetricLine(row=row) for row in layout.ordered)
    if rows is not None:
        body.append(RuleLine())
        aggregate = rows.flat(layout.ordered)
        body.append(AggregateLine(label=aggregate.label, cell=aggregate.cell))
    return body


def aggregate_label_lengths[Metric, Cell](body: Sequence[BodyLine[Metric, Cell]]) -> list[int]:
    """The labels of every non-metric row — what the name column widens for."""
    return [cell_len(line.label) for line in body if isinstance(line, (GroupLine, AggregateLine))]


def widest_header_label[Metric, Cell](body: Sequence[BodyLine[Metric, Cell]]) -> str:
    """The widest label any header row carries, defaulting to the metric-column header."""
    widest = METRIC_COLUMN_HEADER
    for line in body:
        if isinstance(line, HeaderLine):
            label = line.title if line.title is not None else METRIC_COLUMN_HEADER
            if cell_len(label) > cell_len(widest):
                widest = label
    return widest


def compute_column_width(header_len: int, content_lengths: Sequence[int], minimum: int) -> int:
    """The width a column settles on: the widest of its content, its header, and its floor."""
    return max(minimum, header_len, max(content_lengths, default=0))


def section_annotation[Metric](
    section: SectionPlan[Metric],
    config_kinds: Mapping[str, KindEntry] | None,
) -> str | None:
    """A section's informational tag as dimmed markup, or ``None`` when the kind gates."""
    if section.has_gating:
        return None
    return markup(informational_tag(section.kind, config_kinds), "dim")


# ---------------------------------------------------------------------------
# Rendering engine
# ---------------------------------------------------------------------------


def _horizontal(widths: Sequence[int], junction: str) -> str:
    """A dashed rule or border, meeting each column separator at ``junction``.

    The dashes account for the padding rich draws with ``pad_edge`` off: the first
    column carries no left pad and the last no right pad, so their segments are one
    dash narrower than the interior columns'.
    """
    last = len(widths) - 1
    segments = [
        "─" * ((0 if index == 0 else 1) + width + (0 if index == last else 1))
        for index, width in enumerate(widths)
    ]
    return junction.join(segments)


def _make_table(widths: Sequence[int]) -> Table:
    """A rich table configured to draw the inner grid alone, columns fixed to ``widths``."""
    table = Table(
        box=box.SQUARE,
        show_edge=False,
        show_header=False,
        pad_edge=False,
        padding=(0, 1),
    )
    for width in widths:
        table.add_column(width=width, no_wrap=True, justify="left")
    return table


def render_body[Metric, Cell](
    body: Sequence[BodyLine[Metric, Cell]],
    widths: Sequence[int],
    to_cells: Callable[[BodyLine[Metric, Cell]], tuple[str, ...]],
    *,
    color: bool | None,
) -> list[str]:
    """Render a planned body to text, delegating the grid to rich and rules to dashes.

    Consecutive content rows (header, group, metric, aggregate) are drawn as one
    rich table so their ``│`` separators line up; blanks, rules, borders and
    titles break the run and are emitted as their own lines. Every column is fixed
    to ``widths``, so separate tables across sections stay aligned. Color resolves
    once per rendered fragment through
    :func:`~gymrat.report.style.render_lines`.

    Args:
        body: The planned body lines.
        widths: The fixed content width of each column.
        to_cells: Builds the markup cells of a content row.
        color: The explicit color choice, or ``None`` to defer to the environment.

    Returns:
        The rendered lines, in order.
    """
    out: list[str] = []
    batch: list[tuple[str, ...]] = []

    def flush() -> None:
        if not batch:
            return
        table = _make_table(widths)
        for cells in batch:
            table.add_row(*cells)
        out.extend(render_lines(table, color=color, width=RENDER_WIDTH).split("\n"))
        batch.clear()

    for line in body:
        if isinstance(line, (HeaderLine, GroupLine, MetricLine, AggregateLine)):
            batch.append(to_cells(line))
            continue
        flush()
        if isinstance(line, BlankLine):
            out.append("")
        elif isinstance(line, RuleLine):
            out.append(_horizontal(widths, "┼"))
        elif isinstance(line, BorderLine):
            out.append(_horizontal(widths, "┬"))
        elif isinstance(line, TitleLine):
            out.extend(
                render_lines(Text.from_markup(line.text), color=color, width=RENDER_WIDTH).split(
                    "\n"
                )
            )
        else:
            assert_never(line)

    flush()
    return out


__all__ = [
    "CELL_GUTTER",
    "METRIC_COLUMN_MIN",
    "VALUE_COLUMN_MIN",
    "VERDICT_COLUMN_MIN",
    "AggregateColumnCell",
    "AggregateLine",
    "AggregateRow",
    "AggregateRows",
    "BodyLine",
    "GroupLine",
    "HeaderLine",
    "MetricLine",
    "ValueWidths",
    "VerdictParts",
    "VerdictWidths",
    "aggregate_label_lengths",
    "compute_column_width",
    "geomean_column_cell",
    "group_metric_cell",
    "header_metric_cell",
    "indented_section_label",
    "join_value_cell",
    "join_verdict_cell",
    "plan_body",
    "render_body",
    "section_annotation",
    "style_verdict_cell",
    "value_widths",
    "verdict_parts",
    "verdict_widths",
    "widest_header_label",
]
