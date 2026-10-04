"""The shared machinery both text tables draw through.

The data half is the body planner that lays a
:class:`~gymrat.report.table.markup.SectionLayout` out as titles, borders, rules
and rows. The cell builders that pad a value cell's magnitude and spread, and a
verdict cell's glyph, delta and band, into fields of their own live in
:mod:`gymrat.report.table.markup`.

The rendering half draws the grid. The box chrome — column padding, the ``│``
separators, and the ``┼`` rules closing a header or a run of rows — is delegated
to a :class:`rich.table.Table`; only a section's ``┬`` top border is drawn by
hand. Cells are styled rich :class:`~rich.text.Text` (or markup strings, which
rich parses), resolved to color once by
:func:`~gymrat.report.style.render_lines`. In-cell sub-field alignment stays in
the cell builders, because that is the behavior the tests pin; only the grid
around the cells is rich's.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Protocol, assert_never

from rich import box
from rich.cells import cell_len
from rich.table import Table
from rich.text import Text

from gymrat.report.style import SCOPE_SEPARATOR, markup, render_lines
from gymrat.report.table.markup import (
    METRIC_COLUMN_HEADER,
    METRIC_COLUMN_MIN,
    VALUE_COLUMN_MIN,
    GroupBlock,
    MetricBlock,
    informational_tag,
    join_value_cell,
    value_widths,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from gymrat.config import KindEntry
    from gymrat.report.format import MetricCellParts
    from gymrat.report.table.markup import SectionLayout, SectionPlan


type TableCell = str | Text
"""A content row's cell: styled ``Text`` renders literally, a string is parsed as markup."""


# ---------------------------------------------------------------------------
# Body planning
# ---------------------------------------------------------------------------


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
    """A single metric's data row in the table body."""

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

    group: Callable[[str, str, Sequence[Metric]], AggregateLine[Cell]]
    kind: Callable[[str, Sequence[Metric]], AggregateLine[Cell]]
    flat: Callable[[Sequence[Metric]], AggregateLine[Cell]]


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
    group_label: Callable[[str], str],
) -> list[BodyLine[Metric, Cell]]:
    """The lines one section's blocks produce: groups, standalone metrics, sub-geomeans.

    Args:
        section: The section whose blocks to lay out.
        rows: The aggregate-row builders, or ``None`` to draw no per-group aggregate.
        group_label: Turns a group's name into the label its group line shows.

    Returns:
        The block lines, in draw order.
    """
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

        lines.append(GroupLine(label=group_label(block.group)))
        lines.extend(MetricLine(row=metric) for metric in block.metrics)
        if rows is not None:
            lines.append(rows.group(section.kind, block.group, block.metrics))
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
        lines.extend(_plan_blocks(section, rows, lambda group: group))
        if rows is not None:
            lines.append(RuleLine())
            lines.append(rows.kind(section.kind, _section_metrics(section)))
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
        kind = section.kind
        body.extend(_plan_blocks(section, None, lambda group: f"{group} {SCOPE_SEPARATOR} {kind}"))
    else:
        body.extend(MetricLine(row=row) for row in layout.ordered)
    if rows is not None:
        body.append(RuleLine())
        body.append(rows.flat(layout.ordered))
    return body


def compute_column_width(header_len: int, content_lengths: Sequence[int], minimum: int) -> int:
    """The width a column settles on: the widest of its content, its header, and its floor."""
    return max(minimum, header_len, max(content_lengths, default=0))


def section_annotation[Metric](
    section: SectionPlan[Metric],
    config_kinds: Mapping[str, KindEntry] | None,
) -> str | None:
    """A section's informational tag as dimmed markup.

    Args:
        section: The planned section to tag.
        config_kinds: The config's ``kinds`` entries, or ``None`` when the run
            carries no kind metadata.

    Returns:
        The dimmed tag, or ``None`` when the section has a gating metric.
    """
    if section.has_gating:
        return None
    return markup(informational_tag(section.kind, config_kinds), "dim")


# ---------------------------------------------------------------------------
# Table skeleton
# ---------------------------------------------------------------------------


class NamedRow(Protocol):
    """A table row exposing the ungrouped name and the grouped, indented label."""

    @property
    def name(self) -> str:
        """The metric's bare name, shown when the table has nothing to group under it."""

    @property
    def label(self) -> str:
        """The metric's section label, indented under its group."""


@dataclass(frozen=True, slots=True)
class TableSkeleton[Row]:
    """The body, name/value cells, and widths a measurement and probe table share.

    Attributes:
        body: The planned body lines, ready for :func:`render_body`.
        name_cell: The metric-column cell for one row — its indented label once
            the table is grouped (see :func:`is_grouped`), its bare name otherwise.
        value_cell: The value-column cell for one row.
        metric_width: The metric column's settled width.
        value_width: The value column's settled width.
    """

    body: list[BodyLine[Row, object]]
    name_cell: Callable[[Row], str]
    value_cell: Callable[[Row], str]
    metric_width: int
    value_width: int


def is_grouped[Metric, Cell](
    layout: SectionLayout[Metric],
    body: Sequence[BodyLine[Metric, Cell]],
) -> bool:
    """Whether a row shows its indented, grouped label rather than its bare name.

    True once the run spans more than one kind, or its one section has a group
    holding more than one metric — the same test a measurement, probe, and
    comparison table all apply to their body.

    Args:
        layout: The sectioned rows the body was planned from.
        body: The planned body lines to check for a multi-metric group.

    Returns:
        Whether rows show their grouped label.
    """
    return len(layout.sections) > 1 or any(isinstance(line, GroupLine) for line in body)


def metric_column_width[Metric, Cell](
    body: Sequence[BodyLine[Metric, Cell]],
    name_lengths: Sequence[int],
) -> int:
    """The metric column's width: the widest of its name cells, its header, and its floor.

    Args:
        body: The planned body lines, whose section titles and group and
            aggregate labels also size the column.
        name_lengths: Each row's name-cell width, in terminal cells.

    Returns:
        The metric column's settled width.
    """
    label_lengths = [
        cell_len(line.label) for line in body if isinstance(line, (GroupLine, AggregateLine))
    ]
    title_lengths = [
        cell_len(line.title)
        for line in body
        if isinstance(line, HeaderLine) and line.title is not None
    ]
    return compute_column_width(
        cell_len(METRIC_COLUMN_HEADER),
        [*name_lengths, *label_lengths, *title_lengths],
        METRIC_COLUMN_MIN,
    )


def plan_table_skeleton[Row: NamedRow](
    layout: SectionLayout[Row],
    config_kinds: Mapping[str, KindEntry] | None,
    value_of: Callable[[Row], MetricCellParts],
    label: str,
) -> TableSkeleton[Row]:
    """The shared skeleton a measurement and a probe table both build their columns on.

    Both tables plan the same body, decide grouping the same way, and size their
    metric and value columns identically; a probe table appends a reference and a
    delta column of its own on top of what this returns.

    Args:
        layout: The sectioned rows to plan a body for.
        config_kinds: The run's configured kinds, threaded through to
            :func:`section_annotation`, or ``None`` where the table states no kind
            metadata (the probe table).
        value_of: Reads a row's value cell parts.
        label: The value column's header, sizing the value column.

    Returns:
        The planned body, the name and value cell builders, and the metric and
        value column widths.
    """
    value_fields = value_widths([value_of(row) for row in layout.ordered])

    body: list[BodyLine[Row, object]] = plan_body(
        layout,
        None,
        lambda section: section_annotation(section, config_kinds),
    )
    grouped = is_grouped(layout, body)

    def name_cell(row: Row) -> str:
        return row.label if grouped else row.name

    def value_cell(row: Row) -> str:
        return join_value_cell(value_of(row), value_fields)

    metric_width = metric_column_width(body, [cell_len(name_cell(row)) for row in layout.ordered])
    value_width = compute_column_width(
        cell_len(label),
        [cell_len(value_cell(row)) for row in layout.ordered],
        VALUE_COLUMN_MIN,
    )
    return TableSkeleton(
        body=body,
        name_cell=name_cell,
        value_cell=value_cell,
        metric_width=metric_width,
        value_width=value_width,
    )


def build_cell_dispatcher[Row, Cell](
    header: Callable[[str | None], tuple[TableCell, ...]],
    group: Callable[[str], tuple[TableCell, ...]],
    metric: Callable[[Row], tuple[TableCell, ...]],
    aggregate: Callable[[AggregateLine[Cell]], tuple[TableCell, ...]] | None = None,
) -> Callable[[BodyLine[Row, Cell]], tuple[TableCell, ...]]:
    """A ``to_cells`` callable dispatching each content line to its cells.

    The tables' ``to_cells`` differ only in how many columns each line states;
    the dispatch itself is identical.

    Args:
        header: Builds a header row's cells from its section title.
        group: Builds a group row's cells from its label.
        metric: Builds a metric row's cells from its row.
        aggregate: Builds an aggregate row's cells, or ``None`` for a table
            whose body plans no aggregate.

    Returns:
        The dispatching ``to_cells`` callable. Calling it raises
        ``AssertionError`` for a ``BlankLine``, ``RuleLine``, ``BorderLine`` or
        ``TitleLine``, and for an ``AggregateLine`` when ``aggregate`` is
        ``None``.
    """

    def to_cells(line: BodyLine[Row, Cell]) -> tuple[TableCell, ...]:
        if isinstance(line, HeaderLine):
            return header(line.title)
        if isinstance(line, GroupLine):
            return group(line.label)
        if isinstance(line, MetricLine):
            return metric(line.row)
        if isinstance(line, AggregateLine) and aggregate is not None:
            return aggregate(line)
        msg = f"unexpected body line {line!r}"
        raise AssertionError(msg)

    return to_cells


# ---------------------------------------------------------------------------
# Rendering engine
# ---------------------------------------------------------------------------


def _horizontal(widths: Sequence[int], junction: str) -> str:
    """A dashed rule or border, meeting each column separator at ``junction``.

    The dashes account for the padding rich draws with ``pad_edge`` off: the first
    column carries no left pad and the last no right pad, so their segments are one
    dash narrower than the interior columns'.

    Args:
        widths: The rendered width of each column, in order.
        junction: The character drawn at each column boundary.

    Returns:
        A dashed rule string with junctions at every column boundary.
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


@dataclass(frozen=True, slots=True)
class _BatchedRow:
    """A content row waiting to be drawn, and whether a rule closes it."""

    cells: tuple[TableCell, ...]
    end_section: bool = False


def _flush_batch(
    batch: list[_BatchedRow],
    widths: Sequence[int],
    *,
    color: bool | None,
) -> list[str]:
    """Draw the batched rows as one rich table; append a closing ``┼`` rule if wanted."""
    if not batch:
        return []
    table = _make_table(widths)
    for row in batch:
        table.add_row(*row.cells, end_section=row.end_section)
    out = render_lines(table, color=color).split("\n")
    # rich draws a section end only between rows, never after the last one.
    if batch[-1].end_section:
        out.append(_horizontal(widths, "┼"))
    batch.clear()
    return out


def render_body[Metric, Cell](
    body: Sequence[BodyLine[Metric, Cell]],
    widths: Sequence[int],
    to_cells: Callable[[BodyLine[Metric, Cell]], tuple[TableCell, ...]],
    *,
    color: bool | None,
) -> list[str]:
    """Render a planned body to text, delegating the grid and its rules to rich.

    Consecutive content rows (header, group, metric, aggregate) are drawn as one
    rich table so their ``│`` separators line up, and a rule following a row is
    drawn by rich as that row's section end. Blanks, borders and titles break the
    run and are emitted as their own lines; a rule with no row of its own to
    close — the second of two in a row, or one ending a run — is drawn by hand.
    Every column is fixed to ``widths``, so separate tables across sections stay
    aligned. Color resolves once per rendered fragment through
    :func:`~gymrat.report.style.render_lines`.

    Args:
        body: The planned body lines.
        widths: The fixed content width of each column.
        to_cells: Builds the cells of a content row.
        color: The explicit color choice, or ``None`` to defer to the environment.

    Returns:
        The rendered lines, in order.
    """
    out: list[str] = []
    batch: list[_BatchedRow] = []

    for line in body:
        if isinstance(line, (HeaderLine, GroupLine, MetricLine, AggregateLine)):
            batch.append(_BatchedRow(cells=to_cells(line)))
            continue
        if isinstance(line, RuleLine) and batch and not batch[-1].end_section:
            batch[-1] = replace(batch[-1], end_section=True)
            continue
        out.extend(_flush_batch(batch, widths, color=color))
        if isinstance(line, BlankLine):
            out.append("")
        elif isinstance(line, RuleLine):
            out.append(_horizontal(widths, "┼"))
        elif isinstance(line, BorderLine):
            out.append(_horizontal(widths, "┬"))
        elif isinstance(line, TitleLine):
            out.extend(render_lines(line.text, color=color).split("\n"))
        else:
            assert_never(line)

    out.extend(_flush_batch(batch, widths, color=color))
    return out


__all__ = [
    "AggregateLine",
    "AggregateRows",
    "BodyLine",
    "GroupLine",
    "HeaderLine",
    "MetricLine",
    "NamedRow",
    "RuleLine",
    "TableCell",
    "TableSkeleton",
    "build_cell_dispatcher",
    "compute_column_width",
    "is_grouped",
    "metric_column_width",
    "plan_body",
    "plan_table_skeleton",
    "render_body",
    "section_annotation",
]
