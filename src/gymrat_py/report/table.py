"""The shared machinery both text tables draw through.

The data half is ported from the TypeScript ``text-table-core``: the string
builders that pad a value cell's magnitude and spread, and a verdict cell's
glyph, delta and band, into fields of their own, plus the body planner that lays
a :class:`~gymrat_py.report.sections.SectionLayout` out as titles, borders, rules
and rows.

The rendering half is not a port. The box chrome — column padding, the ``│``
separators, the ``─`` rules and their ``┼``/``┬`` junctions — is delegated to a
:class:`rich.table.Table`, and styling is carried as rich markup that
:func:`~gymrat_py.report.style.render_lines` resolves to color once. In-cell
sub-field alignment stays in the string builders, because that is the behavior
the tests pin; only the grid around the cells is rich's.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from rich import box
from rich.markup import escape
from rich.table import Table
from rich.text import Text

from gymrat_py.report.format import (
    PLUS_MINUS,
    SPREAD_SEPARATOR,
    VERDICT_GLOSSES,
    display_class,
    format_delta,
    format_noise_band_value,
    format_pair_count,
    get_glyph,
)
from gymrat_py.report.sections import (
    GroupBlock,
    MetricBlock,
    informational_tag,
    section_label,
)
from gymrat_py.report.style import render_lines

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from gymrat_py.config import KindEntry
    from gymrat_py.model import MetricVerdict
    from gymrat_py.report.format import MetricCellParts
    from gymrat_py.report.sections import SectionLayout, SectionPlan

# ---------------------------------------------------------------------------
# Column constants
# ---------------------------------------------------------------------------

#: The metric-name column's header text, and its fallback when a section has none.
METRIC_COLUMN_HEADER = "metric"

#: Minimum column widths in characters, enforced regardless of content length.
METRIC_COLUMN_MIN = 16
VALUE_COLUMN_MIN = 12
VERDICT_COLUMN_MIN = 12

#: Gap between the fields of one verdict cell: glyph, delta, band and pair count.
CELL_GUTTER = "  "

#: Indent a metric row carries under the group sub-header above it.
GROUP_INDENT = "  "

# A wide render width so the capture console never soft-wraps a table line.
_RENDER_WIDTH = 200


# ---------------------------------------------------------------------------
# Value-cell string builders
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ValueWidths:
    """Widths a value column pads its two fields to, measured on plain text."""

    magnitude: int
    spread: int


def value_widths(cells: Sequence[MetricCellParts]) -> ValueWidths:
    """The widest magnitude and the widest spread a column of value cells holds."""
    return ValueWidths(
        magnitude=max((len(cell.magnitude) for cell in cells), default=0),
        spread=max((len(cell.spread) for cell in cells), default=0),
    )


def join_value_cell(parts: MetricCellParts, widths: ValueWidths) -> str:
    """A value cell with its magnitude and spread each right-aligned in its own field."""
    magnitude = parts.magnitude.rjust(widths.magnitude)
    if widths.spread == 0:
        return magnitude
    spread = "" if parts.spread == "" else f"{SPREAD_SEPARATOR}{parts.spread.rjust(widths.spread)}"
    return f"{magnitude}{spread}".ljust(widths.magnitude + len(SPREAD_SEPARATOR) + widths.spread)


# ---------------------------------------------------------------------------
# Verdict-cell string builders
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class VerdictParts:
    """One verdict's fields, with the noise band only where the caller shows one.

    Attributes:
        glyph: The verdict glyph, or a slot the caller fills.
        delta: The signed percentage, right-aligned among the column's deltas.
        word: The word standing in for a delta too noisy to report, empty
            otherwise.
        band: The noise band's figure, without the ``±`` the column pins.
        pairs: The ``n=N`` pair count, empty when the verdict rests on every pair.
    """

    glyph: str
    delta: str
    word: str
    band: str
    pairs: str


@dataclass(frozen=True, slots=True)
class VerdictWidths:
    """Widths a verdict column pads its delta and band to, measured on plain text."""

    delta: int
    band: int


def verdict_parts(verdict: MetricVerdict, samples: int, *, with_band: bool) -> VerdictParts:
    """Take a verdict apart into the fields a verdict column pads and styles.

    Args:
        verdict: The verdict to render.
        samples: The run's sample count, so a full-count verdict drops its ``n=N``.
        with_band: Whether the caller shows a noise band (the compact
            multi-candidate table drops it).

    Returns:
        The verdict's fields.
    """
    shown = display_class(verdict)
    unstable = verdict.verdict == "unstable"
    band = ""
    if with_band and not unstable and shown != "inconclusive" and verdict.method != "exact":
        band = format_noise_band_value(verdict.noise_pct)
    return VerdictParts(
        glyph=get_glyph(shown),
        delta="" if unstable else format_delta(verdict.delta),
        word=VERDICT_GLOSSES["unstable"] if unstable else "",
        band=band,
        pairs="" if verdict.n == samples else format_pair_count(verdict.n),
    )


def verdict_widths(cells: Sequence[VerdictParts]) -> VerdictWidths:
    """The widest delta and band a column of verdict cells holds.

    The word standing in for a delta is not measured: it is wider than any
    percentage, and sizing the field from it would push a whole column of bands
    right for the sake of the one row that has none.
    """
    return VerdictWidths(
        delta=max((len(cell.delta) for cell in cells), default=0),
        band=max((len(cell.band) for cell in cells), default=0),
    )


def band_field(band: str, width: int) -> str:
    """The band as it prints: the ``±`` pinned, its figure right-aligned behind it."""
    return "" if band == "" else f"{PLUS_MINUS}{band.rjust(width)}"


def join_verdict_cell(parts: VerdictParts, widths: VerdictWidths) -> str:
    """A verdict cell, each field padded to the width its column settled on."""
    delta = parts.word if parts.word != "" else parts.delta.rjust(widths.delta)
    band = band_field(parts.band, widths.band)
    band_cell = " " * (len(PLUS_MINUS) + widths.band) if band == "" and widths.band > 0 else band
    fields = [parts.glyph, delta, band_cell, parts.pairs]
    return CELL_GUTTER.join(field for field in fields if field != "").rstrip()


def indented_section_label(short_name: str, group: str | None) -> str:
    """A metric's name cell inside a section: its short name, indented under its group."""
    label = section_label(short_name, group)
    return label if group is None else f"{GROUP_INDENT}{label}"


# ---------------------------------------------------------------------------
# Markup helpers
# ---------------------------------------------------------------------------


def markup(text: str, style: str) -> str:
    """``text`` wrapped in a rich-markup span carrying ``style``, escaping the text."""
    return f"[{style}]{escape(text)}[/]"


def style_verdict_cell(
    parts: VerdictParts,
    widths: VerdictWidths,
    *,
    glyph_style: str | None,
    delta_style: str | None,
    band_style: str | None,
) -> str:
    """A verdict cell rendered to markup, each field wrapped in the style it carries.

    Reproduces :func:`join_verdict_cell`'s layout — same visible text, so a column
    sized on the plain join renders it flush — wrapping the glyph, the delta (or
    the word standing in for it) and the band in the styles the caller passes.
    A ``None`` style leaves that field plain.

    Args:
        parts: The verdict's fields.
        widths: The column widths the fields pad to.
        glyph_style: The style the glyph wears.
        delta_style: The style the delta or word wears.
        band_style: The style the noise band wears.

    Returns:
        The cell as rich markup.
    """
    glyph = _wrap(parts.glyph, glyph_style)
    if parts.word != "":
        delta = _wrap(parts.word, delta_style)
    else:
        pad = " " * max(0, widths.delta - len(parts.delta))
        delta = f"{pad}{_wrap(parts.delta, delta_style)}" if parts.delta != "" else ""
    band = band_field(parts.band, widths.band)
    styled_band = _wrap(band, band_style)
    band_cell = (
        " " * (len(PLUS_MINUS) + widths.band) if band == "" and widths.band > 0 else styled_band
    )
    fields = [glyph, delta, band_cell, escape(parts.pairs)]
    plain_fields = [parts.glyph, parts.delta if parts.word == "" else parts.word, band, parts.pairs]
    return _join_styled(fields, plain_fields)


def _wrap(text: str, style: str | None) -> str:
    """``text`` wrapped in ``style`` as markup, or escaped plain when ``style`` is ``None``."""
    if text == "":
        return ""
    return escape(text) if style is None else markup(text, style)


def _join_styled(fields: Sequence[str], plain_fields: Sequence[str]) -> str:
    """Join styled fields by the cell gutter, dropping the ones whose plain text is empty.

    The join mirrors :func:`join_verdict_cell`: a field is kept only when its
    plain text is non-empty, and the trailing gutter is trimmed. Trimming works on
    the markup because the gutter is plain spaces at the end.
    """
    kept = [styled for styled, plain in zip(fields, plain_fields, strict=True) if plain != ""]
    return CELL_GUTTER.join(kept).rstrip()


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
    body.extend(MetricLine(row=row) for row in layout.ordered)
    if rows is not None:
        body.append(RuleLine())
        aggregate = rows.flat(layout.ordered)
        body.append(AggregateLine(label=aggregate.label, cell=aggregate.cell))
    return body


def aggregate_label_lengths[Metric, Cell](body: Sequence[BodyLine[Metric, Cell]]) -> list[int]:
    """The labels of every non-metric row — what the name column widens for."""
    return [len(line.label) for line in body if isinstance(line, (GroupLine, AggregateLine))]


def widest_header_label[Metric, Cell](body: Sequence[BodyLine[Metric, Cell]]) -> str:
    """The widest label any header row carries, defaulting to the metric-column header."""
    widest = METRIC_COLUMN_HEADER
    for line in body:
        if isinstance(line, HeaderLine):
            label = line.title if line.title is not None else METRIC_COLUMN_HEADER
            if len(label) > len(widest):
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
    :func:`~gymrat_py.report.style.render_lines`.

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
        out.extend(render_lines(table, color=color, width=_RENDER_WIDTH).split("\n"))
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
                render_lines(Text.from_markup(line.text), color=color, width=_RENDER_WIDTH).split(
                    "\n"
                )
            )

    flush()
    return out
