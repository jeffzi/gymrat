"""Fixed-width cell text builders and the styled rich ``Text`` cells of table columns."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from rich.text import Text

from gymrat.report.display import VERDICT_GLOSSES, display_class, get_glyph
from gymrat.report.format import (
    PLUS_MINUS,
    SPREAD_SEPARATOR,
    format_delta,
    format_noise_band_value,
    format_pair_count,
)
from gymrat.report.geomean_label import (
    NO_GEOMEAN_CELL,
    NO_GEOMEAN_FIGURE,
    NO_STABLE_METRICS,
    geomean_parts,
    geomean_value_style,
)
from gymrat.report.sections import section_label
from gymrat.report.style import AGGREGATE_LABEL_STYLE, GROUP_LABEL_STYLE, VARIANT_NAME_STYLE

if TYPE_CHECKING:
    from collections.abc import Sequence

    from gymrat.model import GeomeanResult, MetricVerdict
    from gymrat.report.display import DisplayClass
    from gymrat.report.format import MetricCellParts

CELL_GUTTER = "  "

GROUP_INDENT = "  "

METRIC_COLUMN_HEADER = "metric"

METRIC_COLUMN_MIN = 16
VALUE_COLUMN_MIN = 12
VERDICT_COLUMN_MIN = 12


def header_metric_cell(title: str | None) -> Text:
    """The metric-column cell for a section header row."""
    return _field(title, "bold") if title is not None else Text(METRIC_COLUMN_HEADER)


def group_metric_cell(label: str) -> Text:
    """The metric-column cell for a group separator row."""
    return _field(label, GROUP_LABEL_STYLE)


def aggregate_label_cell(label: str) -> Text:
    """The metric-column cell for an aggregate row's scope label."""
    return _field(label, AGGREGATE_LABEL_STYLE)


def variant_name_cell(name: str) -> Text:
    """A variant's name, styled as a column header."""
    return _field(name, VARIANT_NAME_STYLE)


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

    Args:
        cells: The verdict cells to measure.

    Returns:
        The maximum delta and band widths across all cells.
    """
    return VerdictWidths(
        delta=max((len(cell.delta) for cell in cells), default=0),
        band=max((len(cell.band) for cell in cells), default=0),
    )


def band_field(band: str, width: int) -> str:
    """The band as it prints: the ``±`` pinned, its figure right-aligned behind it."""
    return "" if band == "" else f"{PLUS_MINUS}{band.rjust(width)}"


def _empty_band_cell(width: int) -> str:
    """The blank a row with no band reserves where its column shows one: ``±`` plus figure width."""
    return " " * (len(PLUS_MINUS) + width)


def indented_section_label(short_name: str, group: str | None) -> str:
    """A metric's name cell inside a section: its short name, indented under its group."""
    label = section_label(short_name, group)
    return label if group is None else f"{GROUP_INDENT}{label}"


_PROVENANCE_SEPARATOR = "·"


def verdict_cell(
    parts: VerdictParts,
    widths: VerdictWidths,
    *,
    glyph_style: str | None,
    delta_style: str | None,
    band_style: str | None,
) -> Text:
    """A verdict cell, each field padded to its column's width and styled on its own.

    The fields — glyph, delta (or the word standing in for it), band and pair
    count — are joined by the cell gutter, a field with no text is dropped, and
    trailing space is trimmed. Only a field's text carries its style, never the
    padding around it, and the cell's plain text is what its column is sized on.

    Args:
        parts: The verdict's fields.
        widths: The column widths the delta and band pad to.
        glyph_style: The style the glyph wears, or ``None`` to leave it plain.
        delta_style: The style the delta or word wears, or ``None`` to leave it
            plain.
        band_style: The style the noise band wears, or ``None`` to leave it
            plain.

    Returns:
        The styled cell.
    """
    if parts.word != "":
        delta = _field(parts.word, delta_style)
    else:
        pad = " " * max(0, widths.delta - len(parts.delta))
        delta = Text(pad).append(parts.delta, delta_style)
    band = band_field(parts.band, widths.band)
    band_cell = (
        Text(_empty_band_cell(widths.band))
        if band == "" and widths.band > 0
        else _field(band, band_style)
    )
    fields = [_field(parts.glyph, glyph_style), delta, band_cell, Text(parts.pairs)]
    cell = Text(CELL_GUTTER).join(field for field in fields if field.plain != "")
    cell.rstrip()
    return cell


def _field(text: str, style: str | None) -> Text:
    """``text`` as a styled span, or plain when ``style`` is ``None``."""
    return Text().append(text, style)


def geomean_column_cell(
    geomean: GeomeanResult,
    outcomes: Sequence[DisplayClass | None],
) -> Text:
    """The geomean of one candidate column: the aggregate, then how many metrics back it.

    The multi-candidate table names the scope once in its label column and states
    each candidate's own figure and count in the candidate columns, so this builds
    one column's cell.

    Args:
        geomean: The candidate's aggregate over the scope's metrics.
        outcomes: The display class of each metric behind the figure, for vetoing
            the figure's color when every one is quiet.

    Returns:
        The styled cell: the delta by
        :func:`~gymrat.report.geomean_label.geomean_value_style`, the provenance
        dimmed. An empty geomean shows the ``no stable metrics`` stand-in rather
        than the ``0.0%`` it computes to.
    """
    parts = geomean_parts(geomean)
    if parts is None:
        cell = Text(NO_GEOMEAN_CELL)
        cell.stylize("bold", 0, len(NO_GEOMEAN_FIGURE))
        cell.stylize("dim", len(NO_GEOMEAN_CELL) - len(NO_STABLE_METRICS))
        return cell
    return Text.assemble(
        (parts.delta, geomean_value_style(geomean, outcomes)),
        f" {_PROVENANCE_SEPARATOR} ",
        (parts.provenance, "dim"),
    )
