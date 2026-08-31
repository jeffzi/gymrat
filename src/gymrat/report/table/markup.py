"""Fixed-width cell text builders and Rich markup helpers for table columns."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from rich.markup import escape

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
from gymrat.report.style import GROUP_LABEL_STYLE, markup

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


def header_metric_cell(title: str | None) -> str:
    """The metric-column cell for a section header row."""
    return markup(title, "bold") if title is not None else escape(METRIC_COLUMN_HEADER)


def group_metric_cell(label: str) -> str:
    """The metric-column cell for a group separator row."""
    return markup(label, GROUP_LABEL_STYLE)


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


def join_verdict_cell(parts: VerdictParts, widths: VerdictWidths) -> str:
    """A verdict cell, each field padded to the width its column settled on."""
    delta = parts.word if parts.word != "" else parts.delta.rjust(widths.delta)
    band = band_field(parts.band, widths.band)
    band_cell = _empty_band_cell(widths.band) if band == "" and widths.band > 0 else band
    fields = [parts.glyph, delta, band_cell, parts.pairs]
    return CELL_GUTTER.join(field for field in fields if field != "").rstrip()


def indented_section_label(short_name: str, group: str | None) -> str:
    """A metric's name cell inside a section: its short name, indented under its group."""
    label = section_label(short_name, group)
    return label if group is None else f"{GROUP_INDENT}{label}"


_PROVENANCE_SEPARATOR = "·"


@dataclass(frozen=True, slots=True)
class StyledSpan:
    """A run of an already-built cell, and the style it wears."""

    text: str
    style: str


@dataclass(frozen=True, slots=True)
class AggregateColumnCell:
    """One candidate column's aggregate cell: its text, and the spans that style it.

    Attributes:
        text: The cell's plain text, which the column is sized on.
        spans: The runs of that text carrying a style, in the order they appear.
    """

    text: str
    spans: tuple[StyledSpan, ...]


def style_verdict_cell(
    parts: VerdictParts,
    widths: VerdictWidths,
    *,
    glyph_style: str | None,
    delta_style: str | None,
    band_style: str | None,
) -> str:
    """A verdict cell rendered to markup, each field wrapped in the style it carries.

    Reproduces :func:`join_verdict_cell`'s layout —
    same visible text, so a column sized on the plain join renders it flush —
    wrapping the glyph, the delta (or the word standing in for it) and the band
    in the styles the caller passes. A ``None`` style leaves that field plain.

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
        plain_delta = parts.word
    elif parts.delta != "":
        pad = " " * max(0, widths.delta - len(parts.delta))
        delta = f"{pad}{_wrap(parts.delta, delta_style)}"
        plain_delta = parts.delta
    else:
        delta = " " * widths.delta if widths.delta > 0 else ""
        plain_delta = delta
    band = band_field(parts.band, widths.band)
    styled_band = _wrap(band, band_style)
    band_cell = _empty_band_cell(widths.band) if band == "" and widths.band > 0 else styled_band
    fields = [glyph, delta, band_cell, escape(parts.pairs)]
    plain_fields = [parts.glyph, plain_delta, band, parts.pairs]
    return _join_styled(fields, plain_fields)


def _wrap(text: str, style: str | None) -> str:
    """``text`` wrapped in ``style`` as markup, or escaped plain when ``style`` is ``None``."""
    if text == "":
        return ""
    return escape(text) if style is None else markup(text, style)


def _join_styled(fields: Sequence[str], plain_fields: Sequence[str]) -> str:
    """Join styled fields by the cell gutter, dropping the ones whose plain text is empty.

    The join mirrors :func:`join_verdict_cell`: a field
    is kept only when its plain text is non-empty, and the trailing gutter is
    trimmed. Trimming works on the markup because the gutter is plain spaces at
    the end.
    """
    kept = [styled for styled, plain in zip(fields, plain_fields, strict=True) if plain != ""]
    return CELL_GUTTER.join(kept).rstrip()


def geomean_column_cell(
    geomean: GeomeanResult,
    outcomes: Sequence[DisplayClass | None],
) -> AggregateColumnCell:
    """The geomean of one candidate column: the aggregate, then how many metrics back it.

    The multi-candidate table names the scope once in its label column and states
    each candidate's own figure and count in the candidate columns, so this builds
    one column's cell — the empty case falling back to the ``no stable metrics``
    stand-in rather than the ``0.0%`` an empty geomean computes to.

    Args:
        geomean: The candidate's aggregate over the scope's metrics.
        outcomes: The display class of each metric behind the figure, for vetoing
            the figure's color when every one is quiet.

    Returns:
        The cell's text, and the spans styling it: the delta by
        :func:`~gymrat.report.geomean_label.geomean_value_style`, the provenance
        dimmed.
    """
    parts = geomean_parts(geomean)
    if parts is None:
        return AggregateColumnCell(
            text=NO_GEOMEAN_CELL,
            spans=(
                StyledSpan(text=NO_GEOMEAN_FIGURE, style="bold"),
                StyledSpan(text=NO_STABLE_METRICS, style="dim"),
            ),
        )
    return AggregateColumnCell(
        text=f"{parts.delta} {_PROVENANCE_SEPARATOR} {parts.provenance}",
        spans=(
            StyledSpan(text=parts.delta, style=geomean_value_style(geomean, outcomes)),
            StyledSpan(text=parts.provenance, style="dim"),
        ),
    )
