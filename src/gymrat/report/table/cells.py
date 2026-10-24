"""What a table's columns show.

The geomean row's label, parts and value styling, the fixed-width cell text
builders, and the styled rich ``Text`` cells.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from rich.text import Text

from gymrat.report.display import QUIET_VERDICTS
from gymrat.report.format import (
    PLUS_MINUS,
    format_noise_band_value,
    format_percent_delta,
)
from gymrat.report.style import (
    AGGREGATE_LABEL_STYLE,
    EMPHASIZED_DIRECTION_STYLES,
    GROUP_LABEL_STYLE,
    SCOPE_SEPARATOR,
    VARIANT_NAME_STYLE,
)
from gymrat.utils import pluralize

if TYPE_CHECKING:
    from collections.abc import Sequence

    from gymrat.model import GeomeanResult
    from gymrat.report.display import DisplayClass

CELL_GUTTER = "  "

GROUP_INDENT = "  "

METRIC_COLUMN_HEADER = "metric"

VERDICT_COLUMN_MIN = 12

DETECTS_PREFIX = "detects ≥"

GEOMEAN_LABEL = "geomean"


NO_GEOMEAN_FIGURE = "—"

NO_STABLE_METRICS = "no stable metrics"

# The stand-in's figure wears bold, as a geomean figure does, and its words recede.
NO_GEOMEAN_FIGURE_STYLE = "bold"
NO_STABLE_METRICS_STYLE = "dim"


def scope_label(left: str, right: str) -> str:
    """Name a scoped label, such as a kind and a metric, as one string.

    Args:
        left: The label's first part.
        right: The label's second part.

    Returns:
        The two parts joined by the spaced scope separator.
    """
    return f"{left} {SCOPE_SEPARATOR} {right}"


def geomean_scope_label(scope: str) -> str:
    """The label of an aggregate row covering one scope — a group or a kind."""
    return scope_label(GEOMEAN_LABEL, scope)


@dataclass(frozen=True, slots=True)
class GeomeanParts:
    """The geomean's delta, the count behind it, and the band propagated from its metrics.

    Attributes:
        delta: The signed percentage the geomean moved.
        provenance: How many stable metrics stand behind the figure.
        band: The propagated band's figure, without the ``±`` a column pins in
            front of it, and empty where the metrics left it nothing to state.
    """

    delta: str
    provenance: str
    band: str


def geomean_parts(geomean: GeomeanResult) -> GeomeanParts | None:
    """Take a geomean row's figure apart into the fields its cell pads and styles.

    A band of zero is what an aggregate over exact-only metrics propagates: there
    is no noise to state, and ``±0.0%`` would read as a measurement, so the band
    field is left empty.

    Args:
        geomean: The aggregate to take apart.

    Returns:
        The geomean's delta, band, and provenance, or ``None`` when the geomean
        covers no metrics.
    """
    if geomean.n == 0:
        return None
    return GeomeanParts(
        delta=format_percent_delta(geomean.value),
        provenance=stable_metric_count(geomean.n),
        band=format_noise_band_value(geomean.band) if geomean.band > 0 else "",
    )


def stable_metric_count(n: int) -> str:
    """How many stable metrics stand behind a geomean figure, as the tables phrase it."""
    return pluralize(n, "stable metric")


def _is_quiet_row(outcomes: Sequence[DisplayClass | None]) -> bool:
    """Whether every defined display class in a row is a quiet one.

    Args:
        outcomes: The display class of each metric behind the row, in order.

    Returns:
        Whether every defined outcome is quiet, and at least one is defined:
        a row with no verdicts at all is not counted as quiet.
    """
    defined = [outcome for outcome in outcomes if outcome is not None]
    return len(defined) > 0 and all(outcome in QUIET_VERDICTS for outcome in defined)


def geomean_value_style(
    geomean: GeomeanResult,
    outcomes: Sequence[DisplayClass | None],
) -> str:
    """How a geomean's figure is styled: bold always, colored once it clears the noise band.

    The figure is an average of ratios, so it moves whether or not anything did.
    A value inside the band is emboldened and left uncolored. When every metric
    behind the figure is quiet the color is vetoed, since coloring it would
    announce a win the rows all decline to claim.

    Args:
        geomean: The aggregate whose figure is being styled.
        outcomes: The display class of each metric behind the figure; empty
            leaves the band deciding alone.

    Returns:
        A rich style string: ``"bold"``, or bold in the improved or regressed
        verdict color.
    """
    if _is_quiet_row(outcomes):
        return "bold"
    if geomean.value < -geomean.band:
        return EMPHASIZED_DIRECTION_STYLES["improved"]
    if geomean.value > geomean.band:
        return EMPHASIZED_DIRECTION_STYLES["regressed"]
    return "bold"


# ---------------------------------------------------------------------------
# Metric-column cells
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Verdict cells and section labels
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
        detects: The detectable change's figure, without the ``detects ≥`` the
            column pins; empty when the row states none.
    """

    glyph: str
    delta: str
    word: str
    band: str
    pairs: str
    detects: str = ""


@dataclass(frozen=True, slots=True)
class VerdictWidths:
    """Widths a verdict column pads its fields to, measured on plain text.

    Attributes:
        delta: The width every delta pads to.
        band: The width every noise band's figure pads to.
        detects: The width every detectable change's figure pads to.
        pairs: The width a pair count pads to in a cell that states a
            detectable change after it, so those changes line up.
    """

    delta: int
    band: int
    detects: int = 0
    pairs: int = 0


def verdict_widths(cells: Sequence[VerdictParts]) -> VerdictWidths:
    """Measure a column of verdict cells so every cell pads to the same fields.

    Only the deltas set the delta width; a word standing in for a delta is not
    measured. A caller that wants the delta field to fit a word widens it
    itself. Only the cells stating a detectable change set the pair-count
    width, since only those cells pad their pair count.

    Args:
        cells: The verdict cells to measure.

    Returns:
        The widest delta, band and detectable change across all cells, and the
        widest pair count among the cells stating a detectable change.
    """
    return VerdictWidths(
        delta=max((len(cell.delta) for cell in cells), default=0),
        band=max((len(cell.band) for cell in cells), default=0),
        detects=max((len(cell.detects) for cell in cells), default=0),
        pairs=max((len(cell.pairs) for cell in cells if cell.detects != ""), default=0),
    )


def indented_section_label(short_name: str, group: str | None) -> str:
    """A metric's name cell inside a section: its short name, indented under its group.

    Args:
        short_name: The metric's short name.
        group: The group the metric sits under, or ``None`` when it has none.

    Returns:
        The short name as is when ungrouped, else the name with its group prefix
        stripped and indented.
    """
    return short_name if group is None else f"{GROUP_INDENT}{short_name[len(group) + 1 :]}"


def verdict_cell(
    parts: VerdictParts,
    widths: VerdictWidths,
    *,
    glyph_style: str | None,
    delta_style: str | None,
    band_style: str | None,
) -> Text:
    """A verdict cell, each field padded to its column's width and styled on its own.

    The fields — glyph, delta (or the word standing in for it), band, pair
    count and detectable change — are joined by the cell gutter, a field with
    no text is dropped, and trailing space is trimmed. Only a field's text
    carries its style, never the padding around it, and the cell's plain text is
    what its column is sized on.

    Args:
        parts: The verdict's fields.
        widths: The column widths the fields pad to.
        glyph_style: The style the glyph wears, or ``None`` to leave it plain.
        delta_style: The style the delta or word wears, or ``None`` to leave it
            plain.
        band_style: The style the noise band and the detectable change wear,
            or ``None`` to leave them plain.

    Returns:
        The styled cell.
    """
    if parts.word != "":
        # Padded past its end so a field after the word starts where the
        # column's deltas end.
        delta = _field(parts.word, delta_style).append(" " * max(0, widths.delta - len(parts.word)))
    else:
        pad = " " * max(0, widths.delta - len(parts.delta))
        delta = Text(pad).append(parts.delta, delta_style)
    # The `±` and `detects ≥` are pinned and each figure right-aligned behind
    # its prefix; a row with no band reserves the same width blank where its
    # column shows one, so the pair counts line up. The detectable change comes
    # last so a row without one reads unchanged: its blank is trailing space,
    # trimmed below. A row stating one pads its pair count instead, so the
    # detectable changes line up whether or not a row shows a pair count.
    pairs = parts.pairs.ljust(widths.pairs) if parts.detects != "" else parts.pairs
    fields = [
        _field(parts.glyph, glyph_style),
        delta,
        _pinned_field(PLUS_MINUS, parts.band, widths.band, band_style),
        Text(pairs),
        _pinned_field(DETECTS_PREFIX, parts.detects, widths.detects, band_style),
    ]
    cell = Text(CELL_GUTTER).join(field for field in fields if field.plain != "")
    cell.rstrip()
    return cell


def _pinned_field(prefix: str, figure: str, width: int, style: str | None) -> Text:
    """``figure`` right-aligned to ``width`` behind ``prefix``, or a blank of that width."""
    if figure != "":
        return _field(f"{prefix}{figure.rjust(width)}", style)
    return Text(" " * (len(prefix) + width) if width > 0 else "")


def _field(text: str, style: str | None) -> Text:
    """``text`` as a styled span, or plain when ``style`` is ``None``."""
    return Text().append(text, style)
