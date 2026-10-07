"""The sectioned layout a table is drawn in, its geomean labels, and its cells.

Sorting a run's metrics into kinds and groups, and reading a candidate's
aggregate back out for each scope, live here rather than inside a renderer: it is
what keeps a row and the geomean closing it describing the same set of metrics. A
comparison and a single-target measurement agree on nothing but their metadata,
so the planner is stated over that alone (:class:`SectionedMetric`) and draws
both in the same sections.

The rest is what a column shows: the geomean row's label, parts and value
styling, the fixed-width cell text builders, and the styled rich ``Text`` cells.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from rich.text import Text

from gymrat.metric_name import parse as parse_metric_name
from gymrat.model import GeomeanResult
from gymrat.report.display import GLYPHS, QUIET_VERDICTS, VERDICT_GLOSSES, display_class
from gymrat.report.format import (
    PLUS_MINUS,
    SPREAD_SEPARATOR,
    format_noise_band_value,
    format_pair_count,
    format_percent_delta,
)
from gymrat.report.style import (
    AGGREGATE_LABEL_STYLE,
    GROUP_LABEL_STYLE,
    SCOPE_SEPARATOR,
    VARIANT_NAME_STYLE,
)
from gymrat.utils import pluralize

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from gymrat.config import KindEntry
    from gymrat.model import MetricVerdict, ResolvedMetricMeta
    from gymrat.report.display import DisplayClass
    from gymrat.report.format import MetricCellParts
    from gymrat.report.types import CandidateComparison
    from gymrat.verdict import KindAggregate

CELL_GUTTER = "  "

GROUP_INDENT = "  "

METRIC_COLUMN_HEADER = "metric"

METRIC_COLUMN_MIN = 16
VALUE_COLUMN_MIN = 12
VERDICT_COLUMN_MIN = 12

GEOMEAN_LABEL = "geomean"

GATED_GEOMEAN_LABEL = "gated geomean"

NO_GEOMEAN_FIGURE = "—"

NO_STABLE_METRICS = "no stable metrics"

_INFORMATIONAL_TAG = "informational — gating off"

# The aggregate stated where a candidate reported none. Every section is drawn from
# the same metadata the aggregates were computed from, so this stands in for
# nothing the renderers can produce — and if one ever does, the row says it
# aggregated nothing rather than inventing a figure.
NO_AGGREGATE: GeomeanResult = GeomeanResult(value=math.nan, n=0, band=0, excluded=())


class SectionedMetric(Protocol):
    """All a layout needs of a metric entry: the metadata that decides where it lands."""

    @property
    def meta(self) -> ResolvedMetricMeta:
        """The resolved metadata that sorts the metric into a kind and a group."""


@dataclass(slots=True)
class GroupBlock[Row]:
    """A group of one section's metrics, gathered under the prefix they share."""

    group: str
    metrics: list[Row]


@dataclass(slots=True)
class MetricBlock[Row]:
    """A single metric of a section that belongs to no group."""

    metric: Row


#: One block of a section: either a named group or a single ungrouped metric.
type SectionBlock[Row] = GroupBlock[Row] | MetricBlock[Row]


@dataclass(slots=True)
class SectionPlan[Row]:
    """One kind's slice of the table: what it holds, and whether the run is judged on it.

    Attributes:
        kind: The metric kind this section covers.
        has_gating: Whether any of the section's metrics gate the run.
        blocks: The section's groups and standalone metrics, in first-appearance
            order.
    """

    kind: str
    has_gating: bool
    blocks: list[SectionBlock[Row]]


@dataclass(frozen=True, slots=True)
class SectionLayout[Row]:
    """The run's metrics as sections, and as the flat list a single-kind run draws.

    Attributes:
        sections: One plan per kind, in first-appearance order.
        ordered: Every metric in the order the run reported it, whatever section
            it landed in.
    """

    sections: tuple[SectionPlan[Row], ...]
    ordered: tuple[Row, ...]


def plan_sections[Row, Metric: SectionedMetric](
    metrics: Mapping[str, Metric],
    measure: Callable[[str, str | None, Metric], Row],
) -> SectionLayout[Row]:
    """Sort the run's metrics into one section per kind, and each section into its groups.

    Kinds, groups and metrics keep first-appearance order — the order the
    aggregates were computed in — so a section reads in the same order as the rows
    its geomean covers. A group block sits where its first metric appeared and
    gathers the rest of the group with it, rather than letting a metric of another
    group split it.

    Rows are built here rather than looked up later, so every row a section names
    is the row the table draws. ``measure`` receives the inferred group rather
    than a finished label, since what a renderer does with the prefix is its own
    business.

    Args:
        metrics: Every metric of the run, keyed by name, in first-appearance order.
        measure: Builds a row from a metric's name, inferred group, and entry.

    Returns:
        The sectioned layout and the flat ordered rows.
    """
    sections: dict[str, SectionPlan[Row]] = {}
    ordered: list[Row] = []

    for name, metric in metrics.items():
        meta = metric.meta
        section = sections.get(meta.kind)
        if section is None:
            section = SectionPlan(kind=meta.kind, has_gating=False, blocks=[])
            sections[meta.kind] = section
        if meta.gating:
            section.has_gating = True

        group = parse_metric_name(name).group
        row = measure(name, group, metric)
        ordered.append(row)

        if group is None:
            section.blocks.append(MetricBlock(metric=row))
            continue

        opened = _open_group(section, group)
        if opened is not None:
            opened.metrics.append(row)
        else:
            section.blocks.append(GroupBlock(group=group, metrics=[row]))

    return SectionLayout(sections=tuple(sections.values()), ordered=tuple(ordered))


def _open_group[Row](section: SectionPlan[Row], group: str) -> GroupBlock[Row] | None:
    """The section's already-opened block for ``group``, or ``None`` when none is open."""
    return next(
        (
            block
            for block in section.blocks
            if isinstance(block, GroupBlock) and block.group == group
        ),
        None,
    )


def spans_many_kinds(metrics: Mapping[str, SectionedMetric]) -> bool:
    """Whether the run spans several kinds, and so is reported in sections.

    Read straight off the metrics rather than off a :class:`SectionLayout`, so the
    parts of a report drawn outside the table can ask without building rows they
    have no use for.

    Args:
        metrics: The run's metrics, keyed by name.

    Returns:
        Whether the metrics span more than one kind.
    """
    return len({metric.meta.kind for metric in metrics.values()}) > 1


def informational_tag(kind: str, config_kinds: Mapping[str, KindEntry] | None) -> str:
    """The tag a non-gating kind's title carries, naming the config key that decided it.

    Gating is resolved per metric before the report sees it, so only the config
    distinguishes a kind switched off wholesale from one whose metrics were each
    switched off by name. Naming the key is what lets the reader switch it back.

    Args:
        kind: The kind whose title the tag decorates.
        config_kinds: The configured kinds, keyed by name, or ``None`` when
            absent.

    Returns:
        The informational tag, optionally naming the config key that switched
        gating off.
    """
    entry = config_kinds.get(kind) if config_kinds is not None else None
    switched_off = entry is not None and entry.gating is False
    source = f" (config: kinds.{kind}.gating = false)" if switched_off else ""
    return f"{_INFORMATIONAL_TAG}{source}"


def _kind_aggregate_of(candidate: CandidateComparison, kind: str) -> KindAggregate | None:
    """The aggregate a candidate reported for one kind, or ``None`` when it reported none."""
    return next((aggregate for aggregate in candidate.kinds if aggregate.kind == kind), None)


def kind_geomean_of(candidate: CandidateComparison, kind: str) -> GeomeanResult:
    """The geomean over every metric of ``kind``, gating or not."""
    aggregate = _kind_aggregate_of(candidate, kind)
    return NO_AGGREGATE if aggregate is None else aggregate.geomean


def group_geomean_of(candidate: CandidateComparison, kind: str, group: str) -> GeomeanResult:
    """The geomean over one group of ``kind``'s metrics.

    Args:
        candidate: The candidate whose geomean to read.
        kind: The metric kind the group belongs to.
        group: The group whose geomean to read.

    Returns:
        The group's geomean, or :data:`NO_AGGREGATE` when the candidate
        reported no aggregate for that kind or group.
    """
    aggregate = _kind_aggregate_of(candidate, kind)
    if aggregate is None:
        return NO_AGGREGATE
    return next((entry.geomean for entry in aggregate.groups if entry.group == group), NO_AGGREGATE)


def flat_geomean_of(candidate: CandidateComparison) -> GeomeanResult:
    """The geomean a flat table closes on: the single kind's gating metrics.

    A run reporting one kind has no section to name, so its geomean row states
    what the run is judged on without saying which kind that was.

    Args:
        candidate: The candidate whose geomean to read.

    Returns:
        The gated geomean of the single kind, or :data:`NO_AGGREGATE` when
        none is available.
    """
    if not candidate.kinds:
        return NO_AGGREGATE
    gated = candidate.kinds[0].gated_geomean
    return NO_AGGREGATE if gated is None else gated


def geomean_label(n: int) -> str:
    """The geomean row's label, carrying the count of metrics behind the figure.

    A table with one candidate names the count here, which frees its cells of
    everything but the aggregate itself.

    Args:
        n: The count of metrics behind the aggregate figure.

    Returns:
        The label with the metric count, or the bare :data:`GEOMEAN_LABEL` when
        ``n`` is zero.
    """
    return GEOMEAN_LABEL if n == 0 else f"{GEOMEAN_LABEL} ({pluralize(n, 'stable metric')})"


def geomean_scope_label(scope: str) -> str:
    """The label of an aggregate row covering one scope — a group or a kind."""
    return f"{GEOMEAN_LABEL} {SCOPE_SEPARATOR} {scope}"


def _geomean_provenance(geomean: GeomeanResult) -> str:
    """The provenance suffix behind a scope's figure.

    ``(n)`` when every scope metric stands behind the figure, ``(n/m)`` when
    exclusions thinned them.

    Args:
        geomean: The scope's aggregate result.

    Returns:
        ``"(n)"`` or ``"(n/m)"`` when exclusions reduced the count.
    """
    total = geomean.n + len(geomean.excluded)
    return f"({geomean.n})" if total == geomean.n else f"({geomean.n}/{total})"


def scoped_geomean_label(scope: str, geomean: GeomeanResult) -> str:
    """A sectioned table's aggregate label with the provenance behind its figure."""
    return f"{geomean_scope_label(scope)} {_geomean_provenance(geomean)}"


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
    """The geomean's delta, band, and provenance, or ``None`` when nothing survived.

    A band of zero is what an aggregate over exact-only metrics propagates: there
    is no noise to state, and ``±0.0%`` would read as a measurement, so the band
    field is left empty.

    Args:
        geomean: The aggregate to take apart.

    Returns:
        The parts, or ``None`` when the geomean covers no metrics.
    """
    if geomean.n == 0:
        return None
    return GeomeanParts(
        delta=format_percent_delta(geomean.value),
        provenance=pluralize(geomean.n, "stable metric"),
        band=format_noise_band_value(geomean.band) if geomean.band > 0 else "",
    )


def _is_quiet_row(outcomes: Sequence[DisplayClass | None]) -> bool:
    """Whether every defined display class in a row is a quiet one.

    A row with no verdicts at all is left alone rather than counted as quiet.

    Args:
        outcomes: The display class of each metric behind the row, in order.

    Returns:
        Whether every defined outcome is quiet, and at least one is defined.
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
        A rich style string: ``"bold"``, ``"bold green"``, or ``"bold red"``.
    """
    if _is_quiet_row(outcomes):
        return "bold"
    if geomean.value < -geomean.band:
        return "bold green"
    if geomean.value > geomean.band:
        return "bold red"
    return "bold"


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


@dataclass(frozen=True, slots=True)
class ShownVerdict:
    """A verdict's pre-split parts and display class, always present together.

    Attributes:
        parts: The fields the verdict column pads and styles.
        outcome: The display class the verdict presents as.
    """

    parts: VerdictParts
    outcome: DisplayClass


def shown_verdict(
    verdict: MetricVerdict | None, samples: int, *, with_band: bool
) -> ShownVerdict | None:
    """Take a verdict apart into the fields a verdict column pads and styles.

    Args:
        verdict: The verdict to render, or ``None`` when the metric has none.
        samples: The run's sample count, so a full-count verdict drops its ``n=N``.
        with_band: Whether the caller shows a noise band (the compact
            multi-candidate table drops it).

    Returns:
        The verdict's fields with its display class, or ``None`` when there is
        no verdict to show.
    """
    if verdict is None:
        return None
    shown = display_class(verdict)
    unstable = verdict.verdict == "unstable"
    band = ""
    if with_band and not unstable and shown != "inconclusive" and verdict.method != "exact":
        band = format_noise_band_value(verdict.noise_pct)
    return ShownVerdict(
        parts=VerdictParts(
            glyph=GLYPHS[shown],
            delta="" if unstable else format_percent_delta(verdict.delta),
            word=VERDICT_GLOSSES["unstable"] if unstable else "",
            band=band,
            pairs="" if verdict.n == samples else format_pair_count(verdict.n),
        ),
        outcome=shown,
    )


def verdict_widths(cells: Sequence[VerdictParts]) -> VerdictWidths:
    """The widest delta and band a column of verdict cells holds.

    Only the deltas set the delta width; a word standing in for a delta is not
    measured. A caller that wants the delta field to fit a word widens it
    itself.

    Args:
        cells: The verdict cells to measure.

    Returns:
        The maximum delta and band widths across all cells.
    """
    return VerdictWidths(
        delta=max((len(cell.delta) for cell in cells), default=0),
        band=max((len(cell.band) for cell in cells), default=0),
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
        # Padded past its end so a field after the word starts where the
        # column's deltas end.
        delta = _field(parts.word, delta_style).append(" " * max(0, widths.delta - len(parts.word)))
    else:
        pad = " " * max(0, widths.delta - len(parts.delta))
        delta = Text(pad).append(parts.delta, delta_style)
    # The `±` is pinned and the figure right-aligned behind it; a row with no
    # band reserves the same width blank where its column shows one.
    if parts.band != "":
        band_cell = _field(f"{PLUS_MINUS}{parts.band.rjust(widths.band)}", band_style)
    else:
        band_cell = Text(" " * (len(PLUS_MINUS) + widths.band) if widths.band > 0 else "")
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
        :func:`geomean_value_style`, the provenance
        dimmed. An empty geomean shows the ``no stable metrics`` stand-in rather
        than the ``0.0%`` it computes to.
    """
    parts = geomean_parts(geomean)
    if parts is None:
        return Text.assemble((NO_GEOMEAN_FIGURE, "bold"), "  ", (NO_STABLE_METRICS, "dim"))
    return Text.assemble(
        (parts.delta, geomean_value_style(geomean, outcomes)),
        f" {SCOPE_SEPARATOR} ",
        (parts.provenance, "dim"),
    )
