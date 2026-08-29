"""Formatting and classification primitives for the report package.

These turn the model's numbers and verdicts into the strings a renderer draws:
scaled measurements, signed deltas, display-class glyphs, per-candidate
highlight selection, geomean labels, the verdict summary line, and the footer.

Functions that carry color return rich-markup strings rather than raw ANSI, so a
renderer decides color once through :func:`gymrat_py.report.style.render_lines`.
The dynamic text inside a styled span is escaped so a metric named ``[i]`` renders
literally.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, assert_never

from rich.markup import escape

from gymrat_py.model import (
    BAND_DESCRIPTOR,
    PERMUTATION_DESCRIPTOR,
    Effect,
    GeomeanResult,
    MetricUnit,
    MetricVerdict,
)
from gymrat_py.report.style import VERDICT_STYLES, markup

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from gymrat_py.report.types import CandidateMetric, MetricComparison, MetricComparisons

#: Minimum pairs the permutation test needs; below it a run falls to the band.
MIN_PERMUTATION_N = PERMUTATION_DESCRIPTOR.min_n
#: Minimum pairs the band method needs to measure a spread at all.
MIN_BAND_N = BAND_DESCRIPTOR.min_n


# ---------------------------------------------------------------------------
# Value and cell formatting
# ---------------------------------------------------------------------------

# A tier: the threshold below which it applies, the divisor into its unit, the
# suffix it prints, and how many decimals it carries.
type _Tier = tuple[float, float, str, int]

_NS_TIERS: tuple[_Tier, ...] = (
    (1000, 1, "ns", 0),
    (1e6, 1000, "µs", 1),
    (1e9, 1e6, "ms", 1),
    (math.inf, 1e9, "s", 1),
)

_BYTE_TIERS: tuple[_Tier, ...] = (
    (1000, 1, "B", 0),
    (1e6, 1000, "KB", 1),
    (1e9, 1e6, "MB", 1),
    (math.inf, 1e9, "GB", 1),
)

_TIER_MAP: dict[MetricUnit, tuple[_Tier, ...]] = {"ns": _NS_TIERS, "bytes": _BYTE_TIERS}


def _non_finite_token(value: float) -> str:
    """The token a non-finite reading prints, matching the differ/JSON contract."""
    if math.isnan(value):
        return "NaN"
    return "Infinity" if value > 0 else "-Infinity"


def _scale_tier(value: float, tiers: tuple[_Tier, ...]) -> str:
    """Scale ``value`` into the first tier whose rounded figure stays below its threshold.

    The tier is chosen on the figure *as rounded* rather than as measured: a
    value just under a threshold rounds up onto it (999.5 bytes to ``1000B``),
    which is a four-digit magnitude in a column sized for three, so it is
    promoted to the tier above. The threshold is compared against the magnitude,
    since a sign is not a size: a negative reading picks the tier its magnitude
    names.
    """
    magnitude = abs(value)
    for threshold, divisor, suffix, decimals in tiers:
        rounded = float(format(magnitude / divisor, f".{decimals}f"))
        if rounded * divisor < threshold:
            return f"{value / divisor:.{decimals}f}{suffix}"
    return str(value)


def format_value(value: float, unit: MetricUnit | None = None) -> str:
    """Scale a measurement into its unit's tier, or round it when the metric has no unit.

    Args:
        value: The measurement to format.
        unit: The metric's unit, or ``None`` for a unitless figure.

    Returns:
        The scaled, suffixed figure (``"1.7µs"``), the rounded integer for a
        unitless value, or ``"Infinity"`` / ``"-Infinity"`` / ``"NaN"`` for a
        non-finite reading.
    """
    if not math.isfinite(value):
        return _non_finite_token(value)
    if unit is None:
        return str(round(value))
    return _scale_tier(value, _TIER_MAP[unit])


def format_delta(effect: Effect) -> str:
    """A signed percentage, or nothing when the effect is not a number.

    A delta that rounds to zero prints as an unsigned ``0.0%``: at display
    precision there is no direction to report, so ``-0.0%`` would claim one.

    Args:
        effect: The effect to render. Its ``value`` carries the number and its
            ``unit`` the scale (percent today).

    Returns:
        A signed percentage such as ``"+2.2%"``, an unsigned ``"0.0%"`` for a
        value that rounds to zero, or ``""`` when the value is ``NaN``.
    """
    value = effect.value
    if math.isnan(value):
        return ""
    magnitude = f"{abs(value):.1f}"
    if magnitude == "0.0":
        return "0.0%"
    sign = "+" if value > 0 else "-"
    return f"{sign}{magnitude}%"


def _delta_of(value: float) -> str:
    """A raw percentage figure as a signed delta, wrapping it in a percent effect.

    The aggregate figures carry a bare ratio rather than an :class:`Effect`, so
    they reuse :func:`format_delta` through this adapter.
    """
    return format_delta(Effect(value=value, unit="percent"))


def is_improvement(effect: Effect) -> bool:
    """Whether an effect's move counts as an improvement, keyed on its unit.

    This is the single place the sign-of-improvement rule lives, so a caller
    judging a direction-aware metric combines this with the metric's own
    direction rather than re-deriving the sign.

    For a ``"percent"`` delta the default is lower-is-better: a strictly negative
    value improves. A value of exactly zero does not — at rest a figure moved in
    no direction to call good.

    Args:
        effect: The effect to judge; its ``unit`` selects the rule.

    Returns:
        ``True`` when the effect's value is an improvement for its unit.
    """
    if effect.unit == "percent":
        return effect.value < 0
    assert_never(effect.unit)


#: The sign every spread and noise band is stated behind.
PLUS_MINUS = "±"

#: What separates a value from the spread that follows it. Exported because the
#: text table pads a cell's magnitude and spread into fields of their own and has
#: to rebuild the join around them.
SPREAD_SEPARATOR = f" {PLUS_MINUS} "


@dataclass(frozen=True, slots=True)
class MetricCellParts:
    """A value cell taken apart, so a table can pad each field to its own column width.

    Both fields are empty when the side reported nothing, and the spread alone is
    empty when the measurement carries no scatter.

    Attributes:
        magnitude: The scaled measurement.
        spread: What follows the ``±``: a percentage, or absolute units once it
            outgrows the median.
    """

    magnitude: str
    spread: str


def format_metric_cell_parts(
    median: float | None = None,
    spread: float | None = None,
    unit: MetricUnit | None = None,
) -> MetricCellParts:
    """A value cell's fields: the scaled measurement and the spread stated behind it.

    A spread past :data:`_RELATIVE_SPREAD_CAP_PCT` is restated in absolute units,
    so ``5B ± 7620%`` reads ``5B ± 381B`` instead.

    Args:
        median: The measurement, or ``None`` when the side reported nothing.
        spread: The half-range around the median, or ``None`` when none measured.
        unit: The metric's unit, or ``None`` when unitless.

    Returns:
        The magnitude and spread, each empty where there was nothing to state.
    """
    if median is None:
        return MetricCellParts(magnitude="", spread="")
    magnitude = format_value(median, unit)
    if spread is None:
        return MetricCellParts(magnitude=magnitude, spread="")
    if spread > _RELATIVE_SPREAD_CAP_PCT:
        return MetricCellParts(
            magnitude=magnitude, spread=format_value(abs(median * spread / 100), unit)
        )
    return MetricCellParts(magnitude=magnitude, spread=f"{spread:.0f}%")


def baseline_cell_parts(metric: MetricComparison) -> MetricCellParts:
    """A metric's baseline figure, taken apart the way :func:`format_metric_cell_parts` does."""
    return format_metric_cell_parts(
        metric.baseline_median, metric.baseline_spread, metric.meta.unit
    )


def candidate_cell_parts(
    side: CandidateMetric | None,
    unit: MetricUnit | None = None,
) -> MetricCellParts:
    """One candidate's side of a metric, taken apart into padded fields."""
    if side is None:
        return format_metric_cell_parts(None, None, unit)
    return format_metric_cell_parts(side.median, side.spread, unit)


# ---------------------------------------------------------------------------
# Verdict classification
# ---------------------------------------------------------------------------

type DisplayClass = Literal[
    "improved",
    "regressed",
    "unstable",
    "identical",
    "within-noise",
    "inconclusive",
]
"""How a verdict presents itself in the report.

``no-signal`` splits in three here: a metric resting on too few pairs for the
band method reads ``inconclusive``, one whose two sides measured close enough
to identical to starve the permutation test reads ``identical``, and every
other no-signal verdict reads ``within-noise``. The split is presentation only
— the stored verdict stays ``no-signal``.
"""


def display_class(verdict: MetricVerdict) -> DisplayClass:
    """Which display class a verdict reads as.

    The pair count is read before anything else about a band verdict: below
    :data:`MIN_BAND_N` the band is the noise floor constant rather than a
    measurement, so the verdict reads ``inconclusive``. A band verdict with
    enough pairs for the permutation test reads ``identical`` only when every
    one of them tied (``usable_n == 0``); anywhere ``usable_n`` sits between
    zero and :data:`MIN_PERMUTATION_N` some pairs did differ, so that reads
    ``within-noise``. An exact no-signal always reads ``within-noise``.
    """
    if verdict.method == "band" and verdict.n < MIN_BAND_N:
        return "inconclusive"
    if verdict.verdict != "no-signal":
        return verdict.verdict
    return _no_signal_class(verdict)


def _no_signal_class(verdict: MetricVerdict) -> DisplayClass:
    """The display class a no-signal verdict reads as, by the method that produced it.

    The method union is discriminated exhaustively so a new method fails to
    type-check here until it decides what its no-signal reads as. A band verdict
    with enough usable pairs but every one tied measured identical; every other
    no-signal — band or otherwise — reads within noise.
    """
    match verdict.method:
        case "band":
            if verdict.n >= MIN_PERMUTATION_N and verdict.usable_n == 0:
                return "identical"
            return "within-noise"
        case "permutation":
            return "within-noise"
        case "exact":
            return "within-noise"
        case _ as unreachable:  # pragma: no cover — exhaustive match over Method
            assert_never(unreachable)


_GLYPHS: dict[DisplayClass, str] = {
    "improved": "✓",
    "regressed": "✗",
    "unstable": "≈",
    "identical": "=",
    "within-noise": "~",
    "inconclusive": "?",
}


def get_glyph(shown: DisplayClass) -> str:
    """The glyph a display class is drawn with in the report's rows and legend."""
    return _GLYPHS[shown]


def shown_class(verdict: MetricVerdict | None) -> DisplayClass | None:
    """:func:`display_class`, or ``None`` when there is no verdict to show one for."""
    return None if verdict is None else display_class(verdict)


def format_verdict_delta(verdict: MetricVerdict) -> str:
    """The delta cell: the word ``unstable`` for a verdict too noisy to trust, else the delta."""
    return "unstable" if verdict.verdict == "unstable" else format_delta(verdict.delta)


#: The word each display class reads as, shared by the summary line and the legend.
VERDICT_GLOSSES: dict[DisplayClass, str] = {
    "improved": "improved",
    "regressed": "regressed",
    "unstable": "unstable",
    "identical": "identical",
    "within-noise": "within noise",
    "inconclusive": "inconclusive",
}

#: The display classes whose rows carry no news worth keeping above the fold.
QUIET_VERDICTS: frozenset[DisplayClass] = frozenset(
    {"within-noise", "identical", "inconclusive", "unstable"}
)


# ---------------------------------------------------------------------------
# Verdict evidence
# ---------------------------------------------------------------------------

# The scatter, relative to the median, past which a percentage stops informing:
# once the spread outgrows the median the percentage climbs without bound, so the
# evidence restates it in the metric's own units instead.
_RELATIVE_SPREAD_CAP_PCT = 100


def format_noise_band_value(noise_pct: float) -> str:
    """A noise band's figure, without the sign it is stated behind."""
    return f"{noise_pct:.1f}%"


def _format_noise_band(noise_pct: float) -> str:
    """A metric's noise band as the ``±N%`` the rows and highlights share."""
    return f"{PLUS_MINUS}{format_noise_band_value(noise_pct)}"


def format_pair_count(n: int) -> str:
    """How many pairs a verdict rests on, as the ``n=N`` the rows and footer share."""
    return f"n={n}"


def format_evidence(
    verdict: MetricVerdict,
    unit: MetricUnit | None = None,
    baseline_median: float | None = None,
) -> str:
    """The evidence suffix for a highlighted metric.

    Exact entries keep ``(exact)``. Unstable entries show the noise that swamped
    the signal — as a percentage while that stays readable, and against the
    baseline median in the metric's own units past
    :data:`_RELATIVE_SPREAD_CAP_PCT`. Improved/regressed/no-signal entries from
    approximate methods carry no trailing evidence.

    Args:
        verdict: The verdict to describe.
        unit: The metric's unit, for restating noise in absolute terms.
        baseline_median: The baseline median, for the absolute restatement.

    Returns:
        The evidence suffix, or ``""`` when there is nothing to add.
    """
    if verdict.method == "exact":
        return "(exact)"
    if verdict.verdict != "unstable":
        return ""
    if verdict.noise_pct > _RELATIVE_SPREAD_CAP_PCT and baseline_median is not None:
        noise = format_value(verdict.noise_abs, unit)
        return f"{PLUS_MINUS}{noise} noise on a {format_value(baseline_median, unit)} median"
    return f"noise {_format_noise_band(verdict.noise_pct)}"


# ---------------------------------------------------------------------------
# Verdict tallies
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class VerdictCounts:
    """How many metrics landed in each stored verdict class."""

    improved: int
    regressed: int
    unstable: int
    no_signal: int


def _candidate_at(metric: MetricComparison, index: int) -> CandidateMetric | None:
    """The candidate slice at ``index``, or ``None`` when the metric has no such entry."""
    if 0 <= index < len(metric.candidates):
        return metric.candidates[index]
    return None


def _each_verdict(
    metrics: MetricComparisons,
    candidate_index: int,
    fn: Callable[[MetricVerdict], None],
) -> None:
    """Run ``fn`` on one candidate's verdict for every metric that reported one.

    Verdicts belong to a candidate, never to the run, so callers read one
    candidate at a time. Metrics that candidate never reported are skipped.
    """
    for metric in metrics.values():
        candidate = _candidate_at(metric, candidate_index)
        if candidate is not None and candidate.verdict is not None:
            fn(candidate.verdict)


def count_verdicts(metrics: MetricComparisons, candidate_index: int) -> VerdictCounts:
    """Tally the stored verdict classes one candidate earned against the baseline.

    These are the verdicts as decided, not as displayed: ``no_signal`` covers
    every no-signal metric whether the text report shows it as identical or as
    within noise.

    Args:
        metrics: Every metric of the run, keyed by name.
        candidate_index: Which candidate's verdicts to count.

    Returns:
        The per-class tally.
    """
    tally = {"improved": 0, "regressed": 0, "unstable": 0, "no-signal": 0}

    def bump(verdict: MetricVerdict) -> None:
        tally[verdict.verdict] += 1

    _each_verdict(metrics, candidate_index, bump)
    return VerdictCounts(
        improved=tally["improved"],
        regressed=tally["regressed"],
        unstable=tally["unstable"],
        no_signal=tally["no-signal"],
    )


# ---------------------------------------------------------------------------
# Highlight selection
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MetricHighlight:
    """A metric worth calling out for one candidate, with the name it is reported under.

    Attributes:
        name: The metric name the highlight is reported under.
        metric: The metric's comparison data.
        candidate: The candidate slice that earned the highlight.
    """

    name: str
    metric: MetricComparison
    candidate: CandidateMetric


# Where each highlighted display class sits in the reported order; ``None`` opts
# a class out of highlights entirely.
_HIGHLIGHT_RANK: dict[DisplayClass, int | None] = {
    "regressed": 0,
    "improved": 1,
    "unstable": 2,
    "identical": None,
    "within-noise": None,
    "inconclusive": None,
}


def _highlight_weight(verdict: MetricVerdict) -> float:
    """How loud a highlight is within its class: noise for unstable, delta magnitude otherwise."""
    if verdict.verdict == "unstable":
        return verdict.noise_pct
    magnitude = abs(verdict.delta.value)
    return 0.0 if math.isnan(magnitude) else magnitude


def select_highlights(
    metrics: MetricComparisons,
    candidate_index: int,
) -> list[MetricHighlight]:
    """The metrics worth calling out for one candidate, ordered by class then loudness.

    Regressions come first (by delta magnitude, descending), then improvements
    the same way, then unstable metrics by noise. Ranking is per candidate
    because the verdicts are. Metrics that sat within the noise, measured
    identical, or were never reported carry no news and are left out. Ties keep
    the order the metrics were measured in.

    Args:
        metrics: Every metric of the run, keyed by name.
        candidate_index: Which candidate's verdicts to rank.

    Returns:
        The highlights in report order.
    """
    ranked: list[_RankedHighlight] = []
    for order, (name, metric) in enumerate(metrics.items()):
        candidate = _candidate_at(metric, candidate_index)
        if candidate is None or candidate.verdict is None:
            continue
        rank = _HIGHLIGHT_RANK[display_class(candidate.verdict)]
        if rank is None:
            continue
        highlight = MetricHighlight(name=name, metric=metric, candidate=candidate)
        # Order breaks weight ties toward first appearance; weight is negated so
        # a single ascending sort ranks by class, then descending loudness.
        ranked.append(
            _RankedHighlight(
                rank=rank,
                weight=-_highlight_weight(candidate.verdict),
                order=order,
                highlight=highlight,
            )
        )

    ranked.sort(key=_rank_key)
    return [entry.highlight for entry in ranked]


@dataclass(frozen=True, slots=True)
class _RankedHighlight:
    """A highlight with the keys it is ordered by: class rank, negated loudness, appearance."""

    rank: int
    weight: float
    order: int
    highlight: MetricHighlight


def _rank_key(entry: _RankedHighlight) -> tuple[int, float, int]:
    """The sort key ranking by class, then descending loudness, then appearance order."""
    return entry.rank, entry.weight, entry.order


#: The note closing a highlights block that held any metric the noise swamped.
UNSTABLE_FUTILITY_NOTE = "unstable metrics won't stabilize with more samples"


def highlight_label(highlight: MetricHighlight, *, qualify: bool) -> str:
    """The name a highlight is reported under, its kind named ahead of it when qualified.

    The highlights sit below the table, away from the section titles that told the
    reader which kind a row belonged to, so a multi-kind run has to carry the kind
    on the line itself. A single-kind run would say the same word on every line,
    which tells the reader nothing and only pushes the deltas right, so it stays
    with the bare metric name.

    Args:
        highlight: The highlight to name.
        qualify: Whether to prefix the metric's kind and short name, as a run
            spanning several kinds does.

    Returns:
        The reported name.
    """
    if qualify:
        meta = highlight.metric.meta
        return f"{meta.kind} {_SCOPE_SEPARATOR} {meta.short_name}"
    return highlight.name


def has_unstable_highlight(highlights: Sequence[MetricHighlight]) -> bool:
    """Whether any highlight is one the noise swamped, so it carries no usable delta."""
    return any(shown_class(highlight.candidate.verdict) == "unstable" for highlight in highlights)


@dataclass(frozen=True, slots=True)
class HighlightBlock:
    """One candidate's highlight entries, and whether the noise swamped any of them.

    Attributes:
        entries: The rendered highlight lines, gate trips included.
        unstable: Whether any entry is an unstable metric, so the block earns the
            futility note.
        label: The candidate sub-label the block sits under, or ``None`` for a
            single-candidate report that lists its entries directly.
    """

    entries: tuple[str, ...]
    unstable: bool
    label: str | None = None


# ---------------------------------------------------------------------------
# Geomean labeling and styling
# ---------------------------------------------------------------------------

#: The name the geomean row is reported under, in every renderer.
GEOMEAN_LABEL = "geomean"

#: The label a gate-trip line names the gated geomean under.
GATED_GEOMEAN_LABEL = "gated geomean"

#: The figure standing in for a geomean with nothing to aggregate.
NO_GEOMEAN_FIGURE = "—"

#: What an empty geomean says in place of the count behind a figure.
NO_STABLE_METRICS = "no stable metrics"

#: The whole cell a geomean with nothing left to aggregate prints, so it says so
#: rather than printing the ``0.0%`` an empty geomean computes to.
NO_GEOMEAN_CELL = f"{NO_GEOMEAN_FIGURE}  {NO_STABLE_METRICS}"

_SCOPE_SEPARATOR = "·"


#: Endings that take ``-es`` rather than a bare ``-s`` ("pass" → "passes").
_SIBILANT_ENDINGS = ("s", "x", "z", "ch", "sh")

_VOWELS = "aeiou"  # cspell:disable-line -- the vowel set, not a word


def _regular_plural(noun: str) -> str:
    """``noun`` under the regular English suffix rules.

    A sibilant ending takes ``-es``; a consonant followed by ``y`` takes
    ``-ies``; everything else takes ``-s``. Multi-word nouns inflect on their
    last word, which the suffix tests already look at.
    """
    if noun.endswith(_SIBILANT_ENDINGS):
        return f"{noun}es"
    if len(noun) > 1 and noun.endswith("y") and noun[-2] not in _VOWELS:
        return f"{noun[:-1]}ies"
    return f"{noun}s"


def pluralize(count: int, noun: str, plural: str | None = None) -> str:
    """Inflect ``noun`` for ``count``, or use the explicit ``plural`` instead.

    A count of one keeps ``noun`` as given; any other count — zero and
    negatives included — takes the plural form.
    """
    if count == 1:
        return f"{count} {noun}"
    return f"{count} {plural if plural is not None else _regular_plural(noun)}"


def geomean_label(n: int) -> str:
    """The geomean row's label, carrying the count of metrics behind the figure.

    A table with one candidate names the count here, which frees its cells of
    everything but the aggregate itself. An empty geomean has no count to name
    and takes :data:`GEOMEAN_LABEL` alone.
    """
    return GEOMEAN_LABEL if n == 0 else f"{GEOMEAN_LABEL} ({pluralize(n, 'stable metric')})"


def geomean_scope_label(scope: str) -> str:
    """The label of an aggregate row covering one scope — a group or a kind."""
    return f"{GEOMEAN_LABEL} {_SCOPE_SEPARATOR} {scope}"


def _geomean_provenance(geomean: GeomeanResult) -> str:
    """The provenance suffix behind a scope's figure.

    ``(n)`` when every scope metric stands behind the figure, ``(n/m)`` when
    exclusions thinned them.
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
        delta=_delta_of(geomean.value),
        provenance=pluralize(geomean.n, "stable metric"),
        band=format_noise_band_value(geomean.band) if geomean.band > 0 else "",
    )


def _is_quiet_row(outcomes: Sequence[DisplayClass | None]) -> bool:
    """Whether every defined display class in a row is a quiet one.

    A row with no verdicts at all is left alone rather than counted as quiet.
    """
    defined = [outcome for outcome in outcomes if outcome is not None]
    return len(defined) > 0 and all(outcome in QUIET_VERDICTS for outcome in defined)


def geomean_value_style(
    geomean: GeomeanResult,
    outcomes: Sequence[DisplayClass | None],
) -> str:
    """How a geomean's figure is styled: bold always, colored once it clears the noise band.

    The figure is an average of ratios, so it moves whether or not anything did.
    A value inside the band is emboldened and left uncolored. ``outcomes`` — the
    display class of each metric behind the figure — vetoes the color when every
    one is quiet, since coloring that would announce a win the rows all decline
    to claim. An empty ``outcomes`` leaves the band deciding alone.

    Args:
        geomean: The aggregate whose figure is being styled.
        outcomes: The display class of each metric behind the figure.

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


# ---------------------------------------------------------------------------
# Verdict summary
# ---------------------------------------------------------------------------

# The order the summary line and the legend list display classes in.
_DISPLAY_CLASS_ORDER: tuple[DisplayClass, ...] = (
    "improved",
    "regressed",
    "unstable",
    "identical",
    "within-noise",
    "inconclusive",
)


def _display_counts(
    metrics: MetricComparisons,
    candidate_index: int,
) -> dict[DisplayClass, int]:
    """How many metrics one candidate landed in each display class."""
    counts: dict[DisplayClass, int] = dict.fromkeys(_DISPLAY_CLASS_ORDER, 0)

    def bump(verdict: MetricVerdict) -> None:
        counts[display_class(verdict)] += 1

    _each_verdict(metrics, candidate_index, bump)
    return counts


def verdict_summary_parts(metrics: MetricComparisons, candidate_index: int) -> list[str]:
    """One tally part per display class, in legend order, as rich-markup strings.

    The count is the news, so it alone carries the class color while the glyph
    and the word beside it stay in the surrounding style. A part counting nothing
    is dimmed end to end whatever its class — a zero is not news either way.
    Counts are padded to the widest count's digit width so a renderer can align
    them.

    Args:
        metrics: Every metric of the run, keyed by name.
        candidate_index: Which candidate's verdicts to summarize.

    Returns:
        One markup string per display class.
    """
    counts = _display_counts(metrics, candidate_index)
    max_width = max(len(str(count)) for count in counts.values())

    parts: list[str] = []
    for shown in _DISPLAY_CLASS_ORDER:
        count = counts[shown]
        padded = str(count).rjust(max_width)
        glyph = get_glyph(shown)
        gloss = VERDICT_GLOSSES[shown]
        if count == 0:
            parts.append(markup(f"{glyph} {padded} {gloss}", "dim"))
        else:
            colored = markup(padded, VERDICT_STYLES[shown])
            parts.append(f"{escape(glyph)} {colored} {escape(gloss)}")
    return parts


# ---------------------------------------------------------------------------
# Footer generation
# ---------------------------------------------------------------------------


def _samples_hint(command: str) -> str:
    """The hint asking for more samples, naming the command that would carry them.

    The command is stated whole and backtick-marked so
    :func:`gymrat_py.report.style.format_hint` sets it apart from the prose: a
    reader copies the line rather than assembling the invocation themselves.
    """
    return (
        f"re-run with `gymrat {command} --samples {MIN_PERMUTATION_N}` "
        f"or more for statistical verdicts"
    )


# Shown when the run had enough samples but some metrics paired fewer rounds than
# the permutation test needs: suggesting more samples would be wrong.
_DROPPED_ROUNDS_HINT = (
    "some rounds were dropped — not all samples produced paired measurements for every metric"
)

# How the band method names itself wherever the footer describes a fallback.
_BAND_METHOD = "noise band ±(half-range × K)"


@dataclass(frozen=True, slots=True)
class _FooterData:
    """The pair counts the footer sorts by the cause that forced each fallback.

    ``permutation`` carries the pair counts of every permutation verdict.
    ``shortage`` and ``ties`` split the band-method verdicts by cause: too few
    total pairs, or too many of them tied away.
    """

    permutation: list[int]
    shortage: list[int]
    ties: list[int]


def _classify_verdict(verdict: MetricVerdict, data: _FooterData) -> None:
    """Sort one verdict's pair count into the footer cause it belongs to.

    The method union is discriminated exhaustively: exact verdicts contribute
    nothing to the footer by decision, an explicit arm rather than a fall-through
    a new method could slip past unnoticed.
    """
    match verdict.method:
        case "permutation":
            data.permutation.append(verdict.n)
        case "band":
            if verdict.n < MIN_PERMUTATION_N:
                data.shortage.append(verdict.n)
            else:
                data.ties.append(verdict.usable_n)
        case "exact":
            return
        case _ as unreachable:  # pragma: no cover — exhaustive match over Method
            assert_never(unreachable)


def _collect_footer_data(metrics: MetricComparisons) -> _FooterData:
    """Sort every verdict's pair count into the cause it belongs to, in one pass."""
    data = _FooterData(permutation=[], shortage=[], ties=[])
    for metric in metrics.values():
        for candidate in metric.candidates:
            if candidate.verdict is not None:
                _classify_verdict(candidate.verdict, data)
    return data


def _method_lines(data: _FooterData) -> list[str]:
    """The verbose method lines naming how each verdict was decided, each dimmed.

    A band fallback gets one line per cause: the highest total pair count for a
    shortage — even the best-off metric fell this far short — and the lowest
    usable pair count for ties, so each line stays true of every metric behind
    it.
    """
    lines: list[str] = []
    if data.permutation:
        desc = (
            f"verdicts: sign-flip permutation test on pairs "
            f"({format_pair_count(min(data.permutation))} ≥ {MIN_PERMUTATION_N}) "
            f"· ~ = no signal at α=0.05"
        )
        lines.append(markup(desc, "dim"))
    if data.shortage:
        desc = (
            f"{_BAND_METHOD} — {format_pair_count(max(data.shortage))} "
            f"below permutation floor ({MIN_PERMUTATION_N} pairs)"
        )
        lines.append(markup(desc, "dim"))
    if data.ties:
        desc = (
            f"{_BAND_METHOD} — ties left {format_pair_count(min(data.ties))} "
            f"usable pairs ({MIN_PERMUTATION_N} needed)"
        )
        lines.append(markup(desc, "dim"))
    return lines


def _shortage_hint(shortage: Sequence[int], samples: int | None, command: str) -> str | None:
    """The hint for metrics that fell to the band because their paired count was short.

    When the run's own sample count is below the floor, more samples are the
    fix. When it had enough samples but rounds were dropped during pairing,
    suggesting more samples is misleading.
    """
    if not shortage:
        return None
    if samples is not None and samples >= MIN_PERMUTATION_N:
        return _DROPPED_ROUNDS_HINT
    return _samples_hint(command)


def footer_lines(
    metrics: MetricComparisons,
    *,
    verbose: bool,
    format_hint: Callable[[str], str],
    command: str,
    samples: int | None = None,
) -> list[str]:
    """The footer: how each verdict was decided when verbose, and the samples hint.

    When ``samples`` is provided the hint distinguishes insufficient samples from
    dropped rounds. Renderers differ only in how they format the hint line, which
    ``format_hint`` owns.

    Args:
        metrics: Every metric of the run, keyed by name.
        verbose: Whether to include the method lines naming each verdict's basis.
        format_hint: Turns a bare hint into the renderer's own hint line.
        command: The subcommand the report was produced by, so a hint suggesting
            a re-run names the whole invocation.
        samples: The run's sample count, to distinguish shortage from dropped
            rounds. Left ``None``, a shortage always suggests more samples.

    Returns:
        The footer lines, method lines (when verbose) first, then the hint.
    """
    data = _collect_footer_data(metrics)
    hint = _shortage_hint(data.shortage, samples, command)
    lines = _method_lines(data) if verbose else []
    if hint is not None:
        lines.append(format_hint(hint))
    return lines
