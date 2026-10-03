"""The human-readable text reports: the compare report and the measure report.

A compare report is the run header, the comparison table, a one-line verdict
summary per candidate, a highlights block, the ``--fail-on`` gate trips, the
verbose method footer, and the worktree-cleanup footer. A measure report is the
run header, the measurement table, and the worktree-cleanup footer — it carries
no verdict machinery, since a single run has nothing to compare against.

The comparison tables come from :mod:`.single` and :mod:`.multi`. The measurement
table, the selection of the metrics a compare report highlights, and the footer
lines naming how each verdict was decided are built here.

The table renderers return lines already resolved to text (ANSI or plain) for the
run's color choice; the summary, highlights, and footer blocks are built as rich
markup here and resolved the same way, so color is decided once per block through
:func:`gymrat.report.style.render_lines`.
"""

from __future__ import annotations

import math
import operator
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, assert_never

from rich.cells import cell_len, set_cell_size
from rich.markup import escape
from rich.text import Text

from gymrat.metric_name import format_inline, parse
from gymrat.model import PERMUTATION_MIN_N, PERMUTATION_P_THRESHOLD
from gymrat.plural import pluralize
from gymrat.report.display import GLYPHS, DisplayClass, display_class
from gymrat.report.format import (
    format_evidence,
    format_metric_cell_parts,
    format_pair_count,
    format_percent_delta,
    format_verdict_delta,
)
from gymrat.report.geomean_label import GATED_GEOMEAN_LABEL
from gymrat.report.sections import plan_sections, spans_many_kinds
from gymrat.report.style import (
    SCOPE_SEPARATOR,
    VARIANT_NAME_STYLE,
    VERDICT_STYLES,
    format_hint,
    join_header_parts,
    markup,
    render_lines,
    render_markup_line,
    truncate_labels,
)
from gymrat.report.table.markup import group_metric_cell, header_metric_cell, indented_section_label
from gymrat.report.table.render import build_cell_dispatcher, plan_table_skeleton, render_body
from gymrat.report.tally import verdict_summary_parts
from gymrat.report.text.multi import render_comparison_table
from gymrat.report.text.single import render_table
from gymrat.report.types import GeomeanFailOn, ReportOptions
from gymrat.report.types import candidate_at as _candidate_at

if TYPE_CHECKING:
    from collections.abc import Sequence

    from gymrat.model import MetricVerdict
    from gymrat.report.format import MetricCellParts
    from gymrat.report.types import (
        CandidateComparison,
        ComparisonResult,
        FailOnCondition,
        MeasurementResult,
        MetricComparison,
        MetricComparisons,
    )
    from gymrat.targets import WorktreeRemovalFailure

# The default presentation flags: detect color, no header override. Immutable, so
# one shared instance is safe as a default argument.
_DEFAULT_OPTIONS = ReportOptions()

# Gap between the longest highlighted metric name and the delta that follows it.
_HIGHLIGHT_NAME_GUTTER = 2

# Width the highlights block right-aligns its deltas in — the length of a `±NN.N%`.
_HIGHLIGHT_DELTA_WIDTH = 6

_HIGHLIGHTS_HEADING = "highlights"

# The glyph flagging a gate the run's own `--fail-on` conditions would trip.
_GATE_TRIP_GLYPH = "⚑"


def _render_block(markup_lines: Sequence[str], *, color: bool | None) -> list[str]:
    """Resolve a block of markup lines to rendered text, one output line per input.

    Each line is resolved through the same color choice as the rest of the report,
    so a block built here sits flush against the table lines the table renderers
    already resolved.

    Args:
        markup_lines: The rich-markup lines to resolve.
        color: The explicit color choice, or ``None`` to defer to the
            environment and TTY detection.

    Returns:
        One rendered text line per input markup line.
    """
    if not markup_lines:
        return []
    return render_lines(*markup_lines, color=color).split("\n")


def with_display_labels(result: ComparisonResult) -> ComparisonResult:
    """``result`` with every variant label replaced by the name the report prints.

    The baseline and candidate labels are shortened together, so a label prints
    the same way wherever the report names it — the header, the column it heads,
    the geomean row. Metric names are left whole; only the variant labels shorten.

    Args:
        result: The comparison to relabel.

    Returns:
        A copy with shortened variant labels.
    """
    labels = truncate_labels([result.baseline_label, *(c.label for c in result.candidates)])
    return replace(
        result,
        baseline_label=labels[0],
        candidates=tuple(
            replace(candidate, label=labels[index + 1])
            for index, candidate in enumerate(result.candidates)
        ),
    )


def paired_samples(samples: int) -> str:
    """The ``N paired samples`` label the comparison report header carries."""
    return pluralize(samples, "paired sample")


def _compare_header(display: ComparisonResult) -> str:
    """The compare report's run header as markup: the baseline's role, the variants, the run."""
    candidate_names = ", ".join(
        markup(candidate.label, VARIANT_NAME_STYLE) for candidate in display.candidates
    )
    return join_header_parts([
        markup("gymrat compare", "bold"),
        f"baseline {markup(display.baseline_label, VARIANT_NAME_STYLE)} ↔ {candidate_names}",
        escape(paired_samples(display.samples)),
        f"adapter: {escape(display.adapter)}",
    ])


# ---------------------------------------------------------------------------
# Verdict summary
# ---------------------------------------------------------------------------


def _render_summary(metrics: MetricComparisons, candidate_index: int) -> str:
    """One markup line tallying every verdict class one candidate earned."""
    return "   ".join(verdict_summary_parts(metrics, candidate_index))


def _render_summaries(result: ComparisonResult) -> list[str]:
    """One markup summary line per candidate, each behind that candidate's bold label."""
    label_width = max(cell_len(candidate.label) for candidate in result.candidates)
    return [
        f"{markup(set_cell_size(candidate.label, label_width), 'bold')}  "
        f"{_render_summary(result.metrics, index)}"
        for index, candidate in enumerate(result.candidates)
    ]


# ---------------------------------------------------------------------------
# Highlights and gate trips
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MetricHighlight:
    """A metric worth calling out for one candidate, with the name it is reported under.

    Attributes:
        name: The metric name the highlight is reported under.
        metric: The metric's comparison data.
        verdict: The candidate's verdict that earned the highlight.
    """

    name: str
    metric: MetricComparison
    verdict: MetricVerdict


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
    magnitude = abs(verdict.delta)
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
    ranked: list[tuple[int, float, MetricHighlight]] = []
    for name, metric in metrics.items():
        candidate = _candidate_at(metric, candidate_index)
        if candidate is None or candidate.verdict is None:
            continue
        rank = _HIGHLIGHT_RANK[display_class(candidate.verdict)]
        if rank is None:
            continue
        highlight = MetricHighlight(name=name, metric=metric, verdict=candidate.verdict)
        ranked.append((rank, -_highlight_weight(candidate.verdict), highlight))

    ranked.sort(key=operator.itemgetter(0, 1))
    return [highlight for _, _, highlight in ranked]


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
        return f"{escape(meta.kind)} {SCOPE_SEPARATOR} {escape(meta.short_name)}"
    return format_inline(parse(highlight.name))


def has_unstable_highlight(highlights: Sequence[MetricHighlight]) -> bool:
    """Whether any highlight is one the noise swamped, so it carries no usable delta."""
    return any(display_class(highlight.verdict) == "unstable" for highlight in highlights)


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


def _highlight_entries(metrics: MetricComparisons, candidate_index: int) -> HighlightBlock:
    """The highlight entries one candidate earned, and whether the noise swamped any."""
    highlights = select_highlights(metrics, candidate_index)
    if not highlights:
        return HighlightBlock(entries=(), unstable=False)

    qualify = spans_many_kinds(metrics)
    labels = [highlight_label(highlight, qualify=qualify) for highlight in highlights]
    label_widths = [cell_len(Text.from_markup(label).plain) for label in labels]
    name_width = max(label_widths) + _HIGHLIGHT_NAME_GUTTER

    entries: list[str] = []
    for highlight, label, width in zip(highlights, labels, label_widths, strict=True):
        verdict = highlight.verdict
        shown = display_class(verdict)
        style = VERDICT_STYLES[shown]
        delta = format_verdict_delta(verdict)
        evidence = format_evidence(
            verdict, highlight.metric.meta.unit, highlight.metric.baseline_median
        )

        label_field = f"{label}{' ' * (name_width - width)}"
        delta_field = (
            f"{' ' * max(0, _HIGHLIGHT_DELTA_WIDTH - cell_len(delta))}{markup(delta, style)}"
        )
        suffix = "" if evidence == "" else f"  {markup(evidence, 'dim')}"
        entries.append(f"  {markup(GLYPHS[shown], style)} {label_field}{delta_field}{suffix}")

    return HighlightBlock(entries=tuple(entries), unstable=has_unstable_highlight(highlights))


def _gate_trip_lines(
    candidate: CandidateComparison,
    conditions: Sequence[FailOnCondition],
) -> list[str]:
    """The gate-trip lines for a candidate whose gated geomean cleared a ``--fail-on`` threshold.

    Only the geomean conditions gate here; the regressed condition contributes no
    line. A kind with no gated geomean, or one aggregating nothing, never trips —
    an informational kind cannot fail a gate it does not stand behind.

    Args:
        candidate: The candidate to check.
        conditions: The run's ``--fail-on`` conditions.

    Returns:
        One markup line per kind whose gated geomean exceeded a threshold.
    """
    thresholds = [condition.pct for condition in conditions if isinstance(condition, GeomeanFailOn)]
    style = VERDICT_STYLES["regressed"]

    lines: list[str] = []
    for kind in candidate.kinds:
        geomean = kind.gated_geomean
        if geomean is None or geomean.n == 0:
            continue
        delta = format_percent_delta(geomean.value)
        # `:g` states the threshold as it was written, dropping a trailing `.0`.
        lines.extend(
            f"  {markup(_GATE_TRIP_GLYPH, style)} {escape(kind.kind)} "
            f"{GATED_GEOMEAN_LABEL} {markup(delta, style)} "
            f"exceeded --fail-on geomean:{pct:g}"
            for pct in thresholds
            if geomean.value >= pct
        )
    return lines


def _highlight_section(blocks: Sequence[HighlightBlock]) -> list[str]:
    """The highlights block: a heading, each candidate's entries, and the futility note.

    A block with a label heads its entries with the bold label and indents them
    under it; a block with no label lists its entries directly.

    Args:
        blocks: One highlight block per candidate.

    Returns:
        The heading, highlight entries, and futility note, or an empty list
        when nothing highlighted.
    """
    non_empty = [block for block in blocks if block.entries]
    if not non_empty:
        return []

    lines = [markup(_HIGHLIGHTS_HEADING, "bold")]
    for block in non_empty:
        if block.label is None:
            lines.extend(block.entries)
        else:
            lines.append(f"  {markup(block.label, 'bold')}")
            lines.extend(f"  {entry}" for entry in block.entries)
    if any(block.unstable for block in non_empty):
        lines.append(f"  {markup(UNSTABLE_FUTILITY_NOTE, 'dim')}")
    return lines


def _render_highlights(
    result: ComparisonResult,
    conditions: Sequence[FailOnCondition],
) -> list[str]:
    """The highlights block, gate trips folded in: one labeled subsection per candidate.

    A single-candidate report lists its entries directly, with no sub-label.

    Args:
        result: The comparison whose candidates to highlight.
        conditions: The run's ``--fail-on`` conditions.

    Returns:
        The highlights block lines, or an empty list when nothing highlighted.
    """
    multi = len(result.candidates) > 1
    blocks: list[HighlightBlock] = []
    for index, candidate in enumerate(result.candidates):
        block = _highlight_entries(result.metrics, index)
        blocks.append(
            HighlightBlock(
                entries=(*block.entries, *_gate_trip_lines(candidate, conditions)),
                unstable=block.unstable,
                label=candidate.label if multi else None,
            )
        )
    return _highlight_section(blocks)


# ---------------------------------------------------------------------------
# Footers
# ---------------------------------------------------------------------------


_DROPPED_ROUNDS_HINT = (
    "some rounds were dropped — not all samples produced paired measurements for every metric"
)

_BAND_METHOD = "noise band ±(half-range × K)"


@dataclass(slots=True)
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

    Args:
        verdict: The verdict to classify.
        data: The footer tallies, updated in place.
    """
    match verdict.method:
        case "permutation":
            data.permutation.append(verdict.n)
        case "band":
            if verdict.n < PERMUTATION_MIN_N:
                data.shortage.append(verdict.n)
            else:
                data.ties.append(verdict.usable_n)
        case "exact":
            return
        case _ as unreachable:  # pragma: no cover — exhaustive match over VerdictMethod
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

    Args:
        data: The pair counts sorted by the cause that forced each fallback.

    Returns:
        One dimmed line per method that contributed a verdict.
    """
    lines: list[str] = []
    if data.permutation:
        desc = (
            f"verdicts: sign-flip permutation test on pairs "
            f"({format_pair_count(min(data.permutation))} ≥ {PERMUTATION_MIN_N}) "
            f"· ~ = no signal at α={PERMUTATION_P_THRESHOLD}"
        )
        lines.append(markup(desc, "dim"))
    if data.shortage:
        desc = (
            f"{_BAND_METHOD} — {format_pair_count(max(data.shortage))} "
            f"below permutation floor ({PERMUTATION_MIN_N} pairs)"
        )
        lines.append(markup(desc, "dim"))
    if data.ties:
        desc = (
            f"{_BAND_METHOD} — ties left {format_pair_count(min(data.ties))} "
            f"usable pairs ({PERMUTATION_MIN_N} needed)"
        )
        lines.append(markup(desc, "dim"))
    return lines


def _shortage_hint(shortage: Sequence[int], samples: int, command: str) -> str | None:
    """The hint for metrics that fell to the band because their paired count was short.

    When the run's own sample count is below the floor, more samples are the
    fix. When it had enough samples but rounds were dropped during pairing,
    suggesting more samples is misleading.

    The suggested command is stated whole and backtick-marked so
    :func:`gymrat.report.style.format_hint` sets it apart from the prose: a
    reader copies the line rather than assembling the invocation themselves.

    Args:
        shortage: The pair counts of verdicts that fell to the band method
            for lack of pairs.
        samples: The run's own sample count.
        command: The subcommand name to embed in the suggested re-run.

    Returns:
        The samples hint, the dropped-rounds hint, or ``None`` when the
        shortage list is empty.
    """
    if not shortage:
        return None
    if samples >= PERMUTATION_MIN_N:
        return _DROPPED_ROUNDS_HINT
    return (
        f"re-run with `gymrat {command} --samples {PERMUTATION_MIN_N}` "
        f"or more for statistical verdicts"
    )


def footer_lines(
    metrics: MetricComparisons,
    *,
    verbose: bool,
    command: str,
    samples: int,
) -> list[str]:
    """The footer: how each verdict was decided when verbose, and the samples hint.

    Args:
        metrics: Every metric of the run, keyed by name.
        verbose: Whether to include the method lines naming each verdict's basis.
        command: The subcommand the report was produced by, so a hint suggesting
            a re-run names the whole invocation.
        samples: The run's sample count, to distinguish shortage from dropped
            rounds.

    Returns:
        The footer lines, method lines (when verbose) first, then the hint.
    """
    data = _collect_footer_data(metrics)
    hint = _shortage_hint(data.shortage, samples, command)
    lines = _method_lines(data) if verbose else []
    if hint is not None:
        lines.append(format_hint(hint))
    return lines


def _to_single_line(text: str) -> str:
    return " ".join(text.split())


def format_cleanup_failures(
    left_behind: Sequence[WorktreeRemovalFailure],
    prune_error: str | None,
) -> list[str]:
    """Format worktree removal failures and a prune error into indented diagnostic lines.

    Each git diagnostic is collapsed onto one line. The lines carry no styling, so
    a caller outside the report — the sampling layer that logs a dirty cleanup —
    can print them as they are.

    Args:
        left_behind: The worktrees the run could not remove, with git's reason.
        prune_error: git's reason the prune step failed, or ``None`` when it did
            not.

    Returns:
        One line per left-behind worktree, then the prune-failure line when
        present.
    """
    lines = [
        f"  left behind: {failure.dir} ({_to_single_line(failure.error)})"
        for failure in left_behind
    ]
    if prune_error is not None:
        lines.append(f"  worktree prune failed: {_to_single_line(prune_error)}")
    return lines


def _render_worktree_footer(result: ComparisonResult | MeasurementResult) -> list[str]:
    """The worktree-cleanup footer as markup, or nothing when the cleanup was clean.

    The lines are plain text escaped for markup rendering: the cleanup footer is
    the same color on or off.

    Args:
        result: The comparison or measurement result to draw the footer from.

    Returns:
        Markup lines describing the cleanup failures, or an empty list when
        the cleanup was clean.
    """
    details = format_cleanup_failures(result.worktrees_left_behind, result.worktree_prune_error)
    if not details:
        return []
    left_behind = len(result.worktrees_left_behind)
    header = (
        f"{pluralize(result.worktrees_removed, 'worktree')} removed · {left_behind} left behind"
    )
    return [escape(line) for line in (header, *details)]


# ---------------------------------------------------------------------------
# Measurement table
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _MeasureRow:
    """One measured metric's name, its section label, and its padded value fields."""

    name: str
    label: str
    value: MetricCellParts


def render_measure_table(
    result: MeasurementResult,
    label: str,
    *,
    color: bool | None,
) -> list[str]:
    """Render a single-revision measurement table with a median and spread per metric.

    Args:
        result: The measurement to draw.
        label: The target's display label, already truncated, heading the value
            column.
        color: The explicit color choice, or ``None`` to defer to the environment.

    Returns:
        The rendered table lines.
    """
    layout = plan_sections(
        result.metrics,
        lambda name, group, metric: _MeasureRow(
            name=name,
            label=indented_section_label(metric.meta.short_name, group),
            value=format_metric_cell_parts(metric.median, metric.spread, metric.meta.unit),
        ),
    )
    skeleton = plan_table_skeleton(layout, result.config_kinds, lambda row: row.value, label)
    widths = [skeleton.metric_width, skeleton.value_width]

    def metric_cells(row: _MeasureRow) -> tuple[str, str]:
        return escape(skeleton.name_cell(row)), escape(skeleton.value_cell(row))

    to_cells = build_cell_dispatcher(
        header=lambda title: (header_metric_cell(title), markup(label, VARIANT_NAME_STYLE)),
        group=lambda group_label: (group_metric_cell(group_label), ""),
        metric=metric_cells,
    )

    return render_body(skeleton.body, widths, to_cells, color=color)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def render_report(result: ComparisonResult, options: ReportOptions = _DEFAULT_OPTIONS) -> str:
    """Render a full comparison report.

    The report is the run header, the comparison table (single- or
    multi-candidate), a one-line verdict summary per candidate, the highlights
    block with any ``--fail-on`` gate trips, and — when non-empty — the verbose
    method footer and the worktree-cleanup footer.

    Args:
        result: The comparison to draw.
        options: The presentation flags. ``options.header`` replaces the run
            header verbatim; ``options.color`` forces color on or off, or defers
            to the environment when ``None``; ``options.verbose`` adds the method
            footer; ``options.fail_on`` names the gate conditions;
            ``options.command`` names the subcommand a re-run hint suggests.

    Returns:
        The rendered report.
    """
    color = options.color
    display = with_display_labels(result)
    conditions = options.fail_on or ()

    if options.header is not None:
        lines = [options.header]
    else:
        lines = [render_markup_line(_compare_header(display), color=color)]

    if len(display.candidates) > 1:
        lines.extend(render_comparison_table(display, color=color))
        lines.append("")
        lines.extend(_render_block(_render_summaries(display), color=color))
    elif len(display.candidates) == 1:
        lines.extend(render_table(display, display.candidates[0], 0, color=color))
        lines.append("")
        lines.extend(_render_block([_render_summary(display.metrics, 0)], color=color))

    highlights = _render_highlights(display, conditions)
    if highlights:
        lines.append("")
        lines.extend(_render_block(highlights, color=color))

    footer = [
        *footer_lines(
            display.metrics,
            verbose=bool(options.verbose),
            command=options.command,
            samples=display.samples,
        ),
        *_render_worktree_footer(display),
    ]
    if footer:
        lines.append("")
        lines.extend(_render_block(footer, color=color))

    return "\n".join(lines)


def render_measure_report(
    result: MeasurementResult,
    options: ReportOptions = _DEFAULT_OPTIONS,
) -> str:
    """Render a single-target measurement report.

    The report is the run header, the measurement table, and — when the cleanup
    left something behind — the worktree-cleanup footer. A single run has nothing
    to compare against, so it carries no verdict summary, highlights, or geomean.

    Args:
        result: The measurement to draw.
        options: The presentation flags. ``options.color`` forces color on or off,
            or defers to the environment when ``None``.

    Returns:
        The rendered report.
    """
    color = options.color
    label = truncate_labels([result.label])[0]
    header = join_header_parts([
        markup("gymrat measure", "bold"),
        markup(label, VARIANT_NAME_STYLE),
        escape(pluralize(result.samples, "sample")),
        f"adapter: {escape(result.adapter)}",
    ])

    lines = [render_markup_line(header, color=color)]
    lines.extend(render_measure_table(result, label, color=color))

    footer = _render_worktree_footer(result)
    if footer:
        lines.append("")
        lines.extend(_render_block(footer, color=color))

    return "\n".join(lines)
