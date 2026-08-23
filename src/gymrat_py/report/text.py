"""The human-readable text reports: the compare report and the measure report.

A compare report is the run header, the comparison table, a one-line verdict
summary per candidate, a highlights block, the ``--fail-on`` gate trips, the
verbose method footer, and the worktree-cleanup footer. A measure report is the
run header, the measurement table, and the worktree-cleanup footer — it carries
no verdict machinery, since a single run has nothing to compare against.

The table renderers return lines already resolved to text (ANSI or plain) for the
run's color choice; the summary, highlights, and footer blocks are built as rich
markup here and resolved the same way, so color is decided once per block through
:func:`gymrat_py.report.style.render_lines`.
"""

from __future__ import annotations

import re
from dataclasses import replace
from typing import TYPE_CHECKING

from rich.markup import escape
from rich.text import Text

from gymrat_py.model import Effect
from gymrat_py.report.format import (
    GATED_GEOMEAN_LABEL,
    UNSTABLE_FUTILITY_NOTE,
    HighlightBlock,
    display_class,
    footer_lines,
    format_delta,
    format_evidence,
    format_verdict_delta,
    get_glyph,
    has_unstable_highlight,
    highlight_label,
    pluralize,
    select_highlights,
    verdict_summary_parts,
)
from gymrat_py.report.sections import spans_many_kinds
from gymrat_py.report.style import (
    VARIANT_NAME_STYLE,
    VERDICT_STYLES,
    format_hint_label,
    render_lines,
    truncate_labels,
)
from gymrat_py.report.table import markup
from gymrat_py.report.text_measure import render_measure_table
from gymrat_py.report.text_multi import render_comparison_table
from gymrat_py.report.text_single import render_table
from gymrat_py.report.types import GeomeanFailOn, ReportOptions

if TYPE_CHECKING:
    from collections.abc import Sequence

    from gymrat_py.report.types import (
        CandidateComparison,
        ComparisonResult,
        FailOnCondition,
        MeasurementResult,
        MetricComparisons,
    )
    from gymrat_py.targets import WorktreeRemovalFailure

# A wide render width so no line soft-wraps.
_RENDER_WIDTH = 200

# The default presentation flags: detect color, no header override. Immutable, so
# one shared instance is safe as a default argument.
_DEFAULT_OPTIONS = ReportOptions()

# The `·` separator every report header joins its parts with, dimmed in color.
_HEADER_SEPARATOR = "·"

# Gap between the longest highlighted metric name and the delta that follows it.
_HIGHLIGHT_NAME_GUTTER = 2

# Width the highlights block right-aligns its deltas in — the length of a `±NN.N%`.
_HIGHLIGHT_DELTA_WIDTH = 6

# The heading a non-empty highlights block opens with.
_HIGHLIGHTS_HEADING = "highlights"

# The glyph flagging a gate the run's own `--fail-on` conditions would trip.
_GATE_TRIP_GLYPH = "⚑"


def _join_header_parts(parts: list[str]) -> str:
    """Join header parts with the dimmed ``·`` separator every report header shares."""
    return f" {markup(_HEADER_SEPARATOR, 'dim')} ".join(parts)


def _render_line(text: str, *, color: bool | None) -> str:
    """Resolve a markup line to text once, deferring wrapping so the line stays whole."""
    return render_lines(Text.from_markup(text), color=color, width=_RENDER_WIDTH)


def _render_block(markup_lines: Sequence[str], *, color: bool | None) -> list[str]:
    """Resolve a block of markup lines to rendered text, one output line per input.

    Each line is resolved through the same color choice as the rest of the report,
    so a block built here sits flush against the table lines the table renderers
    already resolved.
    """
    if not markup_lines:
        return []
    rendered = render_lines(*markup_lines, color=color, width=_RENDER_WIDTH)
    return rendered.split("\n")


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
    return _join_header_parts(
        [
            markup("gymrat compare", "bold"),
            f"baseline {markup(display.baseline_label, VARIANT_NAME_STYLE)} ↔ {candidate_names}",
            escape(paired_samples(display.samples)),
            f"adapter: {escape(display.adapter)}",
        ]
    )


# ---------------------------------------------------------------------------
# Verdict summary
# ---------------------------------------------------------------------------


def _render_summary(metrics: MetricComparisons, candidate_index: int) -> str:
    """One markup line tallying every verdict class one candidate earned."""
    return "   ".join(verdict_summary_parts(metrics, candidate_index))


def _render_summaries(result: ComparisonResult) -> list[str]:
    """One markup summary line per candidate, each behind that candidate's bold label."""
    label_width = max(len(candidate.label) for candidate in result.candidates)
    return [
        f"{markup(candidate.label.ljust(label_width), 'bold')}  "
        f"{_render_summary(result.metrics, index)}"
        for index, candidate in enumerate(result.candidates)
    ]


# ---------------------------------------------------------------------------
# Highlights and gate trips
# ---------------------------------------------------------------------------


def _highlight_entries(metrics: MetricComparisons, candidate_index: int) -> HighlightBlock:
    """The highlight entries one candidate earned, and whether the noise swamped any."""
    highlights = select_highlights(metrics, candidate_index)
    if not highlights:
        return HighlightBlock(entries=(), unstable=False)

    qualify = spans_many_kinds(metrics)
    labels = [highlight_label(highlight, qualify=qualify) for highlight in highlights]
    name_width = max(len(label) for label in labels) + _HIGHLIGHT_NAME_GUTTER

    entries: list[str] = []
    for highlight, label in zip(highlights, labels, strict=True):
        verdict = highlight.candidate.verdict
        if verdict is None:  # pragma: no cover - select_highlights only keeps judged slices
            msg = f"highlight {highlight.name!r} carries no verdict"
            raise ValueError(msg)
        shown = display_class(verdict)
        style = VERDICT_STYLES[shown]
        delta = format_verdict_delta(verdict)
        evidence = format_evidence(
            verdict, highlight.metric.meta.unit, highlight.metric.baseline_median
        )

        label_field = f"{escape(label)}{' ' * (name_width - len(label))}"
        delta_field = f"{' ' * max(0, _HIGHLIGHT_DELTA_WIDTH - len(delta))}{markup(delta, style)}"
        suffix = "" if evidence == "" else f"  {markup(evidence, 'dim')}"
        entries.append(f"  {markup(get_glyph(shown), style)} {label_field}{delta_field}{suffix}")

    return HighlightBlock(entries=tuple(entries), unstable=has_unstable_highlight(highlights))


def _futility_line() -> str:
    """The futility note, indented to sit under the entries it qualifies."""
    return f"  {markup(UNSTABLE_FUTILITY_NOTE, 'dim')}"


def _format_threshold(pct: float) -> str:
    """A ``--fail-on`` threshold as it was written, dropping a trailing ``.0``."""
    return f"{pct:g}"


def _gate_trip_lines(
    candidate: CandidateComparison,
    conditions: Sequence[FailOnCondition],
) -> list[str]:
    """The gate-trip lines for a candidate whose gated geomean cleared a ``--fail-on`` threshold.

    Only the geomean conditions gate here; the regressed condition contributes no
    line. A kind with no gated geomean, or one aggregating nothing, never trips —
    an informational kind cannot fail a gate it does not stand behind.
    """
    thresholds = [condition.pct for condition in conditions if isinstance(condition, GeomeanFailOn)]
    style = VERDICT_STYLES["regressed"]

    lines: list[str] = []
    for kind in candidate.kinds:
        geomean = kind.gated_geomean
        if geomean is None or geomean.n == 0:
            continue
        delta = format_delta(Effect(value=geomean.value, unit="percent"))
        lines.extend(
            f"  {markup(_GATE_TRIP_GLYPH, style)} {escape(kind.kind)} "
            f"{GATED_GEOMEAN_LABEL} {markup(delta, style)} "
            f"exceeded --fail-on geomean:{_format_threshold(pct)}"
            for pct in thresholds
            if geomean.value >= pct
        )
    return lines


def _highlight_section(blocks: Sequence[HighlightBlock]) -> list[str]:
    """The highlights block: a heading, each candidate's entries, and the futility note.

    A block with a label heads its entries with the bold label and indents them
    under it; a block with no label lists its entries directly. The whole block is
    dropped when no candidate had anything to highlight.
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
        lines.append(_futility_line())
    return lines


def _render_highlights(
    metrics: MetricComparisons,
    candidate_index: int,
    gate_trips: Sequence[str],
) -> list[str]:
    """The highlights block for a single-candidate report, gate trips folded in."""
    block = _highlight_entries(metrics, candidate_index)
    combined = HighlightBlock(
        entries=(*block.entries, *gate_trips),
        unstable=block.unstable,
    )
    return _highlight_section([combined])


def _render_candidate_highlights(
    result: ComparisonResult,
    conditions: Sequence[FailOnCondition],
) -> list[str]:
    """The highlights block for a multi-candidate report: one subsection per candidate."""
    blocks: list[HighlightBlock] = []
    for index, candidate in enumerate(result.candidates):
        block = _highlight_entries(result.metrics, index)
        blocks.append(
            HighlightBlock(
                entries=(*block.entries, *_gate_trip_lines(candidate, conditions)),
                unstable=block.unstable,
                label=candidate.label,
            )
        )
    return _highlight_section(blocks)


# ---------------------------------------------------------------------------
# Footers
# ---------------------------------------------------------------------------


def _render_method_footer(result: ComparisonResult, *, verbose: bool) -> list[str]:
    """The verbose method lines naming how each verdict was decided, and the samples hint."""
    return footer_lines(
        result.metrics,
        verbose=verbose,
        format_hint=lambda hint: f"{format_hint_label()} {escape(hint)}",
        samples=result.samples,
    )


def _to_single_line(text: str) -> str:
    """Collapse a git diagnostic onto one line.

    git routinely emits several lines for one failure — a ``warning:`` line before
    the ``fatal:`` line, plus indented continuations — so the runs of whitespace
    fold to a single space.
    """
    return re.sub(r"\s+", " ", text).strip()


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
            footer; ``options.fail_on`` names the gate conditions.

    Returns:
        The rendered report.
    """
    color = options.color
    display = with_display_labels(result)
    conditions = options.fail_on or ()

    if options.header is not None:
        lines = [options.header]
    else:
        lines = [_render_line(_compare_header(display), color=color)]

    if len(display.candidates) > 1:
        lines.extend(render_comparison_table(display, color=color))
        lines.append("")
        lines.extend(_render_block(_render_summaries(display), color=color))
        highlights = _render_candidate_highlights(display, conditions)
        if highlights:
            lines.append("")
            lines.extend(_render_block(highlights, color=color))
    elif len(display.candidates) == 1:
        candidate = display.candidates[0]
        lines.extend(render_table(display, candidate, 0, color=color))
        lines.append("")
        lines.extend(_render_block([_render_summary(display.metrics, 0)], color=color))
        highlights = _render_highlights(display.metrics, 0, _gate_trip_lines(candidate, conditions))
        if highlights:
            lines.append("")
            lines.extend(_render_block(highlights, color=color))

    footer = [
        *_render_method_footer(display, verbose=bool(options.verbose)),
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
    header = _join_header_parts(
        [
            markup("gymrat measure", "bold"),
            markup(label, VARIANT_NAME_STYLE),
            escape(pluralize(result.samples, "sample")),
            f"adapter: {escape(result.adapter)}",
        ]
    )

    lines = [_render_line(header, color=color)]
    lines.extend(render_measure_table(result, label, color=color))

    footer = _render_worktree_footer(result)
    if footer:
        lines.append("")
        lines.extend(_render_block(footer, color=color))

    return "\n".join(lines)
