"""The human-readable text reports: the compare report and the measure report.

This module owns only the run header and the table dispatch: one candidate draws
the single-candidate table, two or more the multi-candidate table. The verdict
summary, highlights, gate trips and footers that sit around a full comparison
report are added by a later task; a comparison report here is the header and the
comparison table alone, and a measure report the header and the measurement
table.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from rich.markup import escape
from rich.text import Text

from gymrat_py.report.format import pluralize
from gymrat_py.report.style import VARIANT_NAME_STYLE, render_lines, truncate_labels
from gymrat_py.report.table import markup
from gymrat_py.report.text_measure import render_measure_table
from gymrat_py.report.text_multi import render_comparison_table
from gymrat_py.report.text_single import render_table
from gymrat_py.report.types import ReportOptions

if TYPE_CHECKING:
    from gymrat_py.report.types import ComparisonResult, MeasurementResult

# A wide render width so the header line never soft-wraps.
_RENDER_WIDTH = 200

# The default presentation flags: detect color, no header override. Immutable, so
# one shared instance is safe as a default argument.
_DEFAULT_OPTIONS = ReportOptions()

# The `·` separator every report header joins its parts with, dimmed in color.
_HEADER_SEPARATOR = "·"


def _join_header_parts(parts: list[str]) -> str:
    """Join header parts with the dimmed ``·`` separator every report header shares."""
    return f" {markup(_HEADER_SEPARATOR, 'dim')} ".join(parts)


def _render_line(text: str, *, color: bool | None) -> str:
    """Resolve a markup line to text once, deferring wrapping so the header stays whole."""
    return render_lines(Text.from_markup(text), color=color, width=_RENDER_WIDTH)


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


def render_report(result: ComparisonResult, options: ReportOptions = _DEFAULT_OPTIONS) -> str:
    """Render a comparison report: the run header and the comparison table.

    One candidate draws the single-candidate table; two or more draw the
    multi-candidate table. The verdict summary, highlights, gate trips and footers
    arrive with a later task. ``options.header`` replaces the run header verbatim;
    ``options.color`` forces color on or off, or defers to the environment when
    ``None``.

    Args:
        result: The comparison to draw.
        options: The presentation flags.

    Returns:
        The rendered report.
    """
    color = options.color
    display = with_display_labels(result)

    if options.header is not None:
        lines = [options.header]
    else:
        lines = [_render_line(_compare_header(display), color=color)]

    if len(display.candidates) == 1:
        lines.extend(render_table(display, display.candidates[0], 0, color=color))
    elif len(display.candidates) > 1:
        lines.extend(render_comparison_table(display, color=color))

    return "\n".join(lines)


def render_measure_report(
    result: MeasurementResult,
    options: ReportOptions = _DEFAULT_OPTIONS,
) -> str:
    """Render a single-target measurement report: the run header and the measurement table.

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
    return "\n".join(lines)
