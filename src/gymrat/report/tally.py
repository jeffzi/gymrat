"""Verdict tallies and summary-line rendering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from rich.markup import escape

from gymrat.report.display import (
    VERDICT_GLOSSES,
    DisplayClass,
    display_class,
    get_glyph,
)
from gymrat.report.style import VERDICT_STYLES, markup

if TYPE_CHECKING:
    from collections.abc import Callable

    from gymrat.model import MetricVerdict
    from gymrat.report.types import CandidateMetric, MetricComparison, MetricComparisons


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
