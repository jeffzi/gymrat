"""Highlight selection: picking the metrics worth calling out for one candidate."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from rich.markup import escape

from gymrat.metric_name import format_inline, parse
from gymrat.report.display import DisplayClass, display_class, shown_class
from gymrat.report.style import SCOPE_SEPARATOR
from gymrat.report.types import candidate_at as _candidate_at

if TYPE_CHECKING:
    from collections.abc import Sequence

    from gymrat.model import MetricVerdict
    from gymrat.report.types import CandidateMetric, MetricComparison, MetricComparisons


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
    return format_inline(parse(highlight.name), color=True)


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
