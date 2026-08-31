"""Rendering input contracts for the report package.

These frozen dataclasses are the shape a renderer draws from. A comparison run
produces a :class:`ComparisonResult`; a single-target run a
:class:`MeasurementResult`. Both carry the worktree cleanup outcome the run
managed and the ``kinds`` section of the config it resolved, so a renderer can
name where a gating decision was made rather than guess.

Field names are snake_case here; the camelCase JSON keys the report serializes
to are a serializer's concern, not this contract's.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from gymrat.config import KindEntry
    from gymrat.model import MetricVerdict, ResolvedMetricMeta
    from gymrat.targets import WorktreeRemovalFailure
    from gymrat.verdict import KindAggregate


@dataclass(frozen=True, slots=True)
class CandidateMetric:
    """One candidate's side of a metric, and the verdict it earned against the baseline.

    Attributes:
        median: The candidate's median measurement, or ``None`` when the
            candidate never reported this metric.
        spread: The half-range around the median, or ``None`` when a lone sample
            left no run-to-run jitter to report.
        verdict: The verdict the candidate earned against the baseline, or
            ``None`` when there was nothing to compare.
    """

    median: float | None = None
    spread: float | None = None
    verdict: MetricVerdict | None = None


@dataclass(frozen=True, slots=True)
class MetricComparison:
    """One metric across the run: the baseline once, then a candidate entry per candidate.

    ``candidates`` is positional — entry *i* belongs to
    ``ComparisonResult.candidates[i]``.

    Attributes:
        baseline_median: The baseline's median measurement, or ``None`` when it
            reported none.
        baseline_spread: The half-range around the baseline median, or ``None``.
        candidates: One entry per candidate, in candidate order.
        meta: The metadata that shaped this metric's readings.
    """

    baseline_median: float | None
    baseline_spread: float | None
    candidates: tuple[CandidateMetric, ...]
    meta: ResolvedMetricMeta


type MetricComparisons = dict[str, MetricComparison]
"""Every metric a comparison produced, keyed by metric name."""


@dataclass(frozen=True, slots=True)
class WorktreeCleanupOutcome:
    """Worktree cleanup outcome shared by every run result.

    Comparison and measurement runs manage the same worktrees the same way.

    Attributes:
        worktrees_removed: How many worktrees cleanup removed.
        worktrees_left_behind: Worktrees cleanup could not remove, each with the
            reason git gave.
        worktree_prune_error: The reason the ``git worktree prune`` sweep failed,
            or ``None`` when it succeeded.
    """

    worktrees_removed: int
    worktrees_left_behind: tuple[WorktreeRemovalFailure, ...]
    worktree_prune_error: str | None


@dataclass(frozen=True, slots=True)
class CandidateComparison:
    """One candidate's run-level results, judged against the shared baseline.

    Attributes:
        label: The candidate's display label.
        kinds: One entry per kind the run reported, in first-appearance order.
    """

    label: str
    kinds: tuple[KindAggregate, ...]


@dataclass(frozen=True, slots=True)
class ComparisonResult(WorktreeCleanupOutcome):
    """Everything a renderer needs to draw a comparison — the rendering input contract.

    The shape is a star, not a mesh: every candidate is compared with the
    baseline and never with another candidate. Those comparisons all reuse the
    same baseline samples, so the candidate verdicts of one metric are
    statistically correlated. Read a candidate's verdict as evidence about that
    candidate alone; the difference between two candidates' deltas is not itself
    a tested quantity.

    Attributes:
        baseline_label: Label of the target every candidate is judged against.
        candidates: The candidates, in command-line order.
        samples: How many samples the run collected.
        adapter: The adapter that produced the measurements.
        metrics: Every metric the comparison produced, keyed by name.
        config_kinds: The ``kinds`` section of the resolved config, when it had
            one, so the report can name the config line behind a gating decision.
    """

    baseline_label: str
    candidates: tuple[CandidateComparison, ...]
    samples: int
    adapter: str
    metrics: MetricComparisons
    config_kinds: dict[str, KindEntry] | None = None


@dataclass(frozen=True, slots=True)
class MetricMeasurement:
    """One metric of a single-target run: what it measured, and how steady it was.

    ``spread`` is the same half-range figure a comparison prints beside a side's
    median, and is absent for the same reasons — a lone sample has no
    run-to-run jitter, and a zero median has no scale to be a percentage of.

    Attributes:
        median: The metric's median measurement, or ``None`` when none reported.
        spread: The half-range around the median, or ``None``.
        meta: The metadata that shaped the reading.
    """

    median: float | None
    spread: float | None
    meta: ResolvedMetricMeta


type MetricMeasurements = dict[str, MetricMeasurement]
"""Every metric a measurement produced, keyed by metric name."""


@dataclass(frozen=True, slots=True)
class MeasurementResult(WorktreeCleanupOutcome):
    """Everything a single-target run measured — the rendering input contract for a measurement.

    There is nothing to judge against, so no verdicts and no aggregates: a
    measurement states what the target reported and how steady it was.

    Attributes:
        label: The target's explicit label, or its ref name / directory base
            name.
        samples: How many samples the run collected.
        adapter: The adapter that produced the measurements.
        metrics: Every metric the measurement produced, keyed by name.
        rounds: What each round reported, in the order the rounds ran, so a
            reader can compute statistics the report never printed.
        config_kinds: The ``kinds`` section of the resolved config, when it had
            one.
    """

    label: str
    samples: int
    adapter: str
    metrics: MetricMeasurements
    rounds: tuple[dict[str, float], ...]
    config_kinds: dict[str, KindEntry] | None = None


@dataclass(frozen=True, slots=True)
class RegressedFailOn:
    """A ``--fail-on`` condition that trips on any regression."""

    kind: Literal["regressed"] = "regressed"


@dataclass(frozen=True, slots=True)
class GeomeanFailOn:
    """A ``--fail-on`` condition that trips when a geomean crosses a threshold.

    Attributes:
        pct: The percentage threshold the geomean must cross to trip the gate.
    """

    pct: float
    kind: Literal["geomean"] = "geomean"


def candidate_at(metric: MetricComparison, index: int) -> CandidateMetric | None:
    """The candidate slice at ``index``, or ``None`` when the metric has no such entry."""
    if 0 <= index < len(metric.candidates):
        return metric.candidates[index]
    return None


type FailOnCondition = RegressedFailOn | GeomeanFailOn
"""A parsed ``--fail-on`` condition: a regression check or a geomean threshold.

Shared by the gate that decides the exit code and the report that echoes which
gate a candidate tripped, so both read the conditions the user wrote.
"""


@dataclass(frozen=True, slots=True)
class ReportOptions:
    """What the human-readable renderers print beyond the report itself.

    The JSON renderer takes none of these: its consumers parse fields rather
    than read prose, so its output must not vary with a presentation flag.

    Attributes:
        verbose: Name the statistical method behind each verdict in the footer.
            Off by default — the glyphs and the summary line carry the reading.
        color: Force ANSI color on (``True``) or off (``False``) rather than
            detecting it. Left ``None``, the renderers defer to detection.
        fail_on: The ``--fail-on`` conditions the run is gated on, so the report
            can say which gate a candidate tripped. Display only.
        header: A run header to open the report with, printed verbatim, in place
            of the one the renderer would build for itself.
        command: The subcommand the report is printed under, so a hint suggesting
            a re-run names the invocation the reader would repeat. Defaults to
            the ``compare`` the renderer is named for; a command borrowing the
            renderer — ``iterate`` — names itself.
    """

    verbose: bool | None = None
    color: bool | None = None
    fail_on: tuple[FailOnCondition, ...] | None = None
    header: str | None = None
    command: str = "compare"
