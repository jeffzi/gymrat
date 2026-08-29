"""Fail-on gate evaluation: which conditions trip a non-zero exit code.

Only gating metrics participate — informational verdicts never trip an
exit-code gate. Conditions are OR-ed: any one that trips fails the run.
"""

import sys

from gymrat.cli.shared import write_and_flush
from gymrat.model import GeomeanResult
from gymrat.report.format import count_verdicts
from gymrat.report.types import (
    CandidateComparison,
    ComparisonResult,
    FailOnCondition,
    GeomeanFailOn,
    MetricComparisons,
    RegressedFailOn,
)


def _gating_metrics(metrics: MetricComparisons) -> MetricComparisons:
    """The gating subset of ``metrics`` — the only metrics a gate may judge."""
    return {name: metric for name, metric in metrics.items() if metric.meta.gating}


def _gated_geomeans_of(candidate: CandidateComparison) -> list[GeomeanResult]:
    """The gated geomean of every kind that gates, one entry per such kind."""
    return [kind.gated_geomean for kind in candidate.kinds if kind.gated_geomean is not None]


def should_fail_gate(conditions: tuple[FailOnCondition, ...], result: ComparisonResult) -> bool:
    """Return ``True`` when any condition trips — meaning the process should exit non-zero."""
    if not conditions:
        return False

    gating = _gating_metrics(result.metrics)

    for condition in conditions:
        if isinstance(condition, RegressedFailOn) and any(
            count_verdicts(gating, index).regressed > 0 for index in range(len(result.candidates))
        ):
            return True
        if isinstance(condition, GeomeanFailOn) and any(
            geomean.n > 0 and geomean.value >= condition.pct
            for candidate in result.candidates
            for geomean in _gated_geomeans_of(candidate)
        ):
            return True

    return False


def warn_empty_geomean_gates(
    conditions: tuple[FailOnCondition, ...], result: ComparisonResult
) -> None:
    """Warn once per candidate whose geomean gate had nothing stable to judge.

    Runs only when a geomean condition is present; such a candidate never trips
    the gate, so the warning is how the user learns the gate was inert for it.
    """
    if not any(isinstance(condition, GeomeanFailOn) for condition in conditions):
        return

    for candidate in result.candidates:
        if all(geomean.n == 0 for geomean in _gated_geomeans_of(candidate)):
            write_and_flush(
                sys.stderr,
                f'warning: geomean gate for "{candidate.label}" '
                "had no stable gating metrics to measure\n",
            )
