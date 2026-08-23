"""JSON report document builders for the compare and measure commands.

Both builders assemble a plain nested structure keyed by the parity JSON surface
(camelCase string literals, distinct from the snake_case dataclass fields) and
serialize it with a two-space indent. Neither takes presentation options, so the
output never carries ANSI, whatever the ambient environment forces.

``json.dumps`` emits invalid ``NaN``/``Infinity`` literals by default, whereas
the contract is a JSON ``null`` for any non-finite float (matching JavaScript's
``JSON.stringify``). A pre-pass replaces every non-finite float with ``None``
before the dump, and ``allow_nan=False`` guards against any that slip through.
"""

from __future__ import annotations

import json
import math
from typing import TYPE_CHECKING

from gymrat_py.model import BandVerdict, SignedRankVerdict
from gymrat_py.report.format import count_verdicts
from gymrat_py.report.types import CandidateMetric
from gymrat_py.verdict import infer_group

if TYPE_CHECKING:
    from gymrat_py.model import GeomeanResult, MetricVerdict
    from gymrat_py.report.format import VerdictCounts
    from gymrat_py.report.types import (
        CandidateComparison,
        ComparisonResult,
        MeasurementResult,
        MetricComparison,
        MetricMeasurement,
        WorktreeCleanupOutcome,
    )
    from gymrat_py.verdict import KindAggregate

_COMPARE_SCHEMA_VERSION = 2
_MEASURE_SCHEMA_VERSION = 1


def render_json(result: ComparisonResult) -> str:
    """Serialize a comparison result to the compare JSON document.

    Args:
        result: The comparison to render.

    Returns:
        The document as a two-space-indented JSON string.
    """
    document: dict[str, object] = {
        "schemaVersion": _COMPARE_SCHEMA_VERSION,
        "baseline": result.baseline_label,
        "candidates": [candidate.label for candidate in result.candidates],
        "samples": result.samples,
        "adapter": result.adapter,
        "metrics": {
            name: _serialize_metric(metric, result.candidates)
            for name, metric in result.metrics.items()
        },
        "perCandidate": _serialize_per_candidate(result),
        "worktrees": _serialize_worktrees(result),
    }
    return _dump(document)


def render_measure_json(result: MeasurementResult) -> str:
    """Serialize a measurement result to the measure JSON document.

    Args:
        result: The single-target measurement to render.

    Returns:
        The document as a two-space-indented JSON string.
    """
    document: dict[str, object] = {
        "schemaVersion": _MEASURE_SCHEMA_VERSION,
        "label": result.label,
        "samples": result.samples,
        "adapter": result.adapter,
        "metrics": {
            name: _serialize_measure_metric(metric) for name, metric in result.metrics.items()
        },
        "worktrees": _serialize_worktrees(result),
    }
    return _dump(document)


def _serialize_metric(
    metric: MetricComparison,
    candidates: tuple[CandidateComparison, ...],
) -> dict[str, object]:
    """One metric's meta, baseline, and positional per-candidate rows."""
    empty = CandidateMetric()
    rows = [
        _candidate_row(
            candidate.label,
            metric.candidates[index] if index < len(metric.candidates) else empty,
        )
        for index, candidate in enumerate(candidates)
    ]
    return {
        "unit": metric.meta.unit,
        "direction": metric.meta.direction,
        "gating": metric.meta.gating,
        "kind": metric.meta.kind,
        "group": infer_group(metric.meta.short_name),
        "baseline": {"median": metric.baseline_median, "spreadPct": metric.baseline_spread},
        "candidates": rows,
    }


def _candidate_row(label: str, candidate: CandidateMetric) -> dict[str, object]:
    """One candidate's measured value and verdict for a metric."""
    return {
        "label": label,
        "median": candidate.median,
        "spreadPct": candidate.spread,
        **_verdict_fields(candidate.verdict),
    }


def _verdict_fields(verdict: MetricVerdict | None) -> dict[str, object]:
    """The verdict-derived fields, padded per method.

    Signed-rank carries a p-value and no band; band carries a noise figure in
    both ``noisePct`` and ``band``; exact carries none. A candidate with no
    verdict leaves every field null while its measurements survive on the row.
    """
    if verdict is None:
        return {
            "verdict": None,
            "method": None,
            "delta": None,
            "noisePct": None,
            "p": None,
            "band": None,
        }
    noise_pct: float | None = None
    p: float | None = None
    band: float | None = None
    if isinstance(verdict, SignedRankVerdict):
        noise_pct = verdict.noise_pct
        p = verdict.p
    elif isinstance(verdict, BandVerdict):
        noise_pct = verdict.noise_pct
        band = verdict.noise_pct
    return {
        "verdict": verdict.verdict,
        "method": verdict.method,
        "delta": verdict.delta.value,
        "noisePct": noise_pct,
        "p": p,
        "band": band,
    }


def _serialize_per_candidate(result: ComparisonResult) -> list[dict[str, object]]:
    """Per-candidate kind aggregates and verdict tallies, in candidate order."""
    return [
        {
            "label": candidate.label,
            "kinds": [_serialize_kind(kind) for kind in candidate.kinds],
            "verdictCounts": _serialize_counts(count_verdicts(result.metrics, index)),
        }
        for index, candidate in enumerate(result.candidates)
    ]


def _serialize_kind(kind: KindAggregate) -> dict[str, object]:
    """One kind's section geomean, group geomeans, and gated geomean."""
    gated = kind.gated_geomean
    return {
        "kind": kind.kind,
        "hasGating": gated is not None,
        "geomean": _serialize_geomean(kind.geomean),
        "groups": [
            {"group": group.group, "geomean": _serialize_geomean(group.geomean)}
            for group in kind.groups
        ],
        "gatedGeomean": _serialize_geomean(gated) if gated is not None else None,
    }


def _serialize_geomean(geomean: GeomeanResult) -> dict[str, object]:
    """A geomean's value, contributing count, exclusions, and band."""
    return {
        "value": geomean.value,
        "n": geomean.n,
        "excluded": [
            {"metric": exclusion.metric, "reason": exclusion.reason}
            for exclusion in geomean.excluded
        ],
        "band": geomean.band,
    }


def _serialize_counts(counts: VerdictCounts) -> dict[str, int]:
    """The per-candidate verdict tally under its camelCase keys."""
    return {
        "improved": counts.improved,
        "regressed": counts.regressed,
        "unstable": counts.unstable,
        "noSignal": counts.no_signal,
    }


def _serialize_measure_metric(metric: MetricMeasurement) -> dict[str, object]:
    """One metric's measurement beside the metadata behind it."""
    return {
        "median": metric.median,
        "spreadPct": metric.spread,
        "unit": metric.meta.unit,
        "direction": metric.meta.direction,
        "gating": metric.meta.gating,
        "exact": metric.meta.exact,
        "kind": metric.meta.kind,
        "group": infer_group(metric.meta.short_name),
    }


def _serialize_worktrees(result: WorktreeCleanupOutcome) -> dict[str, object]:
    """The cleanup outcome: count removed, failures, and any prune error."""
    return {
        "removed": result.worktrees_removed,
        "leftBehind": [
            {"path": failure.dir, "reason": failure.error}
            for failure in result.worktrees_left_behind
        ],
        "pruneError": result.worktree_prune_error,
    }


def _dump(document: dict[str, object]) -> str:
    """Serialize with a two-space indent after nulling every non-finite float."""
    return json.dumps(_null_non_finite(document), indent=2, allow_nan=False)


def _null_non_finite(value: object) -> object:
    """Replace every non-finite float in a nested JSON structure with ``None``."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _null_non_finite(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_null_non_finite(item) for item in value]
    return value
