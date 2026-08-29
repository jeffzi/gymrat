"""Verdict engine: pairing, delta computation, and per-metric method dispatch."""

from gymrat.verdict.aggregate import (
    GroupAggregate,
    KindAggregate,
    compute_kind_aggregates,
    infer_group,
)
from gymrat.verdict.engine import compute_verdicts
from gymrat.verdict.geomean import compute_geomean

__all__ = [
    "GroupAggregate",
    "KindAggregate",
    "compute_geomean",
    "compute_kind_aggregates",
    "compute_verdicts",
    "infer_group",
]
