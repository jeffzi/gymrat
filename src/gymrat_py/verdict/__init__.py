"""Verdict engine: pairing, delta computation, and per-metric method dispatch."""

from gymrat_py.verdict.aggregate import (
    GroupAggregate,
    KindAggregate,
    compute_kind_aggregates,
    infer_group,
)
from gymrat_py.verdict.engine import compute_verdicts
from gymrat_py.verdict.geomean import compute_geomean

__all__ = [
    "GroupAggregate",
    "KindAggregate",
    "compute_geomean",
    "compute_kind_aggregates",
    "compute_verdicts",
    "infer_group",
]
