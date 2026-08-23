"""Verdict engine: pairing, delta computation, and per-metric method dispatch."""

from gymrat_py.verdict.engine import compute_verdicts
from gymrat_py.verdict.geomean import compute_geomean

__all__ = [
    "compute_geomean",
    "compute_verdicts",
]
