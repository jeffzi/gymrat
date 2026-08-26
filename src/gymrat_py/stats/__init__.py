"""Pure descriptive-statistics helpers, free of I/O, config, and model types."""

from gymrat_py.model.metrics import Direction
from gymrat_py.stats.descriptive import (
    GeomeanCombination,
    RatioExclusion,
    RatioOutcome,
    combine_geomean,
    compute_half_range,
    compute_median,
    normalize_ratio,
)
from gymrat_py.stats.permutation import (
    PERMUTATION_SEED,
    RESAMPLE_BUDGET,
    sign_flip_permutation_test,
)
from gymrat_py.stats.results import SignificanceResult

__all__ = [
    "PERMUTATION_SEED",
    "RESAMPLE_BUDGET",
    "Direction",
    "GeomeanCombination",
    "RatioExclusion",
    "RatioOutcome",
    "SignificanceResult",
    "combine_geomean",
    "compute_half_range",
    "compute_median",
    "normalize_ratio",
    "sign_flip_permutation_test",
]
