"""Core model value types: pure, immutable data with no logic."""

from gymrat_py.model.aggregate import (
    Aggregate,
    Exclusion,
    ExclusionReason,
    GeomeanResult,
)
from gymrat_py.model.effect import Effect, EffectUnit
from gymrat_py.model.methods import (
    BAND_DESCRIPTOR,
    DEFAULT_UNSTABLE_NOISE_PCT,
    EXACT_DESCRIPTOR,
    NOISE_FLOOR_PCT,
    NOISE_K,
    PERMUTATION_DESCRIPTOR,
    Method,
    MethodDescriptor,
)
from gymrat_py.model.metrics import (
    Direction,
    MetricMeta,
    MetricUnit,
    ResolvedMetricMeta,
)
from gymrat_py.model.observations import (
    DROP_UNPAIRED,
    Observations,
    PairingKey,
    PairResult,
    Repeat,
    UnpairedPolicy,
    pair_metric,
)
from gymrat_py.model.verdicts import (
    ApproximateVerdict,
    BandVerdict,
    ExactVerdict,
    MetricVerdict,
    PermutationVerdict,
    Verdict,
)

__all__ = [
    "BAND_DESCRIPTOR",
    "DEFAULT_UNSTABLE_NOISE_PCT",
    "DROP_UNPAIRED",
    "EXACT_DESCRIPTOR",
    "NOISE_FLOOR_PCT",
    "NOISE_K",
    "PERMUTATION_DESCRIPTOR",
    "Aggregate",
    "ApproximateVerdict",
    "BandVerdict",
    "Direction",
    "Effect",
    "EffectUnit",
    "ExactVerdict",
    "Exclusion",
    "ExclusionReason",
    "GeomeanResult",
    "Method",
    "MethodDescriptor",
    "MetricMeta",
    "MetricUnit",
    "MetricVerdict",
    "Observations",
    "PairResult",
    "PairingKey",
    "PermutationVerdict",
    "Repeat",
    "ResolvedMetricMeta",
    "UnpairedPolicy",
    "Verdict",
    "pair_metric",
]
