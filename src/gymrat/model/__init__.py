"""Core model value types: pure, immutable data with no logic."""

from gymrat.model.aggregate import (
    Exclusion,
    ExclusionReason,
    GeomeanResult,
)
from gymrat.model.effect import Effect, EffectUnit
from gymrat.model.methods import (
    BAND_DESCRIPTOR,
    DEFAULT_UNSTABLE_NOISE_PCT,
    NOISE_FLOOR_PCT,
    NOISE_K,
    PERMUTATION_DESCRIPTOR,
    Method,
    MethodDescriptor,
)
from gymrat.model.metrics import (
    Direction,
    MetricMeta,
    MetricUnit,
    ResolvedMetricMeta,
)
from gymrat.model.observations import (
    Observations,
    PairingKey,
    PairResult,
    Repeat,
    pair_metric,
)
from gymrat.model.verdicts import (
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
    "NOISE_FLOOR_PCT",
    "NOISE_K",
    "PERMUTATION_DESCRIPTOR",
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
    "Verdict",
    "pair_metric",
]
