"""Core model value types and pairing logic for benchmark observations.

An :class:`Observations` value holds, per pairing-axis key, one or more repeats — each repeat a
mapping of metric name to value. The repeat axis is structurally distinct from the pairing axis: a
key maps to a *sequence* of repeat-mappings, not a single mapping.

:func:`pair_metric` aligns two containers on the keys they share, for a single metric, dropping any
shared key where either side is missing that metric.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, Self

__all__ = [
    "BAND_FLOORS",
    "DEFAULT_UNSTABLE_NOISE_PCT",
    "NOISE_FLOOR_PCT",
    "NOISE_K",
    "PERMUTATION_FLOORS",
    "ApproximateVerdict",
    "BandVerdict",
    "Direction",
    "Effect",
    "EffectUnit",
    "ExactVerdict",
    "Exclusion",
    "ExclusionReason",
    "GeomeanResult",
    "MethodFloors",
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
    "VerdictMethod",
    "pair_metric",
]

# ---------------------------------------------------------------------------
# Metric metadata
# ---------------------------------------------------------------------------

Direction = Literal["lower", "higher"]
"""Whether a lower or higher raw value is the better outcome for a metric."""

MetricUnit = Literal["ns", "bytes"]
"""Physical unit a metric is measured in."""


@dataclass(frozen=True, slots=True)
class MetricMeta:
    """Static metadata describing how a single metric is interpreted.

    Attributes:
        direction: Whether lower or higher is better.
        gating: Whether the metric can gate (fail) a comparison.
        exact: Whether the metric is compared exactly rather than statistically.
        unit: The metric's physical unit, or ``None`` when unitless.
    """

    direction: Direction
    gating: bool
    exact: bool
    unit: MetricUnit | None


@dataclass(frozen=True, slots=True)
class ResolvedMetricMeta(MetricMeta):
    """A :class:`MetricMeta` resolved against a concrete metric, with display metadata.

    Extends the static metadata with the metric's classification and label. Being a subclass, a
    ``ResolvedMetricMeta`` is usable anywhere a :class:`MetricMeta` is.

    Attributes:
        kind: The metric's classification (e.g. ``"time"``, ``"memory"``).
        short_name: The metric's display label.
    """

    kind: str
    short_name: str


# ---------------------------------------------------------------------------
# Effect size
# ---------------------------------------------------------------------------

EffectUnit = Literal["percent"]
"""Unit an :class:`Effect` is expressed in.

Today only ``"percent"`` is admissible; the alias is shaped as a ``Literal`` union so a
percentage-point (``"pp"``) member can be added later without touching call sites.
"""


@dataclass(frozen=True, slots=True)
class Effect:
    """An observed effect size, immutable and compared by value.

    Attributes:
        value: The magnitude of the effect.
        unit: The unit ``value`` is expressed in.
    """

    value: float
    unit: EffectUnit


# ---------------------------------------------------------------------------
# Verdict methods and noise model
# ---------------------------------------------------------------------------

VerdictMethod = Literal["permutation", "band", "exact"]
"""Tag identifying which statistical method produced a verdict."""


@dataclass(frozen=True, slots=True)
class MethodFloors:
    """Statistical floors for one method, carried as data.

    Attributes:
        method: The method these floors apply to.
        min_n: Minimum usable sample size the method requires.
        p_threshold: Significance threshold, or ``None`` when the method has none.
    """

    method: VerdictMethod
    min_n: int
    p_threshold: float | None


PERMUTATION_FLOORS = MethodFloors(method="permutation", min_n=6, p_threshold=0.05)
"""Floors for the sign-flip permutation method."""

BAND_FLOORS = MethodFloors(method="band", min_n=2, p_threshold=None)
"""Floors for the band method, which has no significance threshold."""

NOISE_K = 1.5
"""Multiplier applied to the noise floor when deriving the instability band."""

NOISE_FLOOR_PCT = 0.5
"""Minimum noise level, as a percentage, below which measurements are treated as floor noise."""

DEFAULT_UNSTABLE_NOISE_PCT = 200
"""Noise percentage assigned to a metric flagged unstable when no measured value applies."""


# ---------------------------------------------------------------------------
# Verdict records
# ---------------------------------------------------------------------------

Verdict = Literal["improved", "regressed", "no-signal"]
"""Outcome of a comparison that cannot be flagged unstable."""

ApproximateVerdict = Verdict | Literal["unstable"]
"""Outcome of an approximate comparison, which may additionally be ``"unstable"``."""


@dataclass(frozen=True, slots=True)
class PermutationVerdict:
    """Verdict from the sign-flip permutation method.

    Attributes:
        method: Discriminant tag, always ``"permutation"``.
        verdict: The approximate outcome.
        p: The sign-flip permutation test p-value.
        noise_pct: Estimated noise as a percentage.
        noise_abs: Estimated noise in absolute units.
        delta: The unit-tagged effect size of the comparison.
        n: Number of paired samples.
    """

    method: Literal["permutation"]
    verdict: ApproximateVerdict
    p: float
    noise_pct: float
    noise_abs: float
    delta: Effect
    n: int


@dataclass(frozen=True, slots=True)
class BandVerdict:
    """Verdict from the band method.

    Attributes:
        method: Discriminant tag, always ``"band"``.
        verdict: The approximate outcome.
        usable_n: Number of usable samples.
        noise_pct: Estimated noise as a percentage.
        noise_abs: Estimated noise in absolute units.
        delta: The unit-tagged effect size of the comparison.
        n: Number of paired samples.
    """

    method: Literal["band"]
    verdict: ApproximateVerdict
    usable_n: int
    noise_pct: float
    noise_abs: float
    delta: Effect
    n: int


@dataclass(frozen=True, slots=True)
class ExactVerdict:
    """Verdict from the exact method, which is never unstable and carries no noise.

    Attributes:
        method: Discriminant tag, always ``"exact"``.
        verdict: The (non-approximate) outcome.
        delta: The unit-tagged effect size of the comparison.
        n: Number of samples.
    """

    method: Literal["exact"]
    verdict: Verdict
    delta: Effect
    n: int


MetricVerdict = PermutationVerdict | BandVerdict | ExactVerdict
"""Union of per-method verdict records, discriminated on ``method``."""


# ---------------------------------------------------------------------------
# Aggregate exclusions and geomean result
# ---------------------------------------------------------------------------

ExclusionReason = Literal["no-verdict", "unstable", "undefined-ratio", "infinite-rho"]
"""Why a metric was excluded from an aggregate."""


@dataclass(frozen=True, slots=True)
class Exclusion:
    """A metric excluded from an aggregate, paired with the reason.

    Attributes:
        metric: The excluded metric's name.
        reason: Why it was excluded.
    """

    metric: str
    reason: ExclusionReason


@dataclass(frozen=True, slots=True)
class GeomeanResult:
    """Geometric-mean aggregate over many metrics.

    Attributes:
        value: The geometric-mean value.
        n: Number of metrics contributing to ``value``.
        band: The instability band around ``value``.
        excluded: Metrics left out of the aggregate, with reasons.
    """

    value: float
    n: int
    band: float
    excluded: tuple[Exclusion, ...]


# ---------------------------------------------------------------------------
# Observations and pairing
# ---------------------------------------------------------------------------

type PairingKey = int
"""Pairing-axis key: a round index."""

type Repeat = Mapping[str, float]
"""One repeat: a mapping of metric name to value."""


@dataclass(frozen=True, slots=True)
class PairResult:
    """Aligned metric values for two containers, plus the count of shared keys that were dropped.

    Attributes:
        left: Values from the left container, in shared-key order.
        right: Values from the right container, aligned one-to-one with ``left``.
        dropped: Count of shared keys where exactly one side carried the metric. A shared key where
            neither side has the metric is not a drop; a key present in only one container is not
            shared and is not counted.
    """

    left: tuple[float, ...]
    right: tuple[float, ...]
    dropped: int


@dataclass(frozen=True, slots=True)
class Observations:
    """A frozen wrapper over an ordered mapping from pairing-axis key to a tuple of repeats."""

    by_key: dict[PairingKey, tuple[Repeat, ...]]

    @classmethod
    def from_rounds(cls, samples: Sequence[Repeat]) -> Self:
        """Build a container from per-round samples, preserving round order.

        Args:
            samples: One repeat per round, in round order.

        Returns:
            A container keyed by 0-based round index, one repeat per round.
        """
        return cls(by_key={index: (sample,) for index, sample in enumerate(samples)})


def _require_single_repeat(observations: Observations) -> None:
    """Reject a container that carries more than one repeat for any key.

    A multi-repeat container is constructible, but pairing over one is not defined — surface it
    rather than silently taking the first repeat.

    Args:
        observations: The container to check.

    Raises:
        ValueError: When any key carries more than one repeat.
    """
    for key, repeats in observations.by_key.items():
        if len(repeats) != 1:
            message = (
                f"pair_metric requires single-repeat observations; "
                f"key {key!r} has {len(repeats)} repeats"
            )
            raise ValueError(message)


def pair_metric(
    left: Observations,
    right: Observations,
    metric: str,
) -> PairResult:
    """Align two containers on their shared keys, in order, for a single metric.

    Iterates the keys ``left`` and ``right`` share, in ``left``'s order. Pairing over a multi-repeat
    container is not defined, so both must be single-repeat.

    Args:
        left: The baseline observation container.
        right: The candidate observation container.
        metric: The metric name to pair across both containers.

    Returns:
        The paired samples and drop count for the requested metric. A shared key where either
        side's repeat lacks ``metric`` is left out of both sequences, so they are always equal
        length; two empty sequences — the metric absent from every shared key — are the caller's
        skip-metric signal. ``dropped`` counts the shared keys where exactly one side carried the
        metric; a key where neither side has it is not a drop.

    Raises:
        ValueError: If any key in either container carries more than one repeat.
    """
    _require_single_repeat(left)
    _require_single_repeat(right)

    left_values: list[float] = []
    right_values: list[float] = []
    dropped = 0
    for key, left_repeats in left.by_key.items():
        right_repeats = right.by_key.get(key)
        if right_repeats is None:
            continue
        left_repeat = left_repeats[0]
        right_repeat = right_repeats[0]
        in_left = metric in left_repeat
        in_right = metric in right_repeat
        if in_left and in_right:
            left_values.append(left_repeat[metric])
            right_values.append(right_repeat[metric])
        elif in_left != in_right:
            dropped += 1
    return PairResult(left=tuple(left_values), right=tuple(right_values), dropped=dropped)
