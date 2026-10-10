"""Core model value types and pairing logic for benchmark observations.

A side's observations are a sequence of repeats, one per round — each repeat a mapping of metric
name to value.

:func:`pair_metric` aligns two sides round by round, for a single metric, dropping any round where
either side is missing that metric.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from gymrat.session import schema

__all__ = [
    "DEFAULT_UNSTABLE_NOISE_PCT",
    "NOISE_FLOOR_PCT",
    "PERMUTATION_MIN_N",
    "PERMUTATION_P_THRESHOLD",
    "ApproximateVerdict",
    "BandVerdict",
    "Direction",
    "ExactVerdict",
    "Exclusion",
    "ExclusionReason",
    "GeomeanResult",
    "MetricMeta",
    "MetricUnit",
    "MetricVerdict",
    "PairResult",
    "PermutationVerdict",
    "Repeat",
    "ResolvedMetricMeta",
    "Verdict",
    "is_improvement",
    "pair_metric",
]

# ---------------------------------------------------------------------------
# Metric metadata
# ---------------------------------------------------------------------------

Direction = Literal["lower", "higher"]
"""Whether a lower or higher raw value is the better outcome for a metric."""


def is_improvement(delta: float, direction: Direction) -> bool:
    """Whether a percentage delta moved the way its direction calls an improvement.

    This is the single place the sign-of-improvement rule lives. A value of
    exactly zero never improves — at rest a figure moved in no direction to call
    good — and neither does ``NaN``.

    Args:
        delta: The percentage delta to judge.
        direction: Whether a lower or higher value is the better outcome.

    Returns:
        ``True`` when ``delta`` is strictly negative for ``"lower"`` or strictly
        positive for ``"higher"``.
    """
    return delta < 0 if direction == "lower" else delta > 0


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
# Verdict methods and noise model
# ---------------------------------------------------------------------------

PERMUTATION_MIN_N = 6
"""Minimum count of differing pairs the sign-flip permutation method requires."""

PERMUTATION_P_THRESHOLD = 0.05
"""Significance threshold a permutation p-value must fall below."""

NOISE_FLOOR_PCT = 0.5
"""Minimum noise level, as a percentage, below which measurements are treated as floor noise."""

DEFAULT_UNSTABLE_NOISE_PCT = 200
"""Noise band width, as a percentage, above which a verdict is forced unstable."""


# ---------------------------------------------------------------------------
# Verdict records
# ---------------------------------------------------------------------------

Verdict = schema.Outcome
"""Outcome of a comparison that cannot be flagged unstable.

The session log's vocabulary in :mod:`gymrat.session.schema` is the one home of
the verdict words; the in-memory verdicts reuse it so the two cannot drift.
"""

ApproximateVerdict = schema.Verdict
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
        delta: The percentage delta between the paired medians.
        n: Number of paired samples.
    """

    method: Literal["permutation"]
    verdict: ApproximateVerdict
    p: float
    noise_pct: float
    noise_abs: float
    delta: float
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
        delta: The percentage delta between the paired medians.
        n: Number of paired samples.
    """

    method: Literal["band"]
    verdict: ApproximateVerdict
    usable_n: int
    noise_pct: float
    noise_abs: float
    delta: float
    n: int


@dataclass(frozen=True, slots=True)
class ExactVerdict:
    """Verdict from the exact method, which is never unstable and carries no noise.

    Attributes:
        method: Discriminant tag, always ``"exact"``.
        verdict: The (non-approximate) outcome.
        delta: The percentage delta between the paired medians.
        n: Number of samples.
    """

    method: Literal["exact"]
    verdict: Verdict
    delta: float
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
# Repeats and pairing
# ---------------------------------------------------------------------------

type Repeat = Mapping[str, float]
"""One repeat: a mapping of metric name to value."""


@dataclass(frozen=True, slots=True)
class PairResult:
    """Aligned metric values for two sides, plus the count of shared rounds that were dropped.

    Attributes:
        left: Values from the left side, in round order.
        right: Values from the right side, aligned one-to-one with ``left``.
        dropped: Count of shared rounds where exactly one side carried the metric. A shared round
            where neither side has the metric is not a drop; a round only the longer side reached
            is not shared and is not counted.
    """

    left: tuple[float, ...]
    right: tuple[float, ...]
    dropped: int


def pair_metric(
    left: Sequence[Repeat],
    right: Sequence[Repeat],
    metric: str,
) -> PairResult:
    """Align two sides round by round, over the shorter of the two, for a single metric.

    Args:
        left: The baseline repeats, one per round.
        right: The candidate repeats, one per round.
        metric: The metric name to pair across both sides.

    Returns:
        The paired samples and drop count for the requested metric. A shared round where either
        side's repeat lacks ``metric`` is left out of both sequences, so they are always equal
        length; two empty sequences — the metric absent from every shared round — are the caller's
        skip-metric signal. ``dropped`` counts the shared rounds where exactly one side carried the
        metric; a round where neither side has it is not a drop.
    """
    left_values: list[float] = []
    right_values: list[float] = []
    dropped = 0
    for left_repeat, right_repeat in zip(left, right, strict=False):
        in_left = metric in left_repeat
        in_right = metric in right_repeat
        if in_left and in_right:
            left_values.append(left_repeat[metric])
            right_values.append(right_repeat[metric])
        elif in_left != in_right:
            dropped += 1
    return PairResult(left=tuple(left_values), right=tuple(right_values), dropped=dropped)
