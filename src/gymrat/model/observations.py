"""Observation container and metric-pairing strategy.

An :class:`Observations` value holds, per pairing-axis key, one or more repeats — each repeat a
mapping of metric name to value. The repeat axis is structurally distinct from the pairing axis: a
key maps to a *sequence* of repeat-mappings, not a single mapping.

:func:`pair_metric` aligns two containers on the keys they share, for a single metric, dropping any
shared key where either side is missing that metric.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Self

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
        """Build a container keyed by 0-based round index, one repeat per round, order preserved."""
        return cls(by_key={index: (sample,) for index, sample in enumerate(samples)})


def _require_single_repeat(observations: Observations) -> None:
    """Raise ``ValueError`` if any key carries more than one repeat.

    A multi-repeat container is constructible, but pairing over one is not defined — surface it
    rather than silently taking the first repeat.
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

    Iterates the keys ``left`` and ``right`` share, in ``left``'s order. A shared key where either
    side's repeat lacks ``metric`` is dropped from both output sequences. The two returned sequences
    are always equal length. A metric absent from every shared key yields two empty sequences — the
    caller's skip-metric signal.

    The returned :class:`PairResult` also reports ``dropped``: the count of shared keys where
    exactly one side carried the metric. A shared key where neither side has the metric is not a
    drop.

    Both containers must be single-repeat; a multi-repeat container raises ``ValueError``.
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
