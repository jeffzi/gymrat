"""Observation container and metric-pairing strategy.

An :class:`Observations` value holds, per pairing-axis key, one or more repeats — each repeat a
mapping of metric name to value. The repeat axis is structurally distinct from the pairing axis: a
key maps to a *sequence* of repeat-mappings, not a single mapping.

:func:`pair_metric` aligns two containers on the keys they share, for a single metric, dropping any
shared key where either side is missing that metric.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, Self, assert_never

type PairingKey = int
"""Pairing-axis key. A round index today; a general key for future eval-mode shapes."""

type Repeat = Mapping[str, float]
"""One repeat: a mapping of metric name to value."""

type UnpairedPolicy = Literal["drop-unpaired"]
"""How :func:`pair_metric` treats shared keys missing the metric on a side."""

DROP_UNPAIRED: UnpairedPolicy = "drop-unpaired"
"""Silently drop any shared key where either side lacks the metric.

Named explicitly so a future caller can hang a warning on this policy rather than rediscovering the
drops.
"""


@dataclass(frozen=True, slots=True)
class Observations:
    """A frozen wrapper over an ordered mapping from pairing-axis key to a tuple of repeats."""

    by_key: Mapping[PairingKey, tuple[Repeat, ...]]

    @classmethod
    def from_rounds(cls, samples: Sequence[Repeat]) -> Self:
        """Build a container keyed by 0-based round index, one repeat per round, order preserved."""
        return cls(by_key={index: (sample,) for index, sample in enumerate(samples)})


def _require_single_repeat(observations: Observations) -> None:
    """Raise ``ValueError`` if any key carries more than one repeat.

    The repeat-reducer is future scope: a multi-repeat container is constructible, but pairing over
    one is not defined yet — surface it rather than silently taking the first repeat.
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
    *,
    policy: UnpairedPolicy = DROP_UNPAIRED,
) -> tuple[list[float], list[float]]:
    """Align two containers on their shared keys, in order, for a single metric.

    Iterates the keys ``left`` and ``right`` share, in ``left``'s order. A shared key where either
    side's repeat lacks ``metric`` is dropped from both output sequences. The two returned sequences
    are always equal length. A metric absent from every shared key yields two empty sequences — the
    caller's skip-metric signal.

    Both containers must be single-repeat; a multi-repeat container raises ``ValueError``.
    """
    if policy != DROP_UNPAIRED:
        assert_never(policy)

    _require_single_repeat(left)
    _require_single_repeat(right)

    left_values: list[float] = []
    right_values: list[float] = []
    for key, left_repeats in left.by_key.items():
        right_repeats = right.by_key.get(key)
        if right_repeats is None:
            continue
        left_repeat = left_repeats[0]
        right_repeat = right_repeats[0]
        if metric not in left_repeat or metric not in right_repeat:
            continue
        left_values.append(left_repeat[metric])
        right_values.append(right_repeat[metric])
    return left_values, right_values
