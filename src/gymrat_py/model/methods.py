"""Method tag union, statistical descriptors, and noise-model constants."""

from dataclasses import dataclass
from typing import Literal

Method = Literal["signed-rank", "band", "exact"]
"""Tag identifying which statistical method produced a verdict."""


@dataclass(frozen=True, slots=True)
class MethodDescriptor:
    """Statistical floors for one method, carried as data.

    Attributes:
        method: The method this descriptor describes.
        min_n: Minimum usable sample size the method requires.
        p_threshold: Significance threshold, or ``None`` when the method has none.
    """

    method: Method
    min_n: int
    p_threshold: float | None


SIGNED_RANK_DESCRIPTOR = MethodDescriptor(method="signed-rank", min_n=6, p_threshold=0.05)
"""Signed-rank floors: at least 6 pairs, gated at p ≤ 0.05."""

BAND_DESCRIPTOR = MethodDescriptor(method="band", min_n=2, p_threshold=None)
"""Band floors: at least 2 usable entries; no significance gate."""

EXACT_DESCRIPTOR = MethodDescriptor(method="exact", min_n=1, p_threshold=None)
"""Exact floors: a single sample suffices; no significance gate."""

NOISE_K = 1.5
"""Multiplier applied to the noise floor when deriving the instability band."""

NOISE_FLOOR_PCT = 0.5
"""Minimum noise level, as a percentage, below which measurements are treated as floor noise."""

DEFAULT_UNSTABLE_NOISE_PCT = 200
"""Noise percentage assigned to a metric flagged unstable when no measured value applies."""
