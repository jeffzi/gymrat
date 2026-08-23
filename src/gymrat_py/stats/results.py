"""Shared significance-test result shape and pairing helper.

:class:`SignificanceResult` is deliberately test-agnostic: it carries only the
two-sided p-value and the effective paired sample count, with no reference to
which test produced them. Every significance test in this package — the
Wilcoxon signed-rank wrapper and the sign-flip permutation test — returns this
same shape so callers can consume verdicts uniformly. :func:`count_nonzero_pairs`
is the pairing logic every such test starts from.
"""

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SignificanceResult:
    """The outcome of a paired significance test.

    Attributes:
        p: The two-sided p-value, always within ``[0.0, 1.0]``.
        n: The number of paired entries that actually entered the test, after
            dropping zero-difference pairs.
    """

    p: float
    n: int


def count_nonzero_pairs(x: Sequence[float], y: Sequence[float]) -> tuple[int, int]:
    """Pair ``x`` and ``y`` positionally over the shorter input and count non-zero diffs.

    Shared by every paired significance test: each pairs ``x[i]`` with ``y[i]``
    for ``i`` below the shorter input's length and drops pairs whose difference
    is zero before invoking its test.

    Args:
        x: The first sample.
        y: The second sample, paired positionally with ``x``.

    Returns:
        A ``(m, n)`` pair: ``m`` is the number of positions paired (the shorter
        input's length), and ``n`` is how many of those pairs have a non-zero
        difference.
    """
    m = min(len(x), len(y))
    n = sum(1 for i in range(m) if x[i] - y[i] != 0)
    return m, n
