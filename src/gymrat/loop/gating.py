"""The rule that decides whether one metric's verdict is a gating regression.

The iteration judge reads it from live verdicts and ``keep`` reads it back from
the recorded ones, so both answer the same question the same way.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gymrat.model import ApproximateVerdict


def is_gating_regression(*, gating: bool, verdict: ApproximateVerdict | None) -> bool:
    """Whether a metric that gates the run carries a ``regressed`` verdict.

    Args:
        gating: Whether the metric gates the run.
        verdict: The metric's verdict word, or ``None`` when it has none.

    Returns:
        ``True`` for a gating metric whose verdict is ``regressed``.
    """
    return gating and verdict == "regressed"
