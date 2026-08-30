"""Value and cell formatting, and verdict evidence primitives.

These turn the model's numbers and verdicts into the strings a renderer draws:
scaled measurements, signed deltas, and evidence suffixes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, assert_never

if TYPE_CHECKING:
    from gymrat.model import Effect, MetricUnit, MetricVerdict
    from gymrat.report.types import CandidateMetric, MetricComparison


# ---------------------------------------------------------------------------
# Value and cell formatting
# ---------------------------------------------------------------------------

type _Tier = tuple[float, float, str, int]

_NS_TIERS: tuple[_Tier, ...] = (
    (1000, 1, "ns", 0),
    (1e6, 1000, "µs", 1),
    (1e9, 1e6, "ms", 1),
    (math.inf, 1e9, "s", 1),
)

_BYTE_TIERS: tuple[_Tier, ...] = (
    (1000, 1, "B", 0),
    (1e6, 1000, "KB", 1),
    (1e9, 1e6, "MB", 1),
    (math.inf, 1e9, "GB", 1),
)

_TIER_MAP: dict[MetricUnit, tuple[_Tier, ...]] = {"ns": _NS_TIERS, "bytes": _BYTE_TIERS}


def _non_finite_token(value: float) -> str:
    """The token a non-finite reading prints, matching the differ/JSON contract."""
    if math.isnan(value):
        return "NaN"
    return "Infinity" if value > 0 else "-Infinity"


def _scale_tier(value: float, tiers: tuple[_Tier, ...]) -> str:
    """Scale ``value`` into the first tier whose rounded figure stays below its threshold.

    The tier is chosen on the figure *as rounded* rather than as measured: a
    value just under a threshold rounds up onto it (999.5 bytes to ``1000B``),
    which is a four-digit magnitude in a column sized for three, so it is
    promoted to the tier above. The threshold is compared against the magnitude,
    since a sign is not a size: a negative reading picks the tier its magnitude
    names.
    """
    magnitude = abs(value)
    for threshold, divisor, suffix, decimals in tiers:
        rounded = float(format(magnitude / divisor, f".{decimals}f"))
        if rounded * divisor < threshold:
            return f"{value / divisor:.{decimals}f}{suffix}"
    return str(value)


def format_value(value: float, unit: MetricUnit | None = None) -> str:
    """Scale a measurement into its unit's tier, or round it when the metric has no unit.

    Args:
        value: The measurement to format.
        unit: The metric's unit, or ``None`` for a unitless figure.

    Returns:
        The scaled, suffixed figure (``"1.7µs"``), the rounded integer for a
        unitless value, or ``"Infinity"`` / ``"-Infinity"`` / ``"NaN"`` for a
        non-finite reading.
    """
    if not math.isfinite(value):
        return _non_finite_token(value)
    if unit is None:
        return str(round(value))
    return _scale_tier(value, _TIER_MAP[unit])


def format_delta(effect: Effect) -> str:
    """A signed percentage, or nothing when the effect is not a number.

    A delta that rounds to zero prints as an unsigned ``0.0%``: at display
    precision there is no direction to report, so ``-0.0%`` would claim one.

    Args:
        effect: The effect to render. Its ``value`` carries the number and its
            ``unit`` the scale (percent today).

    Returns:
        A signed percentage such as ``"+2.2%"``, an unsigned ``"0.0%"`` for a
        value that rounds to zero, or ``""`` when the value is ``NaN``.
    """
    value = effect.value
    if math.isnan(value):
        return ""
    magnitude = f"{abs(value):.1f}"
    if magnitude == "0.0":
        return "0.0%"
    sign = "+" if value > 0 else "-"
    return f"{sign}{magnitude}%"


def is_improvement(effect: Effect) -> bool:
    """Whether an effect's move counts as an improvement, keyed on its unit.

    This is the single place the sign-of-improvement rule lives, so a caller
    judging a direction-aware metric combines this with the metric's own
    direction rather than re-deriving the sign.

    For a ``"percent"`` delta the default is lower-is-better: a strictly negative
    value improves. A value of exactly zero does not — at rest a figure moved in
    no direction to call good.

    Args:
        effect: The effect to judge; its ``unit`` selects the rule.

    Returns:
        ``True`` when the effect's value is an improvement for its unit.
    """
    if effect.unit == "percent":
        return effect.value < 0
    assert_never(effect.unit)


PLUS_MINUS = "±"

SPREAD_SEPARATOR = f" {PLUS_MINUS} "


@dataclass(frozen=True, slots=True)
class MetricCellParts:
    """A value cell taken apart, so a table can pad each field to its own column width.

    Both fields are empty when the side reported nothing, and the spread alone is
    empty when the measurement carries no scatter.

    Attributes:
        magnitude: The scaled measurement.
        spread: What follows the ``±``: a percentage, or absolute units once it
            outgrows the median.
    """

    magnitude: str
    spread: str


def format_metric_cell_parts(
    median: float | None = None,
    spread: float | None = None,
    unit: MetricUnit | None = None,
) -> MetricCellParts:
    """A value cell's fields: the scaled measurement and the spread stated behind it.

    A spread past :data:`_RELATIVE_SPREAD_CAP_PCT` is restated in absolute units,
    so ``5B ± 7620%`` reads ``5B ± 381B`` instead.

    Args:
        median: The measurement, or ``None`` when the side reported nothing.
        spread: The half-range around the median, or ``None`` when none measured.
        unit: The metric's unit, or ``None`` when unitless.

    Returns:
        The magnitude and spread, each empty where there was nothing to state.
    """
    if median is None:
        return MetricCellParts(magnitude="", spread="")
    magnitude = format_value(median, unit)
    if spread is None:
        return MetricCellParts(magnitude=magnitude, spread="")
    if spread > _RELATIVE_SPREAD_CAP_PCT:
        return MetricCellParts(
            magnitude=magnitude, spread=format_value(abs(median * spread / 100), unit)
        )
    return MetricCellParts(magnitude=magnitude, spread=f"{spread:.0f}%")


def baseline_cell_parts(metric: MetricComparison) -> MetricCellParts:
    """A metric's baseline figure, taken apart the way :func:`format_metric_cell_parts` does."""
    return format_metric_cell_parts(
        metric.baseline_median, metric.baseline_spread, metric.meta.unit
    )


def candidate_cell_parts(
    side: CandidateMetric | None,
    unit: MetricUnit | None = None,
) -> MetricCellParts:
    """One candidate's side of a metric, taken apart into padded fields."""
    if side is None:
        return format_metric_cell_parts(None, None, unit)
    return format_metric_cell_parts(side.median, side.spread, unit)


def format_verdict_delta(verdict: MetricVerdict) -> str:
    """The delta cell: the word ``unstable`` for a verdict too noisy to trust, else the delta."""
    return "unstable" if verdict.verdict == "unstable" else format_delta(verdict.delta)


# ---------------------------------------------------------------------------
# Verdict evidence
# ---------------------------------------------------------------------------

_RELATIVE_SPREAD_CAP_PCT = 100


def format_noise_band_value(noise_pct: float) -> str:
    """A noise band's figure, without the sign it is stated behind."""
    return f"{noise_pct:.1f}%"


def _format_noise_band(noise_pct: float) -> str:
    """A metric's noise band as the ``±N%`` the rows and highlights share."""
    return f"{PLUS_MINUS}{format_noise_band_value(noise_pct)}"


def format_pair_count(n: int) -> str:
    """How many pairs a verdict rests on, as the ``n=N`` the rows and footer share."""
    return f"n={n}"


def format_evidence(
    verdict: MetricVerdict,
    unit: MetricUnit | None = None,
    baseline_median: float | None = None,
) -> str:
    """The evidence suffix for a highlighted metric.

    Exact entries keep ``(exact)``. Unstable entries show the noise that swamped
    the signal — as a percentage while that stays readable, and against the
    baseline median in the metric's own units past
    :data:`_RELATIVE_SPREAD_CAP_PCT`. Improved/regressed/no-signal entries from
    approximate methods carry no trailing evidence.

    Args:
        verdict: The verdict to describe.
        unit: The metric's unit, for restating noise in absolute terms.
        baseline_median: The baseline median, for the absolute restatement.

    Returns:
        The evidence suffix, or ``""`` when there is nothing to add.
    """
    if verdict.method == "exact":
        return "(exact)"
    if verdict.verdict != "unstable":
        return ""
    if verdict.noise_pct > _RELATIVE_SPREAD_CAP_PCT and baseline_median is not None:
        noise = format_value(verdict.noise_abs, unit)
        return f"{PLUS_MINUS}{noise} noise on a {format_value(baseline_median, unit)} median"
    return f"noise {_format_noise_band(verdict.noise_pct)}"
