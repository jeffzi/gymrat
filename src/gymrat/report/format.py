"""Value and cell formatting, and verdict evidence primitives.

These turn the model's numbers and verdicts into the strings a renderer draws:
scaled measurements, signed deltas, and evidence suffixes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from gymrat.report.display import VERDICT_GLOSSES
from gymrat.utils import fraction_of_median

if TYPE_CHECKING:
    from gymrat.model import MetricUnit, MetricVerdict
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

_RELATIVE_SPREAD_CAP_PCT = 100


def _non_finite_token(value: float) -> str:
    """The token the text report prints for a non-finite reading (JSON writes ``null``)."""
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

    Args:
        value: The value to scale.
        tiers: The scale tiers, smallest first.

    Returns:
        The scaled, suffixed figure such as ``"1.7µs"`` or ``"512KB"``.
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


ZERO_PERCENT_DELTA = "0.0%"
"""What a delta that rounds to zero prints as: unsigned, since it points nowhere."""


def format_percent_delta(value: float | None, *, missing: str = "") -> str:
    """A signed percentage, or a placeholder when there is no finite delta to state.

    At display precision a delta that rounds to zero has no direction to
    report, so ``-0.0%`` would claim one.

    Args:
        value: The percentage delta, such as ``2.2`` for ``+2.2%``, or ``None``
            when none was measured.
        missing: What to print for a missing or non-finite delta.

    Returns:
        A signed percentage such as ``"+2.2%"``, an unsigned ``"0.0%"`` for a
        value that rounds to zero, or ``missing`` when the value is ``None`` or
        not finite (``NaN`` or either infinity).
    """
    if value is None or not math.isfinite(value):
        return missing
    magnitude = f"{abs(value):.1f}"
    if magnitude == "0.0":
        return ZERO_PERCENT_DELTA
    sign = "+" if value > 0 else "-"
    return f"{sign}{magnitude}%"


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
    if verdict.verdict == "unstable":
        return VERDICT_GLOSSES["unstable"]
    return format_percent_delta(verdict.delta)


# ---------------------------------------------------------------------------
# Verdict evidence
# ---------------------------------------------------------------------------


def format_noise_band_value(noise_pct: float) -> str:
    """A noise band's figure, without the sign it is stated behind."""
    return f"{noise_pct:.1f}%"


def format_pair_count(n: int) -> str:
    """How many pairs a verdict rests on, as the ``n=N`` the rows and footer share."""
    return f"n={n}"


def _is_ratio_undefined(noise_abs: float, median: float | None) -> bool:
    return median is not None and fraction_of_median(noise_abs, median, 100) is None


def is_noise_percentage_undefined(
    noise_abs: float, baseline_median: float | None, candidate_median: float | None
) -> bool:
    """Whether the noise cannot be stated as a percentage of either side's median.

    The verdict engine drops such a side from the noise percentage, so the
    percentage left behind is only the floor and understates the scatter that
    made the verdict unstable.

    Args:
        noise_abs: The noise in the metric's own units.
        baseline_median: The baseline median, or ``None`` when unknown.
        candidate_median: The candidate median, or ``None`` when unknown.

    Returns:
        ``True`` when either median is zero or the noise as a percentage of it is
        not finite.
    """
    return _is_ratio_undefined(noise_abs, baseline_median) or _is_ratio_undefined(
        noise_abs, candidate_median
    )


def is_noise_absolute(
    noise_pct: float,
    noise_abs: float,
    baseline_median: float,
    candidate_median: float | None = None,
) -> bool:
    """Whether a noise figure reads in the metric's own units rather than as a percentage.

    Past :data:`_RELATIVE_SPREAD_CAP_PCT` a percentage stops being readable, and
    when either median leaves the noise without a finite percentage (see
    :func:`is_noise_percentage_undefined`) it is only the noise floor, standing
    in for a scatter no percentage can express.

    Args:
        noise_pct: The noise as a percentage of the median.
        noise_abs: The noise in the metric's own units.
        baseline_median: The baseline median the noise is stated against.
        candidate_median: The candidate median, or ``None`` when unknown.

    Returns:
        ``True`` when the noise belongs in absolute units.
    """
    return noise_pct > _RELATIVE_SPREAD_CAP_PCT or is_noise_percentage_undefined(
        noise_abs, baseline_median, candidate_median
    )


def format_evidence(
    verdict: MetricVerdict,
    unit: MetricUnit | None = None,
    baseline_median: float | None = None,
    candidate_median: float | None = None,
) -> str:
    """The evidence suffix for a highlighted metric.

    Exact entries keep ``(exact)``. Unstable entries show the noise that swamped
    the signal — as a percentage while that stays readable, and in the metric's
    own units past :data:`_RELATIVE_SPREAD_CAP_PCT` or when either median leaves
    the noise without a finite percentage (a zero median, or one so close to zero
    that the percentage overflows), where the percentage is only the noise floor
    and would understate the scatter. The absolute noise is stated against the
    baseline median, or against the candidate median when only the candidate
    leaves the percentage undefined. Improved/regressed/no-signal entries from
    approximate methods carry no trailing evidence.

    Args:
        verdict: The verdict to describe.
        unit: The metric's unit, for restating noise in absolute terms.
        baseline_median: The baseline median, for the absolute restatement.
        candidate_median: The candidate median, for the absolute restatement
            when the noise has no finite percentage of it.

    Returns:
        The evidence suffix, or ``""`` when there is nothing to add.
    """
    if verdict.method == "exact":
        return "(exact)"
    if verdict.verdict != "unstable":
        return ""
    if baseline_median is None or not is_noise_absolute(
        verdict.noise_pct, verdict.noise_abs, baseline_median, candidate_median
    ):
        return f"noise {PLUS_MINUS}{format_noise_band_value(verdict.noise_pct)}"
    noise = f"{PLUS_MINUS}{format_value(verdict.noise_abs, unit)} noise"
    if (
        candidate_median is not None
        and not _is_ratio_undefined(verdict.noise_abs, baseline_median)
        and _is_ratio_undefined(verdict.noise_abs, candidate_median)
    ):
        return f"{noise} on a {format_value(candidate_median, unit)} candidate median"
    return f"{noise} on a {format_value(baseline_median, unit)} median"
