"""Verdict engine core: pairing, delta computation, and method dispatch.

For each metric the engine pairs its per-round observations, computes a
percentage delta from the paired medians, and classifies the move with one of
three methods:

- **exact** — any difference between medians is signal; no noise band.
- **signed-rank** — the Wilcoxon test decides significance once there are enough
  non-tied pairs; a delta must also clear the metric's measurement resolution.
- **band** — the fallback for short or tied runs; a delta must exceed the
  metric's own noise band.
"""

import dataclasses
import math
from collections.abc import Mapping
from dataclasses import dataclass

from gymrat_py.model import (
    BAND_DESCRIPTOR,
    DEFAULT_UNSTABLE_NOISE_PCT,
    NOISE_FLOOR_PCT,
    NOISE_K,
    SIGNED_RANK_DESCRIPTOR,
    BandVerdict,
    Direction,
    Effect,
    ExactVerdict,
    MetricMeta,
    MetricUnit,
    MetricVerdict,
    Observations,
    SignedRankVerdict,
    Verdict,
    pair_metric,
)
from gymrat_py.stats import compute_half_range, compute_median, wilcoxon_signed_rank
from gymrat_py.warn import WarnSink, warn_to_stderr

# One byte expressed as a percentage of a one-byte median: what a whole-byte
# metric cannot measure below.
ONE_BYTE_PCT = 100.0


@dataclass(frozen=True, slots=True)
class _Noise:
    """Measurement noise of a metric, in the forms the verdict logic needs.

    Attributes:
        pct: Noise as a percentage of the metric's median, never below the floor.
        abs: The same noise in the metric's own unit, with no floor applied.
        resolution_pct: The part of ``pct`` set by the metric's measurement
            resolution rather than by its observed scatter; ``0`` for a unit that
            is not quantized. A delta no larger than this is a step of
            quantization, not a measured move.
    """

    pct: float
    abs: float
    resolution_pct: float


def _determine_verdict(delta: float, direction: Direction) -> Verdict:
    """Classify a delta as improved, regressed, or no-signal for a direction.

    A NaN delta (the ratio is undefined because the baseline median was 0) has no
    direction to read, so it reports no signal rather than falling through to
    "regressed" — every comparison against NaN is false.
    """
    if delta == 0 or math.isnan(delta):
        return "no-signal"

    improved = delta < 0 if direction == "lower" else delta > 0
    return "improved" if improved else "regressed"


def _verdict_if_signal(delta: float, direction: Direction, *, has_signal: bool) -> Verdict:
    """Classify the delta when a signal was found, else ``no-signal``.

    ``has_signal`` already encodes the method's own significance test (the
    signed-rank p-value comparison or the band comparison); this helper exists only
    to keep ``no-signal`` the single fallback path shared by both approximate
    methods.
    """
    return _determine_verdict(delta, direction) if has_signal else "no-signal"


def _compute_delta(median_a: float, median_b: float) -> float:
    """Percentage delta between two medians, normalized by ``|median_a|``.

    Normalizing by the magnitude keeps the delta's sign tied to the direction the
    value moved: a negative-median metric dropping further below zero is a
    decrease, not an increase. When ``median_a`` is 0 the ratio is undefined — 0
    if both medians are 0, ``NaN`` otherwise.
    """
    if median_a == 0 and median_b == 0:
        return 0.0
    if median_a == 0:
        return math.nan
    return (median_b - median_a) / abs(median_a) * 100


def _fraction_of_median(numerator: float, median: float) -> float:
    """``numerator`` as a fraction of a median's magnitude, or 0 with no magnitude.

    A side that measured 0 contributes nothing instead of making the result
    infinite, so each side stands on its own term.
    """
    return 0.0 if median == 0 else numerator / abs(median)


def _compute_noise(
    paired_left: list[float],
    paired_right: list[float],
    unit: MetricUnit | None,
) -> _Noise:
    """Compute the measurement noise of a metric.

    Percentage form:
    ``max(K * 100 * max(spread(A), spread(B)), floor%, byteFloor%)`` where each
    ``spread`` is a side's half-range over its median magnitude, ``K`` is
    :data:`NOISE_K`, and ``floor`` is :data:`NOISE_FLOOR_PCT`. Absolute form:
    ``K * max(halfRange(A), halfRange(B))``.

    A byte-valued metric takes a further floor of one byte against each median: it
    is quantized to whole bytes, so a 4B → 3B move is one step of resolution
    rather than a measured 25% win, however tight its spread. Averaged units such
    as ``ns`` carry no such bound and keep the plain floor.
    """
    median_a = compute_median(paired_left)
    median_b = compute_median(paired_right)
    half_range_a = compute_half_range(paired_left)
    half_range_b = compute_half_range(paired_right)

    spread_a = _fraction_of_median(half_range_a, median_a)
    spread_b = _fraction_of_median(half_range_b, median_b)
    max_spread = max(spread_a, spread_b)

    byte_floor_pct = (
        max(
            _fraction_of_median(ONE_BYTE_PCT, median_a),
            _fraction_of_median(ONE_BYTE_PCT, median_b),
        )
        if unit == "bytes"
        else 0.0
    )

    return _Noise(
        pct=max(NOISE_K * 100 * max_spread, NOISE_FLOOR_PCT, byte_floor_pct),
        abs=NOISE_K * max(half_range_a, half_range_b),
        resolution_pct=byte_floor_pct,
    )


def _compute_approximate_verdict(
    paired_left: list[float],
    paired_right: list[float],
    delta: float,
    meta: MetricMeta,
    unstable_noise_pct: float,
) -> SignedRankVerdict | BandVerdict:
    """Decide a non-exact verdict, applying the unstable override.

    Uses the signed-rank method when at least :attr:`SIGNED_RANK_DESCRIPTOR.min_n`
    pairs differ by a non-zero amount; the noise band otherwise. Tied pairs carry
    no rank information, so a long but mostly identical run falls back to the band
    just as a short one does.

    A significant p-value alone is not enough on the signed-rank path: the delta
    must also clear the metric's measurement resolution, or a one-byte
    quantization step reads as signal however many rounds agree on it.
    """
    result = wilcoxon_signed_rank(paired_left, paired_right)
    noise = _compute_noise(paired_left, paired_right, meta.unit)
    n = len(paired_left)
    effect = Effect(value=delta, unit="percent")

    record: SignedRankVerdict | BandVerdict
    if result.n < SIGNED_RANK_DESCRIPTOR.min_n:
        has_signal = n >= BAND_DESCRIPTOR.min_n and abs(delta) > noise.pct
        verdict = _verdict_if_signal(delta, meta.direction, has_signal=has_signal)
        record = BandVerdict(
            method="band",
            verdict=verdict,
            usable_n=result.n,
            noise_pct=noise.pct,
            noise_abs=noise.abs,
            delta=effect,
            n=n,
        )
    else:
        # The signed-rank descriptor always carries a threshold; the union type
        # admits None only for the exact and band descriptors. Surface a
        # misconfiguration rather than compare against None.
        threshold = SIGNED_RANK_DESCRIPTOR.p_threshold
        if threshold is None:
            message = "SIGNED_RANK_DESCRIPTOR must define a p-value threshold"
            raise ValueError(message)
        has_signal = result.p < threshold and abs(delta) > noise.resolution_pct
        verdict = _verdict_if_signal(delta, meta.direction, has_signal=has_signal)
        record = SignedRankVerdict(
            method="signed-rank",
            verdict=verdict,
            p=result.p,
            noise_pct=noise.pct,
            noise_abs=noise.abs,
            delta=effect,
            n=n,
        )

    # The band is too wide to measure any delta against, so the override is
    # unconditional. Strict comparison keeps a metric sitting exactly on the
    # threshold on its normal verdict.
    if record.noise_pct > unstable_noise_pct:
        return dataclasses.replace(record, verdict="unstable")
    return record


def compute_verdicts(
    left: Observations,
    right: Observations,
    metric_meta: Mapping[str, MetricMeta],
    *,
    unstable_noise_pct: float = DEFAULT_UNSTABLE_NOISE_PCT,
    warn: WarnSink = warn_to_stderr,
) -> dict[str, MetricVerdict]:
    """Compute per-metric verdicts across two observation sets.

    Metrics are visited in ``metric_meta`` insertion order. For each, values are
    paired by round; windows where either side is missing the metric are dropped.
    A metric present on only one side across every window yields no paired samples
    and is skipped silently — it produces no verdict and no warning.

    A metric that did produce a verdict but lost windows to one-sided measurement
    emits a single warning through ``warn`` naming the metric and the dropped
    count. The dropped windows never change the verdict, which is computed from
    the windows that did pair.

    Args:
        left: Baseline observations.
        right: Candidate observations.
        metric_meta: Per-metric metadata, iterated in insertion order.
        unstable_noise_pct: Noise band width, in percent, above which a non-exact
            metric is reported unstable. Compared strictly, so a metric sitting
            exactly on the threshold keeps its normal verdict.
        warn: Sink for the dropped-window divergence warning.

    Returns:
        A mapping from metric name to verdict, holding only metrics that produced
        one.
    """
    result: dict[str, MetricVerdict] = {}

    for metric, meta in metric_meta.items():
        paired = pair_metric(left, right, metric)

        # Both paired sequences grow together, so one length check covers both.
        if not paired.left:
            continue

        median_a = compute_median(paired.left)
        median_b = compute_median(paired.right)
        delta = _compute_delta(median_a, median_b)

        if meta.exact:
            result[metric] = ExactVerdict(
                method="exact",
                verdict=_determine_verdict(delta, meta.direction),
                delta=Effect(value=delta, unit="percent"),
                n=len(paired.left),
            )
        else:
            result[metric] = _compute_approximate_verdict(
                paired.left,
                paired.right,
                delta,
                meta,
                unstable_noise_pct,
            )

        if paired.dropped > 0:
            warn(
                f"{metric}: dropped {paired.dropped} paired window(s) "
                "where the metric was measured on only one side",
            )

    return result
