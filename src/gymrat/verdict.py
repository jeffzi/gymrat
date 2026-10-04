"""Verdict engine: per-metric verdicts and their geometric-mean aggregation.

For each metric, :func:`compute_verdicts` pairs its per-round observations,
computes a percentage delta from the paired medians, and classifies the move
with one of three methods:

- **exact** — any difference between medians is signal; no noise band.
- **permutation** — the sign-flip permutation test decides significance once
  there are enough non-tied pairs. Its statistic is the reported delta functional
  itself, its small-sample null is an exact sign-flip enumeration, and its
  p-value is used unclamped; a delta must also clear the metric's measurement
  resolution.
- **band** — the fallback for short or tied runs; a delta must exceed the
  metric's own noise band.

Verdicts are then averaged hierarchically — one geometric mean per kind, per
group, and per gating subset. Each mean covers exactly the metrics in its scope,
in log space over each metric's normalized ratio, and propagates their noise
bands in quadrature. Metrics that cannot contribute a usable ratio — never
judged, judged unstable, or yielding a degenerate ratio — are reported as
exclusions instead, keeping each aggregate accountable for every metric the
caller asked it to cover.
"""

import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

from gymrat.metric_name import parse as parse_metric_name
from gymrat.model import (
    DEFAULT_UNSTABLE_NOISE_PCT,
    NOISE_FLOOR_PCT,
    PERMUTATION_MIN_N,
    PERMUTATION_P_THRESHOLD,
    BandVerdict,
    Direction,
    ExactVerdict,
    Exclusion,
    GeomeanResult,
    MetricMeta,
    MetricUnit,
    MetricVerdict,
    PermutationVerdict,
    Repeat,
    ResolvedMetricMeta,
    Verdict,
    is_improvement,
    pair_metric,
)
from gymrat.stats import (
    combine_geomean,
    compute_half_range,
    count_nonzero_pairs,
    normalize_ratio,
    percent_delta,
    sign_flip_permutation_test,
)
from gymrat.utils import WarnSink, warn_to_stderr

__all__ = [
    "BAND_MIN_N",
    "NOISE_K",
    "GroupAggregate",
    "KindAggregate",
    "compute_geomean",
    "compute_kind_aggregates",
    "compute_verdicts",
]

# ---------------------------------------------------------------------------
# Per-metric verdicts
# ---------------------------------------------------------------------------

ONE_BYTE_PCT = 100.0
"""One byte expressed as a percentage of a one-byte median: what a whole-byte metric
cannot measure below."""

BAND_MIN_N = 2
"""Minimum count of differing pairs the band method requires."""

NOISE_K = 1.5
"""Multiplier applied to the noise floor when deriving the instability band."""


@dataclass(frozen=True, slots=True)
class _PairedSamples:
    """A metric's round-paired samples with the per-side medians already computed.

    Bundles what every downstream step needs — :func:`compute_verdicts` computes
    the medians once for the delta, and the noise and verdict paths reuse them
    instead of recomputing.
    """

    left: Sequence[float]
    right: Sequence[float]
    median_left: float
    median_right: float


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
        force_unstable: Whether a side scattered around a median too close to
            zero to express its noise as a percentage, which understates ``pct``.
    """

    pct: float
    abs: float
    resolution_pct: float
    force_unstable: bool


def _determine_verdict(delta: float, direction: Direction, *, has_signal: bool = True) -> Verdict:
    """Classify a delta as improved, regressed, or no-signal for a direction.

    A NaN delta (the ratio is undefined because the baseline median was 0) has no
    direction to read, so it reports no signal rather than falling through to
    "regressed" — every comparison against NaN is false.

    Args:
        delta: The percentage delta between the two medians.
        direction: Which sign of delta counts as an improvement for this metric.
        has_signal: Whether the delta cleared the statistical test; ``False``
            reports no signal whatever the delta.

    Returns:
        The verdict for the given delta and direction.
    """
    if not has_signal or delta == 0 or math.isnan(delta):
        return "no-signal"

    return "improved" if is_improvement(delta, direction) else "regressed"


def _fraction_of_median(numerator: float, median: float, scale: float = 1.0) -> float | None:
    """A value as a scaled fraction of a median's magnitude.

    The scale is applied after the division so a large numerator alone cannot
    overflow.

    Args:
        numerator: The value to express as a fraction of the median.
        median: The median whose magnitude is the denominator.
        scale: The factor applied to the fraction.

    Returns:
        The scaled fraction, or ``None`` when *median* is zero or the scaled
        fraction is not finite.
    """
    if median == 0:
        return None
    fraction = (numerator / abs(median)) * scale
    return fraction if math.isfinite(fraction) else None


def _largest_term(*fractions: float | None) -> float:
    return max((f for f in fractions if f is not None), default=0.0)


def _compute_noise(samples: _PairedSamples, unit: MetricUnit | None) -> _Noise:
    """The percentage and absolute noise thresholds a paired delta must clear.

    Percentage form:
    ``max(K * 100 * max(spread(A), spread(B)), floor%, byteFloor%)`` where each
    ``spread`` is a side's half-range over its median magnitude, ``K`` is
    :data:`NOISE_K`, and ``floor`` is :data:`NOISE_FLOOR_PCT`. Absolute form:
    ``K * max(halfRange(A), halfRange(B))``.

    A side whose median is 0, or so close to 0 that its ratio overflows, contributes
    no term; with a non-zero half-range it also forces the verdict unstable.

    A byte-valued metric takes a further floor of one byte against each median: it
    is quantized to whole bytes, so a 4B → 3B move is one step of resolution
    rather than a measured 25% win, however tight its spread. Averaged units such
    as ``ns`` carry no such bound and keep the plain floor.

    Args:
        samples: The paired baseline/candidate samples for the metric.
        unit: The metric's unit, or ``None`` when it carries no unit.

    Returns:
        The ``_Noise`` thresholds and the forced-unstable flag.
    """
    half_range_a = compute_half_range(samples.left)
    half_range_b = compute_half_range(samples.right)

    noise_pct_a = _fraction_of_median(half_range_a, samples.median_left, NOISE_K * 100)
    noise_pct_b = _fraction_of_median(half_range_b, samples.median_right, NOISE_K * 100)

    byte_floor_pct = 0.0
    if unit == "bytes":
        byte_pct_a = _fraction_of_median(ONE_BYTE_PCT, samples.median_left)
        byte_pct_b = _fraction_of_median(ONE_BYTE_PCT, samples.median_right)
        byte_floor_pct = _largest_term(byte_pct_a, byte_pct_b)

    # A side with scatter but no median magnitude cannot express its noise as a
    # percentage and adds no term to `pct`, which would understate the band; force
    # unstable instead of reporting a verdict against an understated band.
    force_unstable = (noise_pct_a is None and half_range_a != 0) or (
        noise_pct_b is None and half_range_b != 0
    )

    return _Noise(
        pct=max(_largest_term(noise_pct_a, noise_pct_b), NOISE_FLOOR_PCT, byte_floor_pct),
        abs=NOISE_K * max(half_range_a, half_range_b),
        resolution_pct=byte_floor_pct,
        force_unstable=force_unstable,
    )


def _compute_approximate_verdict(
    samples: _PairedSamples,
    delta: float,
    meta: MetricMeta,
    unstable_noise_pct: float,
) -> PermutationVerdict | BandVerdict:
    """Decide a non-exact verdict, applying the unstable override.

    Uses the sign-flip permutation test when at least
    :data:`~gymrat.model.PERMUTATION_MIN_N` pairs differ by a non-zero amount; the
    noise band otherwise. Tied pairs contribute no sign to flip, so a long but
    mostly identical run falls back to the band just as a short one does.

    A significant p-value alone is not enough on the permutation path: the delta
    must also clear the metric's measurement resolution, or a one-byte
    quantization step reads as signal however many rounds agree on it.

    Args:
        samples: The paired baseline/candidate samples for the metric.
        delta: The percentage delta between the two medians.
        meta: The metric's metadata, including its direction and unit.
        unstable_noise_pct: Noise band width, in percent, above which the
            verdict is forced to ``"unstable"``.

    Returns:
        A permutation or band verdict record for the metric.
    """
    nonzero_n = count_nonzero_pairs(samples.left, samples.right)
    noise = _compute_noise(samples, meta.unit)
    n = len(samples.left)

    record: PermutationVerdict | BandVerdict
    if nonzero_n < PERMUTATION_MIN_N:
        has_signal = nonzero_n >= BAND_MIN_N and abs(delta) > noise.pct
        verdict = _determine_verdict(delta, meta.direction, has_signal=has_signal)
        record = BandVerdict(
            method="band",
            verdict=verdict,
            usable_n=nonzero_n,
            noise_pct=noise.pct,
            noise_abs=noise.abs,
            delta=delta,
            n=n,
        )
    else:
        p = sign_flip_permutation_test(samples.left, samples.right)
        has_signal = p < PERMUTATION_P_THRESHOLD and abs(delta) > noise.resolution_pct
        verdict = _determine_verdict(delta, meta.direction, has_signal=has_signal)
        record = PermutationVerdict(
            method="permutation",
            verdict=verdict,
            p=p,
            noise_pct=noise.pct,
            noise_abs=noise.abs,
            delta=delta,
            n=n,
        )

    # The band is too wide to measure any delta against, so the override is
    # unconditional. Strict comparison keeps a metric sitting exactly on the
    # threshold on its normal verdict.
    if noise.force_unstable or record.noise_pct > unstable_noise_pct:
        return replace(record, verdict="unstable")
    return record


def compute_verdicts(
    left: Sequence[Repeat],
    right: Sequence[Repeat],
    metric_meta: Mapping[str, MetricMeta],
    *,
    unstable_noise_pct: float = DEFAULT_UNSTABLE_NOISE_PCT,
    warn: WarnSink = warn_to_stderr,
) -> dict[str, MetricVerdict]:
    """Compute per-metric verdicts across two sides' per-round samples.

    Values of each metric are paired by round; windows where either side is
    missing the metric are dropped. A metric present on only one side across
    every window yields no paired samples and is skipped silently — it produces
    no verdict and no warning.

    A metric that did produce a verdict but lost windows to one-sided measurement
    emits a single warning through ``warn`` naming the metric and the dropped
    count. The dropped windows never change the verdict, which is computed from
    the windows that did pair.

    Args:
        left: Baseline repeats, one per round.
        right: Candidate repeats, one per round.
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

        samples = _PairedSamples(
            left=paired.left,
            right=paired.right,
            median_left=statistics.median(paired.left),
            median_right=statistics.median(paired.right),
        )
        delta = percent_delta(samples.median_left, samples.median_right)

        if meta.exact:
            result[metric] = ExactVerdict(
                method="exact",
                verdict=_determine_verdict(delta, meta.direction),
                delta=delta,
                n=len(paired.left),
            )
        else:
            result[metric] = _compute_approximate_verdict(samples, delta, meta, unstable_noise_pct)

        if paired.dropped > 0:
            warn(
                f"{metric}: dropped {paired.dropped} paired window(s) "
                "where the metric was measured on only one side",
            )

    return result


# ---------------------------------------------------------------------------
# Geometric-mean aggregation
# ---------------------------------------------------------------------------


def compute_geomean(
    verdicts: Mapping[str, MetricVerdict],
    metric_meta: Mapping[str, MetricMeta],
) -> GeomeanResult:
    """Aggregate the metrics named in ``metric_meta`` into a geometric mean.

    Which metrics belong in the geomean is the caller's decision: every metric
    ``metric_meta`` names is in scope, gating or not, and ``verdicts`` may carry
    others that are ignored. Each in-scope metric is either included as a
    ``(rho, noise_pct)`` pair or reported as an exclusion, so ``n`` plus the
    number of exclusions always equals the number of metrics in scope.

    A metric is excluded, in this order, when it has no verdict
    (``"no-verdict"``), when its verdict is unstable (``"unstable"``, decided
    before the ratio so an unstable verdict with a NaN delta is still reported
    unstable), or when its ratio is degenerate (``"undefined-ratio"`` for a NaN
    delta, ``"infinite-rho"`` for a non-positive or non-finite ratio).

    Args:
        verdicts: Per-metric verdicts keyed by metric name.
        metric_meta: The metrics to average, keyed by name, in the order they
            should be considered.

    Returns:
        A :class:`GeomeanResult` carrying the combined value, the count of
        included metrics, the propagated noise band, and every exclusion.
    """
    entries: list[tuple[float, float]] = []
    exclusions: list[Exclusion] = []

    for name, meta in metric_meta.items():
        verdict = verdicts.get(name)
        if verdict is None:
            exclusions.append(Exclusion(metric=name, reason="no-verdict"))
            continue
        if verdict.verdict == "unstable":
            exclusions.append(Exclusion(metric=name, reason="unstable"))
            continue

        rho = normalize_ratio(verdict.delta, meta.direction)
        if isinstance(rho, str):
            exclusions.append(Exclusion(metric=name, reason=rho))
            continue

        noise_pct = 0.0 if verdict.method == "exact" else verdict.noise_pct
        entries.append((rho, noise_pct))

    value, band = combine_geomean(entries)
    return GeomeanResult(value=value, n=len(entries), band=band, excluded=tuple(exclusions))


# ---------------------------------------------------------------------------
# Hierarchical aggregation by kind and group
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class GroupAggregate:
    """The geomean over one group of a kind's metrics.

    Attributes:
        group: Path prefix the group's metrics share — the name minus its last
            segment.
        geomean: Geomean over the group's metrics, gating and non-gating alike.
    """

    group: str
    geomean: GeomeanResult


@dataclass(frozen=True, slots=True)
class KindAggregate:
    """One kind's aggregation for a single candidate.

    ``geomean`` covers every metric of the kind and ``gated_geomean`` only the
    gating ones, so a report can show what the whole section did next to what the
    run is judged on. The two coincide when every metric of the kind gates.

    Attributes:
        kind: The metric kind these aggregates summarize.
        geomean: Over every metric of the kind, gating and non-gating alike.
        groups: One entry per group the kind's metric paths name, empty when
            every path has a single segment.
        gated_geomean: Over the kind's gating metrics alone, ``None`` when the
            kind has none.
    """

    kind: str
    geomean: GeomeanResult
    groups: tuple[GroupAggregate, ...]
    gated_geomean: GeomeanResult | None = None


@dataclass(slots=True)
class _KindBucket:
    """A kind's metrics, and the groups their short names sort them into."""

    metrics: dict[str, ResolvedMetricMeta]
    groups: dict[str, dict[str, ResolvedMetricMeta]]


def _bucket_by_kind(
    metric_meta: Mapping[str, ResolvedMetricMeta],
) -> dict[str, _KindBucket]:
    buckets: dict[str, _KindBucket] = {}

    for name, meta in metric_meta.items():
        bucket = buckets.setdefault(meta.kind, _KindBucket(metrics={}, groups={}))
        bucket.metrics[name] = meta

        group = parse_metric_name(name).group
        if group is not None:
            bucket.groups.setdefault(group, {})[name] = meta

    return buckets


def compute_kind_aggregates(
    verdicts: Mapping[str, MetricVerdict],
    metric_meta: Mapping[str, ResolvedMetricMeta],
) -> list[KindAggregate]:
    """Aggregate one candidate's verdicts into a geomean per kind, group, and gating subset.

    Kinds, groups, and the metrics inside them keep the order ``metric_meta``
    lists them in, which is the order the run first reported each metric — so a
    report drawn from these aggregates reads in the same order as the metric
    table.

    Grouping is decided per kind: a group exists only where a name's path has
    more than one segment, and a kind of single-segment names has no groups at
    all rather than one group per metric. Inside a kind that does have groups, a
    single-segment name joins none of them, yet still counts toward the kind.

    Every geomean here is a plain ``compute_geomean`` call over a chosen subset,
    so the unstable, undefined-ratio and infinite-rho exclusions apply throughout,
    each reported against the subset it was excluded from.

    Args:
        verdicts: The candidate's verdicts, keyed by metric name.
        metric_meta: Metadata for every metric of the run, in first-appearance
            order.

    Returns:
        One :class:`KindAggregate` per kind, in first-appearance order.
    """
    aggregates: list[KindAggregate] = []

    for kind, bucket in _bucket_by_kind(metric_meta).items():
        gating = {name: meta for name, meta in bucket.metrics.items() if meta.gating}
        aggregates.append(
            KindAggregate(
                kind=kind,
                geomean=compute_geomean(verdicts, bucket.metrics),
                groups=tuple(
                    GroupAggregate(group=group, geomean=compute_geomean(verdicts, members))
                    for group, members in bucket.groups.items()
                ),
                gated_geomean=compute_geomean(verdicts, gating) if gating else None,
            ),
        )

    return aggregates
