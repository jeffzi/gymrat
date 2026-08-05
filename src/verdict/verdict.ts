/**
 * Verdict engine core: pairing, delta computation, and verdict determination.
 */

import { computeHalfRange, computeMedian } from "../math.js";
import { metricRecord } from "../metric-record.js";
import { wilcoxonSignedRank } from "./wilcoxon.js";

/**
 * Verdict values every method can reach.
 *
 * Which of `improved` / `regressed` a delta earns is decided by the metric's
 * direction, not by the raw sign of the delta — a rising `ops/sec` improves
 * where a rising `duration` regresses.
 */
export type Verdict = "improved" | "regressed" | "no-signal";

/**
 * Verdict values only the approximate methods can reach.
 *
 * "unstable" means the metric's own noise band is so wide that no delta measured
 * against it carries information — not that the delta was small. The exact
 * method has no noise band, so `ExactVerdict` keeps the narrower `Verdict` and
 * an exact metric can never be reported unstable.
 */
export type ApproximateVerdictValue = Verdict | "unstable";

const P_VALUE_THRESHOLD = 0.05;
/**
 * Minimum paired-sample count required to use the signed-rank method; below
 * this the verdict falls back to the noise band.
 */
export const MIN_WILCOXON_N = 6;

/** Multiplier applied to the observed spread when sizing the noise band. */
const NOISE_K = 1.5;

/** Lower bound on the noise band, in percent, for very stable metrics. */
const NOISE_FLOOR_PCT = 0.5;

/**
 * Minimum paired-sample count the band method needs to report a signal.
 *
 * A single pair has no observable spread, so the band collapses to
 * `NOISE_FLOOR_PCT` — a bound meant for metrics measured many times, not for one
 * observation whose spread is simply unknown. Two pairs already yield a real
 * half-range, so the guard stops there.
 */
export const MIN_BAND_N = 2;

/**
 * Noise band width, in percent, above which a metric is reported "unstable".
 *
 * A band this wide spans a factor of three around the median, so the sign of a
 * delta measured against it is not evidence of anything.
 */
export const DEFAULT_UNSTABLE_NOISE_PCT = 200;

/**
 * Fields every verdict carries, whichever method produced it.
 *
 * `verdict` is deliberately absent: its value set is wider for the approximate
 * methods than for the exact one, so each variant declares its own.
 */
type VerdictBase = {
  delta: number;
  n: number;
};

/**
 * Fields every non-exact verdict carries.
 *
 * `noisePct` is the measurement noise of the metric, expressed as a percentage
 * of its median; `noiseAbs` is that same noise in the metric's own unit — see
 * `computeNoise`. Both are reported whichever non-exact method decided the
 * verdict, so a caller can compare a delta against the noise floor without
 * knowing which test ran.
 */
type ApproximateVerdictBase = VerdictBase & {
  verdict: ApproximateVerdictValue;
  noisePct: number;
  noiseAbs: number;
};

/** A verdict from the Wilcoxon signed-rank test, which always yields a p-value. */
export type SignedRankVerdict = ApproximateVerdictBase & { method: "signed-rank"; p: number };

/**
 * A verdict from the noise-band method, which always yields a band width.
 *
 * `usableN` is how many of the `n` pairs differed by a non-zero amount, which is
 * what the signed-rank test would have had to work with. It separates the two
 * causes of a band fallback: `usableN < n` means tied pairs starved the test,
 * while `usableN === n` below `MIN_WILCOXON_N` means the run was simply short.
 */
export type BandVerdict = ApproximateVerdictBase & {
  method: "band";
  band: number;
  usableN: number;
};

/** A verdict from the exact path, where any difference at all is signal. */
export type ExactVerdict = VerdictBase & { verdict: Verdict; method: "exact" };

/**
 * Result of verdict analysis for a single metric.
 *
 * Discriminated on `method` so the method-specific statistic is required
 * exactly where it exists: reading `p` off a band verdict, or `band` off an
 * exact one, is a compile error rather than a runtime `undefined`.
 */
export type MetricVerdict = SignedRankVerdict | BandVerdict | ExactVerdict;

/**
 * The discriminant of {@link MetricVerdict}, derived rather than restated so a
 * new method variant widens it automatically.
 */
export type Method = MetricVerdict["method"];

/**
 * Unit a metric's values carry, when the adapter could tell.
 *
 * `"bytes"` marks a metric quantized to whole bytes, whose noise can never be
 * finer than one byte; `"ns"` marks an averaged duration, which carries no such
 * bound.
 */
type MetricUnit = "ns" | "bytes";

/**
 * Settled per-metric metadata the engine consumes, produced by `resolveMetricMeta`
 * in `config.ts` from adapter defaults merged with the user's config overrides.
 */
export type MetricMetadata = {
  /** "lower" = smaller values are better; "higher" = larger values are better */
  direction: "lower" | "higher";
  /** Whether this metric participates in geomean aggregation */
  gating: boolean;
  /** Whether to use the exact path (any difference = signal) */
  exact: boolean;
  /** Unit of the metric's values; absent when the adapter could not tell */
  unit?: MetricUnit;
};

/**
 * Why a gating metric was left out of the geomean.
 *
 * - `"no-verdict"` — only one target reported the metric, so there are no paired
 *   samples to compare and no verdict was ever produced
 * - `"unstable"` — the metric's verdict is `"unstable"`, so it is unjudgeable
 *   and cannot be allowed to move the headline number
 * - `"undefined-ratio"` — the delta is NaN (median A was zero, median B was
 *   not), so ρ is undefined
 * - `"infinite-rho"` — ρ is non-positive or non-finite, so its logarithm is not
 *   a real number
 */
export type GeomeanExclusionReason = "no-verdict" | "unstable" | "undefined-ratio" | "infinite-rho";

/** A gating metric the geomean skipped, paired with the reason it was skipped. */
export type GeomeanExclusion = {
  metric: string;
  reason: GeomeanExclusionReason;
};

/**
 * Result of geomean aggregation across gating metrics.
 */
export type GeomeanResult = {
  /** Geomean expressed as percentage change: (geomean(ρ) − 1) × 100 */
  value: number;
  /** Number of gating metrics included in the geomean */
  n: number;
  /** Metrics excluded, each with its reason */
  excluded: GeomeanExclusion[];
  /**
   * Noise of the included metrics propagated onto `value`, in percentage points:
   * √(Σ noisePctᵢ²) ÷ n.
   *
   * Each metric's noise enters the geomean divided by n, so the independent
   * contributions add in quadrature and shrink as more metrics join. Exact
   * metrics carry no noise and contribute nothing; with nothing included the
   * band is 0.
   */
  band: number;
};

/**
 * A NaN delta (the ratio is undefined, because the baseline median is 0) has no
 * direction to read, so it reports no signal rather than falling through to
 * "regressed" — every comparison against NaN is false.
 */
function determineVerdict(delta: number, direction: "lower" | "higher"): Verdict {
  if (Number.isNaN(delta)) return "no-signal";

  const isImproved = direction === "lower" ? delta < 0 : delta > 0;
  return isImproved ? "improved" : "regressed";
}

/**
 * Pair sample windows by index for a single metric, dropping windows where
 * either side is missing it.
 *
 * Exported so callers displaying a metric's median/spread (e.g. the report
 * builder in `compare.ts`) can draw on the same paired windows this module
 * uses to compute a verdict's delta — otherwise a displayed median computed
 * over every sample disagrees with a delta computed over paired samples only,
 * whenever a metric is missing in some rounds.
 *
 * @returns Paired values, one array per side, growing together
 */
export function pairSamples(
  metric: string,
  samplesA: ReadonlyArray<Record<string, number>>,
  samplesB: ReadonlyArray<Record<string, number>>,
): { pairedA: number[]; pairedB: number[] } {
  const pairedA: number[] = [];
  const pairedB: number[] = [];

  for (let i = 0; i < samplesA.length && i < samplesB.length; i++) {
    const valA = samplesA[i]?.[metric];
    const valB = samplesB[i]?.[metric];

    if (valA !== undefined && valB !== undefined) {
      pairedA.push(valA);
      pairedB.push(valB);
    }
  }

  return { pairedA, pairedB };
}

/**
 * Compute the exact-path verdict: any difference between medians is a signal.
 */
function computeExactVerdict(
  medianA: number,
  medianB: number,
  delta: number,
  direction: "lower" | "higher",
  n: number,
): ExactVerdict {
  // Equal medians lack a signal direction; an undefined ratio is caught by
  // `determineVerdict`.
  const verdict: Verdict = medianA === medianB ? "no-signal" : determineVerdict(delta, direction);

  return { verdict, method: "exact", delta, n };
}

/**
 * Percentage delta between two medians, normalized by the magnitude of `medianA`.
 *
 * Normalizing by the magnitude keeps the sign of the delta tied to the
 * direction the value moved: a negative-median metric dropping further below
 * zero is a decrease, not an increase. When `medianA` is 0, the ratio is
 * undefined; the result is 0 if both medians are 0, and `NaN` otherwise.
 */
function computeDelta(medianA: number, medianB: number): number {
  if (medianA === 0 && medianB === 0) {
    return 0;
  }
  if (medianA === 0) {
    return Number.NaN;
  }
  return ((medianB - medianA) / Math.abs(medianA)) * 100;
}

/**
 * Compute verdicts for multiple metrics across two sample sets.
 *
 * Pairing:
 * - Samples are paired by index: samplesA[i] with samplesB[i]
 * - Windows where either side is missing the metric are dropped
 * - Metrics present on only one side across all windows are skipped
 *
 * Delta:
 * - delta% = 100 × (median(B) − median(A)) / |median(A)|
 * - Computed from per-run medians of paired windows
 * - Normalized by the magnitude of median(A), so the sign always follows the
 *   direction the value moved, even for a metric whose medians are negative
 * - Always reported, even for "no-signal" verdicts
 *
 * Exact path:
 * - For exact-flagged metrics: any difference between medians is a signal
 * - Verdict determined by direction: lower delta + direction="lower" = improved
 * - Method is "exact"; single sample (n=1) suffices
 * - Equal medians = "no-signal"
 * - Zero median A against a non-zero median B is also "no-signal": the ratio is
 *   undefined, so there is no direction to report
 *
 * Direction-awareness:
 * - The engine interprets improved/regressed using the direction flag
 * - direction="lower": negative delta = improved, positive delta = regressed
 * - direction="higher": positive delta = improved, negative delta = regressed
 *
 * Unstable metrics:
 * - A non-exact metric whose noisePct exceeds `unstableNoisePct` is "unstable"
 * - The override is unconditional: a delta that clears the band still says
 *   nothing, because the band itself is too wide to measure against
 * - Exact metrics have no noise band and are never unstable
 *
 * @param samplesA Array of metric maps from the first sample set (e.g., baseline)
 * @param samplesB Array of metric maps from the second sample set (e.g., candidate)
 * @param unstableNoisePct Noise band width, in percent, above which a metric is
 *   unstable. Compared strictly, so a metric sitting exactly on the threshold
 *   still gets a normal verdict.
 * @returns Record mapping metric names to verdicts (only metrics with verdicts)
 */
// Pairing, delta, and method dispatch are one pass over each metric; splitting
// them would mean re-walking the samples.
// fallow-ignore-next-line complexity
export function computeVerdicts(
  samplesA: ReadonlyArray<Record<string, number>>,
  samplesB: ReadonlyArray<Record<string, number>>,
  metricMeta: Record<string, MetricMetadata>,
  unstableNoisePct: number = DEFAULT_UNSTABLE_NOISE_PCT,
): Record<string, MetricVerdict> {
  const result = metricRecord<MetricVerdict>();

  for (const [metric, meta] of Object.entries(metricMeta)) {
    const { pairedA, pairedB } = pairSamples(metric, samplesA, samplesB);

    // Both paired arrays grow together, so one length check covers both.
    if (pairedA.length === 0) {
      continue;
    }

    const medianA = computeMedian(pairedA);
    const medianB = computeMedian(pairedB);
    const delta = computeDelta(medianA, medianB);

    result[metric] = meta.exact
      ? computeExactVerdict(medianA, medianB, delta, meta.direction, pairedA.length)
      : computeApproximateVerdict(
          pairedA,
          pairedB,
          delta,
          meta.direction,
          unstableNoisePct,
          meta.unit,
        );
  }

  return result;
}

/**
 * Compute a verdict for the non-exact path: signed-rank when there are at least
 * `MIN_WILCOXON_N` pairs *and* at least that many with a non-zero difference;
 * the noise-band method otherwise.
 *
 * Tied pairs carry no rank information, so `wilcoxonSignedRank` drops them — a
 * long but mostly identical run falls back to the band just as a short one does.
 */
function computeApproximateVerdict(
  pairedA: readonly number[],
  pairedB: readonly number[],
  delta: number,
  direction: "lower" | "higher",
  unstableNoisePct: number,
  unit: MetricUnit | undefined,
): SignedRankVerdict | BandVerdict {
  const wilcoxonResult = wilcoxonSignedRank(pairedA, pairedB);

  let result: SignedRankVerdict | BandVerdict;
  if (wilcoxonResult.n < MIN_WILCOXON_N) {
    result = computeBandMethod(pairedA, pairedB, delta, direction, wilcoxonResult.n, unit);
  } else {
    const verdict: Verdict =
      wilcoxonResult.p < P_VALUE_THRESHOLD ? determineVerdict(delta, direction) : "no-signal";

    const noise = computeNoise(pairedA, pairedB, unit);
    result = {
      verdict,
      method: "signed-rank",
      delta,
      n: pairedA.length,
      p: wilcoxonResult.p,
      noisePct: noise.pct,
      noiseAbs: noise.abs,
    };
  }

  return applyUnstableOverride(result, unstableNoisePct);
}

/**
 * Replace a method's verdict with "unstable" when the metric's noise band is too
 * wide to judge against.
 *
 * Applied after the method has had its say, because the override does not depend
 * on what the method concluded: signal or no signal, a band wider than the
 * threshold makes the call meaningless. The comparison is strict, so a metric
 * sitting exactly on the threshold keeps its verdict.
 *
 * @param verdict Verdict produced by the signed-rank or band method
 * @param unstableNoisePct Noise band width above which the verdict is "unstable"
 * @returns The verdict unchanged, or the same record marked "unstable"
 */
function applyUnstableOverride(
  verdict: SignedRankVerdict | BandVerdict,
  unstableNoisePct: number,
): SignedRankVerdict | BandVerdict {
  if (verdict.noisePct <= unstableNoisePct) return verdict;
  return { ...verdict, verdict: "unstable" };
}

/** The measurement noise of a metric, in both the forms a report can show. */
type Noise = {
  /** Noise as a percentage of the metric's median, never below the floor. */
  pct: number;
  /** The same noise in the metric's own unit, with no floor applied. */
  abs: number;
};

/**
 * One byte expressed as a percentage of a median, or 0 when there is no
 * magnitude to divide by.
 *
 * Each side contributes its own term, so a side that measured 0 drops out
 * instead of making the whole floor infinite.
 */
function quantizationPct(median: number): number {
  return median === 0 ? 0 : 100 / Math.abs(median);
}

/**
 * A half-range expressed as a fraction of a median's magnitude, or 0 when
 * there is no magnitude to divide by.
 */
function relativeSpread(halfRange: number, median: number): number {
  return median === 0 ? 0 : halfRange / Math.abs(median);
}

/**
 * Compute the measurement noise of a metric.
 *
 * Percentage form: max(K × 100 × max(halfRange(A)/|median(A)|, halfRange(B)/|median(B)|), floor%)
 * where K = 1.5 and floor = 0.5%. Absolute form: K × max(halfRange(A), halfRange(B)).
 *
 * The magnitude of the median is the denominator, so a metric centered on a
 * negative value gets the same spread as its positive mirror image.
 *
 * When a median is 0, halfRange/median is undefined; that side's spread
 * contribution to the percentage is treated as 0. The absolute form divides by
 * nothing, so it stays meaningful for such a metric.
 *
 * A byte-valued metric takes a further percentage floor of one byte against each
 * median. Such a metric is quantized to whole bytes, so a 4B → 3B move is one
 * step of resolution rather than a 25% win — however many times it is measured,
 * and however tight its spread. K does not apply: the resolution is a hard bound
 * on what the numbers can express, not an estimate of their scatter. Averaged
 * units such as `ns` carry no such bound and keep the plain floor.
 *
 * @returns The noise as a percentage (never below the floor) and in raw units
 */
function computeNoise(
  pairedA: readonly number[],
  pairedB: readonly number[],
  unit: MetricUnit | undefined,
): Noise {
  const medianA = computeMedian(pairedA);
  const medianB = computeMedian(pairedB);
  const halfRangeA = computeHalfRange(pairedA);
  const halfRangeB = computeHalfRange(pairedB);

  const spreadA = relativeSpread(halfRangeA, medianA);
  const spreadB = relativeSpread(halfRangeB, medianB);
  const maxSpread = Math.max(spreadA, spreadB);

  const byteFloorPct =
    unit === "bytes" ? Math.max(quantizationPct(medianA), quantizationPct(medianB)) : 0;

  return {
    pct: Math.max(NOISE_K * 100 * maxSpread, NOISE_FLOOR_PCT, byteFloorPct),
    abs: NOISE_K * Math.max(halfRangeA, halfRangeB),
  };
}

/**
 * Compute verdict using the noise band method.
 *
 * The band is the metric's noise percentage (see `computeNoise`). Signal when
 * |delta%| > band%; no-signal when |delta%| ≤ band%, or when fewer than
 * `MIN_BAND_N` pairs make the band meaningless.
 *
 * @param usableN Pairs with a non-zero difference, as counted by `wilcoxonSignedRank`
 * @returns A band verdict whose `band` and `noisePct` hold the same value — the
 *   band a delta is judged against *is* the metric's own noise.
 */
function computeBandMethod(
  pairedA: readonly number[],
  pairedB: readonly number[],
  delta: number,
  direction: "lower" | "higher",
  usableN: number,
  unit: MetricUnit | undefined,
): BandVerdict {
  const noise = computeNoise(pairedA, pairedB, unit);

  const absDelta = Math.abs(delta);
  const hasSignal = pairedA.length >= MIN_BAND_N && absDelta > noise.pct;
  const verdict: Verdict = hasSignal ? determineVerdict(delta, direction) : "no-signal";

  return {
    verdict,
    method: "band",
    delta,
    n: pairedA.length,
    usableN,
    band: noise.pct,
    noisePct: noise.pct,
    noiseAbs: noise.abs,
  };
}

/** Either a usable ρ, or the reason the delta yields none. */
type RhoOutcome =
  | { rho: number }
  | { reason: Exclude<GeomeanExclusionReason, "unstable" | "no-verdict"> };

function computeNormalizedRho(delta: number, direction: "lower" | "higher"): RhoOutcome {
  if (Number.isNaN(delta)) return { reason: "undefined-ratio" };

  const rho = direction === "lower" ? 1 + delta / 100 : 1 / (1 + delta / 100);

  if (rho <= 0 || !Number.isFinite(rho)) return { reason: "infinite-rho" };
  return { rho };
}

/** A metric the geomean kept: its ρ, paired with the noise it brings along. */
type IncludedMetric = { rho: number; noisePct: number };

function collectNormalizedRhos(
  verdicts: Record<string, MetricVerdict>,
  metricMeta: Record<string, MetricMetadata>,
): { included: IncludedMetric[]; excluded: GeomeanExclusion[] } {
  const included: IncludedMetric[] = [];
  const excluded: GeomeanExclusion[] = [];

  for (const [metric, meta] of Object.entries(metricMeta)) {
    const verdict = verdicts[metric];
    if (!verdict) {
      excluded.push({ metric, reason: "no-verdict" });
      continue;
    }

    if (verdict.verdict === "unstable") {
      excluded.push({ metric, reason: "unstable" });
      continue;
    }

    const outcome = computeNormalizedRho(verdict.delta, meta.direction);
    if ("rho" in outcome) {
      included.push({
        rho: outcome.rho,
        noisePct: verdict.method === "exact" ? 0 : verdict.noisePct,
      });
    } else {
      excluded.push({ metric, reason: outcome.reason });
    }
  }

  return { included, excluded };
}

/**
 * Compute geometric mean of direction-normalized ratios across a set of metrics.
 *
 * Which metrics belong in the average is the caller's decision: every metric
 * `metricMeta` names participates, whatever its `gating` flag, and `verdicts`
 * may carry more than that. Restricting `metricMeta` is therefore how a subset
 * is selected — see `computeKindAggregates`, which slices a run by kind, by
 * group, and by gating that way.
 *
 * Direction-normalized ratio ρ:
 * - For direction="lower": ρ = 1 + delta/100 (ρ < 1 means improvement)
 * - For direction="higher": ρ = 1 / (1 + delta/100) (ρ < 1 means improvement)
 *
 * Exclusion logic — each exclusion carries the reason it happened, decided in
 * this order:
 * - Metrics `metricMeta` names that `verdicts` has no entry for are excluded:
 *   only one target reported them, so there was never anything to compare
 * - Metrics whose verdict is "unstable" are excluded, however usable their ρ
 *   would have been: the geomean must agree with the per-row verdicts, so a
 *   metric too noisy to judge cannot move the headline number
 * - Metrics with NaN delta are excluded (medianA was zero, medianB non-zero)
 * - Metrics where ρ ≤ 0 or ρ is non-finite are excluded
 *
 * Return value:
 * - value = (geomean(ρ) − 1) × 100 (percentage change)
 * - n = count of included metrics
 * - excluded = excluded metrics, each with its reason
 * - band = √(Σ noisePctᵢ²) ÷ n over the included metrics, the noise of `value`
 *
 * Edge cases:
 * - No metrics or all excluded: returns { value: 0, n: 0, band: 0, excluded: [...] }
 * - Single metric: geomean = that metric's ρ, band = its own noise
 *
 * @param verdicts Verdict record from computeVerdicts
 * @param metricMeta Metadata for exactly the metrics to average
 */
export function computeGeomean(
  verdicts: Record<string, MetricVerdict>,
  metricMeta: Record<string, MetricMetadata>,
): GeomeanResult {
  const { included, excluded } = collectNormalizedRhos(verdicts, metricMeta);

  if (included.length === 0) {
    return { value: 0, n: 0, excluded, band: 0 };
  }

  let sumLnRho = 0;
  let sumSquaredNoise = 0;
  for (const { rho, noisePct } of included) {
    sumLnRho += Math.log(rho);
    sumSquaredNoise += noisePct * noisePct;
  }
  const geomean = Math.exp(sumLnRho / included.length);

  return {
    value: (geomean - 1) * 100,
    n: included.length,
    excluded,
    band: Math.sqrt(sumSquaredNoise) / included.length,
  };
}
