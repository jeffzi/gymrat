/**
 * Verdict engine core: pairing, delta computation, and verdict determination.
 */

import { computeMedian } from "../math.js";
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
 * of its median — see `computeNoisePct`. It is reported whichever non-exact
 * method decided the verdict, so a caller can compare a delta against the noise
 * floor without knowing which test ran.
 */
type ApproximateVerdictBase = VerdictBase & {
  verdict: ApproximateVerdictValue;
  noisePct: number;
};

/** A verdict from the Wilcoxon signed-rank test, which always yields a p-value. */
export type SignedRankVerdict = ApproximateVerdictBase & { method: "signed-rank"; p: number };

/** A verdict from the noise-band method, which always yields a band width. */
export type BandVerdict = ApproximateVerdictBase & { method: "band"; band: number };

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
};

/**
 * Why a gating metric was left out of the geomean.
 *
 * - `"unstable"` — the metric's verdict is `"unstable"`, so it is unjudgeable
 *   and cannot be allowed to move the headline number
 * - `"undefined-ratio"` — the delta is NaN (median A was zero, median B was
 *   not), so ρ is undefined
 * - `"infinite-rho"` — ρ is non-positive or non-finite, so its logarithm is not
 *   a real number
 */
export type GeomeanExclusionReason = "unstable" | "undefined-ratio" | "infinite-rho";

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
  /** Gating metrics excluded, each with its reason */
  excluded: GeomeanExclusion[];
};

/**
 * Determine verdict based on delta sign and direction.
 *
 * @param delta Delta percentage (may be NaN for undefined ratios)
 * @param direction Whether lower or higher values are better
 * @returns "improved" or "regressed" based on direction and delta sign
 */
function determineVerdict(delta: number, direction: "lower" | "higher"): Verdict {
  const isImproved = direction === "lower" ? delta < 0 : delta > 0;
  return isImproved ? "improved" : "regressed";
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
 * - delta% = 100 × (median(B) − median(A)) / median(A)
 * - Computed from per-run medians of paired windows
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
 * @param metricMeta Metadata per metric name
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
  const result: Record<string, MetricVerdict> = {};

  const allMetrics = new Set(Object.keys(metricMeta));

  for (const metric of allMetrics) {
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

    // Both paired arrays grow together, so one length check covers both.
    if (pairedA.length === 0) {
      continue;
    }

    const meta = metricMeta[metric]!;

    const medianA = computeMedian(pairedA);
    const medianB = computeMedian(pairedB);

    let delta: number;
    if (medianA === 0 && medianB === 0) {
      delta = 0;
    } else if (medianA === 0) {
      delta = Number.NaN;
    } else {
      delta = ((medianB - medianA) / medianA) * 100;
    }

    // Exact path: any difference is a signal
    if (meta.exact) {
      let verdict: Verdict;

      if (medianA === medianB) {
        verdict = "no-signal";
      } else if (Number.isNaN(delta)) {
        // When delta is NaN (medianA=0, medianB≠0), we lack a meaningful signal direction
        verdict = "no-signal";
      } else {
        verdict = determineVerdict(delta, meta.direction);
      }

      result[metric] = {
        verdict,
        method: "exact",
        delta,
        n: pairedA.length,
      };
    } else {
      result[metric] = computeApproximateVerdict(
        pairedA,
        pairedB,
        delta,
        meta.direction,
        unstableNoisePct,
      );
    }
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
 *
 * @param pairedA Array of values from sample A
 * @param pairedB Array of values from sample B
 * @param delta Delta percentage already computed
 * @param direction "lower" or "higher"
 * @param unstableNoisePct Noise band width above which the verdict is "unstable"
 * @returns Verdict from either the signed-rank or band method
 */
function computeApproximateVerdict(
  pairedA: readonly number[],
  pairedB: readonly number[],
  delta: number,
  direction: "lower" | "higher",
  unstableNoisePct: number,
): SignedRankVerdict | BandVerdict {
  const pairs = pairedA.map((a, i): readonly [number, number] => [a, pairedB[i]!]);
  const wilcoxonResult = wilcoxonSignedRank(pairs);

  if (pairedA.length < MIN_WILCOXON_N || wilcoxonResult.n < MIN_WILCOXON_N) {
    return applyUnstableOverride(
      computeBandMethod(pairedA, pairedB, delta, direction),
      unstableNoisePct,
    );
  }

  const verdict: Verdict =
    wilcoxonResult.p < P_VALUE_THRESHOLD ? determineVerdict(delta, direction) : "no-signal";

  return applyUnstableOverride(
    {
      verdict,
      method: "signed-rank",
      delta,
      n: pairedA.length,
      p: wilcoxonResult.p,
      noisePct: computeNoisePct(pairedA, pairedB),
    },
    unstableNoisePct,
  );
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

/**
 * Compute the "half-range" (half the spread from min to max) of a numeric array.
 *
 * @param values Array of numbers
 * @returns (max - min) / 2
 */
function computeHalfRange(values: readonly number[]): number {
  if (values.length === 0) return 0;
  return (Math.max(...values) - Math.min(...values)) / 2;
}

/**
 * Compute the measurement noise of a metric as a percentage of its median.
 *
 * Formula: max(K × 100 × max(halfRange(A)/median(A), halfRange(B)/median(B)), floor%)
 * where K = 1.5 and floor = 0.5%.
 *
 * When a median is 0, halfRange/median is undefined; that side's spread
 * contribution is treated as 0.
 *
 * @param pairedA Array of values from sample A
 * @param pairedB Array of values from sample B
 * @returns Noise as a percentage, never below the floor
 */
function computeNoisePct(pairedA: readonly number[], pairedB: readonly number[]): number {
  const medianA = computeMedian(pairedA);
  const medianB = computeMedian(pairedB);
  const halfRangeA = computeHalfRange(pairedA);
  const halfRangeB = computeHalfRange(pairedB);

  const spreadA = medianA === 0 ? 0 : halfRangeA / medianA;
  const spreadB = medianB === 0 ? 0 : halfRangeB / medianB;
  const maxSpread = Math.max(spreadA, spreadB);

  return Math.max(NOISE_K * 100 * maxSpread, NOISE_FLOOR_PCT);
}

/**
 * Compute verdict using the noise band method.
 *
 * The band is the metric's noise percentage (see `computeNoisePct`). Signal when
 * |delta%| > band%; no-signal when |delta%| ≤ band%.
 *
 * @param pairedA Array of values from sample A
 * @param pairedB Array of values from sample B
 * @param delta Delta percentage already computed
 * @param direction "lower" or "higher"
 * @returns A band verdict whose `band` and `noisePct` hold the same value — the
 *   band a delta is judged against *is* the metric's own noise.
 */
function computeBandMethod(
  pairedA: readonly number[],
  pairedB: readonly number[],
  delta: number,
  direction: "lower" | "higher",
): BandVerdict {
  const band = computeNoisePct(pairedA, pairedB);

  const absDelta = Math.abs(delta);
  const verdict: Verdict = absDelta > band ? determineVerdict(delta, direction) : "no-signal";

  return {
    verdict,
    method: "band",
    delta,
    n: pairedA.length,
    band,
    noisePct: band,
  };
}

/** Either a usable ρ, or the reason the delta yields none. */
type RhoOutcome = { rho: number } | { reason: Exclude<GeomeanExclusionReason, "unstable"> };

function computeNormalizedRho(delta: number, direction: "lower" | "higher"): RhoOutcome {
  if (Number.isNaN(delta)) return { reason: "undefined-ratio" };

  const rho = direction === "lower" ? 1 + delta / 100 : 1 / (1 + delta / 100);

  if (rho <= 0 || !Number.isFinite(rho)) return { reason: "infinite-rho" };
  return { rho };
}

function collectNormalizedRhos(
  verdicts: Record<string, MetricVerdict>,
  metricMeta: Record<string, MetricMetadata>,
): { included: number[]; excluded: GeomeanExclusion[] } {
  const included: number[] = [];
  const excluded: GeomeanExclusion[] = [];

  for (const [metric, meta] of Object.entries(metricMeta)) {
    if (!meta.gating) continue;

    const verdict = verdicts[metric];
    if (!verdict) continue;

    if (verdict.verdict === "unstable") {
      excluded.push({ metric, reason: "unstable" });
      continue;
    }

    const outcome = computeNormalizedRho(verdict.delta, meta.direction);
    if ("rho" in outcome) {
      included.push(outcome.rho);
    } else {
      excluded.push({ metric, reason: outcome.reason });
    }
  }

  return { included, excluded };
}

/**
 * Compute geometric mean of direction-normalized ratios across gating metrics.
 *
 * Direction-normalized ratio ρ:
 * - For direction="lower": ρ = 1 + delta/100 (ρ < 1 means improvement)
 * - For direction="higher": ρ = 1 / (1 + delta/100) (ρ < 1 means improvement)
 *
 * Exclusion logic — only gating metrics participate, and each exclusion carries
 * the reason it happened, decided in this order:
 * - Metrics whose verdict is "unstable" are excluded, however usable their ρ
 *   would have been: the geomean must agree with the per-row verdicts, so a
 *   metric too noisy to judge cannot move the headline number
 * - Metrics with NaN delta are excluded (medianA was zero, medianB non-zero)
 * - Metrics where ρ ≤ 0 or ρ is non-finite are excluded
 *
 * Return value:
 * - value = (geomean(ρ) − 1) × 100 (percentage change)
 * - n = count of included metrics
 * - excluded = excluded gating metrics, each with its reason
 *
 * Edge cases:
 * - No gating metrics or all excluded: returns { value: 0, n: 0, excluded: [...] }
 * - Single gating metric: geomean = that metric's ρ
 *
 * @param verdicts Verdict record from computeVerdicts
 * @param metricMeta Metadata per metric name
 * @returns Geomean result with value, count, and exclusion list
 */
export function computeGeomean(
  verdicts: Record<string, MetricVerdict>,
  metricMeta: Record<string, MetricMetadata>,
): GeomeanResult {
  const { included, excluded } = collectNormalizedRhos(verdicts, metricMeta);

  if (included.length === 0) {
    return { value: 0, n: 0, excluded };
  }

  let sumLnRho = 0;
  for (const rho of included) {
    sumLnRho += Math.log(rho);
  }
  const geomean = Math.exp(sumLnRho / included.length);

  return {
    value: (geomean - 1) * 100,
    n: included.length,
    excluded,
  };
}
