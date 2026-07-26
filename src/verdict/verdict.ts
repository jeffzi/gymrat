/**
 * Verdict engine core: pairing, delta computation, and verdict determination.
 */

import { wilcoxonSignedRank } from "./wilcoxon.js";

/**
 * Tri-state verdict on a metric based on statistical analysis.
 */
export type Verdict = "improved" | "regressed" | "no-signal";

/**
 * Statistical thresholds and parameters.
 */
const P_VALUE_THRESHOLD = 0.05;
const MIN_WILCOXON_N = 6;

/**
 * Fields every verdict carries, whichever method produced it.
 */
type VerdictBase = {
  verdict: Verdict;
  delta: number;
  n: number;
};

/** A verdict from the Wilcoxon signed-rank test, which always yields a p-value. */
export type SignedRankVerdict = VerdictBase & { method: "signed-rank"; p: number };

/** A verdict from the noise-band method, which always yields a band width. */
export type BandVerdict = VerdictBase & { method: "band"; band: number };

/** A verdict from the exact path, where any difference at all is signal. */
export type ExactVerdict = VerdictBase & { method: "exact" };

/**
 * Result of verdict analysis for a single metric.
 *
 * Discriminated on `method` so the method-specific statistic is required
 * exactly where it exists: reading `p` off a band verdict, or `band` off an
 * exact one, is a compile error rather than a runtime `undefined`.
 */
export type MetricVerdict = SignedRankVerdict | BandVerdict | ExactVerdict;

/**
 * Statistical method used to compute the verdict.
 */
export type Method = MetricVerdict["method"];

/**
 * Metadata describing how to analyze a metric.
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
 * Result of geomean aggregation across gating metrics.
 */
export type GeomeanResult = {
  /** Geomean expressed as percentage change: (geomean(ρ) − 1) × 100 */
  value: number;
  /** Number of gating metrics included in the geomean */
  n: number;
  /** Names of gating metrics excluded (NaN delta or non-positive ρ) */
  excluded: string[];
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
 *
 * Direction-awareness:
 * - The engine interprets improved/regressed using the direction flag
 * - direction="lower": negative delta = improved, positive delta = regressed
 * - direction="higher": positive delta = improved, negative delta = regressed
 *
 * @param samplesA Array of metric maps from the first sample set (e.g., baseline)
 * @param samplesB Array of metric maps from the second sample set (e.g., candidate)
 * @param metricMeta Metadata per metric name
 * @returns Record mapping metric names to verdicts (only metrics with verdicts)
 */
// fallow-ignore-next-line complexity
export function computeVerdicts(
  samplesA: ReadonlyArray<Record<string, number>>,
  samplesB: ReadonlyArray<Record<string, number>>,
  metricMeta: Record<string, MetricMetadata>,
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

    // Skip metrics with no paired windows (both arrays grow together, check one)
    if (pairedA.length === 0) {
      continue;
    }

    const meta = metricMeta[metric]!;

    const medianA = computeMedian(pairedA);
    const medianB = computeMedian(pairedB);

    // Compute delta: 100 × (B − A) / A
    // When medianA is 0: if medianB is also 0, delta is 0 (no change); else undefined
    const delta =
      medianA === 0 ? (medianB === 0 ? 0 : Number.NaN) : ((medianB - medianA) / medianA) * 100;

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
      // Non-exact path: try signed-rank, fall back to band
      if (pairedA.length >= MIN_WILCOXON_N) {
        const pairs: Array<readonly [number, number]> = [];
        for (let i = 0; i < pairedA.length; i++) {
          const a = pairedA[i]!;
          const b = pairedB[i]!;
          pairs.push([a, b]);
        }

        const wilcoxonResult = wilcoxonSignedRank(pairs);

        // If effective n < threshold after dropping zeros, fall back to band
        if (wilcoxonResult.n < MIN_WILCOXON_N) {
          const bandVerdict = computeBandMethod(pairedA, pairedB, delta, meta.direction);
          result[metric] = {
            ...bandVerdict,
            n: pairedA.length,
          };
        } else {
          let verdict: Verdict;
          if (wilcoxonResult.p < P_VALUE_THRESHOLD) {
            verdict = determineVerdict(delta, meta.direction);
          } else {
            verdict = "no-signal";
          }

          result[metric] = {
            verdict,
            method: "signed-rank",
            delta,
            n: pairedA.length,
            p: wilcoxonResult.p,
          };
        }
      } else {
        // n < 6: use band method
        const bandVerdict = computeBandMethod(pairedA, pairedB, delta, meta.direction);
        result[metric] = {
          ...bandVerdict,
          n: pairedA.length,
        };
      }
    }
  }

  return result;
}

/**
 * Compute the median of a numeric array.
 *
 * @param values Non-empty array of numbers
 * @returns Median value
 */
function computeMedian(values: readonly number[]): number {
  const sorted = values.toSorted((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);

  if (sorted.length % 2 === 1) {
    return sorted[mid]!;
  }
  return (sorted[mid - 1]! + sorted[mid]!) / 2;
}

/**
 * Compute the "half-range" (half the spread from min to max) of a numeric array.
 *
 * @param values Array of numbers
 * @returns (max - min) / 2
 */
function computeHalfRange(values: readonly number[]): number {
  if (values.length === 0) return 0;
  let min = values[0]!;
  let max = values[0]!;
  for (let i = 1; i < values.length; i++) {
    const val = values[i]!;
    if (val < min) min = val;
    if (val > max) max = val;
  }
  return (max - min) / 2;
}

/**
 * Compute verdict using the noise band method.
 *
 * Band formula: band% = max(K × 100 × max(halfRange(A)/median(A), halfRange(B)/median(B)), floor%)
 * where K = 1.5 and floor = 0.5%.
 *
 * When median is 0, halfRange/median is undefined; treat the spread contribution as 0.
 * Signal when |delta%| > band%; no-signal when |delta%| ≤ band%.
 *
 * @param pairedA Array of values from sample A
 * @param pairedB Array of values from sample B
 * @param delta Delta percentage already computed
 * @param direction "lower" or "higher"
 * @returns Verdict object with verdict, method, delta, and band fields
 */
function computeBandMethod(
  pairedA: readonly number[],
  pairedB: readonly number[],
  delta: number,
  direction: "lower" | "higher",
): Omit<BandVerdict, "n"> {
  const K = 1.5;
  const FLOOR = 0.5;

  const medianA = computeMedian(pairedA);
  const medianB = computeMedian(pairedB);
  const halfRangeA = computeHalfRange(pairedA);
  const halfRangeB = computeHalfRange(pairedB);

  // Compute spread contributions, handling division by zero (treat as 0)
  const spreadA = medianA === 0 ? 0 : halfRangeA / medianA;
  const spreadB = medianB === 0 ? 0 : halfRangeB / medianB;
  const maxSpread = Math.max(spreadA, spreadB);

  const band = Math.max(K * 100 * maxSpread, FLOOR);

  const absDelta = Math.abs(delta);
  let verdict: Verdict;

  if (absDelta > band) {
    verdict = determineVerdict(delta, direction);
  } else {
    verdict = "no-signal";
  }

  return {
    verdict,
    method: "band",
    delta,
    band,
  };
}

/**
 * Compute geometric mean of direction-normalized ratios across gating metrics.
 *
 * Direction-normalized ratio ρ:
 * - For direction="lower": ρ = 1 + delta/100 (ρ < 1 means improvement)
 * - For direction="higher": ρ = 1 / (1 + delta/100) (ρ < 1 means improvement)
 *
 * Exclusion logic:
 * - Only gating metrics participate
 * - Metrics with NaN delta are excluded (medianA was zero, medianB non-zero)
 * - Metrics where ρ ≤ 0 are excluded (medianA or medianB is non-positive)
 *
 * Return value:
 * - value = (geomean(ρ) − 1) × 100 (percentage change)
 * - n = count of included metrics
 * - excluded = names of excluded gating metrics
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
  const included: number[] = [];
  const excluded: string[] = [];

  for (const [metric, meta] of Object.entries(metricMeta)) {
    if (!meta.gating) {
      continue;
    }

    const verdict = verdicts[metric];
    if (!verdict) {
      continue;
    }

    const delta = verdict.delta;

    if (Number.isNaN(delta)) {
      excluded.push(metric);
      continue;
    }

    // Direction-normalized ratio ρ: below 1 always means improvement, whichever
    // direction the metric favors.
    let rho: number;
    if (meta.direction === "lower") {
      rho = 1 + delta / 100;
    } else {
      rho = 1 / (1 + delta / 100);
    }

    if (rho <= 0) {
      excluded.push(metric);
      continue;
    }

    included.push(rho);
  }

  if (included.length === 0) {
    return {
      value: 0,
      n: 0,
      excluded,
    };
  }

  // Geometric mean (ρ₁ × ρ₂ × … × ρₙ)^(1/n) computed in log space —
  // exp(mean(ln(ρᵢ))) — so a long metric list cannot overflow the product.
  let sumLnRho = 0;
  for (let i = 0; i < included.length; i++) {
    sumLnRho += Math.log(included[i]!);
  }
  const meanLnRho = sumLnRho / included.length;
  const geomean = Math.exp(meanLnRho);

  return {
    value: (geomean - 1) * 100,
    n: included.length,
    excluded,
  };
}
