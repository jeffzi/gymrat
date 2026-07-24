/**
 * Verdict engine core: pairing, delta computation, and verdict determination.
 */

/**
 * Tri-state verdict on a metric based on statistical analysis.
 */
export type Verdict = "improved" | "regressed" | "no-signal";

/**
 * Statistical method used to compute the verdict.
 */
export type Method = "signed-rank" | "band" | "exact";

/**
 * Result of verdict analysis for a single metric.
 *
 * Always includes verdict, method, delta, and n.
 * p is only present when method is "signed-rank".
 * band is only present when method is "band".
 */
export type MetricVerdict = {
  verdict: Verdict;
  method: Method;
  delta: number;
  n: number;
  p?: number;
  band?: number;
};

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

  // Collect all metric names
  const allMetrics = new Set(Object.keys(metricMeta));

  for (const metric of allMetrics) {
    // Pair samples by index, collecting values where both exist
    const pairedA: number[] = [];
    const pairedB: number[] = [];

    for (let i = 0; i < samplesA.length && i < samplesB.length; i++) {
      const valA = samplesA[i]?.[metric];
      const valB = samplesB[i]?.[metric];

      // Include only if both values exist for this window
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

    // Compute medians
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
      } else {
        // Determine improved/regressed based on direction and sign of delta
        if (meta.direction === "lower") {
          // For lower-is-better: negative delta (decrease) = improved
          verdict = delta < 0 ? "improved" : "regressed";
        } else {
          // For higher-is-better: positive delta (increase) = improved
          verdict = delta > 0 ? "improved" : "regressed";
        }
      }

      result[metric] = {
        verdict,
        method: "exact",
        delta,
        n: pairedA.length,
      };
    }
    // TODO: signed-rank and band methods not yet implemented
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
