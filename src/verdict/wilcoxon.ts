/**
 * Wilcoxon signed-rank test for paired samples.
 *
 * Exact test using sign enumeration for n ≤ 25.
 * Normal approximation with continuity correction for n > 25.
 *
 * No external dependencies — implements normal CDF via rational approximation.
 */

/**
 * Compute the standard normal cumulative distribution function (CDF) using
 * a rational approximation (Abramowitz and Stegun, 1964).
 * Accurate to ~7 decimal places.
 *
 * @param z - Standard normal deviate
 * @returns P(Z ≤ z)
 */
function normalCDF(z: number): number {
  // Constants for rational approximation
  const a1 = 0.254829592;
  const a2 = -0.284496736;
  const a3 = 1.421413741;
  const a4 = -1.453152027;
  const a5 = 1.061405429;
  const p = 0.3275911;

  const sign = z < 0 ? -1 : 1;
  const abz = Math.abs(z);

  const t = 1.0 / (1.0 + p * abz);
  const t2 = t * t;
  const t3 = t2 * t;
  const t4 = t3 * t;
  const t5 = t4 * t;

  const phi = 1.0 - (a1 * t + a2 * t2 + a3 * t3 + a4 * t4 + a5 * t5) * Math.exp(-abz * abz);

  const cdf = 0.5 * (1.0 + sign * (phi - 0.5) * 2.0);
  return Math.max(0, Math.min(1, cdf));
}

interface RankedEntry {
  rank: number;
  sign: number;
}

/**
 * Compute ranks for absolute differences with tie handling, paired with signs.
 *
 * @param entries - Array of {absDiff, sign} entries (zeros already removed)
 * @returns Array of {rank, sign} entries with average-rank tie handling
 */
function rankEntries(entries: Array<{ absDiff: number; sign: number }>): RankedEntry[] {
  const sorted = entries
    .map((e, i) => ({ absDiff: e.absDiff, sign: e.sign, origIdx: i }))
    .toSorted((a, b) => a.absDiff - b.absDiff);

  const result = Array.from<RankedEntry>({ length: sorted.length });

  let i = 0;
  while (i < sorted.length) {
    const current = sorted[i]!;
    let j = i;
    while (j < sorted.length && sorted[j]!.absDiff === current.absDiff) {
      j++;
    }
    // Average rank for this tie group
    const avgRank = (i + 1 + j) / 2;
    for (let k = i; k < j; k++) {
      result[sorted[k]!.origIdx] = { rank: avgRank, sign: sorted[k]!.sign };
    }
    i = j;
  }

  return result;
}

/**
 * Compute exact p-value using sign enumeration (2^n permutations).
 * Enumerate all possible sign assignments and count how many yield T ≥ observed T+.
 *
 * @param ranked - Array of ranked entries with signs
 * @returns Two-sided p-value
 */
function exactPValue(ranked: RankedEntry[]): number {
  const n = ranked.length;

  let observedTPlus = 0;
  for (const entry of ranked) {
    if (entry.sign > 0) observedTPlus += entry.rank;
  }

  // Total sum of ranks is n(n+1)/2
  const maxT = (n * (n + 1)) / 2;

  // Enumerate all 2^n sign assignments
  let countExtreme = 0;
  for (let mask = 0; mask < 1 << n; mask++) {
    let t = 0;
    for (let i = 0; i < n; i++) {
      if ((mask & (1 << i)) !== 0) {
        t += ranked[i]!.rank;
      }
    }
    // Two-sided: count min(T, maxT - T) ≤ min(T+, maxT - T+)
    const minObserved = Math.min(observedTPlus, maxT - observedTPlus);
    const minT = Math.min(t, maxT - t);
    if (minT <= minObserved) {
      countExtreme++;
    }
  }

  const pValue = countExtreme / (1 << n);
  return Math.min(1, pValue);
}

/**
 * Compute p-value using normal approximation with continuity correction.
 *
 * @param ranked - Array of ranked entries with signs
 * @returns Two-sided p-value
 */
function approximatePValue(ranked: RankedEntry[]): number {
  const n = ranked.length;

  let tPlus = 0;
  for (const entry of ranked) {
    if (entry.sign > 0) tPlus += entry.rank;
  }

  // Mean under null
  const mu = (n * (n + 1)) / 4;

  // Variance with tie correction
  // σ² = n(n+1)(2n+1)/24 − Σ(t³−t)/48
  let variance = (n * (n + 1) * (2 * n + 1)) / 24;

  // Compute tie correction
  const tieGroups = new Map<number, number>();
  for (const entry of ranked) {
    tieGroups.set(entry.rank, (tieGroups.get(entry.rank) ?? 0) + 1);
  }
  let tieCorrection = 0;
  for (const groupSize of tieGroups.values()) {
    if (groupSize > 1) {
      tieCorrection += (groupSize * groupSize * groupSize - groupSize) / 48;
    }
  }
  variance -= tieCorrection;

  const sigma = Math.sqrt(variance);

  // Continuity correction: z = (T+ - 0.5 - μ) / σ
  const z = (tPlus - mu - 0.5) / sigma;

  // Two-sided p-value: p = 2 * Φ(−|z|)
  const tailProb = normalCDF(-Math.abs(z));
  const pValue = 2 * tailProb;

  return Math.min(1, pValue);
}

export interface WilcoxonResult {
  p: number;
  n: number;
}

/**
 * Wilcoxon signed-rank test.
 *
 * Performs an exact or approximate two-sided Wilcoxon signed-rank test for paired samples.
 *
 * @param pairs - Array of [a, b] pairs, where each pair is a sample from two conditions
 * @returns Object with p-value and effective sample size (n, after dropping zeros)
 *
 * @example
 * const result = wilcoxonSignedRank([[1, 2], [2, 4], [3, 5]]);
 * console.log(result); // { p: 0.25, n: 3 }
 */
export function wilcoxonSignedRank(
  pairs: ReadonlyArray<readonly [number, number]>,
): WilcoxonResult {
  if (pairs.length === 0) {
    return { p: 1, n: 0 };
  }

  // Compute differences and filter out zeros
  const entries: Array<{ absDiff: number; sign: number }> = [];
  for (const [a, b] of pairs) {
    const diff = b - a;
    if (diff !== 0) {
      entries.push({ absDiff: Math.abs(diff), sign: Math.sign(diff) });
    }
  }

  const n = entries.length;

  // All zeros
  if (n === 0) {
    return { p: 1, n: 0 };
  }

  // Rank absolute differences with tie handling
  const ranked = rankEntries(entries);

  // Compute p-value
  let p: number;
  if (n <= 25) {
    p = exactPValue(ranked);
  } else {
    p = approximatePValue(ranked);
  }

  return { p, n };
}
