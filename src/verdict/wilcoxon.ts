import wilcoxon from "@stdlib/stats-wilcoxon";

/**
 * Result of a Wilcoxon signed-rank test.
 */
export interface WilcoxonResult {
  /** p-value from the signed-rank test. */
  p: number;
  /** Effective sample size after dropping zero-difference pairs. */
  n: number;
}

/**
 * Wilcoxon signed-rank test for paired samples.
 *
 * Thin adapter over `@stdlib/stats-wilcoxon`. Guards edge cases where stdlib
 * throws (empty input, all-zero diffs, single pair) and returns the existing
 * `{p, n}` shape.
 *
 * @param x - Values from the first condition
 * @param y - Values from the second condition, paired by index with `x`
 * @returns Object with p-value and effective sample size (n, after dropping zeros)
 */
export function wilcoxonSignedRank(x: readonly number[], y: readonly number[]): WilcoxonResult {
  let n = 0;
  for (let i = 0; i < x.length; i++) {
    if (y[i]! - x[i]! !== 0) n++;
  }

  if (n === 0 || x.length < 2) {
    return { p: 1, n };
  }

  const result = wilcoxon([...x], [...y]);
  return { p: result.pValue, n };
}
