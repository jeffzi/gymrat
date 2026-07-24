import wilcoxon from "@stdlib/stats-wilcoxon";

export interface WilcoxonResult {
  p: number;
  n: number;
}

/**
 * Wilcoxon signed-rank test for paired samples.
 *
 * Thin adapter over `@stdlib/stats-wilcoxon`. Guards edge cases where stdlib
 * throws (empty input, all-zero diffs, single pair) and returns the existing
 * `{p, n}` shape.
 *
 * @param pairs - Array of [a, b] pairs, where each pair is a sample from two conditions
 * @returns Object with p-value and effective sample size (n, after dropping zeros)
 */
export function wilcoxonSignedRank(
  pairs: ReadonlyArray<readonly [number, number]>,
): WilcoxonResult {
  if (pairs.length === 0) {
    return { p: 1, n: 0 };
  }

  const x: number[] = [];
  const y: number[] = [];
  let n = 0;

  for (const [a, b] of pairs) {
    x.push(a);
    y.push(b);
    if (b - a !== 0) n++;
  }

  if (n === 0) {
    return { p: 1, n: 0 };
  }

  if (pairs.length < 2) {
    return { p: 1, n };
  }

  const result = wilcoxon(x, y);
  return { p: result.pValue, n };
}
