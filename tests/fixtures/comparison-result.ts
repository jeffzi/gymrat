import type { CandidateComparison, ComparisonResult } from "../../src/report/types.js";
import type { ApproximateVerdictValue } from "../../src/verdict/verdict.js";

export type Metrics = ComparisonResult["metrics"];
export type MetricEntry = Metrics[string];

/**
 * One candidate's run-level results, judged against the shared baseline.
 *
 * A test that only cares about the geomean can override that field alone.
 */
export function createCandidate(overrides: Partial<CandidateComparison> = {}): CandidateComparison {
  return {
    label: "perf/faster-decode",
    geomean: {
      value: -5.8,
      n: 10,
      excluded: [],
    },
    ...overrides,
  };
}

/**
 * A comparison result with a clean baseline-plus-one-candidate run and no metrics.
 *
 * Shared by the renderer tests and the CLI tests so both drive the renderer
 * with the same shape `compare()` returns.
 */
export function createComparisonResult(
  overrides: Partial<ComparisonResult> = {},
): ComparisonResult {
  return {
    baselineLabel: "main",
    candidates: [createCandidate()],
    samples: 10,
    adapter: "mitata",
    metrics: {},
    worktreesRemoved: 0,
    worktreesLeftBehind: [],
    worktreePruneError: undefined,
    ...overrides,
  };
}

/** A two-sided metric whose verdict came from the signed-rank method. */
export function signedRankMetric(options: {
  verdict: ApproximateVerdictValue;
  delta: number;
  baselineMedian?: number;
  baselineSpread?: number;
  candidateMedian?: number;
  candidateSpread?: number;
  p?: number;
  noisePct?: number;
  unit?: "ns" | "bytes";
  gating?: boolean;
  n?: number;
}): MetricEntry {
  const {
    verdict,
    delta,
    baselineMedian = 100,
    baselineSpread = 1,
    candidateMedian = baselineMedian * (1 + delta / 100),
    candidateSpread = 1,
    p = 0.01,
    noisePct = 2.5,
    unit,
    gating = true,
    n = 10,
  } = options;
  return {
    baselineMedian,
    baselineSpread,
    candidates: [
      {
        median: candidateMedian,
        spread: candidateSpread,
        verdict: { verdict, method: "signed-rank", delta, n, p, noisePct },
      },
    ],
    meta: { direction: "lower", gating, exact: false, unit },
  };
}

/** A two-sided metric whose verdict fell back to the noise band. */
export function bandMetric(options: {
  verdict: ApproximateVerdictValue;
  delta: number;
  noisePct?: number;
  n?: number;
}): MetricEntry {
  const { verdict, delta, noisePct = 2.5, n = 4 } = options;
  return {
    baselineMedian: 100,
    baselineSpread: 5,
    candidates: [
      {
        median: 100 + delta,
        spread: 4,
        verdict: { verdict, method: "band", delta, n, band: noisePct, noisePct },
      },
    ],
    meta: { direction: "lower", gating: true, exact: false },
  };
}

/** A counted metric, compared exactly rather than statistically. */
export function exactMetric(options: {
  delta: number;
  baselineMedian?: number;
  candidateMedian?: number;
  n?: number;
  unit?: "ns" | "bytes";
}): MetricEntry {
  const {
    delta,
    baselineMedian = 1000,
    candidateMedian = 1000 * (1 + delta / 100),
    n = 10,
    unit = "bytes",
  } = options;
  return {
    baselineMedian,
    candidates: [
      {
        median: candidateMedian,
        verdict: { verdict: delta < 0 ? "improved" : "regressed", method: "exact", delta, n },
      },
    ],
    meta: { direction: "lower", gating: true, exact: true, unit },
  };
}

/** One metric judged for several candidates against a single shared baseline. */
export function nWayMetric(
  candidates: readonly { verdict: ApproximateVerdictValue; delta: number; median: number }[],
): MetricEntry {
  const entries: MetricEntry["candidates"] = candidates.map(({ verdict, delta, median }) => ({
    median,
    spread: 1,
    verdict: { verdict, method: "signed-rank", delta, n: 10, p: 0.01, noisePct: 2.5 },
  }));
  return {
    baselineMedian: 100,
    baselineSpread: 1,
    candidates: entries,
    meta: { direction: "lower", gating: true, exact: false, unit: "ns" },
  };
}
