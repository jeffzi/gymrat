import type {
  CandidateComparison,
  CandidateMetric,
  ComparisonResult,
} from "../../src/report/types.js";
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
  noiseAbs?: number;
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
    noiseAbs = 3.5,
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
        verdict: { verdict, method: "signed-rank", delta, n, p, noisePct, noiseAbs },
      },
    ],
    meta: { direction: "lower", gating, exact: false, unit },
  };
}

/**
 * A two-sided metric whose verdict fell back to the noise band.
 *
 * `n` is the total pair count and `usableN` how many of those pairs survived
 * tie-dropping. `n < 6` means the run was too short for the signed-rank test;
 * `n >= 6` with `usableN < 6` means ties starved it instead.
 */
export function bandMetric(
  options: {
    verdict?: ApproximateVerdictValue;
    delta?: number;
    noisePct?: number;
    n?: number;
    usableN?: number;
    direction?: "lower" | "higher";
  } = {},
): MetricEntry {
  const {
    verdict = "no-signal",
    delta = -1,
    noisePct = 2.5,
    n = 4,
    usableN = n,
    direction = "lower",
  } = options;
  return {
    baselineMedian: 100,
    baselineSpread: 5,
    candidates: [
      {
        median: 100 + delta,
        spread: 4,
        verdict: {
          verdict,
          method: "band",
          delta,
          n,
          usableN,
          band: noisePct,
          noisePct,
          noiseAbs: 3.5,
        },
      },
    ],
    meta: { direction, gating: true, exact: false },
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
    verdict: {
      verdict,
      method: "signed-rank",
      delta,
      n: 10,
      p: 0.01,
      noisePct: 2.5,
      noiseAbs: 3.5,
    },
  }));
  return {
    baselineMedian: 100,
    baselineSpread: 1,
    candidates: entries,
    meta: { direction: "lower", gating: true, exact: false, unit: "ns" },
  };
}

/**
 * A multi-candidate comparison with one metric ("decode/time") judged per candidate.
 *
 * With 3 candidates (default): candidate-a improved, candidate-b regressed,
 * candidate-c unstable (band method). With 2: the first two only.
 */
export function multiCandidateResult(candidateCount: 2 | 3 = 3): ComparisonResult {
  const candidates: CandidateComparison[] = [
    createCandidate({ label: "candidate-a", geomean: { value: -10, n: 1, excluded: [] } }),
    createCandidate({ label: "candidate-b", geomean: { value: 4, n: 1, excluded: [] } }),
  ];

  const metricCandidates: CandidateMetric[] = [
    {
      median: 90,
      spread: 1,
      verdict: {
        verdict: "improved",
        method: "signed-rank",
        delta: -10,
        n: 10,
        p: 0.002,
        noisePct: 2.5,
        noiseAbs: 2.5,
      },
    },
    {
      median: 104,
      spread: 1,
      verdict: {
        verdict: "regressed",
        method: "signed-rank",
        delta: 4,
        n: 10,
        p: 0.002,
        noisePct: 2.5,
        noiseAbs: 2.5,
      },
    },
  ];

  if (candidateCount === 3) {
    candidates.push(
      createCandidate({ label: "candidate-c", geomean: { value: 0, n: 1, excluded: [] } }),
    );
    metricCandidates.push({
      median: 150,
      spread: 3,
      verdict: {
        verdict: "unstable",
        method: "band",
        delta: 50,
        n: 10,
        usableN: 3,
        band: 30,
        noisePct: 30,
        noiseAbs: 30,
      },
    });
  }

  return createComparisonResult({
    baselineLabel: "main",
    candidates,
    metrics: {
      "decode/time": {
        baselineMedian: 100,
        baselineSpread: 1,
        candidates: metricCandidates,
        meta: { direction: "lower", gating: true, exact: false, unit: "ns" },
      },
    },
  });
}
