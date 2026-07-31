import type { CandidateComparison, ComparisonResult } from "../../src/report/types.js";

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
