import type { ComparisonResult } from "../../src/report/types.js";

/**
 * A comparison result with a clean two-branch run and no metrics.
 *
 * Shared by the renderer tests and the CLI tests so both drive the renderer
 * with the same shape `compare()` returns.
 */
export function createComparisonResult(
  overrides: Partial<ComparisonResult> = {},
): ComparisonResult {
  return {
    labels: ["main", "perf/faster-decode"],
    samples: 10,
    adapter: "mitata",
    metrics: {},
    geomean: {
      value: -5.8,
      n: 10,
      excluded: [],
    },
    worktreesRemoved: 0,
    worktreesLeftBehind: [],
    worktreePruneError: undefined,
    ...overrides,
  };
}
