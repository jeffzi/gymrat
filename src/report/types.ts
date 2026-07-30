import type { ResolvedMetricMeta } from "../config.js";
import type { WorktreeRemovalFailure } from "../targets.js";
import type { GeomeanResult, MetricVerdict } from "../verdict/verdict.js";

/** One metric's measured values, spreads and verdict, alongside the metadata that shaped them. */
export interface MetricComparison {
  medianA?: number;
  medianB?: number;
  spreadA?: number;
  spreadB?: number;
  verdict?: MetricVerdict;
  meta: ResolvedMetricMeta;
}

/** Every metric a comparison produced, keyed by metric name. */
export type MetricComparisons = Record<string, MetricComparison>;

/**
 * Everything `renderReport` needs to draw a comparison — the rendering input contract.
 */
export interface ComparisonResult {
  /** The baseline and candidate labels, in that order — everything in the report is judged against `labels[0]`. */
  labels: [string, string];
  samples: number;
  adapter: string;
  metrics: MetricComparisons;
  geomean: GeomeanResult;
  worktreesRemoved: number;

  /** Worktrees cleanup could not remove, each with the reason git gave. */
  worktreesLeftBehind: readonly WorktreeRemovalFailure[];

  /** Reason the `git worktree prune` sweep failed, or `undefined` if it succeeded. */
  worktreePruneError: string | undefined;
}
