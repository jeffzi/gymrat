import type { ResolvedMetricMeta } from "../config.js";
import type { WorktreeRemovalFailure } from "../targets.js";
import type { GeomeanResult, MetricVerdict } from "../verdict/verdict.js";

/**
 * Everything `renderReport` needs to draw a comparison — the rendering input contract.
 */
export interface ComparisonResult {
  labels: [string, string];
  samples: number;
  adapter: string;
  metrics: Record<
    string,
    {
      medianA?: number;
      medianB?: number;
      spreadA?: number;
      spreadB?: number;
      verdict?: MetricVerdict;
      meta: ResolvedMetricMeta;
    }
  >;
  geomean: GeomeanResult;
  worktreesRemoved: number;

  /** Worktrees cleanup could not remove, each with the reason git gave. */
  worktreesLeftBehind: readonly WorktreeRemovalFailure[];

  /** Reason the `git worktree prune` sweep failed, or `undefined` if it succeeded. */
  worktreePruneError: string | undefined;
}
