import type { ResolvedMetricMeta } from "../config.js";
import type { WorktreeRemovalFailure } from "../targets.js";
import type { GeomeanResult, MetricVerdict } from "../verdict/verdict.js";

/** One candidate's side of a metric, and the verdict it earned against the baseline. */
export interface CandidateMetric {
  median?: number;
  spread?: number;
  verdict?: MetricVerdict;
}

/**
 * One metric across the run: the baseline's measurement once, then a candidate
 * entry per candidate, alongside the metadata that shaped them.
 *
 * `candidates` is positional — entry _i_ belongs to `ComparisonResult.candidates[i]`.
 */
export interface MetricComparison {
  baselineMedian?: number;
  baselineSpread?: number;
  candidates: readonly CandidateMetric[];
  meta: ResolvedMetricMeta;
}

/** Every metric a comparison produced, keyed by metric name. */
export type MetricComparisons = Record<string, MetricComparison>;

/** One candidate's run-level results, judged against the shared baseline. */
export interface CandidateComparison {
  label: string;
  geomean: GeomeanResult;
}

/**
 * Everything `renderReport` needs to draw a comparison — the rendering input contract.
 *
 * The shape is a star, not a mesh: every candidate is compared with the baseline
 * and never with another candidate. Those comparisons all reuse the same baseline
 * samples, so the candidate verdicts of one metric are statistically correlated —
 * a baseline round that happened to run slow inflates every candidate's delta at
 * once. Read a candidate's verdict as evidence about that candidate alone; the
 * difference between two candidates' deltas is not itself a tested quantity.
 */
export interface ComparisonResult {
  /** Label of the target every candidate is judged against. */
  baselineLabel: string;

  /** The candidates, in the order they were given on the command line. */
  candidates: readonly CandidateComparison[];

  samples: number;
  adapter: string;
  metrics: MetricComparisons;
  worktreesRemoved: number;

  /** Worktrees cleanup could not remove, each with the reason git gave. */
  worktreesLeftBehind: readonly WorktreeRemovalFailure[];

  /** Reason the `git worktree prune` sweep failed, or `undefined` if it succeeded. */
  worktreePruneError: string | undefined;
}

/**
 * What the human-readable renderers print beyond the report itself.
 *
 * The JSON renderer takes none of these: its consumers parse fields rather than
 * read prose, so its output must not vary with a presentation flag.
 */
export interface ReportOptions {
  /**
   * Name the statistical method behind each verdict in the footer.
   *
   * Off by default — the glyphs and the summary line carry the reading, and the
   * method only matters once a reader questions a verdict.
   */
  verbose?: boolean;

  /**
   * Force ANSI color on (`true`) or off (`false`) rather than detecting it.
   *
   * Left unset, the renderers fall back to `styleText`'s own detection: color
   * when the stream is a TTY and the environment does not forbid it. `false`
   * gives plain text a pipe or a file can hold verbatim; `true` keeps the color
   * through a pager that would otherwise look like a non-TTY.
   */
  color?: boolean;
}
