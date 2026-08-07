import type { ConfigKinds, ResolvedMetricMeta } from "../config.js";
import type { WorktreeRemovalFailure } from "../targets.js";
import type { KindAggregate } from "../verdict/aggregate.js";
import type { MetricVerdict } from "../verdict/verdict.js";

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

/**
 * Worktree cleanup outcome shared by every run result — comparison and
 * measurement runs manage the same worktrees the same way.
 */
interface WorktreeCleanupOutcome {
  worktreesRemoved: number;

  /** Worktrees cleanup could not remove, each with the reason git gave. */
  worktreesLeftBehind: readonly WorktreeRemovalFailure[];

  /** Reason the `git worktree prune` sweep failed, or `undefined` if it succeeded. */
  worktreePruneError: string | undefined;
}

/** One candidate's run-level results, judged against the shared baseline. */
export interface CandidateComparison {
  label: string;

  /** One entry per kind the run reported, in first-appearance order. */
  kinds: readonly KindAggregate[];
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
export interface ComparisonResult extends WorktreeCleanupOutcome {
  /** Label of the target every candidate is judged against. */
  baselineLabel: string;

  /** The candidates, in the order they were given on the command line. */
  candidates: readonly CandidateComparison[];

  samples: number;
  adapter: string;
  metrics: MetricComparisons;

  /**
   * The `kinds` section of the config the run resolved, when it had one.
   *
   * Gating is resolved per metric, which loses where the decision was made. A
   * report that tells the reader a whole section is informational is telling
   * them a config line did it, so it needs the config to name that line rather
   * than guess between a kind-level entry and per-metric overrides.
   */
  configKinds?: ConfigKinds;
}

/**
 * One metric of a single-target run: what it measured, and how steady it was.
 *
 * `spread` is the same half-range figure a comparison prints beside a side's
 * median, and is absent for the same reasons — a lone sample has no run-to-run
 * jitter to report, and a zero median has no scale to be a percentage of.
 */
export interface MetricMeasurement {
  median?: number;
  spread?: number;
  meta: ResolvedMetricMeta;
}

/** Every metric a measurement produced, keyed by metric name. */
export type MetricMeasurements = Record<string, MetricMeasurement>;

/**
 * Everything a single-target run measured — the rendering input contract for a
 * measurement, as `ComparisonResult` is for a comparison.
 *
 * There is nothing to judge against, so no verdicts and no aggregates: a
 * measurement states what the target reported and how steady it was, and stops
 * there.
 */
export interface MeasurementResult extends WorktreeCleanupOutcome {
  /** The target's explicit label, or its ref name / directory base name. */
  label: string;

  samples: number;
  adapter: string;
  metrics: MetricMeasurements;

  /**
   * The `kinds` section of the config the run resolved, when it had one.
   *
   * Carried for the same reason `ComparisonResult` carries it: metadata is
   * resolved per metric, which loses which config line decided a whole section.
   */
  configKinds?: ConfigKinds;
}

/**
 * A parsed `--fail-on` condition: either a `regressed` check or a geomean threshold.
 *
 * Shared by the gate that decides the exit code and the report that echoes which
 * gate a candidate tripped, so both read the same conditions the user wrote.
 */
export type FailOnCondition = { kind: "regressed" } | { kind: "geomean"; pct: number };

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

  /**
   * The `--fail-on` conditions the run is gated on, so the report can say which
   * gate a candidate tripped rather than leaving the exit code to explain itself.
   *
   * Display only. The renderer re-derives the check from the same aggregates the
   * table already shows; the exit code remains the caller's decision alone.
   */
  failOn?: readonly FailOnCondition[];
}
