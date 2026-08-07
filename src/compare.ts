import { getAdapter } from "./adapters/index.js";
import type { WarnSink } from "./adapters/types.js";
import { resolveMetricMeta, type ConfigKinds, type ConfigMetrics } from "./config.js";
import { GymratError } from "./errors.js";
import { metricRecord } from "./metric-record.js";
import type { ComparisonResult, MetricComparison } from "./report/types.js";
import {
  collectMetricNames,
  collectSamples,
  computeMetricStats,
  ownValues,
  resolveDir,
  resolveLabel,
  runWithWorktrees,
} from "./sampling.js";
import type { ProgressStep, TargetContext, TargetSamples, TargetSpec } from "./sampling.js";
import { resolveTarget } from "./targets.js";
import type { CleanupResult, Target } from "./targets.js";
import { computeKindAggregates } from "./verdict/aggregate.js";
import { computeVerdicts, pairSamples } from "./verdict/verdict.js";

export { CommandError } from "./sampling.js";
export type { CommandErrorContext, ProgressStep, TargetSpec } from "./sampling.js";

/**
 * Caller-facing configuration for a single comparison run.
 *
 * One baseline, one or more candidates: every candidate is compared with the
 * baseline and never with another candidate.
 */
export interface CompareOptions {
  baseline: TargetSpec;
  /** Judged against `baseline`, and reported in this order. */
  candidates: readonly TargetSpec[];
  /** Run through the shell in each target's directory. */
  bench: string;
  prepare?: string;
  /** Which output format `bench` writes: `"metric-lines"` or `"mitata"`. */
  adapter: string;
  /** How many rounds to run, each round sampling every target once. */
  samples: number;
  timeoutSeconds: number;
  /**
   * Noise band width, in percent, above which a metric is reported "unstable".
   * Omitted, the verdict engine's own default applies — the CLI always supplies
   * the resolved config value, so only direct callers see the fallback.
   */
  unstableNoisePct?: number;
  configMetrics?: ConfigMetrics;
  configKinds?: ConfigKinds;
  /** Fire-and-forget callback invoked at the start of each prepare or sample step. */
  onProgress?: (step: ProgressStep) => void;
  /**
   * Where the adapter's complaints about unreadable bench output go. Omitted,
   * the adapter falls back to stderr.
   */
  warn?: WarnSink;
}

/** Everything a run measured, in the shape the comparison reads it: one baseline, N candidates. */
interface RunSamples {
  readonly baseline: TargetSamples;
  readonly candidates: readonly TargetSamples[];
}

/**
 * The baseline's values for a metric, restricted to rounds where at least one
 * candidate also reported it — the same rounds `pairSamples` can draw a
 * verdict's delta from for at least one candidate.
 *
 * Falls back to every round the baseline reported the metric in when no
 * candidate ever did: a baseline-only metric has no verdict to stay
 * consistent with, so its displayed median is the baseline's own, unpaired.
 */
function baselinePairableValues(
  baselineSamples: readonly Record<string, number>[],
  candidateSampleSets: readonly (readonly Record<string, number>[])[],
  metricName: string,
): number[] {
  const paired: number[] = [];
  for (const [i, sample] of baselineSamples.entries()) {
    const value = sample[metricName];
    if (value === undefined) continue;
    const isPairable = candidateSampleSets.some(
      (samples) => samples[i]?.[metricName] !== undefined,
    );
    if (isPairable) {
      paired.push(value);
    }
  }
  return paired.length > 0 ? paired : ownValues(baselineSamples, metricName);
}

/**
 * A candidate's values for a metric, restricted to the same rounds
 * `computeVerdicts` pairs against the baseline — so the displayed median
 * matches the median the candidate's verdict delta was computed from.
 *
 * Falls back to every round the candidate reported the metric in when the
 * baseline never did: a candidate-only metric has no verdict to stay
 * consistent with, so its displayed median is the candidate's own, unpaired.
 */
function candidatePairableValues(
  baselineSamples: readonly Record<string, number>[],
  candidateSamples: readonly Record<string, number>[],
  metricName: string,
): number[] {
  const { pairedB } = pairSamples(metricName, baselineSamples, candidateSamples);
  return pairedB.length > 0 ? pairedB : ownValues(candidateSamples, metricName);
}

function buildComparisonResult(
  measurement: Measurement,
  options: Pick<CompareOptions, "samples" | "adapter" | "configKinds">,
  cleanup: CleanupResult,
): ComparisonResult {
  const { baselineLabel, baselineSamples, candidates, metricNames, metricMeta } = measurement;

  const result: ComparisonResult = {
    baselineLabel,
    candidates: candidates.map((candidate) => ({
      label: candidate.label,
      kinds: candidate.kinds,
    })),
    samples: options.samples,
    adapter: options.adapter,
    configKinds: options.configKinds,
    metrics: metricRecord<MetricComparison>(),
    worktreesRemoved: cleanup.removed,
    worktreesLeftBehind: cleanup.failures,
    worktreePruneError: cleanup.pruneError,
  };

  const candidateSampleSets = candidates.map((candidate) => candidate.samples);

  for (const metricName of metricNames) {
    const baseline = computeMetricStats(
      baselinePairableValues(baselineSamples, candidateSampleSets, metricName),
    );
    result.metrics[metricName] = {
      baselineMedian: baseline.median,
      baselineSpread: baseline.spread,
      candidates: candidates.map((candidate) => {
        const stats = computeMetricStats(
          candidatePairableValues(baselineSamples, candidate.samples, metricName),
        );
        return {
          median: stats.median,
          spread: stats.spread,
          verdict: candidate.verdicts[metricName],
        };
      }),
      meta: metricMeta[metricName]!,
    };
  }

  return result;
}

/**
 * Everything the measurement phase produces that the report is built from.
 *
 * Bundling these lets the whole set outlive the `try` block that computes them,
 * so the report can be rendered after worktree cleanup has already run.
 */
interface Measurement {
  baselineLabel: string;
  baselineSamples: Record<string, number>[];
  candidates: CandidateMeasurement[];
  metricNames: Set<string>;
  metricMeta: ReturnType<typeof resolveMetricMeta>;
}

/** One candidate's samples and the pairwise verdicts they earned against the baseline. */
interface CandidateMeasurement {
  label: string;
  samples: Record<string, number>[];
  verdicts: ReturnType<typeof computeVerdicts>;
  kinds: ReturnType<typeof computeKindAggregates>;
}

/**
 * Judge every candidate against the same baseline samples, one pairwise
 * comparison each.
 *
 * The topology is a star: no candidate is ever compared with another. Reusing
 * one set of baseline samples is what keeps that affordable, and it is also why
 * the resulting verdicts are statistically correlated — a baseline round that
 * ran slow inflates every candidate's delta at once. Each verdict is still
 * sound evidence about its own candidate; the gap between two candidates'
 * deltas is not a quantity this test measured.
 */
function measureCandidates(
  baselineSamples: Record<string, number>[],
  candidates: readonly TargetSamples[],
  metricMeta: ReturnType<typeof resolveMetricMeta>,
  unstableNoisePct: number | undefined,
): CandidateMeasurement[] {
  return candidates.map(({ ctx, samples }) => {
    const verdicts = computeVerdicts(baselineSamples, samples, metricMeta, unstableNoisePct);
    return {
      label: ctx.label,
      samples,
      verdicts,
      kinds: computeKindAggregates(verdicts, metricMeta),
    };
  });
}

/**
 * Compare one baseline revision against one or more candidate revisions.
 *
 * Orchestrates the comparison workflow:
 * 1. Resolves target directories/refs and creates worktrees as needed
 * 2. Runs the bench round-robin across every target, parsing output with the configured adapter
 * 3. Computes each candidate's verdicts against the shared baseline (signed-rank or band method)
 *
 * Rendering is the caller's job — the CLI passes the result to `renderReport`.
 *
 * Worktree cleanup and signal handling are `runWithWorktrees`'s job — see its
 * doc comment for the ordering and failure-path guarantees.
 */
export async function compare(options: CompareOptions): Promise<ComparisonResult> {
  return runWithWorktrees(
    async (repoDir, worktrees, signal) => {
      const adapter = getAdapter(options.adapter);

      // Every target is resolved before any worktree is materialized, so an
      // unresolvable candidate fails the run without leaving a directory on disk.
      const resolvedBaseline = {
        spec: options.baseline,
        target: resolveTarget(options.baseline.target, repoDir),
      };
      const resolvedCandidates = options.candidates.map((spec) => ({
        spec,
        target: resolveTarget(spec.target, repoDir),
      }));

      const toContext = (
        { spec, target }: { spec: TargetSpec; target: Target },
        position: "old" | "new",
      ): TargetContext => ({
        target,
        dir: resolveDir(target, repoDir, worktrees),
        label: resolveLabel(spec.label, target),
        position,
      });

      // Baseline first: its worktree is the one a cleanup sweep should find even
      // if a candidate's checkout is what failed.
      const baselineContext = toContext(resolvedBaseline, "old");
      const candidateContexts = resolvedCandidates.map((resolved) => toContext(resolved, "new"));

      const [baseline, ...candidates] = await collectSamples(
        adapter,
        [baselineContext, ...candidateContexts],
        options,
        signal,
      );

      /* v8 ignore if -- defensive check; collectSamples returns one result per target given */
      if (baseline === undefined) {
        throw new GymratError("collectSamples returned no result for the baseline target");
      }

      const collected: RunSamples = { baseline, candidates };

      const metricNames = collectMetricNames([
        collected.baseline.samples,
        ...collected.candidates.map(({ samples }) => samples),
      ]);

      /* v8 ignore if -- defensive check; adapters throw AdapterError for no metrics */
      if (metricNames.size === 0) {
        throw new GymratError("No metrics found in benchmark output");
      }

      const metricMeta = resolveMetricMeta(
        Array.from(metricNames),
        options.configMetrics,
        adapter,
        options.configKinds,
      );

      const measurement: Measurement = {
        baselineLabel: collected.baseline.ctx.label,
        baselineSamples: collected.baseline.samples,
        candidates: measureCandidates(
          collected.baseline.samples,
          collected.candidates,
          metricMeta,
          options.unstableNoisePct,
        ),
        metricNames,
        metricMeta,
      };
      return measurement;
    },
    (measurement, cleanup) => buildComparisonResult(measurement, options, cleanup),
  );
}
