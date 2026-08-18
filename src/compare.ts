import { getAdapter } from "./adapters/index.js";
import type { ResolvedMetricMeta } from "./config.js";
import { metricRecord } from "./metric-record.js";
import type { ComparisonResult, MetricComparison } from "./report/types.js";
import {
  collectSamples,
  computeMetricStats,
  pairedOrOwnValues,
  resolveDir,
  resolveLabel,
  resolveMetricMetaFromSamples,
  runWithWorktrees,
} from "./sampling.js";
import type { RunOptions, TargetContext, TargetSamples, TargetSpec } from "./sampling.js";
import { resolveTarget } from "./targets.js";
import type { CleanupResult, Target } from "./targets.js";
import { computeKindAggregates } from "./verdict/aggregate.js";
import { computeVerdicts, pairSamples } from "./verdict/verdict.js";

export type { ProgressStep, TargetSpec } from "./sampling.js";

/**
 * Caller-facing configuration for a single comparison run.
 *
 * One baseline, one or more candidates: every candidate is compared with the
 * baseline and never with another candidate.
 */
export interface CompareOptions extends RunOptions {
  baseline: TargetSpec;
  /** Judged against `baseline`, and reported in this order. */
  candidates: readonly TargetSpec[];
  /**
   * Noise band width, in percent, above which a metric is reported "unstable".
   * Omitted, the verdict engine's own default applies — the CLI always supplies
   * the resolved config value, so only direct callers see the fallback.
   */
  unstableNoisePct?: number;
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
): readonly number[] {
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
  return pairedOrOwnValues(paired, baselineSamples, metricName);
}

function buildComparisonResult(
  measurement: Measurement,
  options: Pick<CompareOptions, "samples" | "adapter" | "configKinds">,
  cleanup: CleanupResult,
): ComparisonResult {
  const { baselineLabel, baselineSamples, candidates, metricMeta } = measurement;

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

  for (const metricName of Object.keys(metricMeta)) {
    const meta = metricMeta[metricName];
    if (meta === undefined) throw new Error(`missing meta for ${metricName}`);
    const baseline = computeMetricStats(
      baselinePairableValues(baselineSamples, candidateSampleSets, metricName),
    );
    result.metrics[metricName] = {
      baselineMedian: baseline.median,
      baselineSpread: baseline.spread,
      candidates: candidates.map((candidate) => {
        const { pairedB } = pairSamples(metricName, baselineSamples, candidate.samples);
        const stats = computeMetricStats(pairedOrOwnValues(pairedB, candidate.samples, metricName));
        return {
          median: stats.median,
          spread: stats.spread,
          verdict: candidate.verdicts[metricName],
        };
      }),
      meta,
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
  metricMeta: Record<string, ResolvedMetricMeta>;
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
  metricMeta: Record<string, ResolvedMetricMeta>,
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

      const metricMeta = resolveMetricMetaFromSamples(
        [baseline.samples, ...candidates.map(({ samples }) => samples)],
        options.configMetrics,
        adapter,
        options.configKinds,
      );

      const measurement: Measurement = {
        baselineLabel: baseline.ctx.label,
        baselineSamples: baseline.samples,
        candidates: measureCandidates(
          baseline.samples,
          candidates,
          metricMeta,
          options.unstableNoisePct,
        ),
        metricMeta,
      };
      return measurement;
    },
    (measurement, cleanup) => buildComparisonResult(measurement, options, cleanup),
  );
}
