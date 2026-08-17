import { getAdapter } from "./adapters/index.js";
import type { ResolvedMetricMeta } from "./config.js";
import { metricRecord } from "./metric-record.js";
import type { MeasurementResult, MetricMeasurement } from "./report/types.js";
import {
  collectSamples,
  computeMetricStats,
  ownValues,
  resolveDir,
  resolveLabel,
  resolveMetricMetaFromSamples,
  runWithWorktrees,
} from "./sampling.js";
import type { RunOptions, TargetContext, TargetSpec } from "./sampling.js";
import { resolveTarget } from "./targets.js";
import type { CleanupResult } from "./targets.js";

/**
 * Caller-facing configuration for a single measurement run.
 *
 * One target, no baseline: nothing is judged, so there is no noise band to set
 * and no verdict to gate on.
 */
export interface MeasureOptions extends RunOptions {
  target: TargetSpec;
}

/**
 * Everything the measurement phase produced that the result is built from.
 *
 * Bundling these lets the whole set outlive the `try` block that computes them,
 * so the result can be assembled after worktree cleanup has already run.
 */
interface Measurement {
  label: string;
  samples: Record<string, number>[];
  metricMeta: Record<string, ResolvedMetricMeta>;
}

function buildMeasurementResult(
  measurement: Measurement,
  options: Pick<MeasureOptions, "samples" | "adapter" | "configKinds">,
  cleanup: CleanupResult,
): MeasurementResult {
  const result: MeasurementResult = {
    label: measurement.label,
    samples: options.samples,
    adapter: options.adapter,
    configKinds: options.configKinds,
    metrics: metricRecord<MetricMeasurement>(),
    rounds: measurement.samples,
    worktreesRemoved: cleanup.removed,
    worktreesLeftBehind: cleanup.failures,
    worktreePruneError: cleanup.pruneError,
  };

  for (const [metricName, meta] of Object.entries(measurement.metricMeta)) {
    const stats = computeMetricStats(ownValues(measurement.samples, metricName));
    result.metrics[metricName] = {
      ...stats,
      meta,
    };
  }

  return result;
}

/**
 * Measure one revision or directory, with nothing to compare it against.
 *
 * Mirrors `compare()`'s worktree and signal discipline for a single target: the
 * target is resolved (a ref into a throwaway worktree, a directory benched
 * where it sits), `prepare` runs once, `bench` runs `samples` times, and the
 * adapter turns each run's stdout into metrics.
 *
 * Worktree cleanup and signal handling are `runWithWorktrees`'s job — see its
 * doc comment for the ordering and failure-path guarantees.
 *
 * Nothing is written to disk beyond that worktree — recording a run is the
 * caller's business.
 */
export async function measure(options: MeasureOptions): Promise<MeasurementResult> {
  return runWithWorktrees(
    async (repoDir, worktrees, signal) => {
      const adapter = getAdapter(options.adapter);
      const target = resolveTarget(options.target.target, repoDir);

      const ctx: TargetContext = {
        target,
        dir: resolveDir(target, repoDir, worktrees),
        label: resolveLabel(options.target.label, target),
      };

      const [collected] = await collectSamples(adapter, [ctx], options, signal);
      const samples = collected.samples;

      const measurement: Measurement = {
        label: ctx.label,
        samples,
        metricMeta: resolveMetricMetaFromSamples(
          [samples],
          options.configMetrics,
          adapter,
          options.configKinds,
        ),
      };
      return measurement;
    },
    (measurement, cleanup) => buildMeasurementResult(measurement, options, cleanup),
  );
}
