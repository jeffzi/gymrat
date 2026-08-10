import { getAdapter } from "./adapters/index.js";
import type { WarnSink } from "./adapters/types.js";
import { resolveMetricMeta, type ConfigKinds, type ConfigMetrics } from "./config.js";
import { GymratError } from "./errors.js";
import { metricRecord } from "./metric-record.js";
import type { MeasurementResult, MetricMeasurement } from "./report/types.js";
import {
  collectMetricNames,
  collectSamples,
  computeMetricStats,
  ownValues,
  resolveDir,
  resolveLabel,
  runWithWorktrees,
} from "./sampling.js";
import type { ProgressStep, TargetContext, TargetSpec } from "./sampling.js";
import { resolveTarget } from "./targets.js";
import type { CleanupResult } from "./targets.js";

/**
 * Caller-facing configuration for a single measurement run.
 *
 * One target, no baseline: nothing is judged, so there is no noise band to set
 * and no verdict to gate on.
 */
export interface MeasureOptions {
  target: TargetSpec;
  /** Run through the shell in the target's directory. */
  bench: string;
  prepare?: string;
  /** Which output format `bench` writes: `"metric-lines"` or `"mitata"`. */
  adapter: string;
  /** How many times to run `bench`. */
  samples: number;
  timeoutSeconds: number;
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

/**
 * Everything the measurement phase produced that the result is built from.
 *
 * Bundling these lets the whole set outlive the `try` block that computes them,
 * so the result can be assembled after worktree cleanup has already run.
 */
interface Measurement {
  label: string;
  samples: Record<string, number>[];
  metricMeta: ReturnType<typeof resolveMetricMeta>;
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
      median: stats.median,
      spread: stats.spread,
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

      /* v8 ignore if -- defensive check; collectSamples returns one result per target given */
      if (collected === undefined) {
        throw new GymratError("collectSamples returned no result for the target");
      }

      const samples = collected.samples;
      const metricNames = collectMetricNames([samples]);

      /* v8 ignore if -- defensive check; adapters throw AdapterError for no metrics */
      if (metricNames.size === 0) {
        throw new GymratError("No metrics found in benchmark output");
      }

      const measurement: Measurement = {
        label: ctx.label,
        samples,
        metricMeta: resolveMetricMeta(
          Array.from(metricNames),
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
