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
  resolveTarget,
  withCleanupFailures,
  cleanupWorktrees,
  installTerminationCleanup,
} from "./sampling.js";
import type { ProgressStep, TargetContext, TargetSpec, WorktreeInfo } from "./sampling.js";
import type { CleanupResult } from "./targets.js";

export { CommandError } from "./sampling.js";
export type { CommandErrorContext, ProgressStep, TargetSpec } from "./sampling.js";

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
  metricNames: Set<string>;
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
    worktreesRemoved: cleanup.removed,
    worktreesLeftBehind: cleanup.failures,
    worktreePruneError: cleanup.pruneError,
  };

  for (const metricName of measurement.metricNames) {
    const stats = computeMetricStats(ownValues(measurement.samples, metricName));
    result.metrics[metricName] = {
      median: stats.median,
      spread: stats.spread,
      meta: measurement.metricMeta[metricName]!,
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
 * The result is built after the try/catch so it states the cleanup that
 * actually ran, and cleanup is attempted exactly once per path — a second sweep
 * could succeed on a worktree the first one recorded as left behind, handing
 * the user a report the disk contradicts. When the measurement phase fails, no
 * result is returned and the cleanup outcome rides out on the propagating error
 * instead.
 *
 * SIGINT and SIGTERM take a third path: the in-flight command is killed,
 * cleanup is attempted, and the process exits with `128 + signum` without a
 * result.
 *
 * Nothing is written to disk beyond that worktree — recording a run is the
 * caller's business.
 */
export async function measure(options: MeasureOptions): Promise<MeasurementResult> {
  const repoDir = process.cwd();
  const worktrees: WorktreeInfo[] = [];
  const run = new AbortController();

  const uninstallTerminationCleanup = installTerminationCleanup(() => {
    // Kill first: the bench group is detached, so gymrat's own Ctrl-C never
    // reaches it and it would outlive the process. SIGKILL delivery is not
    // awaited — removal may well race a still-dying process — but unlinking a
    // directory out from under a live cwd is legal on POSIX, so the sweep
    // succeeds either way.
    run.abort();
    cleanupWorktrees(worktrees, repoDir);
  });

  const runMeasurement = async (): Promise<MeasurementResult> => {
    let measurement: Measurement;

    try {
      const adapter = getAdapter(options.adapter);
      const target = resolveTarget(options.target.target, repoDir);

      const ctx: TargetContext = {
        target,
        dir: resolveDir(target, repoDir, worktrees),
        label: resolveLabel(options.target.label, target),
      };

      const [collected] = await collectSamples(adapter, [ctx], options, run.signal);
      const samples = collected!.samples;

      const metricNames = collectMetricNames([samples]);

      /* v8 ignore if -- defensive check; adapters throw AdapterError for no metrics */
      if (metricNames.size === 0) {
        throw new GymratError("No metrics found in benchmark output");
      }

      measurement = {
        label: ctx.label,
        samples,
        metricNames,
        metricMeta: resolveMetricMeta(
          Array.from(metricNames),
          options.configMetrics,
          adapter,
          options.configKinds,
        ),
      };
    } catch (error) {
      const cleanup = cleanupWorktrees(worktrees, repoDir);
      throw withCleanupFailures(error, cleanup);
    }

    const cleanup = cleanupWorktrees(worktrees, repoDir);

    return buildMeasurementResult(measurement, options, cleanup);
  };

  return runMeasurement().finally(uninstallTerminationCleanup);
}
