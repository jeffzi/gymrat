import path from "node:path";

import { getAdapter } from "./adapters/index.js";
import type { Adapter } from "./adapters/types.js";
import { resolveMetricMeta } from "./config.js";
import { exec } from "./exec.js";
import { formatCleanupFailures, renderReport } from "./report.js";
import type { ComparisonResult } from "./report.js";
import { resolveTarget, createWorktree, cleanupWorktrees } from "./targets.js";
import type { CleanupResult, Target, WorktreeInfo } from "./targets.js";
import { computeVerdicts, computeGeomean } from "./verdict/verdict.js";

export interface CompareOptions {
  oldTarget: string; // git ref or directory path
  newTarget: string; // git ref or directory path
  oldLabel?: string; // display label for old (defaults to ref name or dirname)
  newLabel?: string; // display label for new
  bench: string; // bench command to run in each target dir
  prepare?: string; // optional prepare command before bench
  adapter: string; // "metric-lines" or "mitata"
  samples: number; // number of paired sample windows
  timeoutSeconds: number;
  configMetrics?: Record<
    string,
    { direction?: "lower" | "higher"; gating?: boolean; exact?: boolean }
  >;
}

/**
 * Compute the median of a numeric array.
 */
function computeMedian(values: readonly number[]): number {
  /* v8 ignore if -- defensive check; never called with empty array */
  if (values.length === 0) {
    throw new Error("Cannot compute median of empty array");
  }

  const sorted = [...values].toSorted((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);

  return sorted.length % 2 === 0 ? (sorted[mid - 1]! + sorted[mid]!) / 2 : sorted[mid]!;
}

/**
 * Compute spread as percentage: (max - min) / (2 * median) * 100
 */
function computeSpread(values: readonly number[]): number {
  /* v8 ignore if -- defensive check; never called with empty array */
  if (values.length === 0) {
    return 0;
  }

  const sorted = [...values].toSorted((a, b) => a - b);
  const min = sorted[0]!;
  const max = sorted[sorted.length - 1]!;
  const median = computeMedian(sorted);

  if (median === 0) {
    return 0;
  }

  return ((max - min) / (2 * median)) * 100;
}

/**
 * Run a shell command and throw on timeout or non-zero exit.
 */
async function runCommand(
  command: string,
  cwd: string,
  timeoutMs: number,
  label: string,
): Promise<string> {
  const result = await exec(command, { cwd, timeoutMs });

  if ("kind" in result) {
    throw new Error(`${label} timed out after ${result.timeoutMs}ms:\n${result.stderr}`);
  }

  if (result.exitCode !== 0) {
    throw new Error(
      `${label} failed with exit code ${result.exitCode}:\n${[result.stderr, result.stdout].filter(Boolean).join("\n")}`,
    );
  }

  return result.stdout;
}

/**
 * Collect paired samples by alternating between old and new targets.
 */
async function collectSamples(
  adapter: Adapter,
  oldDir: string,
  newDir: string,
  options: Pick<CompareOptions, "bench" | "prepare" | "samples" | "timeoutSeconds">,
): Promise<{ samplesA: Record<string, number>[]; samplesB: Record<string, number>[] }> {
  const timeoutMs = options.timeoutSeconds * 1000;

  if (options.prepare) {
    for (const dir of [oldDir, newDir]) {
      await runCommand(options.prepare, dir, timeoutMs, "Prepare command");
    }
  }

  const samplesA: Record<string, number>[] = [];
  const samplesB: Record<string, number>[] = [];

  for (let i = 0; i < options.samples; i++) {
    const oldStdout = await runCommand(
      options.bench,
      oldDir,
      timeoutMs,
      "Bench command on old target",
    );
    samplesA.push(adapter.parse(oldStdout));

    const newStdout = await runCommand(
      options.bench,
      newDir,
      timeoutMs,
      "Bench command on new target",
    );
    samplesB.push(adapter.parse(newStdout));
  }

  return { samplesA, samplesB };
}

/**
 * Resolve the working directory for a target, creating a worktree for ref targets.
 */
function resolveDir(resolved: Target, repoDir: string, worktrees: WorktreeInfo[]): string {
  if (resolved.kind === "ref") {
    const wt = createWorktree(resolved, repoDir);
    worktrees.push(wt);
    return wt.dir;
  }
  return resolved.dir;
}

/**
 * Resolve the display label for a target.
 */
function resolveLabel(explicit: string | undefined, resolved: Target): string {
  if (explicit !== undefined) {
    return explicit;
  }
  return resolved.kind === "ref" ? resolved.ref : path.basename(resolved.dir);
}

/**
 * Build the ComparisonResult from collected samples and computed verdicts.
 */
function buildComparisonResult(
  samplesA: Record<string, number>[],
  samplesB: Record<string, number>[],
  metricNames: Set<string>,
  metricMeta: ReturnType<typeof resolveMetricMeta>,
  verdicts: ReturnType<typeof computeVerdicts>,
  geomean: ReturnType<typeof computeGeomean>,
  labels: [string, string],
  options: Pick<CompareOptions, "samples" | "adapter">,
  cleanup: CleanupResult,
): ComparisonResult {
  const result: ComparisonResult = {
    labels,
    samples: options.samples,
    adapter: options.adapter,
    metrics: {},
    geomean,
    worktreesRemoved: cleanup.removed,
    worktreesLeftBehind: cleanup.failures,
    worktreePruneError: cleanup.pruneError,
  };

  for (const metricName of metricNames) {
    const aValues = samplesA.map((s) => s[metricName]).filter((v) => v !== undefined);
    const bValues = samplesB.map((s) => s[metricName]).filter((v) => v !== undefined);

    result.metrics[metricName] = {
      medianA: aValues.length > 0 ? computeMedian(aValues) : undefined,
      medianB: bValues.length > 0 ? computeMedian(bValues) : undefined,
      spreadA: aValues.length > 0 ? computeSpread(aValues) : undefined,
      spreadB: bValues.length > 0 ? computeSpread(bValues) : undefined,
      verdict: verdicts[metricName],
      meta: metricMeta[metricName]!,
    };
  }

  return result;
}

/**
 * Collect all metric names from sample sets.
 */
function collectMetricNames(
  samplesA: Record<string, number>[],
  samplesB: Record<string, number>[],
): Set<string> {
  const names = new Set<string>();
  for (const sample of [...samplesA, ...samplesB]) {
    for (const name of Object.keys(sample)) {
      names.add(name);
    }
  }
  return names;
}

/**
 * Everything the measurement phase produces that the report is built from.
 *
 * Bundling these lets the whole set outlive the `try` block that computes them,
 * so the report can be rendered after worktree cleanup has already run.
 */
interface Measurement {
  samplesA: Record<string, number>[];
  samplesB: Record<string, number>[];
  metricNames: Set<string>;
  metricMeta: ReturnType<typeof resolveMetricMeta>;
  verdicts: ReturnType<typeof computeVerdicts>;
  geomean: ReturnType<typeof computeGeomean>;
  labels: [string, string];
}

/**
 * Restate a failed run's error so it also names what cleanup left on disk.
 *
 * `cleanupWorktrees` reports rather than throws, so on the failure path the
 * original exception would otherwise carry the run's error and nothing else —
 * the caller prints that, and the user never learns which directories survived.
 * When cleanup was clean there is nothing to add and the original error is
 * returned untouched. The original always becomes the `cause`, keeping its
 * stack reachable.
 */
function withCleanupFailures(error: unknown, cleanup: CleanupResult): unknown {
  const details = formatCleanupFailures(cleanup.failures, cleanup.pruneError);

  if (details.length === 0) {
    return error;
  }

  // The parameter stays `unknown` because that is what a catch clause binds, but
  // everything the measurement phase throws is an Error, so the String() arm never
  // runs. Left unsuppressed: a `v8 ignore` hint only binds to a ternary arm when it
  // sits flush against it with no whitespace, which oxfmt will not keep — and a
  // line-level hint would hide the covered arm along with this one.
  const message = error instanceof Error ? error.message : String(error);

  return new Error([message, "", "cleanup did not finish:", ...details].join("\n"), {
    cause: error,
  });
}

/**
 * Compare performance between two revisions.
 *
 * Orchestrates the comparison workflow:
 * 1. Resolves target directories/refs and creates worktrees as needed
 * 2. Runs bench command multiple times per target, parsing output with the configured adapter
 * 3. Computes verdicts using statistical tests (signed-rank or band method)
 * 4. Cleans up worktrees on both the success and the failure path
 * 5. Renders a formatted report carrying that cleanup's outcome
 *
 * Rendering happens after the try/catch so the report states the cleanup that
 * actually ran, and cleanup is attempted exactly once per path — a second sweep
 * could succeed on a worktree the first one recorded as left behind, handing the
 * user a report the disk contradicts.
 *
 * When the measurement phase fails, no report is rendered and the cleanup
 * outcome rides out on the propagating error instead. A failure raised later,
 * while building or rendering the report, is not carried that way: cleanup has
 * already run by then and its outcome is dropped. That path is defensive-only
 * today, so it is left uncovered rather than guarded.
 */
export async function compare(options: CompareOptions): Promise<string> {
  const repoDir = process.cwd();
  const worktrees: WorktreeInfo[] = [];

  let measurement: Measurement;

  try {
    const adapter = getAdapter(options.adapter);

    const oldResolved = resolveTarget(options.oldTarget, repoDir);
    const newResolved = resolveTarget(options.newTarget, repoDir);

    const oldDir = resolveDir(oldResolved, repoDir, worktrees);
    const newDir = resolveDir(newResolved, repoDir, worktrees);

    const oldLabel = resolveLabel(options.oldLabel, oldResolved);
    const newLabel = resolveLabel(options.newLabel, newResolved);

    const { samplesA, samplesB } = await collectSamples(adapter, oldDir, newDir, options);

    const metricNames = collectMetricNames(samplesA, samplesB);

    /* v8 ignore if -- defensive check; adapters throw AdapterError for no metrics */
    if (metricNames.size === 0) {
      throw new Error("No metrics found in benchmark output");
    }

    const metricMeta = resolveMetricMeta(Array.from(metricNames), options.configMetrics, adapter);
    const verdicts = computeVerdicts(samplesA, samplesB, metricMeta);
    const geomean = computeGeomean(verdicts, metricMeta);

    measurement = {
      samplesA,
      samplesB,
      metricNames,
      metricMeta,
      verdicts,
      geomean,
      labels: [oldLabel, newLabel],
    };
  } catch (error) {
    const cleanup = cleanupWorktrees(worktrees, repoDir);
    throw withCleanupFailures(error, cleanup);
  }

  const cleanup = cleanupWorktrees(worktrees, repoDir);

  const result = buildComparisonResult(
    measurement.samplesA,
    measurement.samplesB,
    measurement.metricNames,
    measurement.metricMeta,
    measurement.verdicts,
    measurement.geomean,
    measurement.labels,
    options,
    cleanup,
  );

  return renderReport(result);
}
