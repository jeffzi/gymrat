import path from "node:path";

import { getAdapter } from "./adapters/index.js";
import type { Adapter } from "./adapters/types.js";
import { resolveMetricMeta, type ConfigMetrics } from "./config.js";
import { GymratError, messageOf } from "./errors.js";
import { exec } from "./exec.js";
import { computeMedian } from "./math.js";
import { formatCleanupFailures, renderReport } from "./report.js";
import type { ComparisonResult } from "./report.js";
import { installTerminationCleanup } from "./signals.js";
import { resolveTarget, planWorktree, materializeWorktree, cleanupWorktrees } from "./targets.js";
import type { CleanupResult, Target, WorktreeInfo } from "./targets.js";
import { computeVerdicts, computeGeomean } from "./verdict/verdict.js";

/** Fields that identify which command failed and where, for structured error reporting. */
export interface CommandErrorContext {
  phase: "prepare" | "bench";
  position: "old" | "new";
  label: string;
  command: string;
  target: Target;
  dir: string;
  sample?: number;
}

interface CommandOutput {
  stderr: string;
  stdout: string;
}

/** A command that terminated with a non-zero exit code. */
export interface ExitFailure extends CommandOutput {
  exitCode: number;
}

/** A command that was killed after exceeding its timeout. */
export interface TimeoutFailure extends CommandOutput {
  timeoutMs: number;
}

function isTimeoutFailure(failure: ExitFailure | TimeoutFailure): failure is TimeoutFailure {
  return "timeoutMs" in failure;
}

function formatCommandError(
  context: CommandErrorContext,
  failure: ExitFailure | TimeoutFailure,
): string {
  const isTimeout = isTimeoutFailure(failure);
  const verb = isTimeout ? "timed out" : "failed";

  const samplePart = context.sample !== undefined ? `, sample ${context.sample}` : "";
  const header = `${context.phase} command ${verb} (${context.position}, "${context.label}"${samplePart})`;

  const lines: string[] = [header];

  if (context.target.kind === "ref") {
    lines.push(`  ref:       ${context.target.ref}`);
    lines.push(`  worktree:  ${context.dir}`);
  } else {
    lines.push(`  dir:       ${context.dir}`);
  }

  lines.push(`  command:   ${context.command}`);

  if (isTimeout) {
    lines.push(`  timeout:   ${failure.timeoutMs}ms`);
  } else {
    lines.push(`  exit code: ${failure.exitCode}`);
  }

  const hasStderr = failure.stderr.length > 0;
  const hasStdout = failure.stdout.length > 0;

  if (hasStderr && hasStdout) {
    lines.push("--- stderr ---", failure.stderr, "--- stdout ---", failure.stdout);
  } else if (hasStderr) {
    lines.push(failure.stderr);
  } else if (hasStdout) {
    lines.push(failure.stdout);
  }

  return lines.join("\n");
}

/**
 * Structured error for a command that failed during a benchmark or prepare phase.
 *
 * Carries the full context — phase, position, target, exit/timeout details — both
 * in the formatted message and as typed fields for programmatic access. Ref-target
 * failures append a hint about the ref possibly lacking the files the command needs.
 */
export class CommandError extends GymratError {
  readonly phase: "prepare" | "bench";
  readonly position: "old" | "new";
  readonly label: string;
  readonly command: string;
  readonly target: Target;
  readonly dir: string;
  readonly sample: number | undefined;
  readonly exitCode: number | undefined;
  readonly timeoutMs: number | undefined;

  constructor(context: CommandErrorContext, failure: ExitFailure | TimeoutFailure) {
    const hint =
      context.target.kind === "ref"
        ? "the worktree only contains files tracked at this ref; untracked, gitignored, or not-yet-committed files are absent"
        : undefined;
    super(formatCommandError(context, failure), hint);

    this.phase = context.phase;
    this.position = context.position;
    this.label = context.label;
    this.command = context.command;
    this.target = context.target;
    this.dir = context.dir;
    this.sample = context.sample;
    const isTimeout = isTimeoutFailure(failure);
    this.exitCode = isTimeout ? undefined : failure.exitCode;
    this.timeoutMs = isTimeout ? failure.timeoutMs : undefined;
  }
}

/**
 * Caller-facing configuration for a single comparison run.
 *
 * `oldTarget` / `newTarget` are either git refs (resolved to worktrees) or
 * filesystem directory paths (benched in place).
 */
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
  configMetrics?: ConfigMetrics;
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

interface TargetContext {
  target: Target;
  dir: string;
  label: string;
  position: "old" | "new";
}

/**
 * Run a shell command and throw on timeout or non-zero exit.
 *
 * Aborting `signal` kills the command's whole process group; bench commands are
 * spawned detached, so a Ctrl-C delivered to gymrat never reaches them by itself.
 */
async function runCommand(
  phase: "prepare" | "bench",
  ctx: TargetContext,
  command: string,
  timeoutMs: number,
  signal: AbortSignal,
  sample?: number,
): Promise<string> {
  const context: CommandErrorContext = {
    phase,
    position: ctx.position,
    label: ctx.label,
    command,
    target: ctx.target,
    dir: ctx.dir,
    sample,
  };

  const result = await exec(command, { cwd: ctx.dir, timeoutMs, signal });

  if ("kind" in result) {
    throw new CommandError(context, {
      timeoutMs: result.timeoutMs,
      stderr: result.stderr,
      stdout: result.stdout,
    });
  }

  if (result.exitCode !== 0) {
    throw new CommandError(context, {
      exitCode: result.exitCode,
      stderr: result.stderr,
      stdout: result.stdout,
    });
  }

  return result.stdout;
}

/**
 * Collect paired samples by alternating between old and new targets.
 */
async function collectSamples(
  adapter: Adapter,
  oldCtx: TargetContext,
  newCtx: TargetContext,
  options: Pick<CompareOptions, "bench" | "prepare" | "samples" | "timeoutSeconds">,
  signal: AbortSignal,
): Promise<{ samplesA: Record<string, number>[]; samplesB: Record<string, number>[] }> {
  const timeoutMs = options.timeoutSeconds * 1000;

  if (options.prepare) {
    for (const ctx of [oldCtx, newCtx]) {
      await runCommand("prepare", ctx, options.prepare, timeoutMs, signal);
    }
  }

  const samplesA: Record<string, number>[] = [];
  const samplesB: Record<string, number>[] = [];

  for (let i = 0; i < options.samples; i++) {
    const oldStdout = await runCommand("bench", oldCtx, options.bench, timeoutMs, signal, i + 1);
    samplesA.push(adapter.parse(oldStdout));

    const newStdout = await runCommand("bench", newCtx, options.bench, timeoutMs, signal, i + 1);
    samplesB.push(adapter.parse(newStdout));
  }

  return { samplesA, samplesB };
}

/**
 * Resolve the working directory for a target, creating a worktree for ref targets.
 *
 * The worktree is registered before `git worktree add` runs, not after: git can
 * be killed once the directory is on disk but before the command returns, and
 * every cleanup path — the failure path and the termination handler alike —
 * sweeps only what `worktrees` already names.
 */
function resolveDir(resolved: Target, repoDir: string, worktrees: WorktreeInfo[]): string {
  if (resolved.kind === "ref") {
    const worktree = planWorktree(resolved);
    worktrees.push(worktree);
    materializeWorktree(worktree, repoDir);
    return worktree.dir;
  }
  return resolved.dir;
}

function resolveLabel(explicit: string | undefined, resolved: Target): string {
  if (explicit !== undefined) {
    return explicit;
  }
  return resolved.kind === "ref" ? resolved.ref : path.basename(resolved.dir);
}

function computeMetricStats(
  samplesA: readonly Record<string, number>[],
  samplesB: readonly Record<string, number>[],
  metricName: string,
): { medianA?: number; medianB?: number; spreadA?: number; spreadB?: number } {
  const aValues = samplesA.map((s) => s[metricName]).filter((v) => v !== undefined);
  const bValues = samplesB.map((s) => s[metricName]).filter((v) => v !== undefined);
  const hasA = aValues.length > 0;
  const hasB = bValues.length > 0;

  return {
    medianA: hasA ? computeMedian(aValues) : undefined,
    medianB: hasB ? computeMedian(bValues) : undefined,
    spreadA: hasA ? computeSpread(aValues) : undefined,
    spreadB: hasB ? computeSpread(bValues) : undefined,
  };
}

function buildComparisonResult(
  measurement: Measurement,
  options: Pick<CompareOptions, "samples" | "adapter">,
  cleanup: CleanupResult,
): ComparisonResult {
  const { samplesA, samplesB, metricNames, metricMeta, verdicts, geomean, labels } = measurement;

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
    result.metrics[metricName] = {
      ...computeMetricStats(samplesA, samplesB, metricName),
      verdict: verdicts[metricName],
      meta: metricMeta[metricName]!,
    };
  }

  return result;
}

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

  const combinedMessage = [messageOf(error), "", "cleanup did not finish:", ...details].join("\n");

  if (error instanceof CommandError && error.hint !== undefined) {
    return new GymratError(combinedMessage, error.hint, { cause: error });
  }

  return new Error(combinedMessage, { cause: error });
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
 *
 * SIGINT and SIGTERM take a third path: the in-flight command is killed,
 * cleanup is attempted, and the process exits with `128 + signum` without a
 * report. Nothing is printed, so a worktree cleanup could not remove on this
 * path is left unreported — the exit code is the whole contract. The handlers
 * are installed before the first worktree exists and uninstalled once the run
 * settles, either way it settled.
 */
export async function compare(options: CompareOptions): Promise<string> {
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

  const runComparison = async (): Promise<string> => {
    let measurement: Measurement;

    try {
      const adapter = getAdapter(options.adapter);

      const oldResolved = resolveTarget(options.oldTarget, repoDir);
      const newResolved = resolveTarget(options.newTarget, repoDir);

      const oldDir = resolveDir(oldResolved, repoDir, worktrees);
      const newDir = resolveDir(newResolved, repoDir, worktrees);

      const oldLabel = resolveLabel(options.oldLabel, oldResolved);
      const newLabel = resolveLabel(options.newLabel, newResolved);

      const oldCtx: TargetContext = {
        target: oldResolved,
        dir: oldDir,
        label: oldLabel,
        position: "old",
      };
      const newCtx: TargetContext = {
        target: newResolved,
        dir: newDir,
        label: newLabel,
        position: "new",
      };

      const { samplesA, samplesB } = await collectSamples(
        adapter,
        oldCtx,
        newCtx,
        options,
        run.signal,
      );

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

    const result = buildComparisonResult(measurement, options, cleanup);

    return renderReport(result);
  };

  return runComparison().finally(uninstallTerminationCleanup);
}
