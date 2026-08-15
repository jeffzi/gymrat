import path from "node:path";

import { AdapterError } from "./adapters/index.js";
import type { Adapter, WarnSink } from "./adapters/types.js";
import type { ConfigKinds, ConfigMetrics, ResolvedMetricMeta } from "./config.js";
import { resolveMetricMeta } from "./config.js";
import { GymratError, messageOf } from "./errors.js";
import { exec } from "./exec.js";
import type { ExecResult, ExecTimeoutError } from "./exec.js";
import { computeHalfRange, computeMedian } from "./math.js";
import { formatCleanupFailures } from "./report/text.js";
import { installTerminationCleanup } from "./signals.js";
import { cleanupWorktrees, planWorktree, materializeWorktree } from "./targets.js";
import type { CleanupResult, Target, WorktreeInfo } from "./targets.js";

/** A prepare step about to run for a target with a prepare script. */
interface PrepareProgressStep {
  kind: "prepare";
  label: string;
}

/** A bench/sample step about to run — 1-based index within the total sample count. */
interface SampleProgressStep {
  kind: "sample";
  index: number;
  total: number;
  label: string;
}

/** Structured step info emitted at the start of each prepare or sample step. */
export type ProgressStep = PrepareProgressStep | SampleProgressStep;

// ---------------------------------------------------------------------------
// Command-error formatting
// ---------------------------------------------------------------------------

/** Fields that identify which command failed and where, for structured error reporting. */
export interface CommandErrorContext {
  phase: "prepare" | "bench";
  position?: "old" | "new";
  label: string;
  command: string;
  target: Target;
  dir: string;
  sample?: number;
}

/** The header line naming the command, its verb, and where it ran. */
function formatCommandErrorHeader(context: CommandErrorContext, isTimeout: boolean): string {
  const verb = isTimeout ? "timed out" : "failed";
  const positionPart = context.position !== undefined ? `${context.position}, ` : "";
  const samplePart = context.sample !== undefined ? `, sample ${context.sample}` : "";
  return `${context.phase} command ${verb} (${positionPart}"${context.label}"${samplePart})`;
}

/** The lines naming where the command ran: a worktree and its ref, or a plain directory. */
function formatCommandErrorLocation(context: CommandErrorContext): string[] {
  return context.target.kind === "ref"
    ? [`  ref:       ${context.target.ref}`, `  worktree:  ${context.dir}`]
    : [`  dir:       ${context.dir}`];
}

/**
 * The captured stdout and stderr, labeled when both are present so a reader can
 * tell them apart, unlabeled when only one stream has anything to show.
 */
function formatStreamEntry(label: string, text: string, totalBytes: number): string[] {
  const capturedBytes = Buffer.byteLength(text, "utf8");
  const suffix = totalBytes > capturedBytes ? ` (truncated, ${totalBytes} bytes total)` : "";
  return [`--- ${label}${suffix} ---`, text];
}

function formatCommandErrorOutput(failure: ExecResult | ExecTimeoutError): string[] {
  const streams = (
    [
      ["stderr", failure.stderr, failure.stderrBytes],
      ["stdout", failure.stdout, failure.stdoutBytes],
    ] satisfies [string, string, number][]
  ).filter(([, text]) => text.length > 0);

  if (streams.length < 2) {
    const entry = streams[0];
    if (entry === undefined) return [];
    const [, text, totalBytes] = entry;
    const capturedBytes = Buffer.byteLength(text, "utf8");
    return totalBytes > capturedBytes ? formatStreamEntry(entry[0], text, totalBytes) : [text];
  }
  return streams.flatMap(([label, text, totalBytes]) => formatStreamEntry(label, text, totalBytes));
}

function formatCommandError(
  context: CommandErrorContext,
  failure: ExecResult | ExecTimeoutError,
): string {
  const isTimeout = "kind" in failure;

  return [
    formatCommandErrorHeader(context, isTimeout),
    ...formatCommandErrorLocation(context),
    `  command:   ${context.command}`,
    isTimeout ? `  timeout:   ${failure.timeoutMs}ms` : `  exit code: ${failure.exitCode}`,
    ...formatCommandErrorOutput(failure),
  ].join("\n");
}

/**
 * Structured error for a command that failed during a benchmark or prepare phase.
 *
 * The formatted message carries the full context: phase, position, target, and the
 * exit code or timeout. Ref-target failures append a hint about the ref possibly
 * lacking the files the command needs.
 */
export class CommandError extends GymratError {
  constructor(context: CommandErrorContext, failure: ExecResult | ExecTimeoutError) {
    const hint =
      context.target.kind === "ref"
        ? "the worktree only contains files tracked at this ref; untracked, gitignored, or not-yet-committed files are absent"
        : undefined;
    super(formatCommandError(context, failure), hint);
  }
}

// ---------------------------------------------------------------------------
// Sampling core
// ---------------------------------------------------------------------------

/**
 * One target of a comparison or measurement run.
 *
 * `target` is either a git ref (resolved to a throwaway worktree) or a
 * filesystem directory path (benched in place).
 */
export interface TargetSpec {
  target: string;
  /** Display label; defaults to the ref name or the directory's base name. */
  label?: string;
}

/** A target resolved to its working directory, label, and optional role marker. */
export interface TargetContext {
  target: Target;
  dir: string;
  label: string;
  position?: "old" | "new";
}

/** One target's measurements, kept beside the target that produced them. */
export interface TargetSamples {
  readonly ctx: TargetContext;
  readonly samples: Record<string, number>[];
}

/**
 * The caller-facing subset of run configuration that `collectSamples` reads.
 *
 * `onProgress` and `warn` are optional sinks: omitted, progress goes
 * unreported and the adapter's own default (stderr) takes warnings.
 */
export interface SamplingOptions {
  /** Run through the shell in each target's directory. */
  bench: string;
  prepare?: string;
  /** How many rounds to run; each round runs `bench` once per target. */
  samples: number;
  timeoutSeconds: number;
  /** Fire-and-forget callback invoked at the start of each prepare or sample step. */
  onProgress?: (step: ProgressStep) => void;
  /**
   * Where the adapter's complaints about unreadable bench output go. Omitted,
   * the adapter falls back to stderr.
   */
  warn?: WarnSink;
}

/**
 * The run settings a comparison and a measurement both take, beyond the targets
 * each names for itself.
 *
 * `SamplingOptions` is what `collectSamples` reads; the three fields added here
 * are what the caller needs to turn raw samples into a report — which parser to
 * read the bench output with, and the per-metric and per-kind overrides that
 * settle each metric's metadata.
 */
export interface RunOptions extends SamplingOptions {
  /** Which output format `bench` writes: `"metric-lines"` or `"mitata"`. */
  adapter: string;
  configMetrics?: ConfigMetrics;
  configKinds?: ConfigKinds;
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
  const context: CommandErrorContext = { ...ctx, phase, command, sample };

  const result = await exec(command, { cwd: ctx.dir, timeoutMs, signal });

  if ("kind" in result || result.exitCode !== 0) {
    throw new CommandError(context, result);
  }

  return result.stdout;
}

/**
 * Collect samples round-robin across the targets in the order given.
 *
 * Each round runs the bench once per target, in the given order, before the next
 * round starts. `prepare` runs once per target, up front.
 *
 * The caller decides the ordering and assigns roles — the sampling core treats
 * every target identically.
 */
export async function collectSamples(
  adapter: Adapter,
  targets: readonly TargetContext[],
  options: SamplingOptions,
  signal: AbortSignal,
): Promise<TargetSamples[]> {
  const timeoutMs = options.timeoutSeconds * 1000;
  const collected: TargetSamples[] = targets.map((ctx) => ({ ctx, samples: [] }));

  if (options.prepare) {
    for (const { ctx } of collected) {
      options.onProgress?.({ kind: "prepare", label: ctx.label });
      await runCommand("prepare", ctx, options.prepare, timeoutMs, signal);
    }
  }

  for (let round = 0; round < options.samples; round++) {
    for (const { ctx, samples } of collected) {
      options.onProgress?.({
        kind: "sample",
        index: round + 1,
        total: options.samples,
        label: ctx.label,
      });
      const stdout = await runCommand("bench", ctx, options.bench, timeoutMs, signal, round + 1);
      samples.push(adapter.parse(stdout, options.warn));
    }
  }

  return collected;
}

// ---------------------------------------------------------------------------
// Target resolution
// ---------------------------------------------------------------------------

/**
 * Resolve the working directory for a target, creating a worktree for ref targets.
 *
 * The worktree is registered before `git worktree add` runs, not after: git can
 * be killed once the directory is on disk but before the command returns, and
 * every cleanup path — the failure path and the termination handler alike —
 * sweeps only what `worktrees` already names.
 */
export function resolveDir(resolved: Target, repoDir: string, worktrees: WorktreeInfo[]): string {
  if (resolved.kind === "ref") {
    const worktree = planWorktree(resolved);
    worktrees.push(worktree);
    materializeWorktree(worktree, repoDir);
    return worktree.dir;
  }
  return resolved.dir;
}

/**
 * The target's display label: the explicit label if one was given, otherwise
 * the ref name or the directory's base name.
 */
export function resolveLabel(explicit: string | undefined, resolved: Target): string {
  return explicit ?? (resolved.kind === "ref" ? resolved.ref : path.basename(resolved.dir));
}

// ---------------------------------------------------------------------------
// Metric statistics
// ---------------------------------------------------------------------------

/**
 * Half the observed range as a percentage of the median — the run-to-run jitter a
 * verdict's noise band is judged against.
 *
 * Half-range rather than full range so the figure reads as "± this much" either
 * side of the median, which is how the report prints it. A zero median has no
 * scale to be a percentage of, so it contributes no spread at all.
 *
 * A single observation has no run-to-run jitter to report: its range is zero
 * only because nothing was ever compared against it, and `± 0%` would state
 * that as a measured result. Such a side has no spread at all.
 */
function computeSpread(values: readonly number[], median: number): number | undefined {
  if (values.length < 2 || median === 0) return undefined;
  const ratio = (computeHalfRange(values) / Math.abs(median)) * 100;
  return Number.isFinite(ratio) ? ratio : undefined;
}

/** A side's median and spread over the given values, or both undefined when there are none. */
export function computeMetricStats(values: readonly number[]): {
  median?: number;
  spread?: number;
} {
  if (values.length === 0) {
    return {};
  }
  const median = computeMedian(values);
  return { median, spread: computeSpread(values, median) };
}

/** Every value a side reported for a metric, regardless of what the other side has. */
export function ownValues(
  samples: readonly Record<string, number>[],
  metricName: string,
): number[] {
  return samples.map((sample) => sample[metricName]).filter((v) => v !== undefined);
}

/**
 * The values one side's displayed median is read from: the rounds the metric
 * paired in, or — when nothing paired — every round that side reported it.
 *
 * A metric that paired has a verdict its median must stay consistent with, so
 * the median comes from the paired rounds alone. A metric only one side ever
 * reported has no verdict to agree with, so falling back to its own rounds
 * shows a real measurement instead of an empty cell.
 */
export function pairedOrOwnValues(
  paired: readonly number[],
  samples: readonly Record<string, number>[],
  metricName: string,
): readonly number[] {
  return paired.length > 0 ? paired : ownValues(samples, metricName);
}

/** Every metric name any target reported, so a one-sided metric still gets a row. */
function collectMetricNames(sampleSets: readonly Record<string, number>[][]): Set<string> {
  return new Set(sampleSets.flat().flatMap(Object.keys));
}

/**
 * Collect metric names from sample sets and resolve their metadata in one step.
 *
 * Every call site that measures — `measure`, `compare`, `iterate` — runs the
 * same sequence: collect names, guard against an empty set, resolve metadata.
 * The `v8 ignore` guard covers the empty-set branch that adapters' own
 * validation makes unreachable in practice.
 */
export function resolveMetricMetaFromSamples(
  sampleSets: readonly Record<string, number>[][],
  configMetrics: ConfigMetrics | undefined,
  adapter: Adapter,
  configKinds?: ConfigKinds,
): Record<string, ResolvedMetricMeta> {
  const metricNames = collectMetricNames(sampleSets);

  /* v8 ignore if -- defensive check; adapters throw AdapterError for no metrics */
  if (metricNames.size === 0) {
    throw new GymratError("No metrics found in benchmark output");
  }

  return resolveMetricMeta(Array.from(metricNames), configMetrics, adapter, configKinds);
}

// ---------------------------------------------------------------------------
// Worktree orchestration
// ---------------------------------------------------------------------------

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
function withCleanupFailures(
  error: unknown,
  cleanup: { failures: readonly { dir: string; error: string }[]; pruneError: string | undefined },
): unknown {
  const details = formatCleanupFailures(cleanup.failures, cleanup.pruneError);

  if (details.length === 0) {
    return error;
  }

  const combinedMessage = [messageOf(error), "", "cleanup did not finish:", ...details].join("\n");

  const hint = error instanceof GymratError ? error.hint : undefined;

  return error instanceof AdapterError
    ? new AdapterError(combinedMessage, hint, { cause: error })
    : new GymratError(combinedMessage, hint, { cause: error });
}

/**
 * Run a target-resolution phase under the worktree/signal discipline `measure()`
 * and `compare()` both need, then build the caller's result from whatever the
 * phase produced.
 *
 * The result is built after `phase` settles so it states the cleanup that
 * actually ran, and cleanup runs exactly once per path — a second sweep could
 * succeed on a worktree the first one recorded as left behind, handing the
 * caller a report the disk contradicts. When `phase` throws, `buildResult` is
 * never called and the cleanup outcome rides out on the propagating error
 * instead.
 *
 * A failure raised later, inside `buildResult` itself, is not carried that
 * way: cleanup has already run by then and its outcome is dropped. That path
 * is defensive-only today, so it is left uncovered rather than guarded.
 *
 * SIGINT and SIGTERM take a third path: the in-flight command is killed,
 * cleanup is attempted, and the process exits with `128 + signum` without a
 * result. The handlers are installed before the first worktree exists and
 * uninstalled once the run settles, either way it settled.
 */
export async function runWithWorktrees<M, R>(
  phase: (repoDir: string, worktrees: WorktreeInfo[], signal: AbortSignal) => Promise<M>,
  buildResult: (measurement: M, cleanup: CleanupResult) => R,
): Promise<R> {
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

  const runPhase = async (): Promise<R> => {
    let measurement: M;

    try {
      measurement = await phase(repoDir, worktrees, run.signal);
    } catch (error) {
      const cleanup = cleanupWorktrees(worktrees, repoDir);
      throw withCleanupFailures(error, cleanup);
    }

    const cleanup = cleanupWorktrees(worktrees, repoDir);
    return buildResult(measurement, cleanup);
  };

  return runPhase().finally(uninstallTerminationCleanup);
}
