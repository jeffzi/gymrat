import path from "node:path";

import { getAdapter, AdapterError } from "./adapters/index.js";
import type { Adapter, WarnSink } from "./adapters/types.js";
import { resolveMetricMeta, type ConfigKinds, type ConfigMetrics } from "./config.js";
import { GymratError, messageOf } from "./errors.js";
import { exec } from "./exec.js";
import type { ExecResult, ExecTimeoutError } from "./exec.js";
import { computeHalfRange, computeMedian } from "./math.js";
import { metricRecord } from "./metric-record.js";
import { formatCleanupFailures } from "./report/text.js";
import type { ComparisonResult, MetricComparison } from "./report/types.js";
import { installTerminationCleanup } from "./signals.js";
import { resolveTarget, planWorktree, materializeWorktree, cleanupWorktrees } from "./targets.js";
import type { CleanupResult, Target, WorktreeInfo } from "./targets.js";
import { computeKindAggregates } from "./verdict/aggregate.js";
import { computeVerdicts, pairSamples } from "./verdict/verdict.js";

/** A prepare step about to run for a target with a prepare script. */
export interface PrepareProgressStep {
  kind: "prepare";
  label: string;
}

/** A bench/sample step about to run — 1-based index within the total sample count. */
export interface SampleProgressStep {
  kind: "sample";
  index: number;
  total: number;
  label: string;
}

/** Structured step info emitted at the start of each prepare or sample step. */
export type ProgressStep = PrepareProgressStep | SampleProgressStep;

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

function formatCommandError(
  context: CommandErrorContext,
  failure: ExecResult | ExecTimeoutError,
): string {
  const isTimeout = "kind" in failure;
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

  if ("kind" in failure) {
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
  constructor(context: CommandErrorContext, failure: ExecResult | ExecTimeoutError) {
    const hint =
      context.target.kind === "ref"
        ? "the worktree only contains files tracked at this ref; untracked, gitignored, or not-yet-committed files are absent"
        : undefined;
    super(formatCommandError(context, failure), hint);
  }
}

/**
 * One target of a comparison run.
 *
 * `target` is either a git ref (resolved to a throwaway worktree) or a
 * filesystem directory path (benched in place).
 */
export interface TargetSpec {
  target: string;
  /** Display label; defaults to the ref name or the directory's base name. */
  label?: string;
}

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
  if (values.length < 2) {
    return undefined;
  }

  if (median === 0) {
    return undefined;
  }

  return (computeHalfRange(values) / Math.abs(median)) * 100;
}

interface TargetContext {
  target: Target;
  dir: string;
  label: string;
  position: "old" | "new";
}

/** One target's measurements, kept beside the target that produced them. */
interface TargetSamples {
  readonly ctx: TargetContext;
  readonly samples: Record<string, number>[];
}

/** Everything a run measured, in the shape the comparison reads it: one baseline, N candidates. */
interface RunSamples {
  readonly baseline: TargetSamples;
  readonly candidates: readonly TargetSamples[];
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
    throw new CommandError(context, result);
  }

  if (result.exitCode !== 0) {
    throw new CommandError(context, result);
  }

  return result.stdout;
}

/**
 * Collect paired samples round-robin across the targets, baseline first.
 *
 * Each round runs the bench once per target, baseline first, before the next
 * round starts, so a round is a block of measurements taken close together in
 * time — that adjacency is what lets the machine's drift cancel out of the
 * pairwise comparisons. `prepare` runs once per target, up front.
 *
 * Baseline and candidates stay apart in the return value rather than arriving as
 * one list the caller has to split at position 0: the comparison only ever reads
 * them in those roles.
 */
async function collectSamples(
  adapter: Adapter,
  baseline: TargetContext,
  candidates: readonly TargetContext[],
  options: Pick<
    CompareOptions,
    "bench" | "prepare" | "samples" | "timeoutSeconds" | "onProgress" | "warn"
  >,
  signal: AbortSignal,
): Promise<RunSamples> {
  const timeoutMs = options.timeoutSeconds * 1000;
  const baselineSamples: TargetSamples = { ctx: baseline, samples: [] };
  const candidateSamples: TargetSamples[] = candidates.map((ctx) => ({ ctx, samples: [] }));
  const collected = [baselineSamples, ...candidateSamples];

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

  return { baseline: baselineSamples, candidates: candidateSamples };
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

/** A side's median and spread over the given values, or both undefined when there are none. */
function computeMetricStats(values: readonly number[]): { median?: number; spread?: number } {
  if (values.length === 0) {
    return {};
  }
  const median = computeMedian(values);
  return { median, spread: computeSpread(values, median) };
}

/** Every value a side reported for a metric, regardless of what the other side has. */
function ownValues(samples: readonly Record<string, number>[], metricName: string): number[] {
  return samples.map((sample) => sample[metricName]).filter((v) => v !== undefined);
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

/** Every metric name any target reported, so a one-sided metric still gets a row. */
function collectMetricNames(sampleSets: readonly Record<string, number>[][]): Set<string> {
  return new Set(sampleSets.flat().flatMap(Object.keys));
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

  const hint = error instanceof GymratError ? error.hint : undefined;

  return error instanceof AdapterError
    ? new AdapterError(combinedMessage, undefined, { cause: error })
    : new GymratError(combinedMessage, hint, { cause: error });
}

/**
 * Compare one baseline revision against one or more candidate revisions.
 *
 * Orchestrates the comparison workflow:
 * 1. Resolves target directories/refs and creates worktrees as needed
 * 2. Runs the bench round-robin across every target, parsing output with the configured adapter
 * 3. Computes each candidate's verdicts against the shared baseline (signed-rank or band method)
 * 4. Cleans up worktrees on both the success and the failure path
 * 5. Returns the comparison data carrying that cleanup's outcome
 *
 * Rendering is the caller's job — the CLI passes the result to `renderReport`.
 *
 * The result is built after the try/catch so it states the cleanup that actually
 * ran, and cleanup is attempted exactly once per path — a second sweep could
 * succeed on a worktree the first one recorded as left behind, handing the user
 * a report the disk contradicts.
 *
 * When the measurement phase fails, no result is returned and the cleanup
 * outcome rides out on the propagating error instead. A failure raised later,
 * while building the result, is not carried that way: cleanup has already run by
 * then and its outcome is dropped. That path is defensive-only today, so it is
 * left uncovered rather than guarded.
 *
 * SIGINT and SIGTERM take a third path: the in-flight command is killed,
 * cleanup is attempted, and the process exits with `128 + signum` without a
 * result. Nothing is returned, so a worktree cleanup could not remove on this
 * path is left unreported — the exit code is the whole contract. The handlers
 * are installed before the first worktree exists and uninstalled once the run
 * settles, either way it settled.
 */
export async function compare(options: CompareOptions): Promise<ComparisonResult> {
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

  const runComparison = async (): Promise<ComparisonResult> => {
    let measurement: Measurement;

    try {
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
        position: TargetContext["position"],
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

      const collected = await collectSamples(
        adapter,
        baselineContext,
        candidateContexts,
        options,
        run.signal,
      );

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

      measurement = {
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
    } catch (error) {
      const cleanup = cleanupWorktrees(worktrees, repoDir);
      throw withCleanupFailures(error, cleanup);
    }

    const cleanup = cleanupWorktrees(worktrees, repoDir);

    return buildComparisonResult(measurement, options, cleanup);
  };

  return runComparison().finally(uninstallTerminationCleanup);
}
