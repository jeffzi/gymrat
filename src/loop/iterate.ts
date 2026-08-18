import { getAdapter } from "../adapters/index.js";
import type { ResolvedConfig, ResolvedMetricMeta } from "../config.js";
import { FILTER_PLACEHOLDER, GEOMEAN_PRIMARY } from "../config.js";
import { GymratError } from "../errors.js";
import { metricRecord } from "../metric-record.js";
import type { LoopOutcome, LoopPrimary, RerunConfirmation } from "../report/loop.js";
import {
  deriveOutcome,
  EXPERIMENT_INDEX,
  formatLoopHeader,
  formatVerdictBlock,
} from "../report/loop.js";
import { renderReport } from "../report/text.js";
import type { ComparisonResult, MetricComparisons } from "../report/types.js";
import {
  collectSamples,
  resolveMetricMetaFromSamples,
  type ProgressStep,
  type SamplingOptions,
  type TargetContext,
  type TargetSamples,
} from "../sampling.js";
import type { IterationRecord, SessionRecord } from "../session/records.js";
import type { SessionState } from "../session/store.js";
import { appendRecord, requireOpenSession } from "../session/store.js";
import type { MetricVerdict } from "../verdict/verdict.js";
import { computeGeomean, computeVerdicts } from "../verdict/verdict.js";
import type { HookInvocation } from "./hooks.js";
import { runHook } from "./hooks.js";
import { buildComparisonResult } from "./iterate-compare.js";

/** What the loop tells the agent to do next, one per outcome. */
const NEXT_STEPS: Record<LoopOutcome, string> = {
  improved: "gymrat keep",
  regressed: "fix or gymrat discard",
  "no-signal": "gymrat keep or gymrat discard",
};

/**
 * A configured stop condition refusing another iteration.
 *
 * Separate from a plain `GymratError` because nothing failed: the loop ran to
 * the end it was configured for, which the CLI reports as a gate trip rather
 * than as a tool failure.
 */
export class LoopStopError extends GymratError {}

/** What a caller can hand an iteration beyond its configuration. */
export interface IterateOptions {
  /** Fire-and-forget callback invoked at the start of each prepare or sample step. */
  onProgress?: (step: ProgressStep) => void;
  /** Aborting it kills the in-flight bench command. Omitted, nothing can interrupt the run. */
  signal?: AbortSignal;
}

/** What a confirmation rerun measured, and which of the metrics it re-measured it stood behind. */
interface Confirmation {
  /** The metrics the rerun was asked to re-measure, in the order the run reported them. */
  readonly filtered: readonly string[];
  /** The rerun's own rounds, kept raw so a later statistics change can re-read them. */
  readonly samples: IterationRecord["samples"];
  /** The subset of `filtered` the rerun also gated as regressed. */
  readonly confirmed: ReadonlySet<string>;
  /**
   * The subset of `filtered` the rerun produced no verdict for at all.
   *
   * Disjoint from `confirmed`, and not the complement of it: a metric can be
   * neither, which is the rerun measuring it and declining to call it regressed.
   */
  readonly absent: ReadonlySet<string>;
}

/** The session, config, and caller options that every iteration step shares. */
interface IterationContext {
  readonly session: SessionRecord;
  readonly config: ResolvedConfig;
  readonly options: IterateOptions;
}

/** A bench run's measurement outputs: both sides' samples, the verdicts drawn from them, and the metric metadata. */
export interface BenchRunOutputs {
  readonly baseline: TargetSamples;
  readonly experiment: TargetSamples;
  readonly verdicts: Record<string, MetricVerdict>;
  readonly metricMeta: Record<string, ResolvedMetricMeta>;
}

/** The iteration's judgment: what happened, what metric drove it, and whether a target was met. */
interface IterationJudgment {
  readonly outcome: LoopOutcome;
  readonly primary: LoopPrimary;
  readonly confirmation: Confirmation | undefined;
  readonly reachedTarget: boolean;
}

/** One measured iteration: what was written to the log, and what to print about it. */
export interface IterateResult {
  /** The record appended to the session log. */
  record: IterationRecord;
  /** The iteration as the agent reads it: header, comparison table, verdict block. */
  report: string;
}

/**
 * Measure the experiment worktree against the baseline worktree, record the
 * result, and phrase it for the agent driving the loop.
 *
 * Sampling is driven here rather than through `compare()` because a session's
 * worktrees are persistent: there is nothing to check out and nothing to sweep
 * afterwards, and the raw samples have to survive the run to reach the log.
 *
 * Holding the repository lock across the call is the caller's job — the bench
 * runs of two concurrent sessions would perturb each other's measurements.
 *
 * @throws GymratError when no session has been started, when the last iteration
 *   is still unsettled, or when the bench command fails.
 * @throws LoopStopError when a configured stop condition has already been met,
 *   before anything is measured or recorded.
 */
export async function iterateSession(
  root: string,
  config: ResolvedConfig,
  options: IterateOptions = {},
): Promise<IterateResult> {
  const { session, state, jsonlPath } = requireOpenSession(root, "measuring an edit");

  if (state.unsettled) {
    throw new GymratError(
      `Iteration ${state.lastSeq} has not been settled`,
      "Run gymrat keep or gymrat discard before measuring the next edit.",
    );
  }

  const stop = stopCondition(config, state);
  if (stop !== undefined) {
    throw stop;
  }

  const seq = state.lastSeq + 1;
  const beforeReport = await fireHook(jsonlPath, config.hooks?.before, {
    stage: "before",
    seq,
    session,
    lastIteration: state.lastIteration ?? null,
    iterationCount: state.iterationCount,
    signal: options.signal,
  });

  const ctx: IterationContext = { session, config, options };
  const { run, result, confirmation, samples } = await measureAndJudge(ctx);
  const primary = resolvePrimary(config.primary, run.verdicts, run.metricMeta);
  const outcome = deriveOutcome(result.metrics, primary);
  const reachedTarget = targetReached(config, primary, result.metrics);

  const record = buildIterationRecord({
    seq,
    samples,
    verdicts: run.verdicts,
    metricMeta: run.metricMeta,
    confirmation,
    primary,
    outcome,
    reachedTarget,
  });
  appendRecord(jsonlPath, record);

  const afterReport = await fireHook(jsonlPath, config.hooks?.after, {
    stage: "after",
    seq,
    session,
    lastIteration: record,
    iterationCount: state.iterationCount + 1,
    signal: options.signal,
  });

  const judgment: IterationJudgment = { outcome, primary, confirmation, reachedTarget };
  const iterationReport = renderIteration(result, seq, judgment);
  return {
    record,
    report: [beforeReport, iterationReport, afterReport].filter((part) => part !== "").join("\n"),
  };
}

async function measureAndJudge(ctx: IterationContext): Promise<{
  run: BenchRunOutputs;
  result: ComparisonResult;
  confirmation: Confirmation | undefined;
  samples: IterationRecord["samples"];
}> {
  const first = await benchAndJudge(ctx, ctx.config.bench);
  const confirmation = await confirmRegressions(ctx, first.verdicts, first.metricMeta);
  const verdicts = applyConfirmation(first.verdicts, confirmation);
  const run: BenchRunOutputs = {
    baseline: first.baseline,
    experiment: first.experiment,
    verdicts,
    metricMeta: first.metricMeta,
  };
  return {
    run,
    result: buildComparisonResult(run, ctx.config),
    confirmation,
    samples: first.samples,
  };
}

function buildIterationRecord(args: {
  seq: number;
  samples: IterationRecord["samples"];
  verdicts: Record<string, MetricVerdict>;
  metricMeta: Record<string, ResolvedMetricMeta>;
  confirmation: Confirmation | undefined;
  primary: LoopPrimary;
  outcome: LoopOutcome;
  reachedTarget: boolean;
}): IterationRecord {
  return {
    type: "iteration",
    seq: args.seq,
    at: new Date().toISOString(),
    samples: args.samples,
    metrics: recordedVerdicts(args.verdicts, args.metricMeta, args.confirmation),
    ...(args.confirmation !== undefined && {
      confirm: {
        ran: true,
        filtered: [...args.confirmation.filtered],
        ...(args.confirmation.absent.size > 0 && { absent: [...args.confirmation.absent] }),
        samples: args.confirmation.samples,
      },
    }),
    primary: args.primary,
    outcome: args.outcome,
    targetReached: args.reachedTarget,
  };
}

/**
 * Run the consumer's hook for this stage, logging what it did.
 *
 * A stage the config leaves out runs nothing at all: no process, no record, no
 * line in the report.
 *
 * @returns What to print for the hook, empty when there was no hook or it said nothing.
 */
async function fireHook(
  jsonlPath: string,
  command: string | undefined,
  invocation: Omit<HookInvocation, "command">,
): Promise<string> {
  if (command === undefined) {
    return "";
  }
  const run = await runHook({ command, ...invocation });
  appendRecord(jsonlPath, run.record);
  return run.report;
}

/** What every stop condition tells the agent to do once the loop is over. */
const STOP_HINT = "The loop is done. Report what the session measured instead of measuring again.";

/**
 * The configured stop condition this session has already met, if any.
 *
 * Read off the folded log alone, so it settles before a bench command runs: an
 * iteration measured past the end of the loop is one the agent would have to
 * throw away.
 *
 * `targetValue` stops the loop only once the target-reaching iteration is
 * **kept**. An unkept one is still the agent's to settle, and discarding it puts
 * the target back out of reach.
 */
function stopCondition(config: ResolvedConfig, state: SessionState): LoopStopError | undefined {
  const maxIterations = config.stop?.maxIterations;
  if (maxIterations !== undefined && state.iterationCount >= maxIterations) {
    return new LoopStopError(
      `Stop condition met: max iterations (${state.iterationCount} of ${maxIterations})`,
      STOP_HINT,
    );
  }

  if (config.stop?.targetValue !== undefined && state.targetReachedAndKept) {
    return new LoopStopError("Stop condition met: target reached and kept", STOP_HINT);
  }
  return undefined;
}

/**
 * Re-measure the gating metrics the first run called regressed, once.
 *
 * Only a rerun that also gates a metric makes its regression stand — the
 * asymmetry is deliberate: a false alarm costs the agent an edit it did not need
 * to make, while a missed regression is caught by the next iteration's baseline.
 *
 * `exact` metrics never take part. One differing sample is already the whole
 * signal for them, so a rerun could only add noise to a decision that has none.
 *
 * With a `filter` template configured the rerun benches just those metrics,
 * which is what makes confirming cheap enough to do on every failure; without
 * one it re-runs the whole bench and reads the same metrics out of it.
 *
 * @returns What the rerun found, or `undefined` when nothing called for one.
 * @throws GymratError when the rerun's bench command fails, exactly as the first
 *   run's failure does — an iteration nobody could confirm is not recorded.
 */
async function confirmRegressions(
  ctx: IterationContext,
  verdicts: Record<string, MetricVerdict>,
  metricMeta: Record<string, ResolvedMetricMeta>,
): Promise<Confirmation | undefined> {
  const filtered = Object.keys(metricMeta).filter((metricName) => {
    const meta = metricMeta[metricName];
    return meta?.gating === true && !meta.exact && verdicts[metricName]?.verdict === "regressed";
  });
  if (filtered.length === 0) {
    return undefined;
  }

  const names = filtered.map(shellQuote).join(" ");
  const bench =
    ctx.config.filter === undefined
      ? ctx.config.bench
      : // Replaced through a function so a `$&` or `$'` inside a metric name is
        // the name's own text rather than a substitution pattern.
        ctx.config.filter.replaceAll(FILTER_PLACEHOLDER, () => names);
  const { verdicts: rerun, samples } = await benchAndJudge(ctx, bench, metricMeta);
  return {
    filtered,
    samples,
    confirmed: new Set(filtered.filter((name) => rerun[name]?.verdict === "regressed")),
    absent: new Set(filtered.filter((name) => rerun[name] === undefined)),
  };
}

/** Characters a POSIX shell passes through untouched, so a word of them alone needs no quoting. */
const SHELL_SAFE_WORD = /^[\w@%+=:,./-]+$/;

/**
 * `value` as a single word of a POSIX shell command.
 *
 * The metric names are the bench's to choose, and mitata's `sort(n=1000)/time`
 * alias shape is an ordinary one: spliced into the filter template raw, the
 * shell either splits the name across arguments or refuses the command as a
 * syntax error — and a rerun that cannot run demotes a real regression to no
 * signal. Single quotes are the only POSIX quoting that suspends every
 * expansion, so a name that is not a plain word is wrapped in them, with each
 * single quote inside closed, escaped and reopened.
 */
function shellQuote(value: string): string {
  return SHELL_SAFE_WORD.test(value) ? value : `'${value.replaceAll("'", String.raw`'\''`)}'`;
}

/**
 * Bench a session's worktrees and judge the resulting samples, in one call.
 *
 * The two sites that measure a pair of worktrees — the first run and a
 * confirmation rerun — both need the same paired-samples literal recorded and
 * the same four `computeVerdicts` arguments in the same order, so both go
 * through here rather than repeating the pairing by hand.
 *
 * `metricMeta` is optional because the first run does not know the metric set
 * until it has samples to read it from; the confirmation rerun already has one
 * from the first run and passes it through unchanged.
 */
async function benchAndJudge(
  ctx: IterationContext,
  bench: string,
  metricMeta?: Record<string, ResolvedMetricMeta>,
): Promise<{
  baseline: TargetSamples;
  experiment: TargetSamples;
  metricMeta: Record<string, ResolvedMetricMeta>;
  verdicts: Record<string, MetricVerdict>;
  samples: IterationRecord["samples"];
}> {
  const [baseline, experiment] = await measure(ctx.session, ctx.config, ctx.options, bench);
  const adapter = getAdapter(ctx.config.adapter);
  const resolvedMeta =
    metricMeta ??
    resolveMetricMetaFromSamples(
      [baseline.samples, experiment.samples],
      ctx.config.metrics,
      adapter,
      ctx.config.kinds,
    );
  const verdicts = computeVerdicts(
    baseline.samples,
    experiment.samples,
    resolvedMeta,
    ctx.config.unstableNoisePct,
  );
  return {
    baseline,
    experiment,
    metricMeta: resolvedMeta,
    verdicts,
    samples: { experiment: experiment.samples, baseline: baseline.samples },
  };
}

/**
 * The verdicts as the iteration is finally read: every re-measured regression
 * the rerun would not stand behind demoted to no signal.
 *
 * A metric the rerun never reported is left regressed. The rerun's job is to
 * disprove a regression, and a metric it stayed silent about disproved nothing —
 * demoting on silence would let a bench that quietly dropped a metric wave a
 * real regression through.
 *
 * Only the verdict word moves. The delta, the noise, and the p-value stay the
 * first run's, because they describe the first run's samples — the ones the
 * record stores under `samples` and the table draws its medians from. The
 * rerun's own rounds are kept separately, under `confirm`.
 */
function applyConfirmation(
  verdicts: Record<string, MetricVerdict>,
  confirmation: Confirmation | undefined,
): Record<string, MetricVerdict> {
  if (confirmation === undefined) {
    return verdicts;
  }

  const settled = metricRecord<MetricVerdict>();
  for (const [metricName, verdict] of Object.entries(verdicts)) {
    const disagreed =
      confirmation.filtered.includes(metricName) &&
      !confirmation.confirmed.has(metricName) &&
      !confirmation.absent.has(metricName);
    settled[metricName] = disagreed ? { ...verdict, verdict: "no-signal" } : verdict;
  }
  return settled;
}

/**
 * Bench both of the session's worktrees, baseline first.
 *
 * The order is the one `compare()` samples in — old side first — so a round of
 * the loop perturbs the two sides in the same sequence a plain comparison would.
 *
 * `bench` is a parameter rather than read off `config` because a confirmation
 * rerun narrows the command to the metrics it is re-measuring, while sampling
 * the same pair of worktrees the same way.
 */
async function measure(
  session: SessionRecord,
  config: ResolvedConfig,
  options: IterateOptions,
  bench: string,
): Promise<readonly [baseline: TargetSamples, experiment: TargetSamples]> {
  const contexts: [TargetContext, TargetContext] = [
    worktreeContext(session.worktrees.baseline, "baseline", "old"),
    worktreeContext(session.worktrees.experiment, "experiment", "new"),
  ];

  const samplingOptions: SamplingOptions = {
    bench,
    prepare: config.prepare,
    samples: config.samples,
    timeoutSeconds: config.timeoutSeconds,
    onProgress: options.onProgress,
  };

  const adapter = getAdapter(config.adapter);
  const result = await collectSamples(
    adapter,
    contexts,
    samplingOptions,
    options.signal ?? new AbortController().signal,
  );
  return result;
}

/** A session worktree, benched where it sits: it is checked out for the whole session. */
function worktreeContext(dir: string, label: string, position: "old" | "new"): TargetContext {
  return { target: { kind: "in-place", dir }, dir, label, position };
}

/**
 * The figure the iteration is read on: the geomean over every gating metric, or
 * the one metric the config named.
 *
 * A named metric the run never measured yields a primary with no delta at all:
 * `null`, the form a figure that has no value takes everywhere in the record.
 * Nothing may stand a zero there — a zero is a measurement, and it would have the
 * report, the log and the keep commit all claim the run held its ground.
 */
function resolvePrimary(
  primary: string,
  verdicts: Record<string, MetricVerdict>,
  metricMeta: Record<string, ResolvedMetricMeta>,
): LoopPrimary {
  if (primary === GEOMEAN_PRIMARY) {
    const gating = metricRecord(Object.entries(metricMeta).filter(([, meta]) => meta.gating));
    const geomean = computeGeomean(verdicts, gating);
    return {
      kind: GEOMEAN_PRIMARY,
      deltaPct: geomean.n === 0 ? null : recordedDelta(geomean.value),
    };
  }
  const measured = verdicts[primary];
  return {
    kind: "metric",
    name: primary,
    deltaPct: measured === undefined ? null : recordedDelta(measured.delta),
  };
}

/**
 * A delta in the form the log keeps it: `null` where the ratio had no value.
 *
 * The engine answers a degenerate ratio — a baseline median of zero — with `NaN`,
 * and `JSON.stringify` writes that as `null` whatever the writer intended. Making
 * the substitution here rather than leaving it to serialization is what keeps the
 * record a caller holds identical to the one read back off the log.
 */
function recordedDelta(delta: number): number | null {
  return Number.isNaN(delta) ? null : delta;
}

/**
 * Whether the experiment has reached the value the loop was told to stop at.
 *
 * The target is a value the primary metric must reach, read in that metric's own
 * direction — so it needs a named primary, which is what config validation
 * already demands of a `stop.targetValue`.
 */
function targetReached(
  config: ResolvedConfig,
  primary: LoopPrimary,
  metrics: MetricComparisons,
): boolean {
  const target = config.stop?.targetValue;
  if (target === undefined || primary.kind !== "metric") {
    return false;
  }

  const metric = metrics[primary.name];
  const median = metric?.candidates[EXPERIMENT_INDEX]?.median;
  if (metric === undefined || median === undefined) {
    return false;
  }
  return metric.meta.direction === "higher" ? median >= target : median <= target;
}

/**
 * The per-metric verdicts as the log keeps them: the fields an agent reads back,
 * flattened out of the method-specific shapes.
 *
 * A key whose value is absent is left out rather than written as `undefined`,
 * so a record handed to a caller matches the one read back off the log.
 */
function recordedVerdicts(
  verdicts: Record<string, MetricVerdict>,
  metricMeta: Record<string, ResolvedMetricMeta>,
  confirmation: Confirmation | undefined,
): IterationRecord["metrics"] {
  const recorded: IterationRecord["metrics"] = metricRecord();
  for (const [metricName, verdict] of Object.entries(verdicts)) {
    recorded[metricName] = {
      deltaPct: recordedDelta(verdict.delta),
      verdict: verdict.verdict,
      method: verdict.method,
      ...(verdict.method === "signed-rank" ? { p: verdict.p } : {}),
      ...(verdict.method === "exact" ? {} : { noisePct: verdict.noisePct }),
      gating: metricMeta[metricName]?.gating ?? true,
      confirmed: confirmation?.confirmed.has(metricName) ?? false,
    };
  }
  return recorded;
}

/** What the rerun answered about `metric`, as the report words it. */
function rerunAnswer(confirmation: Confirmation, metric: string): RerunConfirmation["answer"] {
  if (confirmation.absent.has(metric)) {
    return "absent";
  }
  return confirmation.confirmed.has(metric) ? "confirmed" : "disagreed";
}

/** The iteration as it prints: the loop's header, the comparison table, the verdict. */
function renderIteration(
  result: ComparisonResult,
  seq: number,
  judgment: IterationJudgment,
): string {
  const { confirmation } = judgment;
  const reruns: RerunConfirmation[] =
    confirmation?.filtered.map((metric) => ({
      metric,
      answer: rerunAnswer(confirmation, metric),
    })) ?? [];
  const report = renderReport(result, { header: formatLoopHeader(seq, result.samples) });
  const verdict = formatVerdictBlock({
    outcome: judgment.outcome,
    primary: judgment.primary,
    nextStep: NEXT_STEPS[judgment.outcome],
    reruns,
    targetReached: judgment.reachedTarget,
  });
  return [report, "", ...verdict].join("\n");
}
