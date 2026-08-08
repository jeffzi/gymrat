import { getAdapter } from "../adapters/index.js";
import type { ResolvedConfig, ResolvedMetricMeta } from "../config.js";
import { FILTER_PLACEHOLDER, GEOMEAN_PRIMARY, resolveMetricMeta } from "../config.js";
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
import type { ComparisonResult, MetricComparison, MetricComparisons } from "../report/types.js";
import {
  collectMetricNames,
  collectSamples,
  computeMetricStats,
  ownValues,
  type ProgressStep,
  type SamplingOptions,
  type TargetContext,
  type TargetSamples,
} from "../sampling.js";
import { sessionJsonlPath } from "../session/paths.js";
import type { IterationRecord, SessionRecord } from "../session/records.js";
import { appendRecord, foldSession, readRecords } from "../session/store.js";
import { computeKindAggregates } from "../verdict/aggregate.js";
import type { MetricVerdict } from "../verdict/verdict.js";
import { computeGeomean, computeVerdicts, pairSamples } from "../verdict/verdict.js";

/** What the loop tells the agent to do next, one per outcome. */
const NEXT_STEPS: Record<LoopOutcome, string> = {
  improved: "gymrat keep",
  regressed: "fix or gymrat discard",
  "no-signal": "gymrat keep or gymrat discard",
};

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
 */
export async function iterateSession(
  root: string,
  config: ResolvedConfig,
  options: IterateOptions = {},
): Promise<IterateResult> {
  const jsonlPath = sessionJsonlPath(root);
  const state = foldSession(readRecords(jsonlPath));

  if (state.session === undefined) {
    throw new GymratError(
      `No session in ${root}`,
      "Run gymrat start to open one before measuring an edit.",
    );
  }
  if (state.unsettled) {
    throw new GymratError(
      `Iteration ${state.lastSeq} has not been settled`,
      "Run gymrat keep or gymrat discard before measuring the next edit.",
    );
  }

  const [baseline, experiment] = await measure(state.session, config, options, config.bench);
  const metricMeta = resolveMeta(config, [baseline.samples, experiment.samples]);
  const firstRun = computeVerdicts(
    baseline.samples,
    experiment.samples,
    metricMeta,
    config.unstableNoisePct,
  );

  const confirmation = await confirmRegressions(
    state.session,
    config,
    options,
    firstRun,
    metricMeta,
  );
  const verdicts = applyConfirmation(firstRun, confirmation);

  const seq = state.lastSeq + 1;
  const result = buildComparisonResult(baseline, experiment, verdicts, metricMeta, config);
  const primary = resolvePrimary(config.primary, verdicts, metricMeta);
  const outcome = deriveOutcome(result.metrics, primary);

  const record: IterationRecord = {
    type: "iteration",
    seq,
    at: new Date().toISOString(),
    samples: { experiment: experiment.samples, baseline: baseline.samples },
    metrics: recordedVerdicts(verdicts, metricMeta, confirmation),
    ...(confirmation !== undefined && {
      confirm: {
        ran: true,
        filtered: [...confirmation.filtered],
        samples: confirmation.samples,
      },
    }),
    primary,
    outcome,
    targetReached: targetReached(config, primary, result.metrics),
  };
  appendRecord(jsonlPath, record);

  return { record, report: renderIteration(result, seq, outcome, primary, confirmation) };
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
  session: SessionRecord,
  config: ResolvedConfig,
  options: IterateOptions,
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

  const bench =
    config.filter === undefined
      ? config.bench
      : config.filter.replaceAll(FILTER_PLACEHOLDER, filtered.join(" "));
  const [baseline, experiment] = await measure(session, config, options, bench);

  const rerun = computeVerdicts(
    baseline.samples,
    experiment.samples,
    metricMeta,
    config.unstableNoisePct,
  );
  return {
    filtered,
    samples: { experiment: experiment.samples, baseline: baseline.samples },
    confirmed: new Set(filtered.filter((name) => rerun[name]?.verdict === "regressed")),
  };
}

/**
 * The verdicts as the iteration is finally read: every re-measured regression
 * the rerun would not stand behind demoted to no signal.
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
    const unconfirmed =
      confirmation.filtered.includes(metricName) && !confirmation.confirmed.has(metricName);
    settled[metricName] = unconfirmed ? { ...verdict, verdict: "no-signal" } : verdict;
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
  const contexts: TargetContext[] = [
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
  const [baseline, experiment] = await collectSamples(
    adapter,
    contexts,
    samplingOptions,
    options.signal ?? new AbortController().signal,
  );

  /* v8 ignore if -- defensive check; collectSamples returns one result per target given */
  if (baseline === undefined || experiment === undefined) {
    throw new GymratError("collectSamples returned no result for one of the session's worktrees");
  }
  return [baseline, experiment];
}

/** A session worktree, benched where it sits: it is checked out for the whole session. */
function worktreeContext(dir: string, label: string, position: "old" | "new"): TargetContext {
  return { target: { kind: "in-place", dir }, dir, label, position };
}

/** Settle metadata for every metric either worktree reported. */
function resolveMeta(
  config: ResolvedConfig,
  sampleSets: readonly Record<string, number>[][],
): Record<string, ResolvedMetricMeta> {
  const metricNames = collectMetricNames(sampleSets);

  /* v8 ignore if -- defensive check; adapters throw AdapterError for no metrics */
  if (metricNames.size === 0) {
    throw new GymratError("No metrics found in benchmark output");
  }

  return resolveMetricMeta(
    Array.from(metricNames),
    config.metrics,
    getAdapter(config.adapter),
    config.kinds,
  );
}

/**
 * The comparison the report is drawn from: one baseline, one candidate, and no
 * worktrees to account for.
 *
 * A session's worktrees outlive the run, so the cleanup fields state a sweep
 * that never happened rather than one that found nothing to do.
 */
function buildComparisonResult(
  baseline: TargetSamples,
  experiment: TargetSamples,
  verdicts: Record<string, MetricVerdict>,
  metricMeta: Record<string, ResolvedMetricMeta>,
  config: ResolvedConfig,
): ComparisonResult {
  const metrics = metricRecord<MetricComparison>();
  for (const [metricName, meta] of Object.entries(metricMeta)) {
    metrics[metricName] = compareMetric(metricName, baseline, experiment, verdicts, meta);
  }

  return {
    baselineLabel: baseline.ctx.label,
    candidates: [
      {
        label: experiment.ctx.label,
        kinds: computeKindAggregates(verdicts, metricMeta),
      },
    ],
    samples: Math.min(baseline.samples.length, experiment.samples.length),
    adapter: config.adapter,
    configKinds: config.kinds,
    metrics,
    worktreesRemoved: 0,
    worktreesLeftBehind: [],
    worktreePruneError: undefined,
  };
}

/**
 * One metric's two sides, each measured over the rounds the verdict was drawn
 * from, so a displayed median never disagrees with the delta beside it.
 *
 * A metric only one side reported has no pairs and therefore no verdict to stay
 * consistent with, so that side falls back to every round it did report.
 */
function compareMetric(
  metricName: string,
  baseline: TargetSamples,
  experiment: TargetSamples,
  verdicts: Record<string, MetricVerdict>,
  meta: ResolvedMetricMeta,
): MetricComparison {
  const { pairedA, pairedB } = pairSamples(metricName, baseline.samples, experiment.samples);
  const baselineStats = computeMetricStats(
    pairedA.length > 0 ? pairedA : ownValues(baseline.samples, metricName),
  );
  const experimentStats = computeMetricStats(
    pairedB.length > 0 ? pairedB : ownValues(experiment.samples, metricName),
  );

  return {
    baselineMedian: baselineStats.median,
    baselineSpread: baselineStats.spread,
    candidates: [
      {
        median: experimentStats.median,
        spread: experimentStats.spread,
        verdict: verdicts[metricName],
      },
    ],
    meta,
  };
}

/**
 * The figure the iteration is read on: the geomean over every gating metric, or
 * the one metric the config named.
 *
 * A named metric the run never measured still yields a primary, at rest: the
 * outcome reads it as no signal, which is what a figure nothing was measured for
 * amounts to.
 */
function resolvePrimary(
  primary: string,
  verdicts: Record<string, MetricVerdict>,
  metricMeta: Record<string, ResolvedMetricMeta>,
): LoopPrimary {
  if (primary === GEOMEAN_PRIMARY) {
    const gating = Object.fromEntries(Object.entries(metricMeta).filter(([, meta]) => meta.gating));
    return { kind: GEOMEAN_PRIMARY, deltaPct: computeGeomean(verdicts, gating).value };
  }
  return { kind: "metric", name: primary, deltaPct: verdicts[primary]?.delta ?? 0 };
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
  return Object.fromEntries(
    Object.entries(verdicts).map(([metricName, verdict]) => [
      metricName,
      {
        deltaPct: verdict.delta,
        verdict: verdict.verdict,
        method: verdict.method,
        ...(verdict.method === "signed-rank" ? { p: verdict.p } : {}),
        ...(verdict.method === "exact" ? {} : { noisePct: verdict.noisePct }),
        gating: metricMeta[metricName]?.gating ?? true,
        confirmed: confirmation?.confirmed.has(metricName) ?? false,
      },
    ]),
  );
}

/** The iteration as it prints: the loop's header, the comparison table, the verdict. */
function renderIteration(
  result: ComparisonResult,
  seq: number,
  outcome: LoopOutcome,
  primary: LoopPrimary,
  confirmation: Confirmation | undefined,
): string {
  const reruns: RerunConfirmation[] =
    confirmation?.filtered.map((metric) => ({
      metric,
      confirmed: confirmation.confirmed.has(metric),
    })) ?? [];
  const report = renderReport(result, { header: formatLoopHeader(seq, result.samples) });
  return [report, "", ...formatVerdictBlock(outcome, primary, NEXT_STEPS[outcome], reruns)].join(
    "\n",
  );
}
