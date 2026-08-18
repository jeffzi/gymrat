import type { ResolvedConfig, ResolvedMetricMeta } from "../config.js";
import { metricRecord } from "../metric-record.js";
import type { ComparisonResult, MetricComparison } from "../report/types.js";
import { computeMetricStats, pairedOrOwnValues } from "../sampling.js";
import { computeKindAggregates } from "../verdict/aggregate.js";
import { pairSamples } from "../verdict/verdict.js";
import type { BenchRunOutputs } from "./iterate.js";

/**
 * The comparison the report is drawn from: one baseline, one candidate, and no
 * worktrees to account for.
 *
 * A session's worktrees outlive the run, so the cleanup fields state a sweep
 * that never happened rather than one that found nothing to do.
 */
export function buildComparisonResult(
  run: BenchRunOutputs,
  config: ResolvedConfig,
): ComparisonResult {
  const metrics = metricRecord<MetricComparison>();
  for (const [metricName, meta] of Object.entries(run.metricMeta)) {
    metrics[metricName] = compareMetric(metricName, run, meta);
  }

  return {
    baselineLabel: run.baseline.ctx.label,
    candidates: [
      {
        label: run.experiment.ctx.label,
        kinds: computeKindAggregates(run.verdicts, run.metricMeta),
      },
    ],
    samples: Math.min(run.baseline.samples.length, run.experiment.samples.length),
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
  run: BenchRunOutputs,
  meta: ResolvedMetricMeta,
): MetricComparison {
  const { pairedA, pairedB } = pairSamples(
    metricName,
    run.baseline.samples,
    run.experiment.samples,
  );
  const baselineStats = computeMetricStats(
    pairedOrOwnValues(pairedA, run.baseline.samples, metricName),
  );
  const experimentStats = computeMetricStats(
    pairedOrOwnValues(pairedB, run.experiment.samples, metricName),
  );

  return {
    baselineMedian: baselineStats.median,
    baselineSpread: baselineStats.spread,
    candidates: [
      {
        median: experimentStats.median,
        spread: experimentStats.spread,
        verdict: run.verdicts[metricName],
      },
    ],
    meta,
  };
}
