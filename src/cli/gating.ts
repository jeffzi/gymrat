import { assertNever } from "../errors.js";
import { metricRecord } from "../metric-record.js";
import { countVerdicts } from "../report/format.js";
import type {
  CandidateComparison,
  ComparisonResult,
  FailOnCondition,
  MetricComparisons,
} from "../report/types.js";
import type { GeomeanResult } from "../verdict/verdict.js";
import { writeAndFlush } from "./shared.js";

/**
 * Non-gating metrics never participate in gate evaluation — their verdicts
 * are informational and must not trip an exit-code gate.
 */
function gatingMetrics(metrics: MetricComparisons): MetricComparisons {
  return metricRecord(Object.entries(metrics).filter(([, metric]) => metric.meta.gating));
}

/**
 * The gated geomean of every kind that gates, one entry per such kind.
 */
function gatedGeomeansOf(candidate: CandidateComparison): readonly GeomeanResult[] {
  return candidate.kinds.flatMap((kind) => (kind.gatedGeomean ? [kind.gatedGeomean] : []));
}

/**
 * Returns `true` when any condition trips — meaning the process should exit 1.
 */
export function shouldFailGate(
  conditions: readonly FailOnCondition[],
  result: ComparisonResult,
): boolean {
  if (conditions.length === 0) return false;

  const gating = gatingMetrics(result.metrics);

  return conditions.some((condition) => {
    switch (condition.kind) {
      case "regressed":
        return result.candidates.some((_, i) => countVerdicts(gating, i).regressed > 0);
      case "geomean":
        return result.candidates.some((candidate) =>
          gatedGeomeansOf(candidate).some(
            (geomean) => geomean.n > 0 && geomean.value >= condition.pct,
          ),
        );
      default:
        return assertNever(condition);
    }
  });
}

/**
 * Warn once per candidate whose geomean gate has nothing to judge.
 */
export async function warnEmptyGeomeanGates(
  conditions: readonly FailOnCondition[],
  result: ComparisonResult,
): Promise<void> {
  if (!conditions.some((condition) => condition.kind === "geomean")) return;

  for (const candidate of result.candidates) {
    if (gatedGeomeansOf(candidate).every((geomean) => geomean.n === 0)) {
      await writeAndFlush(
        process.stderr,
        `warning: geomean gate for "${candidate.label}" had no stable gating metrics to measure\n`,
      );
    }
  }
}
