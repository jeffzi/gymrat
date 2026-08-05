import { metricRecord } from "../metric-record.js";
import { inferGroup } from "../verdict/aggregate.js";
import type { GeomeanResult } from "../verdict/verdict.js";
import { countVerdicts, type VerdictCounts } from "./format.js";
import type {
  CandidateComparison,
  CandidateMetric,
  ComparisonResult,
  MetricComparisons,
} from "./types.js";

interface JsonCandidateMetric {
  label: string;
  median: number | null;
  spreadPct: number | null;
  verdict: string | null;
  method: string | null;
  delta: number | null;
  noisePct: number | null;
  p: number | null;
  band: number | null;
}

interface JsonMetric {
  unit: string | null;
  direction: "lower" | "higher";
  gating: boolean;
  /** The kind the metric aggregates under — `perCandidate[].kinds[].kind` keys on this. */
  kind: string;
  /** The group its short name puts it in, `null` when the name names none. */
  group: string | null;
  baseline: {
    median: number | null;
    spreadPct: number | null;
  };
  candidates: JsonCandidateMetric[];
}

/** One group's geomean within a kind, named by the prefix its metrics share. */
interface JsonGroupAggregate {
  group: string;
  geomean: GeomeanResult;
}

/**
 * One kind's aggregation for a candidate.
 *
 * `geomean` covers every metric of the kind and `gatedGeomean` only the gating
 * ones, so a consumer can read what the whole section did next to what the run
 * is judged on. `gatedGeomean` is `null` exactly when `hasGating` is false.
 */
interface JsonKindAggregate {
  kind: string;
  hasGating: boolean;
  geomean: GeomeanResult;
  groups: JsonGroupAggregate[];
  gatedGeomean: GeomeanResult | null;
}

interface JsonPerCandidate {
  label: string;
  /**
   * One entry per kind the run reported, in first-appearance order.
   *
   * A run whose adapter names no kind reports one entry, so a consumer reads
   * single-kind and multi-kind runs through the same shape.
   */
  kinds: JsonKindAggregate[];
  verdictCounts: VerdictCounts;
}

interface JsonWorktrees {
  removed: number;
  leftBehind: Array<{ path: string; reason: string }>;
  pruneError: string | null;
}

interface JsonReport {
  schemaVersion: 2;
  baseline: string;
  candidates: string[];
  samples: number;
  adapter: string;
  metrics: Record<string, JsonMetric>;
  perCandidate: JsonPerCandidate[];
  worktrees: JsonWorktrees;
}

/**
 * One candidate's figures for a metric, verdict included when there is one.
 *
 * A candidate can be measured without being judged — the metric may have paired
 * on no round at all — so what it measured is serialized either way, and only
 * the verdict fields go null.
 */
function serializeCandidateMetric(candidate: CandidateMetric, label: string): JsonCandidateMetric {
  const measured = {
    label,
    median: candidate.median ?? null,
    spreadPct: candidate.spread ?? null,
  };
  const verdict = candidate.verdict;
  if (verdict === undefined) {
    return {
      ...measured,
      verdict: null,
      method: null,
      delta: null,
      noisePct: null,
      p: null,
      band: null,
    };
  }
  return {
    ...measured,
    verdict: verdict.verdict,
    method: verdict.method,
    delta: verdict.delta,
    noisePct: verdict.method === "exact" ? null : verdict.noisePct,
    p: verdict.method === "signed-rank" ? verdict.p : null,
    band: verdict.method === "band" ? verdict.band : null,
  };
}

function serializeMetrics(
  metrics: MetricComparisons,
  candidateLabels: string[],
): Record<string, JsonMetric> {
  const result = metricRecord<JsonMetric>();
  for (const [name, metric] of Object.entries(metrics)) {
    result[name] = {
      unit: metric.meta.unit ?? null,
      direction: metric.meta.direction,
      gating: metric.meta.gating,
      kind: metric.meta.kind,
      group: inferGroup(metric.meta.shortName) ?? null,
      baseline: {
        median: metric.baselineMedian ?? null,
        spreadPct: metric.baselineSpread ?? null,
      },
      candidates: candidateLabels.map((label, i) =>
        serializeCandidateMetric(metric.candidates[i] ?? {}, label),
      ),
    };
  }
  return result;
}

/**
 * One candidate's kind aggregates, each stated in full.
 *
 * Assembles a `JsonKindAggregate` per kind from the internal per-candidate
 * aggregates, null-coalescing `gatedGeomean` to `null` for kinds that don't
 * gate (it's `undefined` on the internal aggregate in that case).
 */
function serializeKinds(candidate: CandidateComparison): JsonKindAggregate[] {
  return candidate.kinds.map((aggregate) => ({
    kind: aggregate.kind,
    hasGating: aggregate.hasGating,
    geomean: aggregate.geomean,
    groups: aggregate.groups.map((group) => ({ group: group.group, geomean: group.geomean })),
    gatedGeomean: aggregate.gatedGeomean ?? null,
  }));
}

function serializePerCandidate(result: ComparisonResult): JsonPerCandidate[] {
  return result.candidates.map((candidate, i) => ({
    label: candidate.label,
    kinds: serializeKinds(candidate),
    verdictCounts: countVerdicts(result.metrics, i),
  }));
}

function serializeWorktrees(result: ComparisonResult): JsonWorktrees {
  return {
    removed: result.worktreesRemoved,
    leftBehind: result.worktreesLeftBehind.map((failure) => ({
      path: failure.dir,
      reason: failure.error,
    })),
    pruneError: result.worktreePruneError ?? null,
  };
}

/**
 * Render a comparison result as stable, machine-readable JSON.
 *
 * The output uses 2-space indentation and `null` for absent values — keys are
 * never omitted. Consumers key on `schemaVersion` to detect breaking shape
 * changes.
 */
export function renderJson(result: ComparisonResult): string {
  const candidateLabels = result.candidates.map((c) => c.label);

  const report: JsonReport = {
    schemaVersion: 2,
    baseline: result.baselineLabel,
    candidates: candidateLabels,
    samples: result.samples,
    adapter: result.adapter,
    metrics: serializeMetrics(result.metrics, candidateLabels),
    perCandidate: serializePerCandidate(result),
    worktrees: serializeWorktrees(result),
  };

  return JSON.stringify(report, null, 2);
}
