import type { GeomeanResult, MetricVerdict } from "../verdict/verdict.js";
import { countVerdicts, type VerdictCounts } from "./format.js";
import type { CandidateMetric, ComparisonResult, MetricComparisons } from "./types.js";

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
  baseline: {
    median: number | null;
    spreadPct: number | null;
  };
  candidates: JsonCandidateMetric[];
}

interface JsonPerCandidate {
  label: string;
  geomean: GeomeanResult;
  verdictCounts: VerdictCounts;
}

interface JsonWorktrees {
  removed: number;
  leftBehind: Array<{ path: string; reason: string }>;
  pruneError: string | null;
}

interface JsonReport {
  schemaVersion: 1;
  baseline: string;
  candidates: string[];
  samples: number;
  adapter: string;
  metrics: Record<string, JsonMetric>;
  perCandidate: JsonPerCandidate[];
  worktrees: JsonWorktrees;
}

function serializeCandidateMetric(candidate: CandidateMetric, label: string): JsonCandidateMetric {
  const verdict = candidate.verdict;
  if (verdict === undefined) {
    return {
      label,
      median: null,
      spreadPct: null,
      verdict: null,
      method: null,
      delta: null,
      noisePct: null,
      p: null,
      band: null,
    };
  }
  return {
    label,
    median: candidate.median ?? null,
    spreadPct: candidate.spread ?? null,
    verdict: verdict.verdict,
    method: verdict.method,
    delta: verdict.delta,
    noisePct: extractNoisePct(verdict),
    p: extractP(verdict),
    band: extractBand(verdict),
  };
}

function extractNoisePct(verdict: MetricVerdict): number | null {
  if (verdict.method === "exact") return null;
  return verdict.noisePct;
}

function extractP(verdict: MetricVerdict): number | null {
  if (verdict.method === "signed-rank") return verdict.p;
  return null;
}

function extractBand(verdict: MetricVerdict): number | null {
  if (verdict.method === "band") return verdict.band;
  return null;
}

function serializeMetrics(
  metrics: MetricComparisons,
  candidateLabels: string[],
): Record<string, JsonMetric> {
  const result: Record<string, JsonMetric> = {};
  for (const [name, metric] of Object.entries(metrics)) {
    result[name] = {
      unit: metric.meta.unit ?? null,
      direction: metric.meta.direction,
      gating: metric.meta.gating,
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

function serializePerCandidate(result: ComparisonResult): JsonPerCandidate[] {
  return result.candidates.map((candidate, i) => ({
    label: candidate.label,
    geomean: candidate.geomean,
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
    schemaVersion: 1,
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
