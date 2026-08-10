import type { ResolvedMetricMeta } from "../../src/config.js";
import type { MetricVerdict } from "../../src/verdict/verdict.js";
import { metricRecord } from "./metrics.js";

/** An exact verdict no exclusion rule drops, so ρ is 1 + delta/100 when lower is better. */
export function exactVerdict(delta: number): MetricVerdict {
  const verdict = delta < 0 ? "improved" : delta > 0 ? "regressed" : "no-signal";
  return { verdict, method: "exact", delta, n: 1 };
}

/** One metric's contribution to a run: what it is, and how it moved. */
export interface MetricSpec {
  /** Full metric name — the key the verdict and exclusion lists report it under. */
  name: string;
  /** Defaults to `name`, which is what an adapter reporting no grouping produces. */
  shortName?: string;
  /** Defaults to "lower", so a spec list without directions describes a lower-is-better run. */
  direction?: "lower" | "higher";
  /** Defaults to true, matching every case that isn't explicitly non-gating. */
  gating?: boolean;
  /** Defaults to "time", so a spec list without kinds describes a single-kind run. */
  kind?: string;
  /** Percentage delta behind an exact verdict. Ignored when `verdict` is given. */
  delta?: number;
  /** A full verdict object, for band and unstable cases {@link exactVerdict} cannot express. */
  verdict?: MetricVerdict;
  /** True to add `name` to `metricMeta` without a matching verdict — the no-verdict case. */
  noVerdict?: boolean;
}

/**
 * Verdicts and metadata keyed by metric name, in the order the specs are listed.
 *
 * Both records are built the way the pipeline builds them — without a prototype —
 * so a metric named after an `Object.prototype` member becomes a key rather than
 * re-parenting the record.
 */
export function buildInputs(specs: readonly MetricSpec[]): {
  verdicts: Record<string, MetricVerdict>;
  metricMeta: Record<string, ResolvedMetricMeta>;
} {
  const verdicts = metricRecord<MetricVerdict>();
  const metricMeta = metricRecord<ResolvedMetricMeta>();

  for (const spec of specs) {
    const meta = {
      direction: spec.direction ?? "lower",
      gating: spec.gating ?? true,
      kind: spec.kind ?? "time",
      shortName: spec.shortName ?? spec.name,
    } as const;

    if (spec.noVerdict === true) {
      metricMeta[spec.name] = { ...meta, exact: false };
      continue;
    }

    const verdict = spec.verdict ?? exactVerdict(spec.delta ?? 0);
    verdicts[spec.name] = verdict;
    metricMeta[spec.name] = { ...meta, exact: verdict.method === "exact" };
  }

  return { verdicts, metricMeta };
}
