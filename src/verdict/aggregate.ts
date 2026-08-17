/**
 * Hierarchical aggregation: one geomean per kind, per group, and per gating subset.
 */

import type { ResolvedMetricMeta } from "../config.js";
import { metricRecord } from "../metric-record.js";
import { computeGeomean } from "./verdict.js";
import type { GeomeanResult, MetricVerdict } from "./verdict.js";

/** The geomean over one group of a kind's metrics. */
export interface GroupAggregate {
  /** Short-name prefix the group's metrics share, dot excluded. */
  group: string;
  geomean: GeomeanResult;
}

/**
 * One kind's aggregation for a single candidate.
 *
 * `geomean` covers every metric of the kind and `gatedGeomean` only the gating
 * ones, so a report can show what the whole section did next to what the run is
 * judged on. The two coincide when every metric of the kind gates.
 */
export interface KindAggregate {
  kind: string;

  /** Over every metric of the kind, gating and non-gating alike. */
  geomean: GeomeanResult;

  /** One entry per group the kind's short names name, empty when none is dotted. */
  groups: readonly GroupAggregate[];

  /** Over the kind's gating metrics alone, absent when the kind has none. */
  gatedGeomean?: GeomeanResult;
}

/** A metric name paired with the metadata resolved for it. */
type MetricEntry = readonly [name: string, meta: ResolvedMetricMeta];

/** A kind's metrics, and the groups their short names sort them into. */
interface KindBucket {
  metrics: MetricEntry[];
  groups: Map<string, MetricEntry[]>;
}

/**
 * The text before a short name's first dot, or `undefined` when it has none.
 *
 * Only the first dot divides: `decode.utf8.time` belongs to `decode`, so a suite
 * reads as one group however deeply its own benchmark names nest. A leading dot
 * would name an empty group, which is no grouping at all.
 *
 * Exported so a renderer laying out group blocks sorts its rows by the same rule
 * the aggregates were computed under — a second rule would put a metric in one
 * group and its geomean in another.
 */
export function inferGroup(shortName: string): string | undefined {
  const dot = shortName.indexOf(".");
  return dot > 0 ? shortName.slice(0, dot) : undefined;
}

/**
 * Sort every metric into its kind, and within that kind into its group.
 *
 * Both levels are `Map`s keyed by name, so iteration yields kinds and groups in
 * the order their first metric introduced them — and a kind or group named after
 * an `Object.prototype` member stays an ordinary key.
 */
function bucketByKind(metricMeta: Record<string, ResolvedMetricMeta>): Map<string, KindBucket> {
  const buckets = new Map<string, KindBucket>();

  for (const [name, meta] of Object.entries(metricMeta)) {
    let bucket = buckets.get(meta.kind);
    if (!bucket) {
      bucket = { metrics: [], groups: new Map() };
      buckets.set(meta.kind, bucket);
    }

    const entry: MetricEntry = [name, meta];
    bucket.metrics.push(entry);

    const group = inferGroup(meta.shortName);
    if (group === undefined) continue;

    const members = bucket.groups.get(group);
    if (members) members.push(entry);
    else bucket.groups.set(group, [entry]);
  }

  return buckets;
}

/**
 * The geomean over `entries` alone, whatever else `verdicts` carries.
 *
 * `computeGeomean` averages the metrics its metadata names, so handing it a
 * restricted metadata record is how a subset is selected.
 */
function geomeanOver(
  entries: readonly MetricEntry[],
  verdicts: Record<string, MetricVerdict>,
): GeomeanResult {
  return computeGeomean(verdicts, metricRecord(entries));
}

/**
 * Aggregate one candidate's verdicts into a geomean per kind, group, and gating subset.
 *
 * Kinds, groups, and the metrics inside them keep the order `metricMeta` lists
 * them in, which is the order the run first reported each metric — so a report
 * drawn from these aggregates reads in the same order as the metric table.
 *
 * Grouping is decided per kind: a group exists only where a short name carries a
 * dot, and a kind whose short names carry none has no groups at all rather than
 * one group per metric. Inside a kind that does have groups, a short name with
 * no dot joins none of them, yet still counts toward the kind.
 *
 * Every geomean here is a plain `computeGeomean` call over a chosen subset, so
 * the unstable, undefined-ratio and infinite-ρ exclusions apply throughout, each
 * reported against the subset it was excluded from.
 *
 * @param verdicts The candidate's verdicts, keyed by metric name
 * @param metricMeta Metadata for every metric of the run, in first-appearance order
 */
export function computeKindAggregates(
  verdicts: Record<string, MetricVerdict>,
  metricMeta: Record<string, ResolvedMetricMeta>,
): KindAggregate[] {
  return Array.from(bucketByKind(metricMeta).entries(), ([kind, bucket]) => {
    const gating = bucket.metrics.filter(([, meta]) => meta.gating);

    return {
      kind,
      geomean: geomeanOver(bucket.metrics, verdicts),
      groups: Array.from(bucket.groups, ([group, members]) => ({
        group,
        geomean: geomeanOver(members, verdicts),
      })),
      ...(gating.length > 0 && { gatedGeomean: geomeanOver(gating, verdicts) }),
    };
  });
}
