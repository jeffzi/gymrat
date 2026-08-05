/**
 * The sectioned layout a multi-kind run is drawn in.
 *
 * Sorting a run's metrics into kinds and groups, and reading a candidate's
 * aggregate back out for each scope, are laid out here rather than inside a
 * renderer: it is what keeps a row and the geomean closing it describing the
 * same set of metrics.
 */

import type { ConfigKinds } from "../config.js";
import { inferGroup, type KindAggregate } from "../verdict/aggregate.js";
import type { GeomeanExclusion, GeomeanResult } from "../verdict/verdict.js";
import type { CandidateComparison, MetricComparison, MetricComparisons } from "./types.js";

/** A group of one section's metrics, or a metric belonging to no group. */
type SectionBlock<Metric> =
  | { readonly type: "group"; readonly group: string; readonly metrics: Metric[] }
  | { readonly type: "metric"; readonly metric: Metric };

/** One kind's slice of the table: what it holds, and whether the run is judged on it. */
export interface SectionPlan<Metric> {
  readonly kind: string;
  hasGating: boolean;
  readonly blocks: SectionBlock<Metric>[];
}

/** The run's metrics as sections, and as the flat list of rows a single-kind run draws. */
export interface SectionLayout<Metric> {
  readonly sections: readonly SectionPlan<Metric>[];
  /** Every metric in the order the run reported it, whatever section it landed in. */
  readonly ordered: readonly Metric[];
}

/** A metric's short name as its section shows it, the group prefix stripped when it has one. */
export function sectionLabel(shortName: string, group: string | undefined): string {
  return group === undefined ? shortName : shortName.slice(group.length + 1);
}

/**
 * Sort the run's metrics into one section per kind, and each section into the
 * groups its short names name.
 *
 * Kinds, groups and metrics keep first-appearance order — the order the
 * aggregates were computed in — so a section reads in the same order as the
 * rows its geomean covers. A group block sits where its first metric appeared
 * and gathers the rest of the group with it: a group split across the section by
 * a metric of another would leave its sub-geomean describing rows the reader has
 * to hunt for.
 *
 * Rows are built here rather than looked up later, so every row a section names
 * is the row the table draws. `measure` receives the inferred group rather than
 * a finished label, since what a renderer does with the prefix — strip it,
 * indent under it — is its own business.
 */
export function planSections<Metric>(
  metrics: MetricComparisons,
  measure: (name: string, group: string | undefined, metric: MetricComparison) => Metric,
): SectionLayout<Metric> {
  const sections = new Map<string, SectionPlan<Metric>>();
  const ordered: Metric[] = [];

  for (const [name, metric] of Object.entries(metrics)) {
    const { kind, shortName, gating } = metric.meta;

    let section = sections.get(kind);
    if (section === undefined) {
      section = { kind, hasGating: false, blocks: [] };
      sections.set(kind, section);
    }
    if (gating) section.hasGating = true;

    const group = inferGroup(shortName);
    const row = measure(name, group, metric);
    ordered.push(row);

    if (group === undefined) {
      section.blocks.push({ type: "metric", metric: row });
      continue;
    }

    const opened = section.blocks.find(
      (block): block is Extract<SectionBlock<Metric>, { type: "group" }> =>
        block.type === "group" && block.group === group,
    );
    if (opened !== undefined) {
      opened.metrics.push(row);
    } else {
      section.blocks.push({ type: "group", group, metrics: [row] });
    }
  }

  return { sections: [...sections.values()], ordered };
}

/**
 * Whether the run spans several kinds, and so is reported in sections.
 *
 * Read straight off the metrics rather than off a {@link SectionLayout}, so the
 * parts of a report drawn outside the table — the highlights above all — can ask
 * the question without building rows they have no use for.
 */
export function spansManyKinds(metrics: MetricComparisons): boolean {
  const kinds = new Set<string>();
  for (const metric of Object.values(metrics)) {
    kinds.add(metric.meta.kind);
  }
  return kinds.size > 1;
}

/** What a section's title says about a kind the run is not judged on. */
const INFORMATIONAL_TAG = "informational — gating off";

/**
 * The tag a non-gating kind's title carries, naming the config key that decided
 * it where one did.
 *
 * Gating is resolved per metric before the report sees it, so only the config
 * distinguishes a kind switched off wholesale from one whose metrics were each
 * switched off by name. Naming the key is what lets the reader switch it back.
 */
export function informationalTag(kind: string, configKinds: ConfigKinds | undefined): string {
  const source =
    configKinds?.[kind]?.gating === false ? ` (config: kinds.${kind}.gating = false)` : "";
  return `${INFORMATIONAL_TAG}${source}`;
}

/** The aggregate a candidate reported for one kind, or nothing when it reported none. */
function kindAggregateOf(candidate: CandidateComparison, kind: string): KindAggregate | undefined {
  return candidate.kinds.find((aggregate) => aggregate.kind === kind);
}

/** The exclusions list for a candidate that reported no aggregate. */
const NO_EXCLUDED: GeomeanExclusion[] = [];
Object.freeze(NO_EXCLUDED);

/**
 * The aggregate stated where a candidate reported none.
 *
 * Every section is drawn from the same metadata the aggregates were computed
 * from, so this stands in for nothing the renderers can produce — and if one
 * ever does, the row says it aggregated nothing rather than inventing a figure.
 */
const NO_AGGREGATE: Readonly<GeomeanResult> = Object.freeze({
  value: Number.NaN,
  n: 0,
  excluded: NO_EXCLUDED,
  band: 0,
});

/** The geomean over every metric of `kind`, gating or not. */
export function kindGeomeanOf(
  candidate: CandidateComparison,
  kind: string,
): Readonly<GeomeanResult> {
  return kindAggregateOf(candidate, kind)?.geomean ?? NO_AGGREGATE;
}

/** The geomean over one group of `kind`'s metrics. */
export function groupGeomeanOf(
  candidate: CandidateComparison,
  kind: string,
  group: string,
): Readonly<GeomeanResult> {
  const groups = kindAggregateOf(candidate, kind)?.groups;
  return groups?.find((entry) => entry.group === group)?.geomean ?? NO_AGGREGATE;
}

/**
 * The geomean a flat table closes on: the single kind's gating metrics.
 *
 * A run reporting one kind has no section to name, so its geomean row states
 * what the run is judged on without saying which kind that was.
 */
export function flatGeomeanOf(candidate: CandidateComparison): Readonly<GeomeanResult> {
  return candidate.kinds[0]?.gatedGeomean ?? NO_AGGREGATE;
}
