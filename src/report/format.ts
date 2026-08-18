import {
  MIN_BAND_N,
  MIN_WILCOXON_N,
  type ApproximateVerdictValue,
  type GeomeanResult,
  type MetricVerdict,
} from "../verdict/verdict.js";
import { formatLabel, type Style, VERDICT_STYLES } from "./format-style.js";
export * from "./format-style.js";
import type { CandidateMetric, MetricComparison, MetricComparisons } from "./types.js";

// ---------------------------------------------------------------------------
// Value and cell formatting
// ---------------------------------------------------------------------------

type Tier = readonly [threshold: number, divisor: number, suffix: string, decimals: number];

const NS_TIERS: readonly Tier[] = [
  [1000, 1, "ns", 0],
  [1e6, 1000, "µs", 1],
  [1e9, 1e6, "ms", 1],
  [Infinity, 1e9, "s", 1],
];

const BYTE_TIERS: readonly Tier[] = [
  [1000, 1, "B", 0],
  [1e6, 1000, "KB", 1],
  [1e9, 1e6, "MB", 1],
  [Infinity, 1e9, "GB", 1],
];

const TIER_MAP: Record<"ns" | "bytes", readonly Tier[]> = {
  ns: NS_TIERS,
  bytes: BYTE_TIERS,
};

/**
 * The tier a value prints in, chosen on the figure as rounded rather than as
 * measured.
 *
 * A value just under a threshold rounds up onto it — 999.5 bytes to `1000B`,
 * 999999.6ns to `1000.0µs` — which is a four-digit magnitude in a column sized
 * for three. Promoting it to the tier above keeps every cell inside the width
 * its tier is measured for.
 *
 * Thresholds are compared against the magnitude, since a sign is not a size: a
 * negative reading measured against the raw figure falls into the smallest tier
 * whatever it is worth, and `-1500000` prints in full where `-1.5MB` belongs.
 */
function scaleTier(value: number, tiers: readonly Tier[]): string {
  if (!Number.isFinite(value)) return value.toString();
  const magnitude = Math.abs(value);
  const tier = tiers.find(
    ([threshold, divisor, , decimals]) =>
      Number((magnitude / divisor).toFixed(decimals)) * divisor < threshold,
  );
  if (tier === undefined) return value.toString();
  const [, divisor, suffix, decimals] = tier;
  return `${(value / divisor).toFixed(decimals)}${suffix}`;
}

/** Scale a measurement into its unit's tier, or round it when the metric has no unit. */
export function formatValue(value: number, unit?: "ns" | "bytes"): string {
  if (!unit) {
    return Math.round(value).toString();
  }
  return scaleTier(value, TIER_MAP[unit]);
}

/** The sign every spread and noise band is stated behind. */
export const PLUS_MINUS = "±";

/**
 * What separates a value from the spread that follows it.
 *
 * Exported because the text table pads a cell's magnitude and spread into fields
 * of their own and has to rebuild the join around them.
 */
export const SPREAD_SEPARATOR = ` ${PLUS_MINUS} `;

/**
 * The scatter, relative to the median, past which a percentage stops informing.
 * Once the spread outgrows the median the percentage climbs without bound —
 * `± 7620%` reads as a rendering fault rather than as a measurement — so both
 * the value cells and the unstable evidence restate it in the metric's own
 * units at that point.
 */
const RELATIVE_SPREAD_CAP_PCT = 100;

/**
 * A value cell taken apart, so a table can pad each field to its own column width.
 *
 * Both fields are empty when the side reported nothing, and the spread alone is
 * empty when the measurement carries no scatter.
 */
export interface MetricCellParts {
  /** The scaled measurement. */
  readonly magnitude: string;
  /** What follows the `±`: a percentage, or absolute units once it outgrows the median. */
  readonly spread: string;
}

/**
 * A value cell's fields: the scaled measurement and the spread stated behind it.
 *
 * A spread past `RELATIVE_SPREAD_CAP_PCT` is restated in absolute units, so
 * `5B ± 7620%` reads `5B ± 381B` instead.
 */
export function formatMetricCellParts(
  median?: number,
  spread?: number,
  unit?: "ns" | "bytes",
): MetricCellParts {
  if (median === undefined) return { magnitude: "", spread: "" };
  const magnitude = formatValue(median, unit);
  if (spread === undefined) return { magnitude, spread: "" };
  if (spread > RELATIVE_SPREAD_CAP_PCT) {
    return { magnitude, spread: formatValue(Math.abs((median * spread) / 100), unit) };
  }
  return { magnitude, spread: `${spread.toFixed(0)}%` };
}

/**
 * A metric's own baseline figure, taken apart the way {@link formatMetricCellParts}
 * does — for a table that pads the magnitude and the spread into columns of
 * their own rather than joining them inline.
 *
 * Every renderer reads a metric's baseline off the same three fields, so this
 * is where that reading lives instead of being repeated at each call site.
 */
export function baselineCellParts(metric: MetricComparison): MetricCellParts {
  return formatMetricCellParts(metric.baselineMedian, metric.baselineSpread, metric.meta.unit);
}

/**
 * One candidate's side of a metric, taken apart the way {@link formatMetricCellParts}
 * does, for a table that pads the magnitude and the spread into columns of
 * their own.
 */
export function candidateCellParts(
  side: CandidateMetric | undefined,
  unit?: "ns" | "bytes",
): MetricCellParts {
  return formatMetricCellParts(side?.median, side?.spread, unit);
}

/**
 * A signed percentage, or nothing when the delta is not a number.
 *
 * A delta that rounds to zero prints as an unsigned `0.0%`: at display
 * precision there is no direction to report, so `-0.0%` would claim one.
 */
export function formatDelta(delta: number): string {
  if (Number.isNaN(delta)) return "";
  const magnitude = Math.abs(delta).toFixed(1);
  if (magnitude === "0.0") return "0.0%";
  return `${delta > 0 ? "+" : "-"}${magnitude}%`;
}

// ---------------------------------------------------------------------------
// Verdict classification
// ---------------------------------------------------------------------------

/**
 * How a verdict presents itself in the report.
 *
 * `no-signal` splits in three here: a metric resting on too few pairs for the
 * band method to report a signal reads `inconclusive`, one whose two sides
 * measured close enough to identical to starve the signed-rank test reads
 * `identical`, and every other no-signal verdict reads `within-noise`. The split
 * is presentation only — the stored verdict stays `no-signal`, so geomean
 * gating, `--fail-on` and the JSON report see one class where the report shows
 * three.
 */
export type DisplayClass =
  | "improved"
  | "regressed"
  | "unstable"
  | "identical"
  | "within-noise"
  | "inconclusive";

/**
 * Which display class a verdict reads as.
 *
 * The pair count is read before anything else about a band verdict: below
 * `MIN_BAND_N` the band is the noise floor constant rather than a measurement,
 * so neither the delta measured against it nor a tie between the two readings
 * carries the news its glyph would claim. That reads `inconclusive`.
 *
 * `usableN` counts pairs, not a proportion of them: a band verdict with enough
 * pairs for the signed-rank test reads `identical` only when every one of them
 * tied (`usableN === 0`), which is the one case where the two sides truly
 * measured the same. Anywhere `usableN` sits between `0` and `MIN_WILCOXON_N`,
 * some pairs did differ — there was signal, just not enough of it for a
 * statistical verdict — so that reads `within-noise` instead. An
 * `ExactVerdict` carries no `usableN`, so an exact no-signal always reads
 * `within-noise`.
 */
export function displayClass(verdict: MetricVerdict): DisplayClass {
  if (verdict.method === "band" && verdict.n < MIN_BAND_N) return "inconclusive";
  if (verdict.verdict !== "no-signal") return verdict.verdict;
  if (verdict.method === "band" && verdict.n >= MIN_WILCOXON_N && verdict.usableN === 0) {
    return "identical";
  }
  return "within-noise";
}

/** `displayClass`, or `undefined` when there is no verdict to show one for. */
export function shownClass(verdict: MetricVerdict | undefined): DisplayClass | undefined {
  return verdict === undefined ? undefined : displayClass(verdict);
}

const GLYPHS: Record<DisplayClass, string> = {
  improved: "✓",
  regressed: "✗",
  unstable: "≈",
  identical: "=",
  "within-noise": "~",
  inconclusive: "?",
};

/**
 * ✓ improved, ✗ regressed, ≈ unstable, = identical, ~ within noise,
 * ? inconclusive — the glyphs the report's rows and legend are written in.
 */
export function getGlyph(shown: DisplayClass): string {
  return GLYPHS[shown];
}

/**
 * The word each display class reads as, shared by the summary line and the legend.
 *
 * Typed as a `Record` over the display union rather than listed inline at each
 * call site: a class added to the union without an entry here is a compile
 * error, instead of a class that renders in the table but is missing from both
 * the summary and the legend.
 */
export const VERDICT_GLOSSES: Record<DisplayClass, string> = {
  improved: "improved",
  regressed: "regressed",
  unstable: "unstable",
  identical: "identical",
  "within-noise": "within noise",
  inconclusive: "inconclusive",
};

/** The delta cell: the word `unstable` for a verdict too noisy to trust, otherwise the signed percentage. */
export function formatVerdictDelta(verdict: MetricVerdict): string {
  return verdict.verdict === "unstable" ? "unstable" : formatDelta(verdict.delta);
}

/**
 * The display classes whose rows carry no news worth keeping above the fold.
 *
 * A metric that sat within the noise, measured identical, rested on too few
 * pairs to judge, or was too jittery to judge, has its verdict styled to recede
 * rather than competing with the rows that moved.
 */
export const QUIET_VERDICTS: ReadonlySet<DisplayClass> = new Set([
  "within-noise",
  "identical",
  "inconclusive",
  "unstable",
]);

// ---------------------------------------------------------------------------
// Geomean labeling
// ---------------------------------------------------------------------------

/** The name the geomean row is reported under, in every renderer. */
export const GEOMEAN_LABEL = "geomean";

/**
 * The name the gate-trip highlight gives its figure.
 *
 * The per-kind geomean row shows the ungated aggregate, but the gate reads
 * the gated one — naming both "geomean" conflates two different numbers.
 */
export const GATED_GEOMEAN_LABEL = "gated geomean";

/**
 * Append an `s` to `noun` when `count` is not exactly one, or use `plural`
 * in its place when the plural form isn't a simple suffix.
 */
export function pluralize(count: number, noun: string, plural?: string): string {
  return `${count} ${count === 1 ? noun : (plural ?? `${noun}s`)}`;
}

/**
 * The geomean row's label, carrying the count of metrics behind the figure.
 *
 * A table with one candidate has a single count to name and names it here,
 * which is what frees its cells of everything but the aggregate itself. A table
 * with several has a count per candidate, no one of which describes the row, so
 * it takes {@link GEOMEAN_LABEL} and carries each count in its own cell — the
 * same form an empty geomean takes, having no count to name.
 */
export function geomeanLabel(n: number): string {
  return n === 0 ? GEOMEAN_LABEL : `${GEOMEAN_LABEL} (${pluralize(n, "stable metric")})`;
}

/** The scope separator inside a sectioned table's aggregate labels. */
const SCOPE_SEPARATOR = "·";

/**
 * The label of an aggregate row covering one scope of a sectioned table — a
 * group or a kind.
 *
 * A sectioned table repeats the geomean row per scope, so each one names what it
 * covers; the flat table has a single geomean and takes {@link geomeanLabel}.
 */
export function geomeanScopeLabel(scope: string): string {
  return `${GEOMEAN_LABEL} ${SCOPE_SEPARATOR} ${scope}`;
}

/**
 * How many of the scope's metrics stand behind its figure: `(n)` when they all
 * do, `(n/m)` when exclusions thinned them.
 *
 * The count alone would read as the size of the scope, which it is only when
 * nothing was excluded — and a reader comparing `geomean · memory (13/14)`
 * with the section above it can see at a glance that a metric dropped out.
 */
function geomeanProvenance(geomean: GeomeanResult): string {
  const total = geomean.n + geomean.excluded.length;
  return total === geomean.n ? `(${geomean.n})` : `(${geomean.n}/${total})`;
}

/**
 * A sectioned table's aggregate label with the provenance behind its figure.
 *
 * The single-candidate table has one count per row and names it here, which is
 * what leaves its cell holding the figure alone — the same division of labor
 * {@link geomeanLabel} makes in the flat table.
 */
export function scopedGeomeanLabel(scope: string, geomean: GeomeanResult): string {
  return `${geomeanScopeLabel(scope)} ${geomeanProvenance(geomean)}`;
}

/** The figure standing in for a geomean with nothing to aggregate. */
export const NO_GEOMEAN_FIGURE = "—";

/** What an empty geomean says in place of the count behind a figure. */
export const NO_STABLE_METRICS = "no stable metrics";

/**
 * The whole cell a geomean with nothing left to aggregate prints.
 *
 * It says so rather than printing the 0.0% an empty geomean computes to, which
 * would read as "no change measured".
 */
export const NO_GEOMEAN_CELL = `${NO_GEOMEAN_FIGURE}  ${NO_STABLE_METRICS}`;

// ---------------------------------------------------------------------------
// Verdict evidence and tallies
// ---------------------------------------------------------------------------

/**
 * The evidence suffix for a highlighted metric.
 *
 * Exact entries keep `(exact)`. Unstable entries show the noise that swamped the
 * signal — as a percentage while that stays readable, and against the baseline
 * median in the metric's own units past `RELATIVE_SPREAD_CAP_PCT`, where the
 * percentage no longer conveys the scale. Improved/regressed/no-signal entries
 * from approximate methods carry no trailing evidence — the glyph and delta
 * already tell the story.
 */
export function formatEvidence(
  verdict: MetricVerdict,
  unit?: "ns" | "bytes",
  baselineMedian?: number,
): string {
  if (verdict.method === "exact") return "(exact)";
  if (verdict.verdict !== "unstable") return "";
  if (verdict.noisePct > RELATIVE_SPREAD_CAP_PCT && baselineMedian !== undefined) {
    const noise = formatValue(verdict.noiseAbs, unit);
    return `±${noise} noise on a ${formatValue(baselineMedian, unit)} median`;
  }
  return `noise ${formatNoiseBand(verdict.noisePct)}`;
}

/**
 * What stops a reader re-running the suite in the hope of a cleaner verdict.
 *
 * An unstable verdict is judged against a half-range band, and a half-range
 * never shrinks as samples accumulate: more rounds can only widen it. Whatever
 * else changes the wording here, it must not promise that more samples help.
 */
export const UNSTABLE_FUTILITY_NOTE = "unstable metrics won't stabilize with more samples";

/**
 * A noise band's figure, without the sign it is stated behind.
 *
 * The text table pins the `±` of a verdict column and right-aligns this behind
 * it, so it needs the two apart.
 */
export function formatNoiseBandValue(noisePct: number): string {
  return `${noisePct.toFixed(1)}%`;
}

/** A metric's noise band, as the `±N%` the row annotations and highlights share. */
function formatNoiseBand(noisePct: number): string {
  return `${PLUS_MINUS}${formatNoiseBandValue(noisePct)}`;
}

/**
 * How many pairs a verdict rests on, as the `n=N` the rows and the method
 * footer share.
 */
export function formatPairCount(n: number): string {
  return `n=${n}`;
}

/** How many metrics landed in each verdict class. */
export interface VerdictCounts {
  improved: number;
  regressed: number;
  unstable: number;
  noSignal: number;
}

/**
 * Maps a verdict's outcome to its {@link VerdictCounts} field.
 *
 * A `Record` over the full {@link ApproximateVerdictValue} union keeps this
 * exhaustive at compile time: a new verdict variant fails to type-check here
 * until this map accounts for it.
 */
const COUNT_KEY: Record<ApproximateVerdictValue, keyof VerdictCounts> = {
  improved: "improved",
  regressed: "regressed",
  unstable: "unstable",
  "no-signal": "noSignal",
};

/**
 * Run `fn` on one candidate's verdict for every metric that reported one.
 *
 * Verdicts belong to a candidate, never to the run, so callers always read one
 * candidate at a time. Metrics that candidate never reported have no verdict
 * and are skipped rather than counted.
 */
function forEachVerdict(
  metrics: MetricComparisons,
  candidateIndex: number,
  fn: (verdict: MetricVerdict) => void,
): void {
  for (const metric of Object.values(metrics)) {
    const verdict = metric.candidates[candidateIndex]?.verdict;
    if (verdict !== undefined) fn(verdict);
  }
}

/**
 * Tally the stored verdict classes one candidate earned against the baseline.
 *
 * These are the verdicts as decided, not as displayed — the JSON report is
 * written from them, so `noSignal` covers every no-signal metric whether the
 * text report shows it as identical or as within noise. The summary line counts
 * through {@link verdictSummaryParts} instead.
 */
export function countVerdicts(metrics: MetricComparisons, candidateIndex: number): VerdictCounts {
  const counts: VerdictCounts = { improved: 0, regressed: 0, unstable: 0, noSignal: 0 };
  forEachVerdict(metrics, candidateIndex, (verdict) => {
    counts[COUNT_KEY[verdict.verdict]] += 1;
  });
  return counts;
}

// ---------------------------------------------------------------------------
// Highlight selection
// ---------------------------------------------------------------------------

/** A candidate's side of a metric, known to carry a verdict. */
interface HighlightedCandidate extends CandidateMetric {
  verdict: MetricVerdict;
}

/** A metric worth calling out for one candidate, with the name it is reported under. */
interface MetricHighlight {
  name: string;
  metric: MetricComparison;
  candidate: HighlightedCandidate;
}

/**
 * Where each highlighted display class sits in the reported order.
 *
 * Total rather than partial: `undefined` is how a class opts out of highlights,
 * so a class added to the union without an entry here is a compile error rather
 * than a class that silently never appears.
 */
const HIGHLIGHT_RANK: Record<DisplayClass, number | undefined> = {
  regressed: 0,
  improved: 1,
  unstable: 2,
  identical: undefined,
  "within-noise": undefined,
  inconclusive: undefined,
};

/**
 * How loud a highlight is within its class: noise for unstable metrics, which
 * have no trustworthy delta, and the delta's magnitude for everything else.
 */
function highlightWeight(verdict: MetricVerdict): number {
  if (verdict.verdict === "unstable") {
    return verdict.noisePct;
  }
  const magnitude = Math.abs(verdict.delta);
  return Number.isNaN(magnitude) ? 0 : magnitude;
}

/**
 * The metrics worth calling out for one candidate, ordered regressions first (by
 * delta magnitude, descending), then improvements the same way, then unstable
 * metrics by noise.
 *
 * Ranking is per candidate because the verdicts are: the same metric can be the
 * loudest regression for one candidate and unremarkable for the next. Metrics
 * that sat within the noise or measured identical carry no news, so they are
 * left out entirely, and so is a metric this candidate never reported: with no
 * verdict it has nothing to rank it against the rest. Ties keep the order the
 * metrics were measured in.
 */
export function selectHighlights(
  metrics: MetricComparisons,
  candidateIndex: number,
): readonly MetricHighlight[] {
  const ranked: { highlight: MetricHighlight; rank: number; weight: number }[] = [];
  for (const [name, metric] of Object.entries(metrics)) {
    const candidate = metric.candidates[candidateIndex];
    if (candidate?.verdict === undefined) continue;
    const rank = HIGHLIGHT_RANK[displayClass(candidate.verdict)];
    if (rank === undefined) continue;
    ranked.push({
      highlight: { name, metric, candidate: { ...candidate, verdict: candidate.verdict } },
      rank,
      weight: highlightWeight(candidate.verdict),
    });
  }
  ranked.sort((a, b) => a.rank - b.rank || b.weight - a.weight);
  return ranked.map(({ highlight }) => highlight);
}

/**
 * The name a highlight is reported under, its kind named ahead of it when the
 * run spans several.
 *
 * The highlights sit below the table, away from the section titles that told the
 * reader which kind a row belonged to, so a multi-kind run has to carry the kind
 * on the line itself. A single-kind run would say the same word on every line,
 * which tells the reader nothing and only pushes the deltas right.
 */
export function highlightLabel(highlight: MetricHighlight, qualify: boolean): string {
  return qualify
    ? `${highlight.metric.meta.kind} ${SCOPE_SEPARATOR} ${highlight.metric.meta.shortName}`
    : highlight.name;
}

/** Whether any of these highlights is one the noise swamped, and so carries no usable delta. */
export function hasUnstableHighlight(highlights: readonly MetricHighlight[]): boolean {
  return highlights.some(({ candidate }) => displayClass(candidate.verdict) === "unstable");
}

/** One candidate's highlight entries, and whether the noise swamped any of them. */
export interface HighlightBlock {
  readonly label?: string;
  readonly entries: readonly string[];
  readonly unstable: boolean;
}

// ---------------------------------------------------------------------------
// Geomean aggregation and styling
// ---------------------------------------------------------------------------

/** The geomean's delta and the provenance describing what stands behind it. */
interface GeomeanParts {
  readonly delta: string;
  readonly provenance: string;
  /**
   * The propagated band's figure, without the `±` a column pins in front of it,
   * and empty where the metrics behind the figure left it nothing to state.
   */
  readonly band: string;
}

/**
 * The geomean's delta, the band propagated from the metrics behind it, and how
 * many those are, or `null` when nothing survived to aggregate.
 *
 * The metrics left out are named nowhere near the figure: an unstable metric is
 * already tallied in the verdict summary and flagged in the highlights, so
 * restating the exclusions here spent the row's width on news the reader has.
 *
 * The renderer wraps the parts into its own cell shape, and an empty geomean
 * into {@link NO_GEOMEAN_CELL}.
 */
export function geomeanParts(geomean: GeomeanResult): GeomeanParts | null {
  if (geomean.n === 0) return null;
  // A band of zero is what an aggregate over exact-only metrics propagates:
  // there is no noise to state, and `±0.0%` would read as a measurement.
  return {
    delta: formatDelta(geomean.value),
    provenance: pluralize(geomean.n, "stable metric"),
    band: geomean.band > 0 ? formatNoiseBandValue(geomean.band) : "",
  };
}

/**
 * How a geomean's figure is styled: emboldened always, and colored only once it
 * clears the noise band propagated from the metrics behind it — and only when
 * one of those metrics reported something.
 *
 * The figure is an average of ratios, so it moves whether or not anything did.
 * Coloring it by sign alone would call a run green on a drift smaller than the
 * noise its own metrics carry; a value inside the band is emboldened and left
 * uncolored, which reads as "measured, nothing to conclude". A geomean with
 * nothing to aggregate has a `NaN` value, and both comparisons below leave it
 * uncolored.
 *
 * The band is not the whole guard, because it is propagated from each metric's
 * own noise and shrinks as metrics are averaged: a run where every metric came
 * back quiet can still produce a figure outside it. Coloring that green would
 * announce a win the rows beneath it all decline to claim, so `outcomes` — the
 * display class of each metric the figure covers, one per metric — vetoes the
 * color when every one of them is quiet. Passing none leaves the band deciding
 * alone, for callers holding a figure with no verdicts behind it.
 */
export function geomeanValueStyle(
  geomean: GeomeanResult,
  outcomes: ReadonlyArray<DisplayClass | undefined> = [],
): Style {
  if (isQuietRow(outcomes)) return ["bold"];
  if (geomean.value < -geomean.band) return ["bold", "green"];
  if (geomean.value > geomean.band) return ["bold", "red"];
  return ["bold"];
}

/**
 * Whether every defined display class in a row belongs to
 * {@link QUIET_VERDICTS}.
 *
 * Used to decide whether a row carries no news worth keeping above the fold.
 * A row with no verdicts at all is left alone rather than counted as quiet.
 */
function isQuietRow(outcomes: ReadonlyArray<DisplayClass | undefined>): boolean {
  const defined = outcomes.flatMap((outcome) => (outcome === undefined ? [] : [outcome]));
  return defined.length > 0 && defined.every((outcome) => QUIET_VERDICTS.has(outcome));
}

// ---------------------------------------------------------------------------
// Verdict summary
// ---------------------------------------------------------------------------

/** How many metrics one candidate landed in each display class. */
function displayCounts(
  metrics: MetricComparisons,
  candidateIndex: number,
): Record<DisplayClass, number> {
  const counts: Record<DisplayClass, number> = {
    improved: 0,
    regressed: 0,
    unstable: 0,
    identical: 0,
    "within-noise": 0,
    inconclusive: 0,
  };
  forEachVerdict(metrics, candidateIndex, (verdict) => {
    counts[displayClass(verdict)] += 1;
  });
  return counts;
}

/** The order the summary line and the legend list display classes in. */
const DISPLAY_CLASS_ORDER: readonly DisplayClass[] = [
  "improved",
  "regressed",
  "unstable",
  "identical",
  "within-noise",
  "inconclusive",
];

/**
 * One tally part per display class, in the order the report legend lists them.
 *
 * The text renderer joins these with `"   "` for aligned columns.
 *
 * Each part carries its class color, and a part counting nothing is dimmed
 * whatever its class — a zero is not news either way. `within noise` and
 * `inconclusive` read dim at any count, their class color being dim. Color is
 * governed by `styleText` auto-detection.
 */
export function verdictSummaryParts(metrics: MetricComparisons, candidateIndex: number): string[] {
  const counts = displayCounts(metrics, candidateIndex);

  return DISPLAY_CLASS_ORDER.map((shown) => {
    const count = counts[shown];
    const text = `${getGlyph(shown)} ${count} ${VERDICT_GLOSSES[shown]}`;
    const style: Style = count === 0 ? ["dim"] : VERDICT_STYLES[shown];
    return formatLabel(text, style);
  });
}

// ---------------------------------------------------------------------------
// Footer generation
// ---------------------------------------------------------------------------

const SAMPLES_HINT = `re-run with --samples ${MIN_WILCOXON_N} or more for statistical verdicts`;

/** How the band method names itself wherever the footer describes a fallback. */
const BAND_METHOD = "noise band ±(half-range × K)";

/**
 * Everything the footer needs from the verdicts, collected in one pass.
 *
 * `signedRank` carries the pair counts of every signed-rank verdict.
 * `shortage` and `ties` sort the band-method verdicts by the cause that
 * forced the fallback: too few total pairs, or too many of them tied away.
 */
interface FooterData {
  signedRank: number[];
  shortage: number[];
  ties: number[];
}

function classifyVerdict(verdict: MetricVerdict, data: FooterData): void {
  if (verdict.method === "signed-rank") {
    data.signedRank.push(verdict.n);
    return;
  }
  if (verdict.method !== "band") return;
  if (verdict.n < MIN_WILCOXON_N) {
    data.shortage.push(verdict.n);
    return;
  }
  data.ties.push(verdict.usableN);
}

function collectFooterData(metrics: MetricComparisons): FooterData {
  const data: FooterData = { signedRank: [], shortage: [], ties: [] };
  for (const metric of Object.values(metrics)) {
    for (const { verdict } of metric.candidates) {
      if (verdict !== undefined) classifyVerdict(verdict, data);
    }
  }
  return data;
}

/**
 * The verbose method lines naming how each verdict was decided.
 *
 * A band fallback gets one line per cause, because the counts that explain a
 * short run and a tie-starved one are different numbers, and each is picked so
 * the line stays true of every metric behind it: the highest total pair count
 * for a shortage — even the best-off metric fell this far short — and the lowest
 * usable pair count for ties. A run that hit both causes on different metrics
 * gets both lines.
 *
 * Every line is dimmed via `styleText` auto-detection.
 */
function methodLinesFrom({ signedRank, shortage, ties }: FooterData): string[] {
  const lines: string[] = [];

  if (signedRank.length > 0) {
    const desc = `verdicts: Wilcoxon signed-rank on pairs (${formatPairCount(Math.min(...signedRank))} ≥ ${MIN_WILCOXON_N}) · ~ = no signal at α=0.05`;
    lines.push(formatLabel(desc, ["dim"]));
  }
  if (shortage.length > 0) {
    const desc = `${BAND_METHOD} — ${formatPairCount(Math.max(...shortage))} below signed-rank floor (${MIN_WILCOXON_N} pairs)`;
    lines.push(formatLabel(desc, ["dim"]));
  }
  if (ties.length > 0) {
    const desc = `${BAND_METHOD} — ties left ${formatPairCount(Math.min(...ties))} usable pairs (${MIN_WILCOXON_N} needed)`;
    lines.push(formatLabel(desc, ["dim"]));
  }

  return lines;
}

/**
 * The footer: how each verdict was decided when verbose, and — either way — the
 * hint telling the reader when more samples would buy a statistical verdict.
 *
 * Renderers differ only in how they format the hint line, which `formatHint`
 * owns: the text renderer prepends a styled label.
 */
export function footerLines(
  metrics: MetricComparisons,
  verbose: boolean,
  formatHint: (hint: string) => string,
): string[] {
  const data = collectFooterData(metrics);
  return [
    ...(verbose ? methodLinesFrom(data) : []),
    ...(data.shortage.length > 0 ? [formatHint(SAMPLES_HINT)] : []),
  ];
}
