import { styleText } from "node:util";

import { assertNever } from "../errors.js";
import {
  MIN_WILCOXON_N,
  type GeomeanResult,
  type Method,
  type MetricVerdict,
} from "../verdict/verdict.js";
import type {
  CandidateMetric,
  ComparisonResult,
  MetricComparison,
  MetricComparisons,
} from "./types.js";

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

function scaleTier(value: number, tiers: readonly Tier[]): string {
  if (!Number.isFinite(value)) return value.toString();
  const tier = tiers.find(([threshold]) => value < threshold);
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

/** The `± N%` suffix that follows a value, or nothing when the spread is unknown. */
export function formatSpread(spread?: number): string {
  if (spread === undefined) return "";
  return `${SPREAD_SEPARATOR}${spread.toFixed(0)}%`;
}

/**
 * The scatter, relative to the median, past which a percentage stops informing.
 *
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
    return { magnitude, spread: formatValue((median * spread) / 100, unit) };
  }
  return { magnitude, spread: `${spread.toFixed(0)}%` };
}

/**
 * A value cell as one string, or nothing when unmeasured.
 *
 * The fields are joined by a single space either side of the `±`: this is the
 * form markdown embeds in its cells, where the renderer does no padding of its
 * own. The text table pads {@link formatMetricCellParts} instead.
 */
export function formatMetricCell(median?: number, spread?: number, unit?: "ns" | "bytes"): string {
  const { magnitude, spread: scatter } = formatMetricCellParts(median, spread, unit);
  if (magnitude === "" || scatter === "") return magnitude;
  return `${magnitude}${SPREAD_SEPARATOR}${scatter}`;
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

/** A p-value at reading precision, collapsed to `p<0.001` below the display floor. */
export function formatPValue(p: number): string {
  if (p < 0.001) return "p<0.001";
  if (p < 0.01) return `p=${p.toFixed(3)}`;
  return `p=${p.toFixed(2)}`;
}

/**
 * How a verdict presents itself in the report.
 *
 * `no-signal` splits in two here: a metric whose two sides measured close enough
 * to identical to starve the signed-rank test reads `identical`, and every other
 * no-signal verdict reads `within-noise`. The split is presentation only — the
 * stored verdict stays `no-signal`, so geomean gating, `--fail-on` and the JSON
 * report see one class where the report shows two.
 */
export type DisplayClass = "improved" | "regressed" | "unstable" | "identical" | "within-noise";

/**
 * Which display class a verdict reads as.
 *
 * A band verdict with enough pairs for the signed-rank test but too few usable
 * ones fell back because tied pairs zeroed the differences out — the two sides
 * measured the same, which `identical` says and `within noise` does not. An
 * `ExactVerdict` carries no `usableN`, so an exact no-signal always reads
 * `within-noise`.
 */
export function displayClass(verdict: MetricVerdict): DisplayClass {
  if (verdict.verdict !== "no-signal") return verdict.verdict;
  if (
    verdict.method === "band" &&
    verdict.n >= MIN_WILCOXON_N &&
    verdict.usableN < MIN_WILCOXON_N
  ) {
    return "identical";
  }
  return "within-noise";
}

const GLYPHS: Record<DisplayClass, string> = {
  improved: "✓",
  regressed: "✗",
  unstable: "≈",
  identical: "=",
  "within-noise": "~",
};

/**
 * ✓ improved, ✗ regressed, ≈ unstable, = identical, ~ within noise — the glyphs
 * the report's rows and legend are written in.
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
};

/** The delta cell: the word `unstable` for a verdict too noisy to trust, otherwise the signed percentage. */
export function formatVerdictDelta(verdict: MetricVerdict): string {
  return verdict.verdict === "unstable" ? "unstable" : formatDelta(verdict.delta);
}

/**
 * The display classes whose rows carry no news worth keeping above the fold.
 *
 * Shared by every renderer: a metric that sat within the noise, measured
 * identical, or was too jittery to judge, has its verdict styled to recede
 * (text) or is collapsed into a `<details>` block (markdown) rather than
 * competing with the rows that moved.
 */
export const QUIET_VERDICTS: ReadonlySet<DisplayClass> = new Set([
  "within-noise",
  "identical",
  "unstable",
]);

/** The name the geomean row is reported under, in every renderer. */
export const GEOMEAN_LABEL = "geomean";

/** How many gating metrics an aggregate rests on, as the reader is told. */
function stableMetrics(n: number): string {
  return `${n} stable metric${n === 1 ? "" : "s"}`;
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
  return n === 0 ? GEOMEAN_LABEL : `${GEOMEAN_LABEL} (${stableMetrics(n)})`;
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
export function formatNoiseBand(noisePct: number): string {
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
 * Tally the stored verdict classes one candidate earned against the baseline.
 *
 * These are the verdicts as decided, not as displayed — the JSON report is
 * written from them, so `noSignal` covers every no-signal metric whether the
 * text report shows it as identical or as within noise. The summary line counts
 * through {@link verdictSummaryParts} instead.
 *
 * Verdicts belong to a candidate, never to the run, so the tally is taken one
 * candidate at a time. Metrics that candidate never reported have no verdict
 * and count towards nothing.
 */
export function countVerdicts(metrics: MetricComparisons, candidateIndex: number): VerdictCounts {
  const counts: VerdictCounts = { improved: 0, regressed: 0, unstable: 0, noSignal: 0 };
  for (const metric of Object.values(metrics)) {
    const verdict = metric.candidates[candidateIndex]?.verdict;
    if (verdict === undefined) continue;
    const outcome = verdict.verdict;
    switch (outcome) {
      case "improved":
        counts.improved += 1;
        break;
      case "regressed":
        counts.regressed += 1;
        break;
      case "unstable":
        counts.unstable += 1;
        break;
      case "no-signal":
        counts.noSignal += 1;
        break;
      default:
        assertNever(outcome);
    }
  }
  return counts;
}

/** A candidate's side of a metric, known to carry a verdict. */
export interface HighlightedCandidate extends CandidateMetric {
  verdict: MetricVerdict;
}

/** A metric worth calling out for one candidate, with the name it is reported under. */
export interface MetricHighlight {
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

/** Whether any of these highlights is one the noise swamped, and so carries no usable delta. */
export function hasUnstableHighlight(highlights: readonly MetricHighlight[]): boolean {
  return highlights.some(({ candidate }) => displayClass(candidate.verdict) === "unstable");
}

/**
 * Width a column needs to hold its header and every cell, never below `minWidth`.
 *
 * The result includes two columns of gutter beyond the widest content, so
 * callers padding cells to this width get a visible gap before the next
 * column's separator for free.
 */
export function computeColumnWidth(
  headerLength: number,
  contentLengths: number[],
  minWidth: number,
): number {
  const maxContent = Math.max(headerLength, ...contentLengths);
  return Math.max(maxContent + 2, minWidth);
}

/** How a cell renders once it has been padded to its column width. */
export type CellStyler = (cell: string, index: number) => string;

/**
 * Pad each cell to its column width and join them with the column separator.
 *
 * `styleCell` runs on the finished line, after the padding and the trailing
 * trim: every width in the table is measured on plain text, and an ANSI escape
 * introduced before padding would be counted as visible width, shifting every
 * column to its right.
 */
export function formatTableLine(
  cells: readonly string[],
  widths: readonly number[],
  styleCell?: CellStyler,
): string {
  const padded = widths.map((width, i) => (cells[i] ?? "").padEnd(width));
  // Only the trailing run is cut: leading padding is the first column's width,
  // which a row opening on an empty cell needs to stay under the header.
  const line = padded.join("│").trimEnd();
  if (styleCell === undefined) return line;
  return line.split("│").map(styleCell).join("│");
}

/** The style-tag union {@link styleText} accepts — the type every renderer's style constants are typed against. */
export type Style = Parameters<typeof styleText>[0];

/**
 * Apply `style` to `label` via `styleText`.
 *
 * `stream` lets callers target `styleText`'s TTY/color auto-detection at a
 * specific stream instead of the default `process.stdout`. Either way,
 * `styleText` returns the bare label when the environment suppresses styling
 * (`NO_COLOR`, a non-TTY stream).
 */
export function formatLabel(label: string, style: Style, stream?: NodeJS.WriteStream): string {
  return stream !== undefined ? styleText(style, label, { stream }) : styleText(style, label);
}

/**
 * Widest a variant label prints, ellipsis included.
 *
 * A branch name is free to be as long as git allows, but every column it heads
 * is sized from it, so an unbounded one pushes the figures the report exists to
 * show off the right edge of the terminal.
 */
const LABEL_DISPLAY_WIDTH = 20;

/** U+2026, one character wide — three periods would cost two more columns. */
const ELLIPSIS = "…";

/** Characters kept from the head of an overlong label. */
const LABEL_HEAD_WIDTH = Math.ceil((LABEL_DISPLAY_WIDTH - ELLIPSIS.length) / 2);

/** Characters kept from the tail of an overlong label before collisions widen it. */
const LABEL_TAIL_WIDTH = LABEL_DISPLAY_WIDTH - ELLIPSIS.length - LABEL_HEAD_WIDTH;

/** One label under the name it prints, keeping `tail` of its trailing characters. */
function shortenLabel(label: string, tail: number): string {
  if (label.length <= LABEL_DISPLAY_WIDTH) return label;
  return `${label.slice(0, LABEL_HEAD_WIDTH)}${ELLIPSIS}${label.slice(-tail)}`;
}

/**
 * Every variant label under the name the report prints for it.
 *
 * Labels are shortened as a set rather than one at a time because the tail is
 * what tells sibling branches apart: `feature/experiment-one-fastpath` and
 * `feature/exploration-two-fastpath` share both ends, so the shortest tail that
 * keeps them distinct is the one worth keeping. The tail grows until the
 * displayed names are as distinct as the labels were — two candidates named
 * identically stay that way, which is the run's own doing, not the display's.
 *
 * Truncation is display-only: `label=` parsing, the config, and the JSON
 * renderer all keep the full label.
 */
export function truncateLabels(labels: readonly string[]): string[] {
  const longest = Math.max(0, ...labels.map((label) => label.length));
  const distinct = new Set(labels).size;
  for (let tail = LABEL_TAIL_WIDTH; tail < longest; tail++) {
    const shortened = labels.map((label) => shortenLabel(label, tail));
    if (new Set(shortened).size === distinct) return shortened;
  }
  return [...labels];
}

/**
 * `result` with every variant label replaced by the name the report prints.
 *
 * Every renderer reads its labels off this copy, so a label is shortened once
 * per report and prints the same way in the header, the column it heads, the
 * geomean row and the legend.
 */
export function withDisplayLabels(result: ComparisonResult): ComparisonResult {
  const [baseline, ...candidates] = truncateLabels([
    result.baselineLabel,
    ...result.candidates.map((candidate) => candidate.label),
  ]);
  return {
    ...result,
    baselineLabel: baseline ?? result.baselineLabel,
    candidates: result.candidates.map((candidate, index) => ({
      ...candidate,
      label: candidates[index] ?? candidate.label,
    })),
  };
}

/** The style a variant name wears where the report names it as a name. */
export const VARIANT_NAME_STYLE: Style = ["bold", "underline"];

/** Probe text whose styled form differs from itself whenever styling is live. */
const STYLE_PROBE = "x";

/**
 * Whether `styleText` is emitting escapes at all right now.
 *
 * `styleText` answers this only by doing it — it returns the bare string when
 * `NO_COLOR`, a non-TTY stream, or anything else suppresses styling — so the
 * probe asks it the same question the renderer is about to.
 */
function stylingActive(stream?: NodeJS.WriteStream): boolean {
  return formatLabel(STYLE_PROBE, VARIANT_NAME_STYLE, stream) !== STYLE_PROBE;
}

/**
 * A variant name as it prints before {@link VARIANT_NAME_STYLE} wraps it.
 *
 * Emphasis is what separates a branch name from the prose around it. Where
 * there is none to be had the quotes do that work instead, so a piped report
 * still reads `baseline "main" ↔ "perf/simd"` rather than running the names
 * into the sentence carrying them.
 *
 * Callers padding a cell need this plain form for the width, then apply
 * {@link VARIANT_NAME_STYLE} to the padded cell; callers writing prose can take
 * {@link formatVariantName} instead.
 */
export function variantName(label: string, stream?: NodeJS.WriteStream): string {
  return stylingActive(stream) ? label : `"${label}"`;
}

/** A variant name, styled — for the unpadded prose of the run header. */
export function formatVariantName(label: string, stream?: NodeJS.WriteStream): string {
  return formatLabel(variantName(label, stream), VARIANT_NAME_STYLE, stream);
}

const HINT_STYLE: Style = ["yellow", "underline"];

/** The `Hint:` label every hint line opens with, styled by `styleText` auto-detection. */
export function formatHintLabel(stream?: NodeJS.WriteStream): string {
  return formatLabel("Hint:", HINT_STYLE, stream);
}

/**
 * The color each display class wears wherever the report states a verdict.
 *
 * Every style here is worn by the verdict itself — a glyph, a delta, a tally —
 * never by the row or the values around it, so a class that recedes has to say
 * so in its own color: within noise dims, identical reads cyan for "measured
 * the same", and unstable keeps its amber warning.
 */
export const VERDICT_STYLES: Record<DisplayClass, Style> = {
  improved: ["green"],
  regressed: ["red"],
  unstable: ["yellow"],
  identical: ["cyan"],
  "within-noise": ["dim"],
};

/**
 * Style `marker` where it sits inside an already-padded cell.
 *
 * Styling a cell before it is padded is the alignment bug this exists to
 * prevent: `padEnd` counts an ANSI escape as visible width, so a styled cell is
 * padded short and every column after it slides left.
 */
export function styleWithin(cell: string, marker: string, style: Style): string {
  return cell.replace(marker, formatLabel(marker, style));
}

/** The geomean's delta and the provenance describing what stands behind it. */
export interface GeomeanParts {
  readonly delta: string;
  readonly provenance: string;
}

/**
 * The geomean's delta and how many metrics stand behind it, or `null` when
 * nothing survived to aggregate.
 *
 * The metrics left out are named nowhere near the figure: an unstable metric is
 * already tallied in the verdict summary and flagged in the highlights, so
 * restating the exclusions here spent the row's width on news the reader has.
 *
 * Shared by every renderer: each wraps the parts into its own cell shape, and
 * an empty geomean into {@link NO_GEOMEAN_CELL}.
 */
export function geomeanParts(geomean: GeomeanResult): GeomeanParts | null {
  if (geomean.n === 0) return null;
  return { delta: formatDelta(geomean.value), provenance: stableMetrics(geomean.n) };
}

/**
 * Whether every defined display class in a row belongs to
 * {@link QUIET_VERDICTS}.
 *
 * Shared by every renderer to decide whether a row carries no news worth
 * keeping above the fold. A row with no verdicts at all is left alone rather
 * than counted as quiet.
 */
export function isQuietRow(outcomes: ReadonlyArray<DisplayClass | undefined>): boolean {
  const defined = outcomes.flatMap((outcome) => (outcome === undefined ? [] : [outcome]));
  return defined.length > 0 && defined.every((outcome) => QUIET_VERDICTS.has(outcome));
}

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
  };
  for (const metric of Object.values(metrics)) {
    const verdict = metric.candidates[candidateIndex]?.verdict;
    if (verdict === undefined) continue;
    counts[displayClass(verdict)] += 1;
  }
  return counts;
}

/**
 * One tally part per display class, in the order the report legend lists them.
 *
 * Renderers join these with their own separator — `"   "` for aligned text
 * columns, `" · "` for inline markdown.
 *
 * Each part carries its class color, and a part counting nothing is dimmed
 * whatever its class — a zero is not news either way. `within noise` reads dim
 * at any count, its class color being dim. Color is governed by `styleText`
 * auto-detection.
 */
export function verdictSummaryParts(metrics: MetricComparisons, candidateIndex: number): string[] {
  const counts = displayCounts(metrics, candidateIndex);

  const stylePart = (shown: DisplayClass): string => {
    const count = counts[shown];
    const text = `${getGlyph(shown)} ${count} ${VERDICT_GLOSSES[shown]}`;
    const style: Style = count === 0 ? ["dim"] : VERDICT_STYLES[shown];
    return formatLabel(text, style);
  };

  return [
    stylePart("improved"),
    stylePart("regressed"),
    stylePart("unstable"),
    stylePart("identical"),
    stylePart("within-noise"),
  ];
}

/**
 * The pair counts behind every verdict a given method decided, across every
 * candidate.
 */
export function pairCounts(metrics: MetricComparisons, method: Method): number[] {
  const counts: number[] = [];
  for (const metric of Object.values(metrics)) {
    for (const { verdict } of metric.candidates) {
      if (verdict?.method === method) counts.push(verdict.n);
    }
  }
  return counts;
}

const SAMPLES_HINT = `re-run with --samples ${MIN_WILCOXON_N} or more for statistical verdicts`;

/** How the band method names itself wherever the footer describes a fallback. */
const BAND_METHOD = "noise band ±(half-range × K)";

/**
 * Why the band method decided a verdict the signed-rank test would otherwise
 * have owned.
 *
 * `shortage` counts the total pairs of every metric that never reached the
 * signed-rank floor; `ties` counts the surviving pairs of every metric that
 * reached it but had too many of them tied away. A run can hit both across
 * different metrics.
 */
interface BandFallbacks {
  shortage: number[];
  ties: number[];
}

/** Sorts every band-method verdict into the cause that forced the fallback. */
function bandFallbacks(metrics: MetricComparisons): BandFallbacks {
  const shortage: number[] = [];
  const ties: number[] = [];

  for (const metric of Object.values(metrics)) {
    for (const { verdict } of metric.candidates) {
      if (verdict?.method !== "band") continue;
      if (verdict.n < MIN_WILCOXON_N) {
        shortage.push(verdict.n);
      } else {
        ties.push(verdict.usableN);
      }
    }
  }

  return { shortage, ties };
}

/**
 * The verbose method lines naming how each verdict was decided.
 *
 * A band fallback gets one line per cause, because the counts that explain a
 * short run and a tie-starved one are different numbers: the worst total pair
 * count for a shortage, the worst usable pair count for ties. A run that hit
 * both causes on different metrics gets both lines.
 *
 * Every line is dimmed via `styleText` auto-detection.
 */
export function methodFooterLines(metrics: MetricComparisons): string[] {
  const signedRank = pairCounts(metrics, "signed-rank");
  const { shortage, ties } = bandFallbacks(metrics);
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
 * The always-on footer hint, told when a metric fell back to the noise band for
 * want of samples and more of them would buy a statistical verdict.
 *
 * The other cause of a band fallback, tied pairs, gets no hint: more samples
 * cannot help it, and the `=` glyph on those rows already reports what happened.
 *
 * `formatHint` turns the shared hint string into a format-appropriate line: the
 * text renderer prepends a styled label, the markdown renderer wraps it in a
 * blockquote with italic emphasis. The line is left unstyled here — its label
 * carries its own color through `formatHint`.
 */
export function hintFooterLines(
  metrics: MetricComparisons,
  formatHint: (hint: string) => string,
): string[] {
  return bandFallbacks(metrics).shortage.length > 0 ? [formatHint(SAMPLES_HINT)] : [];
}
