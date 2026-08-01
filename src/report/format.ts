import { styleText } from "node:util";

import { assertNever } from "../errors.js";
import type { GeomeanExclusion, GeomeanResult, Method, MetricVerdict } from "../verdict/verdict.js";
import type { CandidateMetric, MetricComparison, MetricComparisons } from "./types.js";

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

/** The `± N%` suffix that follows a value, or nothing when the spread is unknown. */
export function formatSpread(spread?: number): string {
  if (spread === undefined) return "";
  return ` ± ${spread.toFixed(0)}%`;
}

/** A value cell: the scaled measurement and its spread, or nothing when unmeasured. */
export function formatMetricCell(median?: number, spread?: number, unit?: "ns" | "bytes"): string {
  if (median === undefined) return "";
  return `${formatValue(median, unit)}${formatSpread(spread)}`;
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

const GLYPHS: Record<MetricVerdict["verdict"], string> = {
  improved: "✓",
  regressed: "✗",
  "no-signal": "~",
  unstable: "≈",
};

/**
 * ✓ improved, ✗ regressed, ~ no signal, ≈ unstable — the glyphs the report
 * footer legend explains.
 */
export function getGlyph(verdict: MetricVerdict["verdict"]): string {
  return GLYPHS[verdict];
}

/**
 * The word each verdict class reads as, shared by the summary line and the legend.
 *
 * Typed as a `Record` over the verdict union rather than listed inline at each
 * call site: a verdict class added to the union without an entry here is a
 * compile error, instead of a class that renders in the table but is missing
 * from both the summary and the legend.
 */
export const VERDICT_GLOSSES: Record<MetricVerdict["verdict"], string> = {
  improved: "improved",
  regressed: "regressed",
  unstable: "unstable",
  "no-signal": "within noise",
};

/** The delta cell: the word `unstable` for a verdict too noisy to trust, otherwise the signed percentage. */
export function formatVerdictDelta(verdict: MetricVerdict): string {
  return verdict.verdict === "unstable" ? "unstable" : formatDelta(verdict.delta);
}

/**
 * The verdicts whose rows carry no news worth keeping above the fold.
 *
 * Shared by every renderer: a metric that sat within the noise, or was too
 * jittery to judge, is dimmed (text) or collapsed into a `<details>` block
 * (markdown) rather than competing with the rows that moved.
 */
export const QUIET_VERDICTS: ReadonlySet<MetricVerdict["verdict"]> = new Set([
  "no-signal",
  "unstable",
]);

/** The label the geomean row is reported under, in every renderer. */
export const GEOMEAN_LABEL = "geomean (gating metrics)";

/**
 * The evidence suffix for a highlighted metric.
 *
 * Exact entries keep `(exact)`. Unstable entries show the noise that swamped the
 * signal. Improved/regressed/no-signal entries from approximate methods carry no
 * trailing evidence — the glyph and delta already tell the story.
 */
export function formatEvidence(verdict: MetricVerdict): string {
  if (verdict.method === "exact") return "(exact)";
  if (verdict.verdict === "unstable") return `noise ${formatNoiseBand(verdict.noisePct)}`;
  return "";
}

/** A metric's noise band, as the `±N%` the row annotations and highlights share. */
export function formatNoiseBand(noisePct: number): string {
  return `±${noisePct.toFixed(1)}%`;
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
 * Tally the verdict classes one candidate earned against the baseline.
 *
 * This is the single source for the report's summary line: a new consumer
 * counts through this function rather than re-walking `metrics` on its own.
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
 * Where each highlighted verdict class sits in the reported order.
 *
 * Total rather than partial: `undefined` is how a class opts out of highlights,
 * so a verdict class added to the union without an entry here is a compile error
 * rather than a class that silently never appears.
 */
const HIGHLIGHT_RANK: Record<MetricVerdict["verdict"], number | undefined> = {
  regressed: 0,
  improved: 1,
  unstable: 2,
  "no-signal": undefined,
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
 * that sat within the noise carry no news, so they are left out entirely, and so
 * is a metric this candidate never reported: with no verdict it has nothing to
 * rank it against the rest. Ties keep the order the metrics were measured in.
 */
export function selectHighlights(
  metrics: MetricComparisons,
  candidateIndex: number,
): readonly MetricHighlight[] {
  const ranked: { highlight: MetricHighlight; rank: number; weight: number }[] = [];
  for (const [name, metric] of Object.entries(metrics)) {
    const candidate = metric.candidates[candidateIndex];
    if (candidate?.verdict === undefined) continue;
    const rank = HIGHLIGHT_RANK[candidate.verdict.verdict];
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
  const line = padded.join("│").trim();
  if (styleCell === undefined) return line;
  return line.split("│").map(styleCell).join("│");
}

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

const HINT_STYLE: Style = ["yellow", "underline"];

/** The `Hint:` label every hint line opens with, styled by `styleText` auto-detection. */
export function formatHintLabel(stream?: NodeJS.WriteStream): string {
  return formatLabel("Hint:", HINT_STYLE, stream);
}

/**
 * The color each verdict wears in a styled report.
 *
 * A metric that carries no news — within noise, or too unstable to call — has
 * its whole row dimmed by the renderer, so `no-signal` asks for no color of its
 * own and `unstable` asks only for amber: the row's dim is what makes it read
 * as dim amber. Nesting a second dim inside the row's would close it at the
 * glyph and leave the rest of the row bright.
 */
export const VERDICT_STYLES: Record<MetricVerdict["verdict"], Style> = {
  improved: ["green"],
  regressed: ["red"],
  unstable: ["yellow"],
  "no-signal": [],
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

/** How many gating metrics the geomean left out, and on what grounds. */
export function formatExclusions(excluded: readonly GeomeanExclusion[]): string {
  if (excluded.length === 0) return "";
  const reasons = [...new Set(excluded.map((exclusion) => exclusion.reason))];
  return `${excluded.length} excluded: ${reasons.join(", ")}`;
}

/** The geomean's delta and the provenance describing what stands behind it. */
export interface GeomeanParts {
  readonly delta: string;
  readonly provenance: string;
}

/**
 * The geomean's delta plus provenance (how many metrics stand behind it and
 * how many were excluded), or `null` when nothing survived to aggregate.
 *
 * Shared by every renderer: each wraps the parts into its own cell shape
 * (text splits `figure`/`text`, markdown joins them into one string).
 */
export function geomeanParts(geomean: GeomeanResult): GeomeanParts | null {
  if (geomean.n === 0) return null;
  const stable = `${geomean.n} stable metric${geomean.n === 1 ? "" : "s"}`;
  const exclusions = formatExclusions(geomean.excluded);
  const provenance = exclusions === "" ? stable : `${stable} · ${exclusions}`;
  return { delta: formatDelta(geomean.value), provenance };
}

/**
 * Whether every defined verdict outcome in a row belongs to
 * {@link QUIET_VERDICTS}.
 *
 * Shared by every renderer to decide whether a row carries no news worth
 * keeping above the fold. A row with no verdicts at all is left alone rather
 * than counted as quiet.
 */
export function isQuietRow(outcomes: ReadonlyArray<MetricVerdict["verdict"] | undefined>): boolean {
  const defined = outcomes.flatMap((outcome) => (outcome === undefined ? [] : [outcome]));
  return defined.length > 0 && defined.every((outcome) => QUIET_VERDICTS.has(outcome));
}

/**
 * One tally part per verdict class, in the order the report legend lists them.
 *
 * Renderers join these with their own separator — `"   "` for aligned text
 * columns, `" · "` for inline markdown.
 *
 * Non-zero improved/regressed/unstable parts carry their class color and
 * zero-count parts are dimmed. The `within noise` segment is always dimmed
 * regardless of count — it carries no news worth highlighting. Color is
 * governed by `styleText` auto-detection.
 */
export function verdictSummaryParts(metrics: MetricComparisons, candidateIndex: number): string[] {
  const counts = countVerdicts(metrics, candidateIndex);

  const stylePart = (verdict: MetricVerdict["verdict"], count: number, gloss: string): string => {
    const text = `${getGlyph(verdict)} ${count} ${gloss}`;
    const style: Style = verdict === "no-signal" || count === 0 ? ["dim"] : VERDICT_STYLES[verdict];
    return formatLabel(text, style);
  };

  return [
    stylePart("improved", counts.improved, VERDICT_GLOSSES.improved),
    stylePart("regressed", counts.regressed, VERDICT_GLOSSES.regressed),
    stylePart("unstable", counts.unstable, VERDICT_GLOSSES.unstable),
    stylePart("no-signal", counts.noSignal, VERDICT_GLOSSES["no-signal"]),
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

const SAMPLES_HINT = "re-run with --samples 6 or more for statistical verdicts";

/**
 * The method-footer lines naming how each verdict was decided, plus — when any
 * metric fell back to the noise band — a hint telling the user how to get a
 * statistical verdict instead.
 *
 * `formatHint` turns the shared hint string into a format-appropriate line:
 * the text renderer prepends a styled label, the markdown renderer wraps it in
 * a blockquote with italic emphasis.
 *
 * Descriptive lines (signed-rank / noise-band) are dimmed via `styleText`
 * auto-detection. The hint line is left unstyled here — its label carries its
 * own color through `formatHint`.
 */
export function methodFooterLines(
  metrics: MetricComparisons,
  formatHint: (hint: string) => string,
): string[] {
  const signedRank = pairCounts(metrics, "signed-rank");
  const band = pairCounts(metrics, "band");
  const lines: string[] = [];

  if (signedRank.length > 0) {
    const desc = `verdicts: Wilcoxon signed-rank on pairs (${formatPairCount(Math.min(...signedRank))} ≥ 6) · ~ = no signal at α=0.05`;
    lines.push(formatLabel(desc, ["dim"]));
  }
  if (band.length > 0) {
    const desc = `noise band ±(half-range × K) — ${formatPairCount(Math.max(...band))} below signed-rank floor (6 pairs)`;
    lines.push(formatLabel(desc, ["dim"]), formatHint(SAMPLES_HINT));
  }

  return lines;
}

/**
 * The glosses line shared by every legend: each verdict class's glyph and word,
 * joined by ` · `.
 *
 * Renderers wrap this into their own format (plain text prefix, blockquote,
 * etc.) and append the baseline attribution.
 *
 * Each glyph is painted in its verdict class color via `styleText`
 * auto-detection. The `~` glyph has no color of its own — it stays the color
 * of whatever wraps the line (typically dim).
 */
export function legendGlosses(): string {
  const glosses: readonly (keyof typeof VERDICT_GLOSSES)[] = [
    "improved",
    "regressed",
    "unstable",
    "no-signal",
  ];
  return glosses
    .map((verdict) => {
      const glyph = formatLabel(getGlyph(verdict), VERDICT_STYLES[verdict]);
      return `${glyph} ${VERDICT_GLOSSES[verdict]}`;
    })
    .join(" · ");
}

/** Paint a verdict's glyph in the color of its class, inside an already-padded cell. */
export function styleGlyph(cell: string, verdict: MetricVerdict["verdict"]): string {
  return styleWithin(cell, GLYPHS[verdict], VERDICT_STYLES[verdict]);
}
