import { styleText } from "node:util";

import { assertNever } from "../errors.js";
import type { MetricVerdict } from "../verdict/verdict.js";
import type { MetricComparison, MetricComparisons } from "./types.js";

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
 * Tally the verdict classes across a comparison's metrics.
 *
 * This is the single source for the report's summary line: a new consumer
 * counts through this function rather than re-walking `metrics` on its own.
 * Metrics measured on one side only have no verdict and count towards nothing.
 */
export function countVerdicts(metrics: MetricComparisons): VerdictCounts {
  const counts: VerdictCounts = { improved: 0, regressed: 0, unstable: 0, noSignal: 0 };
  for (const { verdict } of Object.values(metrics)) {
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

/** A metric comparison known to carry a verdict — what a highlight is built from. */
export interface HighlightedMetric extends MetricComparison {
  verdict: MetricVerdict;
}

/** A metric worth calling out, paired with the name it is reported under. */
export interface MetricHighlight {
  name: string;
  metric: HighlightedMetric;
}

/** Where each highlighted verdict class sits in the reported order. */
const HIGHLIGHT_RANK: Partial<Record<MetricVerdict["verdict"], number>> = {
  regressed: 0,
  improved: 1,
  unstable: 2,
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
 * The metrics worth calling out, ordered regressions first (by delta magnitude,
 * descending), then improvements the same way, then unstable metrics by noise.
 *
 * Metrics that sat within the noise carry no news, so they are left out
 * entirely, and so is a metric measured on one side only: with no verdict it
 * has nothing to rank it against the rest. Ties keep the order the metrics
 * were measured in.
 */
export function selectHighlights(metrics: MetricComparisons): readonly MetricHighlight[] {
  const ranked: { highlight: MetricHighlight; rank: number; weight: number }[] = [];
  for (const [name, metric] of Object.entries(metrics)) {
    if (metric.verdict === undefined) continue;
    const rank = HIGHLIGHT_RANK[metric.verdict.verdict];
    if (rank === undefined) continue;
    ranked.push({
      highlight: { name, metric: { ...metric, verdict: metric.verdict } },
      rank,
      weight: highlightWeight(metric.verdict),
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

/**
 * `validateStream: false` tells `styleText` to skip its own TTY check — the
 * caller already decided whether color is appropriate via the `useColor` flag.
 */
export function formatLabel(
  label: string,
  style: Parameters<typeof styleText>[0],
  useColor: boolean,
): string {
  return useColor ? styleText(style, label, { validateStream: false }) : label;
}

/** The style names `styleText` accepts, in the shape {@link formatLabel} takes them. */
type Style = Parameters<typeof styleText>[0];

const HINT_STYLE: Style = ["yellow", "underline"];

/** The `Hint:` label every hint line opens with, styled when the caller allows it. */
export function formatHintLabel(useColor: boolean): string {
  return formatLabel("Hint:", HINT_STYLE, useColor);
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
const VERDICT_STYLES: Record<MetricVerdict["verdict"], Style> = {
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
export function styleWithin(cell: string, marker: string, style: Style, useColor: boolean): string {
  return cell.replace(marker, formatLabel(marker, style, useColor));
}

/** Paint a verdict's glyph in the color of its class, inside an already-padded cell. */
export function styleGlyph(
  cell: string,
  verdict: MetricVerdict["verdict"],
  useColor: boolean,
): string {
  return styleWithin(cell, GLYPHS[verdict], VERDICT_STYLES[verdict], useColor);
}
