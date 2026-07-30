import { styleText } from "node:util";

import type { MetricVerdict } from "../verdict/verdict.js";

/**
 * Strip trailing zeros but keep at least one decimal place if there's a decimal point.
 * Assumes input from toFixed() which always includes a decimal point.
 */
function stripTrailingZeros(str: string): string {
  return str.replace(/0+$/, "").replace(/\.$/, ".0");
}

type Tier = readonly [threshold: number, divisor: number, suffix: string, decimals: number];

const NS_TIERS: readonly Tier[] = [
  [1000, 1, "n", 0],
  [1e6, 1000, "µ", 3],
  [1e9, 1e6, "m", 1],
  [Infinity, 1e9, "s", 1],
];

const BYTE_TIERS: readonly Tier[] = [
  [1000, 1, "", 0],
  [1e6, 1000, "k", 1],
  [1e9, 1e6, "M", 1],
  [Infinity, 1e9, "G", 1],
];

const TIER_MAP: Record<"ns" | "bytes", readonly Tier[]> = {
  ns: NS_TIERS,
  bytes: BYTE_TIERS,
};

function scaleTier(value: number, tiers: readonly Tier[]): string {
  const tier = tiers.find(([threshold]) => value < threshold)!;
  const [, divisor, suffix, decimals] = tier;
  const scaled = (value / divisor).toFixed(decimals);
  return `${decimals > 0 ? stripTrailingZeros(scaled) : scaled}${suffix}`;
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

/** A signed percentage, or nothing when the delta is not a number. */
export function formatDelta(delta: number): string {
  if (Number.isNaN(delta)) return "";
  const sign = delta > 0 ? "+" : "";
  return `${sign}${delta.toFixed(1)}%`;
}

/** A p-value at reading precision, collapsed to `p<0.001` below the display floor. */
export function formatPValue(p: number): string {
  if (p < 0.001) return "p<0.001";
  if (p < 0.01) return `p=${p.toFixed(3)}`;
  return `p=${p.toFixed(2)}`;
}

/**
 * ✓ improved, ✗ regressed, ~ no signal — the glyphs the report footer legend explains.
 */
export function getGlyph(verdict: MetricVerdict["verdict"]): string {
  if (verdict === "improved") return "✓";
  if (verdict === "regressed") return "✗";
  return "~";
}

/** Width a column needs to hold its header and every cell, never below `minWidth`. */
export function computeColumnWidth(
  headerLength: number,
  contentLengths: number[],
  minWidth: number,
): number {
  const maxContent = Math.max(headerLength, ...contentLengths);
  return Math.max(maxContent + 2, minWidth);
}

/** Pad each cell to its column width and join them with the column separator. */
export function formatTableLine(cells: readonly string[], widths: readonly number[]): string {
  const padded = cells.map((cell, i) => cell.padEnd(widths[i]!)); // widths guaranteed same length as cells by caller
  return padded.join("│").trim();
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
