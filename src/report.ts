import type { ResolvedMetricMeta } from "./config.js";
import type { Method, MetricVerdict } from "./verdict/verdict.js";

export interface ComparisonResult {
  labels: [string, string];
  samples: number;
  adapter: string;
  metrics: Record<
    string,
    {
      medianA?: number;
      medianB?: number;
      spreadA?: number;
      spreadB?: number;
      verdict?: MetricVerdict;
      meta: ResolvedMetricMeta;
    }
  >;
  geomean: {
    value: number;
    n: number;
    excluded: string[];
  };
  worktreesRemoved: number;
  worktreesLeftBehind: number;
}

/**
 * Strip trailing zeros but keep at least one decimal place if there's a decimal point.
 * Assumes input from toFixed() which always includes a decimal point.
 */
function stripTrailingZeros(str: string): string {
  return str.replace(/0+$/, "").replace(/\.$/, ".0");
}

/**
 * Format a number with unit scaling.
 * For ns: n, µ, m, s.
 * For bytes: raw, k, M, G.
 * For no unit: raw value.
 */
function formatValue(value: number, unit?: "ns" | "bytes"): string {
  if (!unit) {
    // No unit specified: return as integer
    return Math.round(value).toString();
  }

  if (unit === "ns") {
    if (value < 1000) return `${value.toFixed(0)}n`;
    if (value < 1e6) return `${stripTrailingZeros((value / 1000).toFixed(3))}µ`;
    if (value < 1e9) return `${stripTrailingZeros((value / 1e6).toFixed(1))}m`;
    return `${stripTrailingZeros((value / 1e9).toFixed(1))}s`;
  }

  // bytes
  if (value < 1000) return value.toFixed(0);
  if (value < 1e6) return `${stripTrailingZeros((value / 1000).toFixed(1))}k`;
  if (value < 1e9) return `${stripTrailingZeros((value / 1e6).toFixed(1))}M`;
  return `${stripTrailingZeros((value / 1e9).toFixed(1))}G`;
}

/**
 * Format spread as percentage.
 */
function formatSpread(spread?: number): string {
  if (spread === undefined) return "";
  return ` ± ${spread.toFixed(0)}%`;
}

/**
 * Format delta percentage with sign.
 */
function formatDelta(delta: number): string {
  if (Number.isNaN(delta)) return "";
  const sign = delta > 0 ? "+" : "";
  return `${sign}${delta.toFixed(1)}%`;
}

/**
 * Format p-value with appropriate precision.
 */
function formatPValue(p: number): string {
  if (p < 0.001) return "p<0.001";
  if (p < 0.01) return `p=${p.toFixed(3)}`;
  return `p=${p.toFixed(2)}`;
}

/**
 * Get the verdict glyph. Only called with non-undefined verdicts.
 */
function getGlyph(verdict: MetricVerdict["verdict"]): string {
  if (verdict === "improved") return "✓";
  if (verdict === "regressed") return "✗";
  return "~";
}

/**
 * Format the annotation for a verdict (p-value, band, or exact). Only called with non-undefined verdicts.
 */
function formatAnnotation(verdict: MetricVerdict): string {
  switch (verdict.method) {
    case "signed-rank": {
      const p = verdict.p ?? 0;
      return `(${formatPValue(p)} n=${verdict.n})`;
    }
    case "band": {
      const band = verdict.band ?? 0;
      return `(band ±${band.toFixed(1)}%, n=${verdict.n})`;
    }
    case "exact":
      return "(exact)";
    default:
      throw new Error("Unknown verdict method");
  }
}

/**
 * Format a metric value cell with optional spread.
 */
function formatMetricCell(median?: number, spread?: number, unit?: "ns" | "bytes"): string {
  if (median === undefined) return "";
  return `${formatValue(median, unit)}${formatSpread(spread)}`;
}

/**
 * Compute column width: max of content lengths plus padding, constrained by minimum.
 */
function computeColumnWidth(
  headerLength: number,
  contentLengths: number[],
  minWidth: number,
): number {
  const maxContent = Math.max(headerLength, ...contentLengths);
  return Math.max(maxContent + 2, minWidth);
}

/**
 * Format a single line of the table with padded cells.
 */
function formatTableLine(cells: string[], widths: number[]): string {
  const padded = cells.map((cell, i) => cell.padEnd(widths[i]!)); // widths guaranteed same length as cells by caller
  return padded.join("│").trim();
}

/**
 * Format the delta cell with glyph and annotation.
 */
function formatDeltaCell(verdict?: MetricVerdict): string {
  if (!verdict) return "";

  const glyph = getGlyph(verdict.verdict);
  const delta = formatDelta(verdict.delta);
  const annotation = formatAnnotation(verdict);

  return `${glyph} ${delta}  ${annotation}`.trim();
}

export function renderReport(result: ComparisonResult): string {
  const lines: string[] = [];

  // Header line
  lines.push(
    `gymrat compare · ${result.labels[0]} ↔ ${result.labels[1]} · ${result.samples} paired samples · adapter: ${result.adapter}`,
  );

  // Column headers
  const headers = ["metric", `old (${result.labels[0]})`, `new (${result.labels[1]})`, "vs old"];

  // Collect all metric rows with formatted values for column width calculation
  const metricEntries = Object.entries(result.metrics);
  const metricRows: Array<{
    name: string;
    oldValue: string;
    newValue: string;
    deltaCell: string;
  }> = [];

  for (const [name, metric] of metricEntries) {
    const oldValue = formatMetricCell(metric.medianA, metric.spreadA, metric.meta.unit);
    const newValue = formatMetricCell(metric.medianB, metric.spreadB, metric.meta.unit);
    const deltaCell = formatDeltaCell(metric.verdict);

    metricRows.push({
      name,
      oldValue,
      newValue,
      deltaCell,
    });
  }

  // Compute column widths
  const metricColWidth = computeColumnWidth(
    headers[0]!.length, // headers is literal 4-element array
    metricRows.map((r) => r.name.length),
    16,
  );
  const oldColWidth = computeColumnWidth(
    headers[1]!.length, // headers is literal 4-element array
    metricRows.map((r) => r.oldValue.length),
    14,
  );
  const newColWidth = computeColumnWidth(
    headers[2]!.length, // headers is literal 4-element array
    metricRows.map((r) => r.newValue.length),
    16,
  );
  const deltaColWidth = computeColumnWidth(
    headers[3]!.length, // headers is literal 4-element array
    metricRows.map((r) => r.deltaCell.length),
    16,
  );

  // Format header line with separators
  const widths = [metricColWidth, oldColWidth, newColWidth, deltaColWidth];
  const headerLine = formatTableLine(headers, widths);
  lines.push(headerLine);

  // Separator line
  const separator = [
    "─".repeat(metricColWidth),
    "─".repeat(oldColWidth),
    "─".repeat(newColWidth),
    "─".repeat(deltaColWidth),
  ].join("┼");
  lines.push(separator);

  // Metric rows
  for (const row of metricRows) {
    const line = formatTableLine([row.name, row.oldValue, row.newValue, row.deltaCell], widths);
    lines.push(line);
  }

  // Geomean row with separator
  lines.push(separator);
  const geomeanDelta = formatDelta(result.geomean.value);
  const geomeanLine = formatTableLine(["geomean (gating metrics)", "", "", geomeanDelta], widths);
  lines.push(geomeanLine);

  // Verdict method footer
  const methods = new Set(
    metricRows
      .map((row) => result.metrics[row.name]?.verdict?.method)
      .filter((method): method is Method => method !== undefined),
  );

  if (methods.has("signed-rank")) {
    lines.push(
      `verdicts: Wilcoxon signed-rank on pairs (n=${result.samples} ≥ 6) · ~ = no signal at α=0.05`,
    );
  } else if (methods.has("band")) {
    lines.push(
      `noise band ±(half-range × K) — n=${result.samples} below signed-rank floor (6 pairs)`,
    );
  }

  // Worktree footer
  lines.push(`worktrees removed · ${result.worktreesLeftBehind} left behind`);

  return lines.join("\n");
}
