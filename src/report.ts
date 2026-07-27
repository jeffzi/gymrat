import type { ResolvedMetricMeta } from "./config.js";
import type { WorktreeRemovalFailure } from "./targets.js";
import type { Method, MetricVerdict } from "./verdict/verdict.js";

function assertNever(x: never): never {
  throw new Error(`Unexpected: ${JSON.stringify(x)}`);
}

/**
 * Everything `renderReport` needs to draw a comparison — the rendering input contract.
 */
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

  /** Worktrees cleanup could not remove, each with the reason git gave. */
  worktreesLeftBehind: readonly WorktreeRemovalFailure[];

  /** Reason the `git worktree prune` sweep failed, or `undefined` if it succeeded. */
  worktreePruneError: string | undefined;
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

function formatSpread(spread?: number): string {
  if (spread === undefined) return "";
  return ` ± ${spread.toFixed(0)}%`;
}

function formatDelta(delta: number): string {
  if (Number.isNaN(delta)) return "";
  const sign = delta > 0 ? "+" : "";
  return `${sign}${delta.toFixed(1)}%`;
}

function formatPValue(p: number): string {
  if (p < 0.001) return "p<0.001";
  if (p < 0.01) return `p=${p.toFixed(3)}`;
  return `p=${p.toFixed(2)}`;
}

/**
 * ✓ improved, ✗ regressed, ~ no signal — the glyphs the report footer legend explains.
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
    case "signed-rank":
      return `(${formatPValue(verdict.p)} n=${verdict.n})`;
    case "band":
      return `(band ±${verdict.band.toFixed(1)}%, n=${verdict.n})`;
    case "exact":
      return "(exact)";
    /* v8 ignore next -- exhaustive switch; compile-time guard via assertNever */
    default:
      return assertNever(verdict);
  }
}

function formatMetricCell(median?: number, spread?: number, unit?: "ns" | "bytes"): string {
  if (median === undefined) return "";
  return `${formatValue(median, unit)}${formatSpread(spread)}`;
}

function computeColumnWidth(
  headerLength: number,
  contentLengths: number[],
  minWidth: number,
): number {
  const maxContent = Math.max(headerLength, ...contentLengths);
  return Math.max(maxContent + 2, minWidth);
}

function formatTableLine(cells: readonly string[], widths: readonly number[]): string {
  const padded = cells.map((cell, i) => cell.padEnd(widths[i]!)); // widths guaranteed same length as cells by caller
  return padded.join("│").trim();
}

function formatDeltaCell(verdict?: MetricVerdict): string {
  if (!verdict) return "";

  const glyph = getGlyph(verdict.verdict);
  const delta = formatDelta(verdict.delta);
  const annotation = formatAnnotation(verdict);

  return `${glyph} ${delta}  ${annotation}`.trim();
}

/**
 * Collapse a git diagnostic onto one line.
 *
 * git routinely emits several lines for one failure — a `warning:` line before the
 * `fatal:` line, plus indented continuations. The report joins its lines with `\n`,
 * so an embedded newline would push git's continuation text flush-left among the
 * table rows, where it reads as report structure rather than as detail.
 */
function toSingleLine(text: string): string {
  return text.replace(/\s+/g, " ").trim();
}

/**
 * Render what cleanup could not do, one line per entry.
 *
 * Shared by the report footer and by the error `compare` throws when the run
 * itself failed, so the user reads the same wording whichever path they land on.
 * Returns an empty array when cleanup was clean, which callers use to decide
 * whether there is anything worth reporting at all.
 */
export function formatCleanupFailures(
  leftBehind: readonly WorktreeRemovalFailure[],
  pruneError: string | undefined,
): string[] {
  const lines = leftBehind.map(
    (failure) => `  left behind: ${failure.dir} (${toSingleLine(failure.error)})`,
  );

  if (pruneError !== undefined) {
    lines.push(`  worktree prune failed: ${toSingleLine(pruneError)}`);
  }

  return lines;
}

/**
 * Render a comparison as the plain-text report the CLI prints.
 *
 * Sections come out in a fixed order: run header, metric table, geomean row,
 * the method footer naming whichever verdict method the run actually used, and
 * the worktree footer with any cleanup failures.
 *
 * Column widths are computed from the widest cell rather than fixed, so long
 * metric names or labels widen the table instead of being truncated.
 */
export function renderReport(result: ComparisonResult): string {
  const lines: string[] = [];

  lines.push(
    `gymrat compare · ${result.labels[0]} ↔ ${result.labels[1]} · ${result.samples} paired samples · adapter: ${result.adapter}`,
  );

  const headers: [string, string, string, string] = [
    "metric",
    `old (${result.labels[0]})`,
    `new (${result.labels[1]})`,
    "vs old",
  ];

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

  const geomeanLabel = "geomean (gating metrics)";
  const metricColWidth = computeColumnWidth(
    headers[0].length,
    [...metricRows.map((r) => r.name.length), geomeanLabel.length],
    16,
  );
  const oldColWidth = computeColumnWidth(
    headers[1].length,
    metricRows.map((r) => r.oldValue.length),
    14,
  );
  const newColWidth = computeColumnWidth(
    headers[2].length,
    metricRows.map((r) => r.newValue.length),
    16,
  );
  const deltaColWidth = computeColumnWidth(
    headers[3].length,
    metricRows.map((r) => r.deltaCell.length),
    16,
  );

  const widths = [metricColWidth, oldColWidth, newColWidth, deltaColWidth];
  const headerLine = formatTableLine(headers, widths);
  lines.push(headerLine);

  const separator = [
    "─".repeat(metricColWidth),
    "─".repeat(oldColWidth),
    "─".repeat(newColWidth),
    "─".repeat(deltaColWidth),
  ].join("┼");
  lines.push(separator);

  for (const row of metricRows) {
    const line = formatTableLine([row.name, row.oldValue, row.newValue, row.deltaCell], widths);
    lines.push(line);
  }

  lines.push(separator);
  const geomeanDelta = formatDelta(result.geomean.value);
  const geomeanLine = formatTableLine([geomeanLabel, "", "", geomeanDelta], widths);
  lines.push(geomeanLine);

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

  const removedNoun = result.worktreesRemoved === 1 ? "worktree" : "worktrees";
  lines.push(
    `${result.worktreesRemoved} ${removedNoun} removed · ${result.worktreesLeftBehind.length} left behind`,
  );

  lines.push(...formatCleanupFailures(result.worktreesLeftBehind, result.worktreePruneError));

  return lines.join("\n");
}
