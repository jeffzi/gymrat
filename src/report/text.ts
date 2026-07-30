import { assertNever } from "../errors.js";
import type { WorktreeRemovalFailure } from "../targets.js";
import type { Method, MetricVerdict } from "../verdict/verdict.js";
import {
  computeColumnWidth,
  formatDelta,
  formatPValue,
  formatSpread,
  formatTableLine,
  formatValue,
  getGlyph,
} from "./format.js";
import type { ComparisonResult } from "./types.js";

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

function renderMethodFooter(methods: Set<Method>, samples: number): string[] {
  if (methods.has("signed-rank")) {
    return [`verdicts: Wilcoxon signed-rank on pairs (n=${samples} ≥ 6) · ~ = no signal at α=0.05`];
  }
  if (methods.has("band")) {
    return [`noise band ±(half-range × K) — n=${samples} below signed-rank floor (6 pairs)`];
  }
  return [];
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

  lines.push(...renderMethodFooter(methods, result.samples));

  const removedNoun = result.worktreesRemoved === 1 ? "worktree" : "worktrees";
  lines.push(
    `${result.worktreesRemoved} ${removedNoun} removed · ${result.worktreesLeftBehind.length} left behind`,
  );

  lines.push(...formatCleanupFailures(result.worktreesLeftBehind, result.worktreePruneError));

  return lines.join("\n");
}
