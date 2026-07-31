import type { GeomeanResult, MetricVerdict } from "../verdict/verdict.js";
import {
  formatDelta,
  formatEvidence,
  formatExclusions,
  formatMetricCell,
  formatVerdictDelta,
  GEOMEAN_LABEL,
  getGlyph,
  legendGlosses,
  methodFooterLines,
  QUIET_VERDICTS,
  selectHighlights,
  verdictSummaryParts,
} from "./format.js";
import type { CandidateComparison, ComparisonResult, MetricComparisons } from "./types.js";

/** One line tallying every verdict class one candidate earned. */
function renderSummary(metrics: MetricComparisons, candidateIndex: number): string {
  return verdictSummaryParts(metrics, candidateIndex).join(" · ");
}

/** Format a single highlight entry as a markdown list item. */
function formatHighlightEntry(name: string, verdict: MetricVerdict): string {
  const glyph = getGlyph(verdict.verdict);
  const delta = formatVerdictDelta(verdict);
  const evidence = formatEvidence(verdict);
  const suffix = evidence === "" ? "" : `  ${evidence}`;
  return `- ${glyph} ${name}  ${delta}${suffix}`;
}

/** Render highlights for a single candidate as markdown list items. */
function renderHighlightEntries(metrics: MetricComparisons, candidateIndex: number): string[] {
  const highlights = selectHighlights(metrics, candidateIndex);
  return highlights.map(({ name, candidate }) => formatHighlightEntry(name, candidate.verdict));
}

/**
 * A verdict rendered as part of a cell: glyph + delta.
 *
 * Unlike the text renderer, this does not include the noise band in the
 * verdict portion — the GFM table cell combines value and verdict in one,
 * keeping it compact.
 */
function formatVerdictPart(verdict: MetricVerdict | undefined): string {
  if (verdict === undefined) return "";
  return `${getGlyph(verdict.verdict)}  ${formatVerdictDelta(verdict)}`;
}

/**
 * The geomean cell: delta plus provenance (how many metrics stand behind it
 * and how many were excluded), or a dash when nothing survived.
 */
function formatGeomeanCell(geomean: GeomeanResult): string {
  if (geomean.n === 0) return "—";
  const delta = formatDelta(geomean.value);
  const stable = `${geomean.n} stable metric${geomean.n === 1 ? "" : "s"}`;
  const exclusions = formatExclusions(geomean.excluded);
  const provenance = exclusions === "" ? stable : `${stable} · ${exclusions}`;
  return `${delta}  ${provenance}`;
}

/**
 * Whether every candidate on a metric row is quiet (no-signal or unstable).
 *
 * Used to decide whether the row belongs inside the `<details>` block.
 * A metric with at least one improved or regressed candidate stays outside.
 */
function isQuietRow(verdicts: ReadonlyArray<MetricVerdict | undefined>): boolean {
  const outcomes = verdicts.flatMap((v) => (v === undefined ? [] : [v.verdict]));
  return outcomes.length > 0 && outcomes.every((outcome) => QUIET_VERDICTS.has(outcome));
}

/**
 * Wrap quiet rows in a `<details>` block with a summary labelling how many
 * are within noise vs unstable. Returns an empty array when there are no quiet
 * rows.
 */
function renderQuietBlock(
  header: string,
  separator: string,
  quietRows: readonly string[],
  unstableCount: number,
): string[] {
  if (quietRows.length === 0) return [];

  const noiseCount = quietRows.length - unstableCount;
  const parts: string[] = [];
  if (noiseCount > 0) parts.push(`${noiseCount} within noise`);
  if (unstableCount > 0) parts.push(`${unstableCount} unstable`);

  return [
    "",
    "<details>",
    `<summary>${parts.join(" / ")}</summary>`,
    "",
    header,
    separator,
    ...quietRows,
    "",
    "</details>",
    "",
  ];
}

/**
 * The first column (metric names) is left-aligned; all numeric columns are
 * right-aligned with the `---:` GFM syntax.
 */
function gfmSeparator(columnCount: number): string {
  const cells = Array.from({ length: columnCount }, (_, i) => (i === 0 ? "---" : "---:"));
  return `| ${cells.join(" | ")} |`;
}

function gfmRow(cells: readonly string[]): string {
  return `| ${cells.join(" | ")} |`;
}

/**
 * Render the single-candidate GFM table.
 *
 * Columns: Metric | baseline | candidate | vs baseline.
 * Within-noise/unstable rows go into a `<details>` block.
 */
function renderSingleCandidateTable(
  result: ComparisonResult,
  candidate: CandidateComparison,
  candidateIndex: number,
): string[] {
  const baseline = result.baselineLabel;
  const header = gfmRow(["Metric", baseline, candidate.label, `vs ${baseline}`]);
  const separator = gfmSeparator(4);

  const prominentRows: string[] = [];
  const quietRows: string[] = [];
  let unstableCount = 0;

  for (const [name, metric] of Object.entries(result.metrics)) {
    const side = metric.candidates[candidateIndex];
    const baselineCell = formatMetricCell(
      metric.baselineMedian,
      metric.baselineSpread,
      metric.meta.unit,
    );
    const candidateCell = formatMetricCell(side?.median, side?.spread, metric.meta.unit);
    const verdictCell = formatVerdictPart(side?.verdict);
    const row = gfmRow([name, baselineCell, candidateCell, verdictCell]);

    if (isQuietRow([side?.verdict])) {
      quietRows.push(row);
      if (side?.verdict?.verdict === "unstable") unstableCount++;
    } else {
      prominentRows.push(row);
    }
  }

  const geomeanDelta = formatGeomeanCell(candidate.geomean);
  const geomeanRow = gfmRow([GEOMEAN_LABEL, baseline, candidate.label, geomeanDelta]);

  return [
    header,
    separator,
    ...prominentRows,
    ...renderQuietBlock(header, separator, quietRows, unstableCount),
    geomeanRow,
  ];
}

/** Format a multi-candidate cell: value + verdict, or just value, or empty. */
function formatMultiCandidateCell(
  side: { median?: number; spread?: number; verdict?: MetricVerdict } | undefined,
  unit?: "ns" | "bytes",
): string {
  const value = formatMetricCell(side?.median, side?.spread, unit);
  const verdictPart = formatVerdictPart(side?.verdict);
  if (value === "" && verdictPart === "") return "";
  if (verdictPart === "") return value;
  return `${value}  ${verdictPart}`;
}

/**
 * Render the multi-candidate GFM table.
 *
 * Columns: Metric | baseline | candidate-1 vs baseline | candidate-2 vs baseline | ...
 * Candidate cells combine value and verdict: `value ± spread%  glyph  delta`.
 */
function renderMultiCandidateTable(result: ComparisonResult): string[] {
  const baseline = result.baselineLabel;
  const candidateHeaders = result.candidates.map((c) => `${c.label} vs ${baseline}`);
  const header = gfmRow(["Metric", baseline, ...candidateHeaders]);
  const separator = gfmSeparator(2 + result.candidates.length);

  const prominentRows: string[] = [];
  const quietRows: string[] = [];
  let unstableCount = 0;

  for (const [name, metric] of Object.entries(result.metrics)) {
    const baselineCell = formatMetricCell(
      metric.baselineMedian,
      metric.baselineSpread,
      metric.meta.unit,
    );
    const candidateCells = result.candidates.map((_, index) =>
      formatMultiCandidateCell(metric.candidates[index], metric.meta.unit),
    );

    const row = gfmRow([name, baselineCell, ...candidateCells]);
    const verdicts = result.candidates.map((_, index) => metric.candidates[index]?.verdict);

    if (isQuietRow(verdicts)) {
      quietRows.push(row);
      if (verdicts.some((v) => v?.verdict === "unstable")) unstableCount++;
    } else {
      prominentRows.push(row);
    }
  }

  const geomeanCells = result.candidates.map((c) => formatGeomeanCell(c.geomean));
  const geomeanRow = gfmRow([GEOMEAN_LABEL, baseline, ...geomeanCells]);

  return [
    header,
    separator,
    ...prominentRows,
    ...renderQuietBlock(header, separator, quietRows, unstableCount),
    geomeanRow,
  ];
}

/** The method footer lines, naming how each verdict was decided. */
function renderMethodFooter(result: ComparisonResult): string[] {
  return methodFooterLines(result.metrics, (hint) => `> *Hint: ${hint}*`);
}

/** The legend as a blockquote: what each glyph means and which target is the baseline. */
function renderLegend(baseline: string): string {
  return `> ${legendGlosses()} — candidates are judged against \`${baseline}\``;
}

/** Summary lines — one per candidate for multi, one total for single. */
function renderSummaryLines(result: ComparisonResult): string[] {
  if (result.candidates.length > 1) {
    return result.candidates.map(
      (candidate, index) => `${candidate.label}: ${renderSummary(result.metrics, index)}`,
    );
  }
  return [renderSummary(result.metrics, 0)];
}

/** Highlight entries, grouped by candidate label when multi-candidate. */
function renderHighlightLines(result: ComparisonResult): string[] {
  if (result.candidates.length > 1) {
    const blocks = result.candidates
      .map((candidate, index) => ({
        label: candidate.label,
        entries: renderHighlightEntries(result.metrics, index),
      }))
      .filter((block) => block.entries.length > 0);

    return blocks.flatMap((block) => [`**${block.label}**:`].concat(block.entries));
  }
  return renderHighlightEntries(result.metrics, 0);
}

/** The metric comparison table — dispatches to single or multi layout. */
function renderTable(result: ComparisonResult): string[] {
  if (result.candidates.length > 1) {
    return renderMultiCandidateTable(result);
  }
  return renderSingleCandidateTable(result, result.candidates[0]!, 0);
}

/**
 * Render a comparison result as GitHub-Flavored Markdown.
 *
 * Summary and highlights sit above the fold for PR comments. The full metric
 * table follows, with within-noise rows collapsed in a `<details>` block.
 * No ANSI codes — glyphs are plain text.
 */
export function renderMarkdown(result: ComparisonResult): string {
  const lines = [...renderSummaryLines(result), ""];

  const highlights = renderHighlightLines(result);
  if (highlights.length > 0) lines.push(...highlights, "");

  lines.push(...renderTable(result), "");
  lines.push(renderLegend(result.baselineLabel));

  const methodLines = renderMethodFooter(result);
  if (methodLines.length > 0) lines.push("", ...methodLines);

  return lines.join("\n");
}
