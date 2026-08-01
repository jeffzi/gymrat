import type { GeomeanResult, MetricVerdict } from "../verdict/verdict.js";
import {
  displayClass,
  type DisplayClass,
  formatEvidence,
  formatMetricCell,
  formatVerdictDelta,
  GEOMEAN_LABEL,
  geomeanParts,
  getGlyph,
  hasUnstableHighlight,
  isQuietRow,
  legendGlosses,
  methodFooterLines,
  selectHighlights,
  UNSTABLE_FUTILITY_NOTE,
  verdictSummaryParts,
  withDisplayLabels,
} from "./format.js";
import type {
  CandidateComparison,
  ComparisonResult,
  MetricComparison,
  MetricComparisons,
} from "./types.js";

/** One line tallying every verdict class one candidate earned. */
function renderSummary(metrics: MetricComparisons, candidateIndex: number): string {
  return verdictSummaryParts(metrics, candidateIndex).join(" · ");
}

/**
 * A variant name as markdown code.
 *
 * Code spans are what the text report's emphasis becomes here: they set a
 * branch name apart from the prose and stop a name carrying `_` or `*` from
 * being read as markup.
 */
function variantName(label: string): string {
  return `\`${label}\``;
}

/** Format a single highlight entry as a markdown list item. */
function formatHighlightEntry(
  name: string,
  metric: MetricComparison,
  verdict: MetricVerdict,
): string {
  const glyph = getGlyph(displayClass(verdict));
  const delta = formatVerdictDelta(verdict);
  const evidence = formatEvidence(verdict, metric.meta.unit, metric.baselineMedian);
  const suffix = evidence === "" ? "" : `  ${evidence}`;
  return `- ${glyph} ${name}  ${delta}${suffix}`;
}

/** One candidate's highlight list items, and whether the noise swamped any of them. */
interface HighlightBlock {
  readonly entries: string[];
  readonly unstable: boolean;
}

/** Render highlights for a single candidate as markdown list items. */
function renderHighlightEntries(
  metrics: MetricComparisons,
  candidateIndex: number,
): HighlightBlock {
  const highlights = selectHighlights(metrics, candidateIndex);
  return {
    entries: highlights.map(({ name, metric, candidate }) =>
      formatHighlightEntry(name, metric, candidate.verdict),
    ),
    unstable: hasUnstableHighlight(highlights),
  };
}

/** The futility note, as the italic line that closes a highlights list. */
const FUTILITY_LINE = `_${UNSTABLE_FUTILITY_NOTE}_`;

/**
 * A verdict rendered as part of a cell: glyph + delta.
 *
 * Unlike the text renderer, this does not include the noise band in the
 * verdict portion — the GFM table cell combines value and verdict in one,
 * keeping it compact.
 */
function formatVerdictPart(verdict: MetricVerdict | undefined): string {
  if (verdict === undefined) return "";
  return `${getGlyph(displayClass(verdict))}  ${formatVerdictDelta(verdict)}`;
}

/**
 * The geomean cell: delta plus provenance (how many metrics stand behind it
 * and how many were excluded), or a dash when nothing survived.
 */
function formatGeomeanCell(geomean: GeomeanResult): string {
  const parts = geomeanParts(geomean);
  if (parts === null) return "—";
  return `${parts.delta}  ${parts.provenance}`;
}

/** How many of the collapsed rows each quiet class beyond within-noise accounts for. */
interface QuietCounts {
  identical: number;
  unstable: number;
}

/**
 * Wrap quiet rows in a `<details>` block with a summary labelling how many are
 * within noise, identical, or unstable. Returns an empty array when there are no
 * quiet rows.
 */
function renderQuietBlock(
  header: string,
  separator: string,
  quietRows: readonly string[],
  counts: QuietCounts,
): string[] {
  if (quietRows.length === 0) return [];

  const noiseCount = quietRows.length - counts.identical - counts.unstable;
  const parts: string[] = [];
  if (noiseCount > 0) parts.push(`${noiseCount} within noise`);
  if (counts.identical > 0) parts.push(`${counts.identical} identical`);
  if (counts.unstable > 0) parts.push(`${counts.unstable} unstable`);

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
  const header = gfmRow([
    "Metric",
    variantName(baseline),
    variantName(candidate.label),
    `vs ${variantName(baseline)}`,
  ]);
  const separator = gfmSeparator(4);

  const prominentRows: string[] = [];
  const quietRows: string[] = [];
  const quietCounts: QuietCounts = { identical: 0, unstable: 0 };

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

    const shown = side?.verdict === undefined ? undefined : displayClass(side.verdict);
    if (isQuietRow([shown])) {
      quietRows.push(row);
      if (shown === "unstable") quietCounts.unstable++;
      else if (shown === "identical") quietCounts.identical++;
    } else {
      prominentRows.push(row);
    }
  }

  const geomeanDelta = formatGeomeanCell(candidate.geomean);
  const geomeanRow = gfmRow([
    GEOMEAN_LABEL,
    variantName(baseline),
    variantName(candidate.label),
    geomeanDelta,
  ]);

  return [
    header,
    separator,
    ...prominentRows,
    ...renderQuietBlock(header, separator, quietRows, quietCounts),
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
  const candidateHeaders = result.candidates.map((c) => variantName(c.label));
  const header = gfmRow(["Metric", variantName(baseline), ...candidateHeaders]);
  const separator = gfmSeparator(2 + result.candidates.length);

  const prominentRows: string[] = [];
  const quietRows: string[] = [];
  const quietCounts: QuietCounts = { identical: 0, unstable: 0 };

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
    const shownList: (DisplayClass | undefined)[] = result.candidates.map((_, index) => {
      const verdict = metric.candidates[index]?.verdict;
      return verdict === undefined ? undefined : displayClass(verdict);
    });

    if (isQuietRow(shownList)) {
      quietRows.push(row);
      if (shownList.includes("unstable")) quietCounts.unstable++;
      else if (shownList.includes("identical")) quietCounts.identical++;
    } else {
      prominentRows.push(row);
    }
  }

  const geomeanCells = result.candidates.map((c) => formatGeomeanCell(c.geomean));
  const geomeanRow = gfmRow([GEOMEAN_LABEL, variantName(baseline), ...geomeanCells]);

  return [
    header,
    separator,
    ...prominentRows,
    ...renderQuietBlock(header, separator, quietRows, quietCounts),
    geomeanRow,
  ];
}

/** The method footer lines, naming how each verdict was decided. */
function renderMethodFooter(result: ComparisonResult): string[] {
  return methodFooterLines(result.metrics, (hint) => `> *Hint: ${hint}*`);
}

/** The legend as a blockquote: what each glyph means and which target is the baseline. */
function renderLegend(baseline: string): string {
  return `> ${legendGlosses()} — candidates are judged against ${variantName(baseline)}`;
}

/** Summary lines — one per candidate for multi, one total for single. */
function renderSummaryLines(result: ComparisonResult): string[] {
  if (result.candidates.length > 1) {
    return result.candidates.map(
      (candidate, index) =>
        `${variantName(candidate.label)}: ${renderSummary(result.metrics, index)}`,
    );
  }
  return [renderSummary(result.metrics, 0)];
}

/**
 * Highlight entries, grouped by candidate label when multi-candidate.
 *
 * The futility note closes the whole section rather than each group — it says
 * the same thing about every unstable metric on the page.
 */
function renderHighlightLines(result: ComparisonResult): string[] {
  if (result.candidates.length > 1) {
    const blocks = result.candidates
      .map((candidate, index) => ({
        label: candidate.label,
        ...renderHighlightEntries(result.metrics, index),
      }))
      .filter((block) => block.entries.length > 0);

    const lines = blocks.flatMap((block) => [`**${variantName(block.label)}**:`, ...block.entries]);
    if (blocks.some((block) => block.unstable)) lines.push(FUTILITY_LINE);
    return lines;
  }

  const { entries, unstable } = renderHighlightEntries(result.metrics, 0);
  if (entries.length === 0) return [];
  return unstable ? [...entries, FUTILITY_LINE] : entries;
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
  const display = withDisplayLabels(result);
  const lines = [...renderSummaryLines(display), ""];

  const highlights = renderHighlightLines(display);
  if (highlights.length > 0) lines.push(...highlights, "");

  lines.push(...renderTable(display), "");
  lines.push(renderLegend(display.baselineLabel));

  const methodLines = renderMethodFooter(display);
  if (methodLines.length > 0) lines.push("", ...methodLines);

  return lines.join("\n");
}
