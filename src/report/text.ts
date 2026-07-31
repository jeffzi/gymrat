import type { WorktreeRemovalFailure } from "../targets.js";
import type { GeomeanResult, MetricVerdict } from "../verdict/verdict.js";
import {
  type CellStyler,
  computeColumnWidth,
  formatEvidence,
  formatHintLabel,
  formatLabel,
  formatMetricCell,
  formatNoiseBand,
  formatPairCount,
  formatTableLine,
  formatVerdictDelta,
  GEOMEAN_LABEL,
  geomeanParts,
  getGlyph,
  isQuietRow,
  legendGlosses,
  methodFooterLines,
  QUIET_VERDICTS,
  selectHighlights,
  styleGlyph,
  styleWithin,
  verdictSummaryParts,
  VERDICT_STYLES,
} from "./format.js";
import type { CandidateComparison, ComparisonResult, MetricComparisons } from "./types.js";

/** The metric name, the two measured values, and the verdict, in column order. */
type Row = readonly [metric: string, baseline: string, candidate: string, verdict: string];

/** Column widths, one per {@link Row} cell. */
type Widths = readonly [number, number, number, number];

/** A metric's rendered cells, kept with the verdict that decides how they are styled. */
interface MetricRow {
  readonly cells: Row;
  readonly verdict: MetricVerdict | undefined;
}

const METRIC_COLUMN_MIN = 16;
const VALUE_COLUMN_MIN = 12;
const VERDICT_COLUMN_MIN = 12;

/** Index of the verdict cell in a {@link Row} — the only cell styled from within. */
const VERDICT_COLUMN = 3;

/**
 * Index of the first candidate column in the multi-candidate table, past the
 * metric name and the baseline's own figures.
 */
const CANDIDATE_COLUMN_OFFSET = 2;

/** Gap between a candidate's own figure and the verdict that follows it in the same cell. */
const CANDIDATE_CELL_GUTTER = "  ";

/** The heading a candidate's non-empty highlights list opens with. */
const HIGHLIGHTS_HEADING = "highlights";

/**
 * Width the highlights block right-aligns its deltas in — the length of a
 * `±NN.N%` percentage.
 *
 * A delta wider than this overflows rather than truncating, so the one word that
 * can stand in for a delta, `unstable`, simply starts two columns further left.
 */
const HIGHLIGHT_DELTA_WIDTH = 6;

/** The `·` separator in the report header, dimmed in colored mode. */
const HEADER_SEPARATOR = "·";

/**
 * Right-align a value in its column, one space clear of the next separator.
 *
 * Padding is measured on the plain string. `padStart` counts an ANSI escape as
 * width it does not occupy, so styling has to wrap a cell once it is padded,
 * never before.
 */
function alignRight(content: string, width: number): string {
  return `${content.padStart(width - 1)} `;
}

/** Indent a left-aligned cell off the separator that precedes it. */
function alignLeft(content: string): string {
  return ` ${content}`;
}

/** Join a verdict cell's parts, dropping whichever ones the verdict has none of. */
function joinCellParts(parts: readonly string[]): string {
  return parts.filter((part) => part !== "").join("  ");
}

/**
 * A verdict as one cell: its glyph, its delta, the noise it was judged against,
 * and — where it rests on fewer pairs than the run took — how many.
 *
 * An unstable metric prints the word in place of the delta — the number exists,
 * but a band that wide makes it a coin toss, and showing it would invite the
 * reader to trust it.
 *
 * The noise band appears only for the two approximate methods: an exact
 * verdict carries no `noisePct`, so its cell ends at the delta.
 *
 * Pairing drops the rounds where either side lacks the metric, so a metric can
 * rest on fewer pairs than the run's sample count. The count is printed exactly
 * where it differs: repeating it on every row would be noise the header already
 * carries, but omitting it where it differs lets the header speak for a metric
 * it does not describe.
 */
function formatVerdictCell(verdict: MetricVerdict | undefined, samples: number): string {
  if (verdict === undefined) return "";
  const delta = formatVerdictDelta(verdict);
  const band = "noisePct" in verdict ? formatNoiseBand(verdict.noisePct) : "";
  const pairs = verdict.n === samples ? "" : formatPairCount(verdict.n);
  return joinCellParts([getGlyph(verdict.verdict), delta, band, pairs]);
}

/** The geomean's verdict cell, and the aggregate inside it that the row exists for. */
interface GeomeanCell {
  readonly figure: string;
  readonly provenance: string;
  readonly text: string;
}

/**
 * The geomean's verdict cell: the aggregate, then how many metrics stand behind
 * it.
 *
 * With nothing left to aggregate the cell says so rather than printing the 0.0%
 * an empty geomean computes to, which would read as "no change measured".
 */
function formatGeomeanCell(geomean: GeomeanResult): GeomeanCell {
  const parts = geomeanParts(geomean);
  if (parts === null) {
    return {
      figure: "—",
      provenance: "no stable gating metrics",
      text: "—  no stable gating metrics",
    };
  }
  return {
    figure: parts.delta,
    provenance: parts.provenance,
    text: `${parts.delta}  ${parts.provenance}`,
  };
}

/**
 * Style a geomean cell's aggregate bold and its provenance dim — the styling
 * both the single- and multi-candidate tables apply to their geomean row.
 */
function styleGeomeanCell(cell: string, geomean: GeomeanCell, useColor: boolean): string {
  const styled = styleWithin(cell, geomean.figure, ["bold"], useColor);
  return styleWithin(styled, geomean.provenance, ["dim"], useColor);
}

/** The style names {@link styleWithin} accepts, in the shape it takes them. */
type Style = Parameters<typeof styleWithin>[2];

/**
 * Style a verdict cell's glyph, and its delta when it has one, the same way.
 *
 * Shared by the multi-candidate table's bright cells and the highlights list:
 * both style a glyph and its trailing delta identically, whether that is the
 * verdict's own class color or a shared dim for a quiet cell on a bright row.
 */
function styleGlyphAndDelta(
  cell: string,
  outcome: MetricVerdict["verdict"],
  delta: string,
  style: Style,
  useColor: boolean,
): string {
  const styled = styleWithin(cell, getGlyph(outcome), style, useColor);
  return delta === "" ? styled : styleWithin(styled, delta, style, useColor);
}

/** Width the metric-name column needs: the widest metric name or the geomean row's label. */
function computeMetricColumnWidth(metricNameLengths: readonly number[]): number {
  return computeColumnWidth(
    "metric".length,
    [...metricNameLengths, GEOMEAN_LABEL.length],
    METRIC_COLUMN_MIN,
  );
}

/** The rule row separating a table's header (or body) from what follows. */
function formatTableRule(widths: readonly number[]): string {
  return widths.map((width) => "─".repeat(width)).join("┼");
}

function formatRow(row: Row, widths: Widths, styleCell?: CellStyler): string {
  return formatTableLine(
    [row[0], alignRight(row[1], widths[1]), alignRight(row[2], widths[2]), alignLeft(row[3])],
    widths,
    styleCell,
  );
}

/** Style the verdict cell of a row, leaving every other cell as it was padded. */
function styleVerdictCell(style: (cell: string) => string): CellStyler {
  return (cell, index) => (index === VERDICT_COLUMN ? style(cell) : cell);
}

/**
 * A metric row, painted by the verdict it reports.
 *
 * The glyph carries the color, and a row with no news to carry is dimmed whole.
 * Both styles land on the finished line, so the column widths behind them were
 * measured on plain text.
 */
function formatMetricRow(
  row: Row,
  widths: Widths,
  verdict: MetricVerdict | undefined,
  useColor: boolean,
): string {
  if (verdict === undefined) return formatRow(row, widths);

  const outcome = verdict.verdict;
  const line = formatRow(
    row,
    widths,
    styleVerdictCell((cell) => {
      let styled = styleGlyph(cell, outcome, useColor);
      if (!QUIET_VERDICTS.has(outcome) && "noisePct" in verdict) {
        styled = styleWithin(styled, formatNoiseBand(verdict.noisePct), ["dim"], useColor);
      }
      return styled;
    }),
  );
  return QUIET_VERDICTS.has(outcome) ? formatLabel(line, ["dim"], useColor) : line;
}

/**
 * The metric table: header, rule, one row per metric, rule, geomean.
 *
 * Widths come from the widest cell rather than a fixed size, so a long metric
 * name or label widens the table instead of being cut. The geomean's own verdict
 * cell is left out of that measurement: it ends its line and is never padded, so
 * letting its provenance text size the column would stretch the rule under every
 * metric row for nothing.
 */
function renderTable(
  result: ComparisonResult,
  candidate: CandidateComparison,
  candidateIndex: number,
  useColor: boolean,
): string[] {
  const baseline = result.baselineLabel;
  const headers: Row = ["metric", baseline, candidate.label, `vs ${baseline}`];

  const rows: MetricRow[] = Object.entries(result.metrics).map(([name, metric]) => {
    const side = metric.candidates[candidateIndex];
    return {
      cells: [
        name,
        formatMetricCell(metric.baselineMedian, metric.baselineSpread, metric.meta.unit),
        formatMetricCell(side?.median, side?.spread, metric.meta.unit),
        formatVerdictCell(side?.verdict, result.samples),
      ],
      verdict: side?.verdict,
    };
  });

  const widths: Widths = [
    computeMetricColumnWidth(rows.map((row) => row.cells[0].length)),
    computeColumnWidth(
      headers[1].length,
      rows.map((row) => row.cells[1].length),
      VALUE_COLUMN_MIN,
    ),
    computeColumnWidth(
      headers[2].length,
      rows.map((row) => row.cells[2].length),
      VALUE_COLUMN_MIN,
    ),
    computeColumnWidth(
      headers[3].length,
      rows.map((row) => row.cells[3].length),
      VERDICT_COLUMN_MIN,
    ),
  ];

  const rule = formatTableRule(widths);
  const geomeanCell = formatGeomeanCell(candidate.geomean);
  const geomean: Row = [GEOMEAN_LABEL, baseline, candidate.label, geomeanCell.text];

  return [
    formatLabel(formatRow(headers, widths), ["bold"], useColor),
    rule,
    ...rows.map((row) => formatMetricRow(row.cells, widths, row.verdict, useColor)),
    rule,
    formatRow(geomean, widths, (cell, index) => {
      if (index === 1) return styleWithin(cell, baseline, ["dim"], useColor);
      if (index === 2) return styleWithin(cell, candidate.label, ["dim"], useColor);
      if (index === VERDICT_COLUMN) return styleGeomeanCell(cell, geomeanCell, useColor);
      return cell;
    }),
  ];
}

/**
 * A verdict as the tail of a candidate's cell in the multi-candidate table.
 *
 * The noise band that ends the single-candidate verdict cell is dropped here: a
 * row already spends one column per candidate, and repeating a per-candidate
 * noise figure in each of them costs the width the deltas need to stay readable.
 * What is left is what differs between candidates: the glyph, the delta, and —
 * where the metric rests on fewer pairs than the run took — how many.
 *
 * The band is not relocated. Highlights carry each verdict's own evidence, which
 * is the band only for band-method verdicts; a signed-rank or no-signal metric's
 * band does not appear anywhere in the N-way report.
 */
function formatCandidateVerdict(verdict: MetricVerdict | undefined, samples: number): string {
  if (verdict === undefined) return "";
  const pairs = verdict.n === samples ? "" : formatPairCount(verdict.n);
  return joinCellParts([getGlyph(verdict.verdict), formatVerdictDelta(verdict), pairs]);
}

/**
 * One candidate's side of a metric: what it measured, how that was judged, and
 * the two joined into a finished cell.
 *
 * `text` is filled during layout rather than at build time because the width it
 * pads to belongs to the whole column, which is not known until every row exists.
 */
interface CandidateCell {
  readonly value: string;
  readonly verdict: string;
  readonly delta: string;
  readonly outcome: MetricVerdict["verdict"] | undefined;
  text: string;
}

/** One metric across every candidate, in the multi-candidate table's column order. */
interface ComparisonRow {
  readonly name: string;
  readonly baseline: string;
  readonly candidates: readonly CandidateCell[];
}

/** One candidate's column, from the header down to the geomean under the rule. */
interface CandidateColumn {
  readonly header: string;
  readonly geomean: GeomeanCell;
  /** This candidate's cell in each metric row, in row order. */
  readonly cells: CandidateCell[];
}

/** The multi-candidate table's contents, held both ways round. */
interface ComparisonGrid {
  readonly rows: readonly ComparisonRow[];
  readonly columns: readonly CandidateColumn[];
}

/**
 * Build the table's contents in both orientations in one pass.
 *
 * Width is a property of a column and dimming a property of a row, so the layout
 * has to read the grid both ways round. Filling both here — rather than
 * transposing one into the other later — keeps every later read a plain
 * iteration, with nothing indexing one array by another array's position and
 * defaulting a miss that cannot happen.
 *
 * Both views hold the same cell objects, so the layout writes each `text` once
 * through the column and the row renders what it wrote.
 */
function buildComparisonGrid(result: ComparisonResult): ComparisonGrid {
  const columns: CandidateColumn[] = result.candidates.map((candidate) => ({
    header: `${candidate.label} vs ${result.baselineLabel}`,
    geomean: formatGeomeanCell(candidate.geomean),
    cells: [],
  }));

  const rows: ComparisonRow[] = [];
  for (const [name, metric] of Object.entries(result.metrics)) {
    const candidates: CandidateCell[] = [];
    for (const [index, column] of columns.entries()) {
      const side = metric.candidates[index];
      const cell: CandidateCell = {
        value: formatMetricCell(side?.median, side?.spread, metric.meta.unit),
        verdict: formatCandidateVerdict(side?.verdict, result.samples),
        delta: side?.verdict ? formatVerdictDelta(side.verdict) : "",
        outcome: side?.verdict?.verdict,
        text: "",
      };
      candidates.push(cell);
      column.cells.push(cell);
    }
    rows.push({
      name,
      baseline: formatMetricCell(metric.baselineMedian, metric.baselineSpread, metric.meta.unit),
      candidates,
    });
  }

  return { rows, columns };
}

/**
 * A candidate's figure and its verdict as one cell, the figures right-aligned
 * among themselves.
 *
 * Padding the figure to the column's widest one keeps the glyphs of a candidate
 * column in a line, so the reader scans one strip of glyphs per candidate rather
 * than hunting for them at whatever offset each row's figure happens to end.
 */
function joinCandidateCell(cell: CandidateCell, valueWidth: number): string {
  const value = cell.value.padStart(valueWidth);
  return cell.verdict === "" ? value : `${value}${CANDIDATE_CELL_GUTTER}${cell.verdict}`;
}

/**
 * The multi-candidate table: the baseline's own figures once, then one column
 * per candidate carrying that candidate's figures and its verdict against the
 * baseline.
 *
 * Every width is measured on plain text and taken from the widest cell the
 * column holds. The geomean's provenance text counts towards that width in every
 * candidate column but the last, which is the one place it can overflow harmlessly:
 * it ends the line there, exactly as it does in the single-candidate table, and
 * letting a long exclusion list size the column instead would stretch the rule
 * under every metric row for one cell's sake.
 */
function renderComparisonTable(result: ComparisonResult, useColor: boolean): string[] {
  const baseline = result.baselineLabel;
  const { rows, columns } = buildComparisonGrid(result);

  for (const column of columns) {
    const valueWidth = Math.max(0, ...column.cells.map((cell) => cell.value.length));
    for (const cell of column.cells) cell.text = joinCandidateCell(cell, valueWidth);
  }

  const metricWidth = computeMetricColumnWidth(rows.map((row) => row.name.length));
  const baselineWidth = computeColumnWidth(
    baseline.length,
    rows.map((row) => row.baseline.length),
    VALUE_COLUMN_MIN,
  );
  const lastColumn = columns.length - 1;
  const candidateWidths = columns.map((column, index) => {
    const contents = column.cells.map((cell) => cell.text.length);
    if (index !== lastColumn) contents.push(column.geomean.text.length);
    return computeColumnWidth(column.header.length, contents, VALUE_COLUMN_MIN);
  });
  const widths = [metricWidth, baselineWidth, ...candidateWidths];

  const line = (
    name: string,
    baselineCell: string,
    candidateCells: readonly string[],
    styleCell?: CellStyler,
  ): string =>
    formatTableLine(
      [name, alignRight(baselineCell, baselineWidth), ...candidateCells.map(alignLeft)],
      widths,
      styleCell,
    );

  const metricRows = rows.map((row) => {
    const rowIsQuiet = isQuietRow(row.candidates.map((cell) => cell.outcome));
    const rendered = line(
      row.name,
      row.baseline,
      row.candidates.map((cell) => cell.text),
      (cell, columnIndex) => {
        const candidateCell = row.candidates[columnIndex - CANDIDATE_COLUMN_OFFSET];
        const outcome = candidateCell?.outcome;
        if (candidateCell === undefined || outcome === undefined) return cell;

        if (!QUIET_VERDICTS.has(outcome)) {
          return styleGlyphAndDelta(
            cell,
            outcome,
            candidateCell.delta,
            VERDICT_STYLES[outcome],
            useColor,
          );
        }

        // Quiet cell on a bright row: dim glyph and delta individually.
        if (!rowIsQuiet && candidateCell.verdict !== "") {
          return styleGlyphAndDelta(cell, outcome, candidateCell.delta, ["dim"], useColor);
        }

        return styleGlyph(cell, outcome, useColor);
      },
    );
    return rowIsQuiet ? formatLabel(rendered, ["dim"], useColor) : rendered;
  });

  const rule = formatTableRule(widths);

  return [
    formatLabel(
      line(
        "metric",
        baseline,
        columns.map((column) => column.header),
      ),
      ["bold"],
      useColor,
    ),
    rule,
    ...metricRows,
    rule,
    line(
      GEOMEAN_LABEL,
      baseline,
      columns.map((column) => column.geomean.text),
      (cell, columnIndex) => {
        if (columnIndex === 1) return styleWithin(cell, baseline, ["dim"], useColor);
        const column = columns[columnIndex - CANDIDATE_COLUMN_OFFSET];
        return column === undefined ? cell : styleGeomeanCell(cell, column.geomean, useColor);
      },
    ),
  ];
}

/** One line tallying every verdict class one candidate earned. */
function renderSummary(
  metrics: MetricComparisons,
  candidateIndex: number,
  useColor = false,
): string {
  return verdictSummaryParts(metrics, candidateIndex, useColor).join("   ");
}

/** Gap between the longest highlighted metric name and the delta that follows it. */
const HIGHLIGHT_NAME_GUTTER = 2;

/**
 * The metrics worth a second look, loudest first, with the evidence behind each.
 *
 * Empty when nothing moved: a heading over an empty list reads as a rendering
 * bug, and a run that changed nothing has nothing to highlight.
 *
 * Padding is measured on the plain strings before any styling wraps them.
 * When `useColor` is true, each entry's glyph and delta (or `unstable` word)
 * carry the verdict's class color, and evidence suffixes are dimmed.
 */
function highlightEntries(
  metrics: MetricComparisons,
  candidateIndex: number,
  useColor = false,
): string[] {
  const highlights = selectHighlights(metrics, candidateIndex);
  if (highlights.length === 0) return [];

  const nameWidth =
    Math.max(...highlights.map((highlight) => highlight.name.length)) + HIGHLIGHT_NAME_GUTTER;

  return highlights.map(({ name, candidate }) => {
    const verdict = candidate.verdict;
    const delta = formatVerdictDelta(verdict);
    const evidence = formatEvidence(verdict);
    const suffix = evidence === "" ? "" : `  ${evidence}`;

    // Pad on plain text, then style the glyph+delta and dim evidence.
    const plain = `  ${getGlyph(verdict.verdict)} ${name.padEnd(nameWidth)}${delta.padStart(
      HIGHLIGHT_DELTA_WIDTH,
    )}${suffix}`;

    if (!useColor) return plain;

    const deltaOrWord = verdict.verdict === "unstable" ? "unstable" : delta;
    const style = VERDICT_STYLES[verdict.verdict];
    let styled = styleGlyphAndDelta(plain, verdict.verdict, deltaOrWord, style, true);
    if (evidence !== "") {
      styled = styleWithin(styled, evidence, ["dim"], true);
    }
    return styled;
  });
}

function renderHighlights(
  metrics: MetricComparisons,
  candidateIndex: number,
  useColor = false,
): string[] {
  const entries = highlightEntries(metrics, candidateIndex, useColor);
  if (entries.length === 0) return [];
  return [formatLabel(HIGHLIGHTS_HEADING, ["bold"], useColor), ...entries];
}

/**
 * The highlights of every candidate, one subsection apiece under a single
 * heading.
 *
 * A candidate whose metrics all sat still contributes no subsection: an empty
 * one under its label reads as a rendering fault rather than as good news.
 */
function renderCandidateHighlights(result: ComparisonResult, useColor: boolean): string[] {
  const blocks = result.candidates
    .map((candidate, index) => ({
      label: candidate.label,
      entries: highlightEntries(result.metrics, index, useColor),
    }))
    .filter((block) => block.entries.length > 0);
  if (blocks.length === 0) return [];

  const lines = [formatLabel(HIGHLIGHTS_HEADING, ["bold"], useColor)];
  for (const block of blocks) {
    lines.push(
      `  ${formatLabel(block.label, ["bold"], useColor)}`,
      ...block.entries.map((entry) => `  ${entry}`),
    );
  }
  return lines;
}

/**
 * One tally per candidate, each behind the label whose verdicts it counts.
 *
 * Labels are padded to a common width so the counts line up under each other:
 * the reader compares candidates by reading down a column of numbers, which a
 * ragged left edge would break. Padding is measured on the plain label, then
 * bold is applied.
 */
function renderSummaries(result: ComparisonResult, useColor: boolean): string[] {
  const labelWidth = Math.max(...result.candidates.map((candidate) => candidate.label.length));
  return result.candidates.map((candidate, index) => {
    const paddedLabel = candidate.label.padEnd(labelWidth);
    const styledLabel = formatLabel(paddedLabel, ["bold"], useColor);
    return `${styledLabel}  ${renderSummary(result.metrics, index, useColor)}`;
  });
}

/**
 * The multi-candidate body: one table holding every candidate, then a tally and
 * a highlights subsection for each.
 *
 * Candidates share the table because they share a baseline — reading two
 * candidates off one row is the comparison the run was for, and a table apiece
 * would leave the reader aligning columns by eye.
 */
function renderComparison(result: ComparisonResult, useColor: boolean): string[] {
  const lines = [
    ...renderComparisonTable(result, useColor),
    "",
    ...renderSummaries(result, useColor),
  ];

  const highlights = renderCandidateHighlights(result, useColor);
  if (highlights.length > 0) {
    lines.push("", ...highlights);
  }

  return lines;
}

/**
 * What each glyph means, and which way round the comparison runs.
 *
 * Printed for every run, whichever method decided the verdicts: the glyphs are
 * the report's whole vocabulary, and which target is the baseline is the one
 * thing a reader cannot infer from the numbers.
 */
function renderLegend(baseline: string, useColor: boolean): string {
  const text = `legend: ${legendGlosses(useColor)} — candidates are judged against ${baseline}`;
  return formatLabel(text, ["dim"], useColor);
}

/**
 * Name the two approximate methods the verdicts came from, and — where any
 * metric fell back to the noise band — how to get a statistical verdict for
 * it instead.
 *
 * `Method` has a third variant, `exact`, but it needs no line here: an exact
 * verdict is self-describing through its `(exact)` evidence, so a run where
 * every metric compared exactly prints no footer at all.
 *
 * A run can reach both approximate methods at once: each metric is paired
 * independently, so a metric that survived enough rounds gets the signed-rank
 * test while one that dropped most of them falls back to the band. Naming
 * only the winner of a precedence order would leave the reader attributing
 * the other metric's verdict to a test that never ran on it.
 *
 * Each line reports the pair count that put its metrics on that side of the
 * six-pair floor: the fewest for signed-rank, the most for the band. Whichever
 * metric the reader picks, the line it read stays true of it.
 */
function renderMethodFooter(result: ComparisonResult, useColor: boolean): string[] {
  return methodFooterLines(
    result.metrics,
    (hint) => `${formatHintLabel(useColor)} ${hint}`,
    useColor,
  );
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
 * What cleanup did, but only when it did something worth saying.
 *
 * A run that created no worktree, or removed every one it created, has nothing
 * to report — and a standing `0 worktrees removed · 0 left behind` trains the
 * reader to skip the line that matters on the run where it does.
 */
function renderWorktreeFooter(result: ComparisonResult): string[] {
  const details = formatCleanupFailures(result.worktreesLeftBehind, result.worktreePruneError);
  if (result.worktreesRemoved === 0 && details.length === 0) return [];

  const noun = result.worktreesRemoved === 1 ? "worktree" : "worktrees";
  return [
    `${result.worktreesRemoved} ${noun} removed · ${result.worktreesLeftBehind.length} left behind`,
    ...details,
  ];
}

/**
 * One candidate's block: its table against the baseline, its verdict counts,
 * and the metrics worth a second look.
 *
 * Everything here is scoped to the candidate, because a verdict is. What speaks
 * for the run as a whole — the legend, the method footer, the cleanup outcome —
 * is rendered once, outside this block.
 */
function renderCandidate(
  result: ComparisonResult,
  candidate: CandidateComparison,
  candidateIndex: number,
  useColor: boolean,
): string[] {
  const lines = [
    ...renderTable(result, candidate, candidateIndex, useColor),
    "",
    renderSummary(result.metrics, candidateIndex, useColor),
  ];

  const highlights = renderHighlights(result.metrics, candidateIndex, useColor);
  if (highlights.length > 0) {
    lines.push("", ...highlights);
  }

  return lines;
}

/**
 * Render a comparison as the plain-text report the CLI prints.
 *
 * The run header comes first, then the body, and finally what speaks for the run
 * as a whole: the legend, the method footer, and any cleanup failure. Terminals
 * anchor on the last lines they printed, so each summary sits below the table it
 * summarizes rather than above it.
 *
 * The body's shape follows the number of candidates. One candidate gets the
 * two-column comparison it is: baseline, candidate, verdict. Two or more share a
 * single table with a column each, since the whole point of running them
 * together is reading them off the same row.
 *
 * Whether the report is styled is the caller's decision, not this renderer's: it
 * never reads `isTTY` or `NO_COLOR`, so the same result renders the same bytes
 * wherever it runs. Color is off unless asked for, which keeps a piped or
 * captured report plain by default.
 */
export function renderReport(result: ComparisonResult, useColor = false): string {
  const candidateLabels = result.candidates.map((candidate) => candidate.label).join(", ");
  const headerPlain = `gymrat compare ${HEADER_SEPARATOR} ${result.baselineLabel} ↔ ${candidateLabels} ${HEADER_SEPARATOR} ${result.samples} paired samples ${HEADER_SEPARATOR} adapter: ${result.adapter}`;
  let header = headerPlain;
  if (useColor) {
    header = styleWithin(header, "gymrat compare", ["bold"], true);
    header = header.replaceAll(HEADER_SEPARATOR, formatLabel(HEADER_SEPARATOR, ["dim"], true));
  }
  const lines = [header];

  if (result.candidates.length > 1) {
    lines.push(...renderComparison(result, useColor));
  } else {
    for (const [index, candidate] of result.candidates.entries()) {
      lines.push(...renderCandidate(result, candidate, index, useColor));
    }
  }

  lines.push(
    "",
    renderLegend(result.baselineLabel, useColor),
    ...renderMethodFooter(result, useColor),
    ...renderWorktreeFooter(result),
  );

  return lines.join("\n");
}
