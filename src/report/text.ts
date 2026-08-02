import type { WorktreeRemovalFailure } from "../targets.js";
import type { GeomeanResult, MetricVerdict } from "../verdict/verdict.js";
import {
  type CellStyler,
  computeColumnWidth,
  displayClass,
  type DisplayClass,
  footerLines,
  formatDelta,
  formatEvidence,
  formatHintLabel,
  formatLabel,
  formatMetricCellParts,
  formatNoiseBandValue,
  formatPairCount,
  formatTableLine,
  formatVariantName,
  formatVerdictDelta,
  GEOMEAN_LABEL,
  geomeanLabel,
  geomeanParts,
  getGlyph,
  hasUnstableHighlight,
  type HighlightBlock,
  type MetricCellParts,
  NO_GEOMEAN_CELL,
  NO_GEOMEAN_FIGURE,
  NO_STABLE_METRICS,
  PLUS_MINUS,
  QUIET_VERDICTS,
  selectHighlights,
  shownClass,
  SPREAD_SEPARATOR,
  type Style,
  styleWithin,
  UNSTABLE_FUTILITY_NOTE,
  VARIANT_NAME_STYLE,
  verdictSummaryParts,
  VERDICT_GLOSSES,
  VERDICT_STYLES,
  withColor,
  withDisplayLabels,
} from "./format.js";
import type {
  CandidateComparison,
  ComparisonResult,
  MetricComparisons,
  ReportOptions,
} from "./types.js";

/** The metric name, the two measured values, and the verdict, in column order. */
type Row = readonly [metric: string, baseline: string, candidate: string, verdict: string];

/** Column widths, one per {@link Row} cell. */
type Widths = readonly [number, number, number, number];

/** A metric's rendered cells, kept with the verdict that decides how they are styled. */
interface MetricRow {
  readonly cells: Row;
  readonly verdict: MetricVerdict | undefined;
  /** The noise band as it was padded into the verdict cell — what the dim style looks for. */
  readonly band: string;
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

/**
 * Gap between the fields of one cell: a candidate's figure and the verdict
 * behind it, and the verdict's own glyph, delta and noise band.
 */
const CELL_GUTTER = "  ";

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

/** Widths a value column pads its two fields to, measured on plain text. */
interface ValueWidths {
  readonly magnitude: number;
  readonly spread: number;
}

/** The widest magnitude and the widest spread a column of value cells holds. */
function valueWidths(cells: readonly MetricCellParts[]): ValueWidths {
  return {
    magnitude: Math.max(0, ...cells.map((cell) => cell.magnitude.length)),
    spread: Math.max(0, ...cells.map((cell) => cell.spread.length)),
  };
}

/**
 * A value cell with its magnitude and its spread each right-aligned in a field
 * of the column's own width.
 *
 * The cell fills both fields whatever it has to put in them, so a measurement
 * with no spread of its own — or none at all — keeps its magnitude under the
 * ones that report one instead of sliding right into their `±`.
 */
function joinValueCell(parts: MetricCellParts, widths: ValueWidths): string {
  const magnitude = parts.magnitude.padStart(widths.magnitude);
  if (widths.spread === 0) return magnitude;
  const spread =
    parts.spread === "" ? "" : `${SPREAD_SEPARATOR}${parts.spread.padStart(widths.spread)}`;
  return `${magnitude}${spread}`.padEnd(widths.magnitude + SPREAD_SEPARATOR.length + widths.spread);
}

/**
 * A verdict cell taken apart: its glyph, its delta, the noise it was judged
 * against, and — where it rests on fewer pairs than the run took — how many.
 *
 * An unstable metric puts the word in `word` rather than in `delta` — the number
 * exists, but a band that wide makes it a coin toss, and showing it would invite
 * the reader to trust it. Its cell ends there: the band it was judged against is
 * what the word already means, and the highlights carry the figure for the
 * reader who wants it.
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
interface VerdictParts {
  readonly glyph: string;
  /** The signed percentage, right-aligned among the column's other deltas. */
  readonly delta: string;
  /** The word standing in for a delta too noisy to report, empty for every other verdict. */
  readonly word: string;
  /** The noise band's figure, without the `±` the column pins in front of it. */
  readonly band: string;
  readonly pairs: string;
}

/**
 * One verdict's fields, with the noise band only where the caller shows one.
 *
 * The multi-candidate table drops the band from its cells, so it asks for the
 * same fields with that one left empty.
 */
function verdictParts(verdict: MetricVerdict, samples: number, withBand: boolean): VerdictParts {
  const unstable = verdict.verdict === "unstable";
  const banded = withBand && !unstable && "noisePct" in verdict;
  return {
    glyph: getGlyph(displayClass(verdict)),
    delta: unstable ? "" : formatDelta(verdict.delta),
    word: unstable ? VERDICT_GLOSSES.unstable : "",
    band: banded ? formatNoiseBandValue(verdict.noisePct) : "",
    pairs: verdict.n === samples ? "" : formatPairCount(verdict.n),
  };
}

/** Widths a verdict column pads its delta and noise band to, measured on plain text. */
interface VerdictWidths {
  readonly delta: number;
  readonly band: number;
}

/**
 * The widest delta and noise band a column of verdict cells holds.
 *
 * The word standing in for a delta is not measured: it is wider than any
 * percentage, and sizing the field from it would push a whole column of bands
 * right for the sake of the one row that has none.
 */
function verdictWidths(cells: readonly VerdictParts[]): VerdictWidths {
  return {
    delta: Math.max(0, ...cells.map((cell) => cell.delta.length)),
    band: Math.max(0, ...cells.map((cell) => cell.band.length)),
  };
}

/** The noise band as it prints: the `±` pinned, its figure right-aligned behind it. */
function bandField(band: string, width: number): string {
  return band === "" ? "" : `${PLUS_MINUS}${band.padStart(width)}`;
}

/**
 * A verdict cell, each field padded to the width its column settled on.
 *
 * A field the verdict has nothing for still holds its column's width, so the
 * fields behind it stay where the rows above put them; the trailing run of
 * padding is cut, since nothing lines up against the end of a line.
 *
 * The band field is padded to the same width even when this verdict has none,
 * as long as some cell in the column does: dropping it outright, as an empty
 * string would when the array is filtered, slides the pairs field behind it
 * out of alignment with every row that does carry a band.
 */
function joinVerdictCell(parts: VerdictParts, widths: VerdictWidths): string {
  const delta = parts.word === "" ? parts.delta.padStart(widths.delta) : parts.word;
  const band = bandField(parts.band, widths.band);
  const bandCell =
    band === "" && widths.band > 0 ? " ".repeat(PLUS_MINUS.length + widths.band) : band;
  return [parts.glyph, delta, bandCell, parts.pairs]
    .filter((field) => field !== "")
    .join(CELL_GUTTER)
    .trimEnd();
}

/** The geomean's aggregate and the fields a cell holding it is styled by. */
interface GeomeanCell {
  /** The aggregate itself, or the dash standing in for one. */
  readonly figure: string;
  /** How many metrics stand behind the figure, where the cell rather than the label carries it. */
  readonly provenance: string;
  /** Whether anything survived to aggregate. */
  readonly aggregated: boolean;
  /** The finished cell of a candidate column in the multi-candidate table. */
  readonly text: string;
}

/**
 * The geomean of one candidate column: the aggregate, then how many metrics
 * stand behind it.
 *
 * Every candidate aggregates its own metrics, so a table holding several has a
 * count per column and carries each beside its own figure. The single-candidate
 * table has one count for the whole row and names it in the label instead,
 * which is what leaves its cell holding the figure alone.
 */
function formatGeomeanCell(geomean: GeomeanResult): GeomeanCell {
  const parts = geomeanParts(geomean);
  if (parts === null) {
    return {
      figure: NO_GEOMEAN_FIGURE,
      provenance: NO_STABLE_METRICS,
      aggregated: false,
      text: NO_GEOMEAN_CELL,
    };
  }
  return {
    figure: parts.delta,
    provenance: parts.provenance,
    aggregated: true,
    text: `${parts.delta} · ${parts.provenance}`,
  };
}

/**
 * The blank the geomean leaves where a metric row carries its glyph.
 *
 * The row reports an aggregate, not a verdict, so it has no glyph to show;
 * holding the slot open is what seats its figure under the deltas above.
 */
const GEOMEAN_GLYPH_SLOT = " ";

/**
 * The geomean as the single-candidate table's verdict cell: the aggregate
 * alone, in the delta field of the column it closes.
 *
 * Reading down the delta column and landing on the run's one aggregate is the
 * comparison the row is there to make, so the figure sits where the deltas sit
 * rather than at the head of the cell.
 */
function geomeanVerdictParts(geomean: GeomeanCell): VerdictParts {
  return {
    glyph: geomean.aggregated ? GEOMEAN_GLYPH_SLOT : geomean.figure,
    delta: geomean.aggregated ? geomean.figure : "",
    word: geomean.aggregated ? "" : geomean.provenance,
    band: "",
    pairs: "",
  };
}

/**
 * Style a geomean cell's aggregate bold and its provenance dim — the styling
 * both the single- and multi-candidate tables apply to their geomean row.
 *
 * A cell whose count rides in the row's label shows no provenance, and styling
 * text a cell does not hold changes nothing.
 */
function styleGeomeanCell(cell: string, geomean: GeomeanCell): string {
  const styled = styleWithin(cell, geomean.figure, ["bold"]);
  return styleWithin(styled, geomean.provenance, ["dim"]);
}

/**
 * Style a verdict cell's glyph, and its delta when it has one, the same way.
 *
 * Shared by every place a verdict is stated in full — the table rows, the
 * candidate columns, the highlights list — so one verdict reads the same
 * wherever the report repeats it.
 */
function styleGlyphAndDelta(
  cell: string,
  shown: DisplayClass,
  delta: string,
  style: Style,
): string {
  const styled = styleWithin(cell, getGlyph(shown), style);
  return delta === "" ? styled : styleWithin(styled, delta, style);
}

/** Width the metric-name column needs: the widest metric name or the geomean row's label. */
function computeMetricColumnWidth(metricNameLengths: readonly number[], label: string): number {
  return computeColumnWidth(
    "metric".length,
    [...metricNameLengths, label.length],
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
 * Only the verdict is painted: the metric name and both figures are the row's
 * evidence, and they read the same whichever way the verdict went. The glyph
 * carries its class color and the noise band behind it is dim, while a quiet
 * class — within noise, identical, unstable — carries its color across the
 * delta as well, since a delta at full brightness is what a reader scanning the
 * column stops on and those rows have nothing to stop for.
 *
 * Every style lands on the finished line, so the column widths behind them were
 * measured on plain text.
 */
function formatMetricRow(row: MetricRow, widths: Widths): string {
  if (row.verdict === undefined) return formatRow(row.cells, widths);

  const outcome = displayClass(row.verdict);
  const delta = QUIET_VERDICTS.has(outcome) ? formatVerdictDelta(row.verdict) : "";
  return formatRow(
    row.cells,
    widths,
    styleVerdictCell((cell) => {
      const styled = styleGlyphAndDelta(cell, outcome, delta, VERDICT_STYLES[outcome]);
      return row.band === "" ? styled : styleWithin(styled, row.band, ["dim"]);
    }),
  );
}

/**
 * The metric table: header, rule, one row per metric, rule, geomean, echo.
 *
 * Widths come from the widest cell rather than a fixed size, so a long metric
 * name or label widens the table instead of being cut. The geomean's own verdict
 * cell is left out of that measurement: it ends its line and is never padded, so
 * letting a long cell size the column would stretch the rule under every metric
 * row for nothing.
 *
 * The echo closing the table repeats the header cell for cell — same quoting,
 * same emphasis, same widths — because a table long enough to scroll leaves the
 * reader at the bottom with columns they can no longer name. It is dimmed whole
 * so it reads as the frame closing rather than as one more row of data, and its
 * name cell stays blank: `metric` heads that column, it does not label the echo.
 */
function renderTable(
  result: ComparisonResult,
  candidate: CandidateComparison,
  candidateIndex: number,
): string[] {
  const baseline = result.baselineLabel;
  const headers: Row = ["metric", baseline, candidate.label, `vs ${baseline}`];

  const measured = Object.entries(result.metrics).map(([name, metric]) => {
    const side = metric.candidates[candidateIndex];
    return {
      name,
      baseline: formatMetricCellParts(
        metric.baselineMedian,
        metric.baselineSpread,
        metric.meta.unit,
      ),
      candidate: formatMetricCellParts(side?.median, side?.spread, metric.meta.unit),
      verdict: side?.verdict,
      parts:
        side?.verdict === undefined ? undefined : verdictParts(side.verdict, result.samples, true),
    };
  });

  const baselineFields = valueWidths(measured.map((row) => row.baseline));
  const candidateFields = valueWidths(measured.map((row) => row.candidate));
  const verdictFields = verdictWidths(
    measured.map((row) => row.parts).filter((parts) => parts !== undefined),
  );

  const rows: MetricRow[] = measured.map((row) => ({
    cells: [
      row.name,
      joinValueCell(row.baseline, baselineFields),
      joinValueCell(row.candidate, candidateFields),
      row.parts === undefined ? "" : joinVerdictCell(row.parts, verdictFields),
    ],
    verdict: row.verdict,
    band: row.parts === undefined ? "" : bandField(row.parts.band, verdictFields.band),
  }));

  const label = geomeanLabel(candidate.geomean.n);
  const widths: Widths = [
    computeMetricColumnWidth(
      rows.map((row) => row.cells[0].length),
      label,
    ),
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
  const geomean: Row = [
    label,
    "",
    "",
    joinVerdictCell(geomeanVerdictParts(geomeanCell), verdictFields),
  ];
  const styleVariantCells: CellStyler = (cell, index) => {
    if (index === 1 || index === 2) {
      return styleWithin(cell, headers[index], VARIANT_NAME_STYLE);
    }
    if (index === VERDICT_COLUMN) {
      // The marker is the bare baseline name, not the whole `vs …` header: the
      // emphasis belongs to the name, and `vs` is prose around it.
      return styleWithin(cell, baseline, VARIANT_NAME_STYLE);
    }
    return cell;
  };

  return [
    formatRow(headers, widths, styleVariantCells),
    rule,
    ...rows.map((row) => formatMetricRow(row, widths)),
    rule,
    formatRow(geomean, widths, (cell, index) =>
      index === VERDICT_COLUMN ? styleGeomeanCell(cell, geomeanCell) : cell,
    ),
    formatLabel(formatRow(["", headers[1], headers[2], headers[3]], widths, styleVariantCells), [
      "dim",
    ]),
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
function formatCandidateVerdict(
  verdict: MetricVerdict | undefined,
  samples: number,
): VerdictParts | undefined {
  return verdict === undefined ? undefined : verdictParts(verdict, samples, false);
}

/**
 * One candidate's side of a metric: what it measured, how that was judged, and
 * the two joined into a finished cell.
 *
 * `text` is filled during layout rather than at build time because the widths it
 * pads to belong to the whole column, which is not known until every row exists.
 */
interface CandidateCell {
  readonly value: MetricCellParts;
  readonly verdict: VerdictParts | undefined;
  readonly delta: string;
  readonly outcome: DisplayClass | undefined;
  text: string;
}

/** One metric across every candidate, in the multi-candidate table's column order. */
interface ComparisonRow {
  readonly name: string;
  readonly baseline: MetricCellParts;
  /** Filled during layout, like {@link CandidateCell.text}: the baseline figure, padded. */
  baselineCell: string;
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
 * Width is a property of a column and a rendered line a property of a row, so
 * the layout has to read the grid both ways round. Filling both here — rather
 * than transposing one into the other later — keeps every later read a plain
 * iteration, with nothing indexing one array by another array's position and
 * defaulting a miss that cannot happen.
 *
 * Both views hold the same cell objects, so the layout writes each `text` once
 * through the column and the row renders what it wrote.
 */
function buildComparisonGrid(result: ComparisonResult): ComparisonGrid {
  const columns: CandidateColumn[] = result.candidates.map((candidate) => ({
    header: candidate.label,
    geomean: formatGeomeanCell(candidate.geomean),
    cells: [],
  }));

  const rows: ComparisonRow[] = [];
  for (const [name, metric] of Object.entries(result.metrics)) {
    const candidates: CandidateCell[] = [];
    for (const [index, column] of columns.entries()) {
      const side = metric.candidates[index];
      const cell: CandidateCell = {
        value: formatMetricCellParts(side?.median, side?.spread, metric.meta.unit),
        verdict: formatCandidateVerdict(side?.verdict, result.samples),
        delta: side?.verdict ? formatVerdictDelta(side.verdict) : "",
        outcome: shownClass(side?.verdict),
        text: "",
      };
      candidates.push(cell);
      column.cells.push(cell);
    }
    rows.push({
      name,
      baseline: formatMetricCellParts(
        metric.baselineMedian,
        metric.baselineSpread,
        metric.meta.unit,
      ),
      baselineCell: "",
      candidates,
    });
  }

  return { rows, columns };
}

/**
 * A candidate's figure and its verdict as one cell, every field padded to the
 * width its column settled on.
 *
 * Padding the figure to the column's own fields keeps the glyphs of a candidate
 * column in a line, so the reader scans one strip of glyphs per candidate rather
 * than hunting for them at whatever offset each row's figure happens to end.
 */
function joinCandidateCell(
  cell: CandidateCell,
  values: ValueWidths,
  verdicts: VerdictWidths,
): string {
  const value = joinValueCell(cell.value, values);
  if (cell.verdict === undefined) return value;
  return `${value}${CELL_GUTTER}${joinVerdictCell(cell.verdict, verdicts)}`;
}

/**
 * The multi-candidate table: the baseline's own figures once, then one column
 * per candidate carrying that candidate's figures and its verdict against the
 * baseline.
 *
 * Every width is measured on plain text and taken from the widest cell the
 * column holds. The geomean cell counts towards that width in every candidate
 * column but the last, which is the one place it can overflow harmlessly: it
 * ends the line there, exactly as it does in the single-candidate table, and
 * letting the count behind a figure size the column instead would stretch the
 * rule under every metric row for one cell's sake.
 *
 * The table closes on the same dimmed echo of its header as the single-candidate
 * one — see {@link renderTable}.
 */
function renderComparisonTable(result: ComparisonResult): string[] {
  const baseline = result.baselineLabel;
  const { rows, columns } = buildComparisonGrid(result);

  for (const column of columns) {
    const values = valueWidths(column.cells.map((cell) => cell.value));
    const verdicts = verdictWidths(
      column.cells.map((cell) => cell.verdict).filter((parts) => parts !== undefined),
    );
    for (const cell of column.cells) cell.text = joinCandidateCell(cell, values, verdicts);
  }

  const baselineFields = valueWidths(rows.map((row) => row.baseline));
  for (const row of rows) row.baselineCell = joinValueCell(row.baseline, baselineFields);

  const baselineHeader = baseline;
  const metricWidth = computeMetricColumnWidth(
    rows.map((row) => row.name.length),
    GEOMEAN_LABEL,
  );
  const baselineWidth = computeColumnWidth(
    baselineHeader.length,
    rows.map((row) => row.baselineCell.length),
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

  const metricRows = rows.map((row) =>
    line(
      row.name,
      row.baselineCell,
      row.candidates.map((cell) => cell.text),
      (cell, columnIndex) => {
        const candidateCell = row.candidates[columnIndex - CANDIDATE_COLUMN_OFFSET];
        const outcome = candidateCell?.outcome;
        if (candidateCell === undefined || outcome === undefined) return cell;
        return styleGlyphAndDelta(cell, outcome, candidateCell.delta, VERDICT_STYLES[outcome]);
      },
    ),
  );

  const rule = formatTableRule(widths);
  const headers = columns.map((column) => column.header);
  const styleVariantCells: CellStyler = (cell, columnIndex) => {
    if (columnIndex === 0) return cell;
    if (columnIndex === 1) return styleWithin(cell, baselineHeader, VARIANT_NAME_STYLE);
    const column = columns[columnIndex - CANDIDATE_COLUMN_OFFSET];
    return column === undefined ? cell : styleWithin(cell, column.header, VARIANT_NAME_STYLE);
  };

  return [
    line("metric", baselineHeader, headers, styleVariantCells),
    rule,
    ...metricRows,
    rule,
    line(
      GEOMEAN_LABEL,
      "",
      columns.map((column) => column.geomean.text),
      (cell, columnIndex) => {
        const column = columns[columnIndex - CANDIDATE_COLUMN_OFFSET];
        return column === undefined ? cell : styleGeomeanCell(cell, column.geomean);
      },
    ),
    formatLabel(line("", baselineHeader, headers, styleVariantCells), ["dim"]),
  ];
}

/** One line tallying every verdict class one candidate earned. */
function renderSummary(metrics: MetricComparisons, candidateIndex: number): string {
  return verdictSummaryParts(metrics, candidateIndex).join("   ");
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
 * Each entry's glyph and delta (or `unstable` word) carry the verdict's class
 * color, and evidence suffixes are dimmed, via `styleText` auto-detection.
 */
function highlightEntries(metrics: MetricComparisons, candidateIndex: number): HighlightBlock {
  const highlights = selectHighlights(metrics, candidateIndex);
  if (highlights.length === 0) return { entries: [], unstable: false };

  const nameWidth =
    Math.max(...highlights.map((highlight) => highlight.name.length)) + HIGHLIGHT_NAME_GUTTER;

  const entries = highlights.map(({ name, metric, candidate }) => {
    const verdict = candidate.verdict;
    const shown = displayClass(verdict);
    const delta = formatVerdictDelta(verdict);
    const evidence = formatEvidence(verdict, metric.meta.unit, metric.baselineMedian);
    const suffix = evidence === "" ? "" : `  ${evidence}`;

    // Pad on plain text, then style the glyph+delta and dim evidence.
    const plain = `  ${getGlyph(shown)} ${name.padEnd(nameWidth)}${delta.padStart(
      HIGHLIGHT_DELTA_WIDTH,
    )}${suffix}`;

    const deltaOrWord = shown === "unstable" ? "unstable" : delta;
    const style = VERDICT_STYLES[shown];
    let styled = styleGlyphAndDelta(plain, shown, deltaOrWord, style);
    if (evidence !== "") {
      styled = styleWithin(styled, evidence, ["dim"]);
    }
    return styled;
  });

  return { entries, unstable: hasUnstableHighlight(highlights) };
}

/** The futility note, indented to sit under the entries it qualifies. */
function futilityLine(): string {
  return `  ${formatLabel(UNSTABLE_FUTILITY_NOTE, ["dim"])}`;
}

function renderHighlights(metrics: MetricComparisons, candidateIndex: number): string[] {
  const { entries, unstable } = highlightEntries(metrics, candidateIndex);
  if (entries.length === 0) return [];
  const lines = [formatLabel(HIGHLIGHTS_HEADING, ["bold"]), ...entries];
  if (unstable) lines.push(futilityLine());
  return lines;
}

/**
 * The highlights of every candidate, one subsection apiece under a single
 * heading.
 *
 * A candidate whose metrics all sat still contributes no subsection: an empty
 * one under its label reads as a rendering fault rather than as good news.
 *
 * The futility note closes the whole section rather than each subsection —
 * it says the same thing about every unstable metric on the page.
 */
function renderCandidateHighlights(result: ComparisonResult): string[] {
  const blocks = result.candidates
    .map((candidate, index) => ({
      label: candidate.label,
      ...highlightEntries(result.metrics, index),
    }))
    .filter((block) => block.entries.length > 0);
  if (blocks.length === 0) return [];

  const lines = [formatLabel(HIGHLIGHTS_HEADING, ["bold"])];
  for (const block of blocks) {
    lines.push(
      `  ${formatLabel(block.label, ["bold"])}`,
      ...block.entries.map((entry) => `  ${entry}`),
    );
  }
  if (blocks.some((block) => block.unstable)) lines.push(futilityLine());
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
function renderSummaries(result: ComparisonResult): string[] {
  const labelWidth = Math.max(...result.candidates.map((candidate) => candidate.label.length));
  return result.candidates.map((candidate, index) => {
    const paddedLabel = candidate.label.padEnd(labelWidth);
    const styledLabel = formatLabel(paddedLabel, ["bold"]);
    return `${styledLabel}  ${renderSummary(result.metrics, index)}`;
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
function renderComparison(result: ComparisonResult): string[] {
  const lines = [...renderComparisonTable(result), "", ...renderSummaries(result)];

  const highlights = renderCandidateHighlights(result);
  if (highlights.length > 0) {
    lines.push("", ...highlights);
  }

  return lines;
}

/**
 * Name the two approximate methods the verdicts came from, and — where any
 * metric fell back to the noise band for want of samples — how to get a
 * statistical verdict for it instead.
 *
 * `Method` has a third variant, `exact`, but it needs no line here: an exact
 * verdict is self-describing through its `(exact)` evidence, so a run where
 * every metric compared exactly prints no method line at all.
 *
 * A run can reach both approximate methods at once: each metric is paired
 * independently, so a metric that survived enough rounds gets the signed-rank
 * test while one that dropped most of them falls back to the band. Naming
 * only the winner of a precedence order would leave the reader attributing
 * the other metric's verdict to a test that never ran on it.
 *
 * The hint outlives `verbose`, because it is the one line that tells a reader
 * to change what they ran rather than explaining what already happened.
 */
function renderMethodFooter(result: ComparisonResult, verbose: boolean): string[] {
  return footerLines(result.metrics, verbose, (hint) => `${formatHintLabel()} ${hint}`);
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
): string[] {
  const lines = [
    ...renderTable(result, candidate, candidateIndex),
    "",
    renderSummary(result.metrics, candidateIndex),
  ];

  const highlights = renderHighlights(result.metrics, candidateIndex);
  if (highlights.length > 0) {
    lines.push("", ...highlights);
  }

  return lines;
}

/**
 * Render a comparison as the plain-text report the CLI prints.
 *
 * The run header comes first, then the body, and finally what speaks for the run
 * as a whole: the method footer and any cleanup failure. Terminals anchor on the
 * last lines they printed, so each summary sits below the table it summarizes
 * rather than above it.
 *
 * The body's shape follows the number of candidates. One candidate gets the
 * two-column comparison it is: baseline, candidate, verdict. Two or more share a
 * single table with a column each, since the whole point of running them
 * together is reading them off the same row.
 *
 * Whether the report is styled follows `options.color`, which `withColor` pins
 * for the whole render: `false` leaves the report free of escapes, `true` forces
 * them, and `undefined` defers to `styleText` auto-detection — the `NO_COLOR` /
 * `FORCE_COLOR` env vars and the stream's TTY status.
 */
export function renderReport(result: ComparisonResult, options: ReportOptions = {}): string {
  return withColor(options.color, () => {
    const display = withDisplayLabels(result);
    const candidateNames = display.candidates
      .map((candidate) => formatVariantName(candidate.label))
      .join(", ");
    let header = `gymrat compare ${HEADER_SEPARATOR} baseline ${formatVariantName(display.baselineLabel)} ↔ ${candidateNames} ${HEADER_SEPARATOR} ${display.samples} paired samples ${HEADER_SEPARATOR} adapter: ${display.adapter}`;
    header = styleWithin(header, "gymrat compare", ["bold"]);
    header = header.replaceAll(HEADER_SEPARATOR, formatLabel(HEADER_SEPARATOR, ["dim"]));
    const lines = [header];

    if (display.candidates.length > 1) {
      lines.push(...renderComparison(display));
    } else {
      for (const [index, candidate] of display.candidates.entries()) {
        lines.push(...renderCandidate(display, candidate, index));
      }
    }

    const footer = [
      ...renderMethodFooter(display, options.verbose ?? false),
      ...renderWorktreeFooter(display),
    ];
    if (footer.length > 0) {
      lines.push("", ...footer);
    }

    return lines.join("\n");
  });
}
