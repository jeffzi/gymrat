import type { ConfigKinds } from "../config.js";
import { assertNever } from "../errors.js";
import type { WorktreeRemovalFailure } from "../targets.js";
import type { GeomeanResult, MetricVerdict } from "../verdict/verdict.js";
import {
  baselineCellParts,
  candidateCellParts,
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
  geomeanScopeLabel,
  geomeanValueStyle,
  getGlyph,
  hasUnstableHighlight,
  type HighlightBlock,
  highlightLabel,
  type MetricCellParts,
  NO_GEOMEAN_CELL,
  NO_GEOMEAN_FIGURE,
  NO_STABLE_METRICS,
  PLUS_MINUS,
  QUIET_VERDICTS,
  scopedGeomeanLabel,
  selectHighlights,
  shownClass,
  SPREAD_SEPARATOR,
  type Style,
  styleWithin,
  truncateLabels,
  UNSTABLE_FUTILITY_NOTE,
  VARIANT_NAME_STYLE,
  verdictSummaryParts,
  VERDICT_GLOSSES,
  VERDICT_STYLES,
  withColor,
  withDisplayLabels,
} from "./format.js";
import {
  flatGeomeanOf,
  groupGeomeanOf,
  informationalTag,
  kindGeomeanOf,
  planSections,
  sectionLabel,
  type SectionLayout,
  type SectionPlan,
  spansManyKinds,
} from "./sections.js";
import type {
  CandidateComparison,
  ComparisonResult,
  FailOnCondition,
  MeasurementResult,
  MetricComparisons,
  ReportOptions,
  WorktreeCleanupOutcome,
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

/** The two names every metric row carries, whatever the table reports beside them. */
interface NamedRow {
  /** The full metric name, which the flat table shows and every other output keys on. */
  readonly name: string;
  /** The short name a section shows the row under, group prefix stripped and indented. */
  readonly label: string;
}

/** Fields shared by every row that names a metric in either comparison layout. */
interface MetricRowBase extends NamedRow {
  readonly gating: boolean;
}

/**
 * One metric of the single-candidate table, measured but not yet padded.
 *
 * The fields every cell pads to are properties of the whole column, so a row is
 * held apart from its widths until every row exists.
 */
interface MeasuredRow extends MetricRowBase {
  readonly baseline: MetricCellParts;
  readonly candidate: MetricCellParts;
  readonly verdict: MetricVerdict | undefined;
  readonly parts: VerdictParts | undefined;
}

/** The metric-name column's header text, and its fallback when a section carries no title. */
const METRIC_COLUMN_HEADER = "metric";

const METRIC_COLUMN_MIN = 16;
const VALUE_COLUMN_MIN = 12;
const VERDICT_COLUMN_MIN = 12;

/** Index of the label cell — the first column of both tables, whatever the row states. */
const LABEL_COLUMN = 0;

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

/** Join a run header's parts with the dimmed `·` separator every report header shares. */
function joinHeaderParts(parts: readonly string[]): string {
  return parts.join(` ${formatLabel(HEADER_SEPARATOR, ["dim"])} `);
}

/** `count`, followed by `noun` pluralized with a trailing `s` unless `count` is exactly one. */
function pluralize(count: number, noun: string): string {
  return `${count} ${noun}${count === 1 ? "" : "s"}`;
}

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
 * verdict carries no `noisePct`, so its cell ends at the delta. So does an
 * inconclusive one, whose band is the floor constant rather than a measurement.
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
  const shown = displayClass(verdict);
  const unstable = verdict.verdict === "unstable";
  // An inconclusive verdict's band is the noise floor constant, not a width
  // anything was measured against, so the row states the delta and stops.
  const banded = withBand && !unstable && shown !== "inconclusive" && "noisePct" in verdict;
  return {
    glyph: getGlyph(shown),
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

/** A span of an already-padded cell, and the style it carries. */
interface StyledSpan {
  readonly text: string;
  readonly style: Style;
}

/**
 * Style each span where it sits inside a finished cell.
 *
 * A span is found by its own text, so the spans of one cell have to be distinct
 * strings — a cell repeating one span's text would style the first occurrence
 * twice and leave the second bare.
 */
function styleSpans(cell: string, spans: readonly StyledSpan[]): string {
  let styled = cell;
  for (const span of spans) {
    styled = styleWithin(styled, span.text, span.style);
  }
  return styled;
}

/** The dash and the phrase standing in for an aggregate that has nothing behind it. */
function emptyGeomeanSpans(): StyledSpan[] {
  return [
    { text: NO_GEOMEAN_FIGURE, style: ["bold"] },
    { text: NO_STABLE_METRICS, style: ["dim"] },
  ];
}

/** One candidate column's aggregate cell in the multi-candidate table. */
interface AggregateColumnCell {
  readonly text: string;
  readonly spans: readonly StyledSpan[];
}

/**
 * The geomean of one candidate column: the aggregate, then how many metrics
 * stand behind it.
 *
 * Every candidate aggregates its own metrics, so a table holding several has a
 * count per column and carries each beside its own figure. The single-candidate
 * table has one count for the whole row and names it in the label instead,
 * which is what leaves its cell holding the figure alone.
 *
 * `outcomes` are that column's own verdicts on the metrics the figure covers,
 * which is what leaves a quiet column uncolored beside a colored one.
 */
function geomeanColumnCell(
  geomean: GeomeanResult,
  outcomes: ReadonlyArray<DisplayClass | undefined>,
): AggregateColumnCell {
  const parts = geomeanParts(geomean);
  if (parts === null) {
    return { text: NO_GEOMEAN_CELL, spans: emptyGeomeanSpans() };
  }
  return {
    text: `${parts.delta} · ${parts.provenance}`,
    spans: [
      { text: parts.delta, style: geomeanValueStyle(geomean, outcomes) },
      { text: parts.provenance, style: ["dim"] },
    ],
  };
}

/**
 * The blank an aggregate row leaves where a metric row carries its glyph.
 *
 * The row reports an aggregate, not a verdict, so it has no glyph to show;
 * holding the slot open is what seats its figure under the deltas above.
 */
const GEOMEAN_GLYPH_SLOT = " ";

/** An aggregate row's verdict cell in the single-candidate table. */
interface AggregateCell {
  readonly parts: VerdictParts;
  readonly spans: readonly StyledSpan[];
}

/**
 * The single-candidate table's aggregate cell: the figure and its band, in the
 * delta and band fields of the column it closes.
 *
 * Reading down the delta column and landing on an aggregate is the comparison
 * the row is there to make, so the figure sits where the deltas sit rather than
 * at the head of the cell, and the band it was judged against sits under the
 * rows' own bands — the figure is colored by that band, so a reader asking why
 * it stayed plain finds the width beside it.
 */
function geomeanVerdictCell(
  geomean: GeomeanResult,
  outcomes: ReadonlyArray<DisplayClass | undefined>,
): AggregateCell {
  const parts = geomeanParts(geomean);
  if (parts === null) {
    return {
      parts: { glyph: NO_GEOMEAN_FIGURE, delta: "", word: NO_STABLE_METRICS, band: "", pairs: "" },
      spans: emptyGeomeanSpans(),
    };
  }
  return {
    parts: { glyph: GEOMEAN_GLYPH_SLOT, delta: parts.delta, word: "", band: parts.band, pairs: "" },
    spans: [{ text: parts.delta, style: geomeanValueStyle(geomean, outcomes) }],
  };
}

/**
 * An aggregate cell's spans, its band among them.
 *
 * The band prints right-aligned behind a pinned `±`, so the text a span has to
 * match only exists once the column has settled on a width — which is why the
 * cell carries the band as a figure and the span is built here.
 */
function aggregateSpans(cell: AggregateCell, widths: VerdictWidths): readonly StyledSpan[] {
  const band = bandField(cell.parts.band, widths.band);
  return band === "" ? cell.spans : [...cell.spans, { text: band, style: ["dim"] }];
}

/** The display class each of `metrics` landed in, in the order the rows are drawn. */
function measuredOutcomes(metrics: readonly MeasuredRow[]): (DisplayClass | undefined)[] {
  return metrics.map((row) => shownClass(row.verdict));
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

/** Width the metric-name column needs: the widest label any of its rows carries. */
function computeMetricColumnWidth(headerLabel: string, labelLengths: number[]): number {
  return computeColumnWidth(headerLabel.length, labelLengths, METRIC_COLUMN_MIN);
}

/** Indent a metric row carries under the group sub-header above it. */
const GROUP_INDENT = "  ";

/** A metric's name cell inside a section: its short name, indented under its group. */
function indentedSectionLabel(shortName: string, group: string | undefined): string {
  const label = sectionLabel(shortName, group);
  return group === undefined ? label : `${GROUP_INDENT}${label}`;
}

/**
 * A section's title: the kind, and — where none of its metrics gates — the tag
 * saying so and the config key that decided it.
 */
function sectionAnnotation<Metric>(
  section: SectionPlan<Metric>,
  configKinds: ConfigKinds | undefined,
): string | undefined {
  if (section.hasGating) return undefined;
  return formatLabel(informationalTag(section.kind, configKinds), ["dim"]);
}

/** An aggregate row: what it covers, and what it states for each candidate column. */
interface AggregateRow<Cell> {
  readonly label: string;
  readonly cell: Cell;
}

/** A line of a table body, held until the column widths that render it are known. */
type BodyLine<Metric, Cell> =
  | { readonly type: "title"; readonly text: string }
  | { readonly type: "blank" }
  | { readonly type: "header"; readonly title?: string }
  | { readonly type: "rule" }
  | { readonly type: "border" }
  | { readonly type: "group"; readonly label: string }
  | { readonly type: "metric"; readonly row: Metric }
  | ({ readonly type: "aggregate" } & AggregateRow<Cell>);

/**
 * The aggregate rows a table fills its body with — the half that differs between
 * the two tables.
 *
 * Each builder is handed the metric rows its figure covers, in the order the
 * section drew them, so a row that reads the verdicts behind an aggregate reads
 * exactly the rows the reader sees above it.
 */
interface AggregateRows<Metric, Cell> {
  group(kind: string, group: string, metrics: readonly Metric[]): AggregateRow<Cell>;
  kind(kind: string, metrics: readonly Metric[]): AggregateRow<Cell>;
  flat(metrics: readonly Metric[]): AggregateRow<Cell>;
}

/** Every metric row a section holds, its group blocks flattened back into row order. */
function sectionMetrics<Metric>(section: SectionPlan<Metric>): Metric[] {
  return section.blocks.flatMap((block) =>
    block.type === "group" ? block.metrics : [block.metric],
  );
}

/** The lines one section's blocks produce: groups, standalone metrics, and sub-geomeans. */
function planBlocks<Metric, Cell>(
  section: SectionPlan<Metric>,
  rows: AggregateRows<Metric, Cell> | undefined,
): BodyLine<Metric, Cell>[] {
  const lines: BodyLine<Metric, Cell>[] = [];
  for (const [blockIndex, block] of section.blocks.entries()) {
    const previous = section.blocks[blockIndex - 1];
    if (previous !== undefined && (previous.type === "group" || block.type === "group")) {
      lines.push({ type: "blank" });
    }

    if (block.type === "metric") {
      lines.push({ type: "metric", row: block.metric });
      continue;
    }
    lines.push({ type: "group", label: block.group });
    for (const metric of block.metrics) lines.push({ type: "metric", row: metric });
    if (rows !== undefined) {
      lines.push({ type: "aggregate", ...rows.group(section.kind, block.group, block.metrics) });
    }
  }
  return lines;
}

/**
 * The body of a table: its header, its metric rows, and the aggregate closing
 * whatever each of them belongs to.
 *
 * A run reporting one kind — every run of an adapter that names none — draws the
 * flat table it always did: one header, the metric rows under it, one geomean.
 * A run reporting several draws a section per kind, each repeating the header so
 * the reader never scrolls back to learn which column is which, and each closed
 * by its own kind's geomean.
 *
 * A repeated header is a landmark rather than a first line, so it is boxed
 * between a top border and a column rule instead of resting on one, which keeps
 * the eye from reading the header as one more row of the section above it.
 *
 * Every gap is a true blank line, at both scales: the one parting two groups
 * inside a section as much as the one parting two sections, or the first section
 * from the banner.
 */
function planBody<Metric, Cell>(
  layout: SectionLayout<Metric>,
  rows: AggregateRows<Metric, Cell> | undefined,
  annotation: (section: SectionPlan<Metric>) => string | undefined,
): BodyLine<Metric, Cell>[] {
  if (layout.sections.length <= 1) {
    const body: BodyLine<Metric, Cell>[] = [
      { type: "header" },
      { type: "rule" },
      ...layout.ordered.map((row) => ({ type: "metric" as const, row })),
    ];
    if (rows !== undefined) {
      body.push({ type: "rule" }, { type: "aggregate", ...rows.flat(layout.ordered) });
    }
    return body;
  }

  const lines: BodyLine<Metric, Cell>[] = [];
  for (const section of layout.sections) {
    lines.push({ type: "blank" });

    const tag = annotation(section);
    if (tag !== undefined) lines.push({ type: "title", text: tag });
    lines.push({ type: "border" }, { type: "header", title: section.kind }, { type: "rule" });
    lines.push(...planBlocks(section, rows));
    if (rows !== undefined) {
      lines.push(
        { type: "rule" },
        { type: "aggregate", ...rows.kind(section.kind, sectionMetrics(section)) },
      );
    }
  }

  return lines;
}

/** The labels of every row that is not a metric — what the name column widens for. */
function aggregateLabelLengths<Metric, Cell>(body: readonly BodyLine<Metric, Cell>[]): number[] {
  return body.flatMap((line) =>
    line.type === "group" || line.type === "aggregate" ? [line.label.length] : [],
  );
}

function widestHeaderLabel<Metric, Cell>(body: readonly BodyLine<Metric, Cell>[]): string {
  const labels = body.flatMap((line) =>
    line.type === "header" ? [line.title ?? METRIC_COLUMN_HEADER] : [],
  );
  return labels.reduce(
    (widest, label) => (label.length > widest.length ? label : widest),
    METRIC_COLUMN_HEADER,
  );
}

/** One dash span per column width, joined by `separator`. */
function formatHorizontalLine(widths: readonly number[], separator: string): string {
  return widths.map((width) => "─".repeat(width)).join(separator);
}

/** The rule row separating a table's header (or body) from what follows. */
function formatTableRule(widths: readonly number[]): string {
  return formatHorizontalLine(widths, "┼");
}

/**
 * The line opening a section — the top of its box, meeting each column at a
 * top-T junction so the border joins the separators of the header below it.
 */
function formatTableBorder(widths: readonly number[]): string {
  return formatHorizontalLine(widths, "┬");
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
 * The style a group sub-header wears.
 *
 * The row names the block beneath it rather than reporting a measurement, and
 * no verdict color is free for that job — every one of them already means an
 * outcome. Blue is the one hue the verdicts leave unclaimed, so a scan down the
 * name column reads the structure without mistaking it for a result.
 */
const GROUP_LABEL_STYLE: Style = ["blue"];

/**
 * The style every geomean label wears, at whatever scope it closes.
 *
 * The figure beside it is already emboldened, so emboldening the label keeps the
 * row reading as one statement and sets it apart from the metric rows it sums.
 */
const AGGREGATE_LABEL_STYLE: Style = ["bold"];

/**
 * Style a row's own label where it sits in the padded label cell, falling
 * through to `rest` for every other cell.
 *
 * An aggregate row styles its label one way and its verdict cell another, so
 * threading both through `rest` is what lets it still hand the table a single
 * styler function.
 */
function styleLabelCell(
  label: string,
  style: Style,
  rest: CellStyler = (cell) => cell,
): CellStyler {
  return (cell, index) =>
    index === LABEL_COLUMN ? styleWithin(cell, label, style) : rest(cell, index);
}

/**
 * A header row's label cell: bold when the row carries a section title,
 * otherwise styled like every other cell in the header.
 *
 * A sectioned run repeats the header once per kind, and each repeat doubles
 * as that section's own title; a flat run's single header carries no title
 * and leaves every column, including the label, to `styleVariantCells`.
 */
function styleHeaderCell(
  label: string,
  hasTitle: boolean,
  styleVariantCells: CellStyler,
): CellStyler {
  return hasTitle ? styleLabelCell(label, ["bold"], styleVariantCells) : styleVariantCells;
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

/** The row-assembly callbacks a table supplies for the lines whose shape is table-specific. */
interface BodyLineRenderers<Metric, Cell> {
  header(label: string, hasTitle: boolean): string;
  group(label: string): string;
  metric(row: Metric): string;
  aggregate(row: AggregateRow<Cell>): string;
}

/**
 * One line of a table body rendered to text.
 *
 * The four constant cases — `title`, `blank`, `rule`, `border` — and exhaustiveness checking
 * are shared here; the two tables differ only in how they assemble the remaining four, which
 * they supply through `renderers`.
 */
function renderBodyLine<Metric, Cell>(
  bodyLine: BodyLine<Metric, Cell>,
  rule: string,
  border: string,
  renderers: BodyLineRenderers<Metric, Cell>,
): string {
  switch (bodyLine.type) {
    case "title":
      return bodyLine.text;
    case "blank":
      return "";
    case "header":
      return renderers.header(bodyLine.title ?? METRIC_COLUMN_HEADER, bodyLine.title !== undefined);
    case "rule":
      return rule;
    case "border":
      return border;
    case "group":
      return renderers.group(bodyLine.label);
    case "metric":
      return renderers.metric(bodyLine.row);
    case "aggregate":
      return renderers.aggregate(bodyLine);
    default:
      return assertNever(bodyLine);
  }
}

/**
 * The metric table: header, rule, one row per metric, rule, geomean.
 *
 * Widths come from the widest cell rather than a fixed size, so a long metric
 * name or label widens the table instead of being cut. The verdict column
 * measures both metric-row and aggregate cells so a geomean band that is wider
 * than any metric band still fits inside the rule.
 */
function renderTable(
  result: ComparisonResult,
  candidate: CandidateComparison,
  candidateIndex: number,
): string[] {
  const baseline = result.baselineLabel;
  const headers: Row = [METRIC_COLUMN_HEADER, baseline, candidate.label, `vs ${baseline}`];

  const layout = planSections(result.metrics, (name, group, metric): MeasuredRow => {
    const side = metric.candidates[candidateIndex];
    return {
      name,
      label: indentedSectionLabel(metric.meta.shortName, group),
      baseline: baselineCellParts(metric),
      candidate: candidateCellParts(side, metric.meta.unit),
      verdict: side?.verdict,
      parts:
        side?.verdict === undefined ? undefined : verdictParts(side.verdict, result.samples, true),
      gating: metric.meta.gating,
    };
  });
  const sectioned = layout.sections.length > 1;

  const baselineFields = valueWidths(layout.ordered.map((row) => row.baseline));
  const candidateFields = valueWidths(layout.ordered.map((row) => row.candidate));

  const scopedRow = (
    scope: string,
    geomean: GeomeanResult,
    metrics: readonly MeasuredRow[],
  ): AggregateRow<AggregateCell> => ({
    label: scopedGeomeanLabel(scope, geomean),
    cell: geomeanVerdictCell(geomean, measuredOutcomes(metrics)),
  });

  const body = planBody<MeasuredRow, AggregateCell>(
    layout,
    {
      group: (kind, group, metrics) =>
        scopedRow(group, groupGeomeanOf(candidate, kind, group), metrics),
      kind: (kind, metrics) => scopedRow(kind, kindGeomeanOf(candidate, kind), metrics),
      flat: (metrics) => {
        const geomean = flatGeomeanOf(candidate);
        return {
          label: geomeanLabel(geomean.n),
          cell: geomeanVerdictCell(geomean, measuredOutcomes(metrics.filter((row) => row.gating))),
        };
      },
    },
    (section) => sectionAnnotation(section, result.configKinds),
  );

  /**
   * Every aggregate cell the body holds, in the fields they will print in.
   *
   * An aggregate states the band its figure was judged against even when every
   * metric behind it was too short to state one of its own, so the column has to
   * be measured on the aggregates as well as the rows: sized on the rows alone,
   * the band would print past the rule those same widths draw.
   */
  const aggregateParts = body.flatMap((line) =>
    line.type === "aggregate" ? [line.cell.parts] : [],
  );
  const verdictFields = verdictWidths([
    ...layout.ordered.map((row) => row.parts).filter((parts) => parts !== undefined),
    ...aggregateParts,
  ]);

  const toMetricRow = (row: MeasuredRow): MetricRow => ({
    cells: [
      sectioned ? row.label : row.name,
      joinValueCell(row.baseline, baselineFields),
      joinValueCell(row.candidate, candidateFields),
      row.parts === undefined ? "" : joinVerdictCell(row.parts, verdictFields),
    ],
    verdict: row.verdict,
    band: row.parts === undefined ? "" : bandField(row.parts.band, verdictFields.band),
  });

  const rows = layout.ordered.map(toMetricRow);
  const widths: Widths = [
    computeMetricColumnWidth(widestHeaderLabel(body), [
      ...rows.map((row) => row.cells[0].length),
      ...aggregateLabelLengths(body),
    ]),
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
      [
        ...rows.map((row) => row.cells[3].length),
        ...aggregateParts.map((parts) => joinVerdictCell(parts, verdictFields).length),
      ],
      VERDICT_COLUMN_MIN,
    ),
  ];

  const rule = formatTableRule(widths);
  const border = formatTableBorder(widths);
  const styleVariantCells: CellStyler = (cell, index) => {
    if (index === 1 || index === 2) {
      return styleWithin(cell, headers[index], VARIANT_NAME_STYLE);
    }
    if (index === VERDICT_COLUMN) {
      // The marker is the bare baseline name, not the whole `vs …` header: the
      // emphasis belongs to the name, and `vs` is prose around it. The name
      // trails that prose, so a baseline of `v`, `s` or `vs` needs the last
      // occurrence to land on the name instead of the prefix.
      return styleWithin(cell, baseline, VARIANT_NAME_STYLE, { last: true });
    }
    return cell;
  };

  const renderers: BodyLineRenderers<MeasuredRow, AggregateCell> = {
    header: (label, hasTitle) => {
      const headerRow: Row = [label, headers[1], headers[2], headers[3]];
      return formatRow(headerRow, widths, styleHeaderCell(label, hasTitle, styleVariantCells));
    },
    group: (label) =>
      formatRow([label, "", "", ""], widths, styleLabelCell(label, GROUP_LABEL_STYLE)),
    metric: (row) => formatMetricRow(toMetricRow(row), widths),
    aggregate: ({ label, cell }) =>
      formatRow(
        [label, "", "", joinVerdictCell(cell.parts, verdictFields)],
        widths,
        styleLabelCell(label, AGGREGATE_LABEL_STYLE, (styledCell, index) =>
          index === VERDICT_COLUMN
            ? styleSpans(styledCell, aggregateSpans(cell, verdictFields))
            : styledCell,
        ),
      ),
  };

  return body.map((bodyLine) => renderBodyLine(bodyLine, rule, border, renderers));
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
interface ComparisonRow extends MetricRowBase {
  readonly baseline: MetricCellParts;
  /** Filled during layout, like {@link CandidateCell.text}: the baseline figure, padded. */
  baselineCell: string;
  readonly candidates: readonly CandidateCell[];
}

/** One candidate's column of cells, in row order. */
interface CandidateColumn {
  readonly header: string;
  readonly cells: CandidateCell[];
}

/** The multi-candidate table's contents, held both ways round. */
interface ComparisonGrid {
  readonly layout: SectionLayout<ComparisonRow>;
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
    cells: [],
  }));

  const layout = planSections(result.metrics, (name, group, metric): ComparisonRow => {
    const candidates: CandidateCell[] = [];
    for (const [index, column] of columns.entries()) {
      const side = metric.candidates[index];
      const cell: CandidateCell = {
        value: candidateCellParts(side, metric.meta.unit),
        verdict:
          side?.verdict === undefined
            ? undefined
            : verdictParts(side.verdict, result.samples, false),
        delta: side?.verdict ? formatVerdictDelta(side.verdict) : "",
        outcome: shownClass(side?.verdict),
        text: "",
      };
      candidates.push(cell);
      column.cells.push(cell);
    }
    return {
      name,
      label: indentedSectionLabel(metric.meta.shortName, group),
      baseline: baselineCellParts(metric),
      baselineCell: "",
      candidates,
      gating: metric.meta.gating,
    };
  });

  return { layout, columns };
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
 * ends the line there, and letting the count behind each figure size the column
 * instead would stretch the rule under every metric row for one cell's sake.
 * The single-candidate table states that count in its label rather than its
 * cell, which is what leaves its aggregate narrow enough to measure.
 */
function renderComparisonTable(result: ComparisonResult): string[] {
  const baselineHeader = result.baselineLabel;
  const { layout, columns } = buildComparisonGrid(result);
  const sectioned = layout.sections.length > 1;

  for (const column of columns) {
    const values = valueWidths(column.cells.map((cell) => cell.value));
    const verdicts = verdictWidths(
      column.cells.map((cell) => cell.verdict).filter((parts) => parts !== undefined),
    );
    for (const cell of column.cells) cell.text = joinCandidateCell(cell, values, verdicts);
  }

  const baselineFields = valueWidths(layout.ordered.map((row) => row.baseline));
  for (const row of layout.ordered) row.baselineCell = joinValueCell(row.baseline, baselineFields);

  /**
   * One aggregate row's cells: the same geomean read off each candidate in turn,
   * each beside the verdicts that candidate's own column reported on `metrics`.
   */
  const columnCells = (
    geomeanOf: (candidate: CandidateComparison) => GeomeanResult,
    metrics: readonly ComparisonRow[],
  ): AggregateColumnCell[] =>
    result.candidates.map((candidate, index) =>
      geomeanColumnCell(
        geomeanOf(candidate),
        metrics.map((row) => row.candidates[index]?.outcome),
      ),
    );

  const body = planBody<ComparisonRow, readonly AggregateColumnCell[]>(
    layout,
    {
      group: (kind, group, metrics) => ({
        label: geomeanScopeLabel(group),
        cell: columnCells((candidate) => groupGeomeanOf(candidate, kind, group), metrics),
      }),
      kind: (kind, metrics) => ({
        label: geomeanScopeLabel(kind),
        cell: columnCells((candidate) => kindGeomeanOf(candidate, kind), metrics),
      }),
      flat: (metrics) => ({
        label: GEOMEAN_LABEL,
        cell: columnCells(
          flatGeomeanOf,
          metrics.filter((row) => row.gating),
        ),
      }),
    },
    (section) => sectionAnnotation(section, result.configKinds),
  );

  const aggregateCells = body.flatMap((line) => (line.type === "aggregate" ? [line.cell] : []));
  const metricWidth = computeMetricColumnWidth(widestHeaderLabel(body), [
    ...layout.ordered.map((row) => (sectioned ? row.label.length : row.name.length)),
    ...aggregateLabelLengths(body),
  ]);
  const baselineWidth = computeColumnWidth(
    baselineHeader.length,
    layout.ordered.map((row) => row.baselineCell.length),
    VALUE_COLUMN_MIN,
  );
  const lastColumn = columns.length - 1;
  const candidateWidths = columns.map((column, index) => {
    const contents = column.cells.map((cell) => cell.text.length);
    if (index !== lastColumn) {
      contents.push(
        ...aggregateCells.flatMap((cells) => {
          const cell = cells[index];
          return cell === undefined ? [] : [cell.text.length];
        }),
      );
    }
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

  const rule = formatTableRule(widths);
  const border = formatTableBorder(widths);
  const headers = columns.map((column) => column.header);
  /** One blank cell per candidate column — what a group row leaves for its candidates. */
  const blankCandidateCells = columns.map(() => "");
  const styleVariantCells: CellStyler = (cell, columnIndex) => {
    if (columnIndex === 0) return cell;
    if (columnIndex === 1) return styleWithin(cell, baselineHeader, VARIANT_NAME_STYLE);
    const column = columns[columnIndex - CANDIDATE_COLUMN_OFFSET];
    return column === undefined ? cell : styleWithin(cell, column.header, VARIANT_NAME_STYLE);
  };

  const renderers: BodyLineRenderers<ComparisonRow, readonly AggregateColumnCell[]> = {
    header: (label, hasTitle) =>
      line(label, baselineHeader, headers, styleHeaderCell(label, hasTitle, styleVariantCells)),
    group: (label) =>
      line(label, "", blankCandidateCells, styleLabelCell(label, GROUP_LABEL_STYLE)),
    metric: (row) =>
      line(
        sectioned ? row.label : row.name,
        row.baselineCell,
        row.candidates.map((cell) => cell.text),
        (cell, columnIndex) => {
          const candidateCell = row.candidates[columnIndex - CANDIDATE_COLUMN_OFFSET];
          const outcome = candidateCell?.outcome;
          if (candidateCell === undefined || outcome === undefined) return cell;
          return styleGlyphAndDelta(cell, outcome, candidateCell.delta, VERDICT_STYLES[outcome]);
        },
      ),
    aggregate: ({ label, cell }) =>
      line(
        label,
        "",
        cell.map((columnCell) => columnCell.text),
        styleLabelCell(label, AGGREGATE_LABEL_STYLE, (styledCell, columnIndex) => {
          const aggregate = cell[columnIndex - CANDIDATE_COLUMN_OFFSET];
          return aggregate === undefined ? styledCell : styleSpans(styledCell, aggregate.spans);
        }),
      ),
  };

  return body.map((bodyLine) => renderBodyLine(bodyLine, rule, border, renderers));
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

  const qualify = spansManyKinds(metrics);
  const labeled = highlights.map((highlight) => ({
    highlight,
    label: highlightLabel(highlight, qualify),
  }));
  const nameWidth = Math.max(...labeled.map(({ label }) => label.length)) + HIGHLIGHT_NAME_GUTTER;

  const entries = labeled.map(({ highlight: { metric, candidate }, label }) => {
    const verdict = candidate.verdict;
    const shown = displayClass(verdict);
    const delta = formatVerdictDelta(verdict);
    const evidence = formatEvidence(verdict, metric.meta.unit, metric.baselineMedian);
    const suffix = evidence === "" ? "" : `  ${evidence}`;

    const plain = `  ${getGlyph(shown)} ${label.padEnd(nameWidth)}${delta.padStart(
      HIGHLIGHT_DELTA_WIDTH,
    )}${suffix}`;

    const style = VERDICT_STYLES[shown];
    let styled = styleGlyphAndDelta(plain, shown, delta, style);
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

/** The glyph flagging a gate the run's own `--fail-on` conditions would trip. */
const GATE_TRIP_GLYPH = "⚑";

/**
 * One line per gating kind whose geomean is at or past a `--fail-on geomean`
 * threshold, naming the kind, its figure and the condition it answered to.
 *
 * A regressed condition earns no line: the metrics that would trip it already
 * carry their own `✗` entries above, and repeating them here would say the same
 * thing twice.
 *
 * The check is display-only, but it reads exactly what the gate that sets the
 * exit code reads: the gated geomean of a kind that gates at all. An
 * informational kind never trips the gate, so flagging one here would announce
 * a failure the run does not have.
 */
function gateTripLines(
  candidate: CandidateComparison,
  conditions: readonly FailOnCondition[],
): string[] {
  const thresholds = conditions.flatMap((condition) =>
    condition.kind === "geomean" ? [condition.pct] : [],
  );

  return candidate.kinds.flatMap((kind) => {
    if (!kind.hasGating) return [];

    const geomean = kind.gatedGeomean;
    if (geomean === undefined || geomean.n === 0) return [];

    const delta = formatDelta(geomean.value);
    const style = VERDICT_STYLES.regressed;
    return thresholds
      .filter((pct) => geomean.value >= pct)
      .map((pct) => {
        const plain = `  ${GATE_TRIP_GLYPH} ${kind.kind} ${GEOMEAN_LABEL} ${delta} exceeded --fail-on geomean:${pct}`;
        return styleSpans(plain, [
          { text: GATE_TRIP_GLYPH, style },
          { text: delta, style },
        ]);
      });
  });
}

/**
 * Renders the highlights section from one or more assembled blocks.
 *
 * A block whose metrics all sat still contributes no subsection: an empty
 * one under its label (or under the bare heading) reads as a rendering fault
 * rather than as good news. The whole section disappears once every block is
 * empty.
 *
 * A labeled block gets its own indented subsection heading, with its entries
 * indented one level further; a label-less block's entries sit directly under
 * the section heading.
 *
 * The futility note closes the whole section rather than each subsection —
 * it says the same thing about every unstable metric on the page.
 */
function highlightSection(blocks: readonly HighlightBlock[]): string[] {
  const nonEmpty = blocks.filter((block) => block.entries.length > 0);
  if (nonEmpty.length === 0) return [];

  const lines = [formatLabel(HIGHLIGHTS_HEADING, ["bold"])];
  for (const block of nonEmpty) {
    if (block.label === undefined) {
      lines.push(...block.entries);
    } else {
      lines.push(
        `  ${formatLabel(block.label, ["bold"])}`,
        ...block.entries.map((entry) => `  ${entry}`),
      );
    }
  }
  if (nonEmpty.some((block) => block.unstable)) lines.push(futilityLine());
  return lines;
}

function renderHighlights(
  metrics: MetricComparisons,
  candidateIndex: number,
  gateTrips: readonly string[],
): string[] {
  const { entries, unstable } = highlightEntries(metrics, candidateIndex);
  return highlightSection([{ entries: [...entries, ...gateTrips], unstable }]);
}

/** The highlights of every candidate, one subsection apiece under a single heading. */
function renderCandidateHighlights(
  result: ComparisonResult,
  conditions: readonly FailOnCondition[],
): string[] {
  const blocks = result.candidates.map((candidate, index) => {
    const { entries, unstable } = highlightEntries(result.metrics, index);
    return {
      label: candidate.label,
      entries: [...entries, ...gateTripLines(candidate, conditions)],
      unstable,
    };
  });
  return highlightSection(blocks);
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
function renderComparison(
  result: ComparisonResult,
  conditions: readonly FailOnCondition[],
): string[] {
  const lines = [...renderComparisonTable(result), "", ...renderSummaries(result)];

  const highlights = renderCandidateHighlights(result, conditions);
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
function renderWorktreeFooter(result: WorktreeCleanupOutcome): string[] {
  const details = formatCleanupFailures(result.worktreesLeftBehind, result.worktreePruneError);
  if (result.worktreesRemoved === 0 && details.length === 0) return [];

  return [
    `${pluralize(result.worktreesRemoved, "worktree")} removed · ${result.worktreesLeftBehind.length} left behind`,
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
  conditions: readonly FailOnCondition[],
): string[] {
  const lines = [
    ...renderTable(result, candidate, candidateIndex),
    "",
    renderSummary(result.metrics, candidateIndex),
  ];

  const highlights = renderHighlights(
    result.metrics,
    candidateIndex,
    gateTripLines(candidate, conditions),
  );
  if (highlights.length > 0) {
    lines.push("", ...highlights);
  }

  return lines;
}

/** How many rounds paired, as the header states it. */
function pairedSamples(samples: number): string {
  return pluralize(samples, "paired sample");
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
    // Joined from its parts rather than rewritten once styled: a `·` is legal in
    // a branch name and in an adapter name, and replacing every one of them in
    // the finished line would splice dim codes into the middle of a name's own
    // style span.
    const header = joinHeaderParts([
      formatLabel("gymrat compare", ["bold"]),
      `baseline ${formatVariantName(display.baselineLabel)} ↔ ${candidateNames}`,
      pairedSamples(display.samples),
      `adapter: ${display.adapter}`,
    ]);
    const lines = [header];

    const conditions = options.failOn ?? [];
    if (display.candidates.length > 1) {
      lines.push(...renderComparison(display, conditions));
    } else {
      for (const [index, candidate] of display.candidates.entries()) {
        lines.push(...renderCandidate(display, candidate, index, conditions));
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

/** The metric name and what the target measured for it, in column order. */
type MeasureRow = readonly [metric: string, value: string];

/** Column widths, one per {@link MeasureRow} cell. */
type MeasureWidths = readonly [number, number];

/** One metric of the measure table, measured but not yet padded to its column. */
interface MeasureMetricRow extends NamedRow {
  readonly value: MetricCellParts;
}

/** A measure row's two cells, the value right-aligned under the target's name. */
function formatMeasureRow(row: MeasureRow, widths: MeasureWidths, styleCell?: CellStyler): string {
  return formatTableLine([row[0], alignRight(row[1], widths[1])], widths, styleCell);
}

/**
 * The measure table: one column of metric names, one of what the target
 * measured for each.
 *
 * Sectioning, group blocks and column sizing are the comparison table's, so a
 * reader who knows one table can read the other. What is missing is everything
 * a second target pays for — the delta, the verdict, the geomean closing a
 * scope — which is why the body is planned with no aggregate rows at all rather
 * than with empty ones.
 */
function renderMeasureTable(result: MeasurementResult, label: string): string[] {
  const layout = planSections(
    result.metrics,
    (name, group, metric): MeasureMetricRow => ({
      name,
      label: indentedSectionLabel(metric.meta.shortName, group),
      value: formatMetricCellParts(metric.median, metric.spread, metric.meta.unit),
    }),
  );
  const sectioned = layout.sections.length > 1;

  const valueFields = valueWidths(layout.ordered.map((row) => row.value));
  const body = planBody<MeasureMetricRow, never>(layout, undefined, (section) =>
    sectionAnnotation(section, result.configKinds),
  );

  const toMeasureRow = (row: MeasureMetricRow): MeasureRow => [
    sectioned ? row.label : row.name,
    joinValueCell(row.value, valueFields),
  ];

  const rows = layout.ordered.map(toMeasureRow);
  const widths: MeasureWidths = [
    computeMetricColumnWidth(widestHeaderLabel(body), [
      ...rows.map((row) => row[0].length),
      ...aggregateLabelLengths(body),
    ]),
    computeColumnWidth(
      label.length,
      rows.map((row) => row[1].length),
      VALUE_COLUMN_MIN,
    ),
  ];

  const rule = formatTableRule(widths);
  const border = formatTableBorder(widths);
  const styleTargetCell: CellStyler = (cell, index) =>
    index === 1 ? styleWithin(cell, label, VARIANT_NAME_STYLE) : cell;

  const renderers: BodyLineRenderers<MeasureMetricRow, never> = {
    header: (headerLabel, hasTitle) =>
      formatMeasureRow(
        [headerLabel, label],
        widths,
        styleHeaderCell(headerLabel, hasTitle, styleTargetCell),
      ),
    group: (groupLabel) =>
      formatMeasureRow([groupLabel, ""], widths, styleLabelCell(groupLabel, GROUP_LABEL_STYLE)),
    // Nothing here is judged, so nothing here is painted: the figure is the
    // whole row, and a color on it would claim a reading the run never made.
    metric: (row) => formatMeasureRow(toMeasureRow(row), widths),
    // `planBody` was handed no aggregate builders, so it planned no aggregate
    // lines and this cell type is uninhabited.
    aggregate: ({ cell }) => assertNever(cell),
  };

  return body.map((bodyLine) => renderBodyLine(bodyLine, rule, border, renderers));
}

/** How many rounds the target ran, as the header states it. */
function measuredSamples(samples: number): string {
  return pluralize(samples, "sample");
}

/**
 * Render a single-target measurement as the plain-text report the CLI prints.
 *
 * Laid out like {@link renderReport} — run header, body, then whatever speaks
 * for the run as a whole — so the two commands read as one tool. The report
 * states what the target measured and stops: with nothing to judge it against
 * there is no delta, no verdict and no geomean to close a section on, and the
 * highlights and method footer that explain those have nothing to explain.
 *
 * `options.verbose` and `options.failOn` are accepted and ignored, both being
 * about verdicts. Whether the report is styled follows `options.color` exactly
 * as it does for a comparison.
 */
export function renderMeasureReport(
  result: MeasurementResult,
  options: ReportOptions = {},
): string {
  return withColor(options.color, () => {
    const label = truncateLabels([result.label])[0] ?? result.label;
    const header = joinHeaderParts([
      formatLabel("gymrat measure", ["bold"]),
      formatVariantName(label),
      measuredSamples(result.samples),
      `adapter: ${result.adapter}`,
    ]);

    const lines = [header, ...renderMeasureTable(result, label)];

    const footer = renderWorktreeFooter(result);
    if (footer.length > 0) {
      lines.push("", ...footer);
    }

    return lines.join("\n");
  });
}
