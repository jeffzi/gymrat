import type { ConfigKinds } from "../config.js";
import { assertNever } from "../errors.js";
import type { GeomeanResult, MetricVerdict } from "../verdict/verdict.js";
import {
  type CellStyler,
  computeColumnWidth,
  displayClass,
  type DisplayClass,
  formatDelta,
  formatLabel,
  formatNoiseBandValue,
  formatPairCount,
  formatTableLine,
  geomeanParts,
  geomeanValueStyle,
  getGlyph,
  type MetricCellParts,
  NO_GEOMEAN_CELL,
  NO_GEOMEAN_FIGURE,
  NO_STABLE_METRICS,
  PLUS_MINUS,
  SPREAD_SEPARATOR,
  type Style,
  styleWithin,
  type StyleWithinOptions,
  VARIANT_NAME_STYLE,
  VERDICT_GLOSSES,
} from "./format.js";
import {
  informationalTag,
  sectionLabel,
  type SectionLayout,
  type SectionPlan,
} from "./sections.js";

// ---------------------------------------------------------------------------
// Shared types
// ---------------------------------------------------------------------------

/** The metric name, the two measured values, and the verdict, in column order. */
export type Row = readonly [metric: string, baseline: string, candidate: string, verdict: string];

/** Column widths, one per {@link Row} cell. */
export type Widths = readonly [number, number, number, number];

/** The two names every metric row carries, whatever the table reports beside them. */
export interface NamedRow {
  /** The full metric name, which the flat table shows and every other output keys on. */
  readonly name: string;
  /** The short name a section shows the row under, group prefix stripped and indented. */
  readonly label: string;
}

/** Fields shared by every row that names a metric in either comparison layout. */
export interface MetricRowBase extends NamedRow {
  readonly gating: boolean;
}

/** Widths a value column pads its two fields to, measured on plain text. */
export interface ValueWidths {
  readonly magnitude: number;
  readonly spread: number;
}

/**
 * One verdict's fields, with the noise band only where the caller shows one.
 *
 * The multi-candidate table drops the band from its cells, so it asks for the
 * same fields with that one left empty.
 */
export interface VerdictParts {
  readonly glyph: string;
  /** The signed percentage, right-aligned among the column's other deltas. */
  readonly delta: string;
  /** The word standing in for a delta too noisy to report, empty for every other verdict. */
  readonly word: string;
  /** The noise band's figure, without the `±` the column pins in front of it. */
  readonly band: string;
  readonly pairs: string;
}

/** Widths a verdict column pads its delta and noise band to, measured on plain text. */
export interface VerdictWidths {
  readonly delta: number;
  readonly band: number;
}

/** A span of an already-padded cell, and the style it carries. */
export interface StyledSpan {
  readonly text: string;
  readonly style: Style;
}

/** One candidate column's aggregate cell in the multi-candidate table. */
export interface AggregateColumnCell {
  readonly text: string;
  readonly spans: readonly StyledSpan[];
}

/** An aggregate row: what it covers, and what it states for each candidate column. */
export interface AggregateRow<Cell> {
  readonly label: string;
  readonly cell: Cell;
}

/** A line of a table body, held until the column widths that render it are known. */
export type BodyLine<Metric, Cell> =
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
 */
export interface AggregateRows<Metric, Cell> {
  group(kind: string, group: string, metrics: readonly Metric[]): AggregateRow<Cell>;
  kind(kind: string, metrics: readonly Metric[]): AggregateRow<Cell>;
  flat(metrics: readonly Metric[]): AggregateRow<Cell>;
}

/** The row-assembly callbacks a table supplies for the lines whose shape is table-specific. */
export interface BodyLineRenderers<Metric, Cell> {
  header(label: string, hasTitle: boolean): string;
  group(label: string): string;
  metric(row: Metric): string;
  aggregate(row: AggregateRow<Cell>): string;
}

/** Styling inputs for the glyph-and-delta span rendered by {@link styleGlyphAndDelta}. */
export interface GlyphDeltaOptions {
  shown: DisplayClass;
  delta: string;
  style: Style;
  deltaSearch?: StyleWithinOptions;
}

// ---------------------------------------------------------------------------
// Shared constants
// ---------------------------------------------------------------------------

/** The metric-name column's header text, and its fallback when a section carries no title. */
export const METRIC_COLUMN_HEADER = "metric";

/** Minimum column widths in characters, enforced regardless of content length. */
export const METRIC_COLUMN_MIN = 16;
export const VALUE_COLUMN_MIN = 12;
export const VERDICT_COLUMN_MIN = 12;

/** Index of the label cell — the first column of both tables, whatever the row states. */
export const LABEL_COLUMN = 0;

/** Index of the verdict cell in a {@link Row} — the only cell styled from within. */
export const VERDICT_COLUMN = 3;

/**
 * Index of the first candidate column in the multi-candidate table, past the
 * metric name and the baseline's own figures.
 */
export const CANDIDATE_COLUMN_OFFSET = 2;

/**
 * Gap between the fields of one cell: a candidate's figure and the verdict
 * behind it, and the verdict's own glyph, delta and noise band.
 */
export const CELL_GUTTER = "  ";

/** Indent a metric row carries under the group sub-header above it. */
export const GROUP_INDENT = "  ";

/**
 * The style a group sub-header wears.
 *
 * The row names the block beneath it rather than reporting a measurement, and
 * no verdict color is free for that job — every one of them already means an
 * outcome. Blue is the one hue the verdicts leave unclaimed, so a scan down the
 * name column reads the structure without mistaking it for a result.
 */
export const GROUP_LABEL_STYLE: Style = ["blue"];

/**
 * The style every geomean label wears, at whatever scope it closes.
 *
 * The figure beside it is already emboldened, so emboldening the label keeps the
 * row reading as one statement and sets it apart from the metric rows it sums.
 */
export const AGGREGATE_LABEL_STYLE: Style = ["bold"];

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/**
 * Right-align a value in its column, one space clear of the next separator.
 *
 * Padding is measured on the plain string. `padStart` counts an ANSI escape as
 * width it does not occupy, so styling has to wrap a cell once it is padded,
 * never before.
 */
export function alignRight(content: string, width: number): string {
  return `${content.padStart(width - 1)} `;
}

/** Indent a left-aligned cell off the separator that precedes it. */
export function alignLeft(content: string): string {
  return ` ${content}`;
}

/** Widest a column of cells needs a text field to be, zero when the column holds no cells. */
export function maxLength<T>(cells: readonly T[], field: (cell: T) => string): number {
  return Math.max(0, ...cells.map((cell) => field(cell).length));
}

/** The widest magnitude and the widest spread a column of value cells holds. */
export function valueWidths(cells: readonly MetricCellParts[]): ValueWidths {
  return {
    magnitude: maxLength(cells, (cell) => cell.magnitude),
    spread: maxLength(cells, (cell) => cell.spread),
  };
}

/**
 * A value cell with its magnitude and its spread each right-aligned in a field
 * of the column's own width.
 */
export function joinValueCell(parts: MetricCellParts, widths: ValueWidths): string {
  const magnitude = parts.magnitude.padStart(widths.magnitude);
  if (widths.spread === 0) return magnitude;
  const spread =
    parts.spread === "" ? "" : `${SPREAD_SEPARATOR}${parts.spread.padStart(widths.spread)}`;
  return `${magnitude}${spread}`.padEnd(widths.magnitude + SPREAD_SEPARATOR.length + widths.spread);
}

export function verdictParts(
  verdict: MetricVerdict,
  samples: number,
  withBand: boolean,
): VerdictParts {
  const shown = displayClass(verdict);
  const unstable = verdict.verdict === "unstable";
  const banded = withBand && !unstable && shown !== "inconclusive" && "noisePct" in verdict;
  return {
    glyph: getGlyph(shown),
    delta: unstable ? "" : formatDelta(verdict.delta),
    word: unstable ? VERDICT_GLOSSES.unstable : "",
    band: banded ? formatNoiseBandValue(verdict.noisePct) : "",
    pairs: verdict.n === samples ? "" : formatPairCount(verdict.n),
  };
}

/**
 * The widest delta and noise band a column of verdict cells holds.
 *
 * The word standing in for a delta is not measured: it is wider than any
 * percentage, and sizing the field from it would push a whole column of bands
 * right for the sake of the one row that has none.
 */
export function verdictWidths(cells: readonly VerdictParts[]): VerdictWidths {
  return {
    delta: maxLength(cells, (cell) => cell.delta),
    band: maxLength(cells, (cell) => cell.band),
  };
}

/** The noise band as it prints: the `±` pinned, its figure right-aligned behind it. */
export function bandField(band: string, width: number): string {
  return band === "" ? "" : `${PLUS_MINUS}${band.padStart(width)}`;
}

/**
 * A verdict cell, each field padded to the width its column settled on.
 */
export function joinVerdictCell(parts: VerdictParts, widths: VerdictWidths): string {
  const delta = parts.word === "" ? parts.delta.padStart(widths.delta) : parts.word;
  const band = bandField(parts.band, widths.band);
  const bandCell =
    band === "" && widths.band > 0 ? " ".repeat(PLUS_MINUS.length + widths.band) : band;
  return [parts.glyph, delta, bandCell, parts.pairs]
    .filter((field) => field !== "")
    .join(CELL_GUTTER)
    .trimEnd();
}

/**
 * Style each span where it sits inside a finished cell.
 */
export function styleSpans(cell: string, spans: readonly StyledSpan[]): string {
  let styled = cell;
  for (const span of spans) {
    styled = styleWithin(styled, span.text, span.style);
  }
  return styled;
}

/** The dash and the phrase standing in for an aggregate that has nothing behind it. */
export function emptyGeomeanSpans(): StyledSpan[] {
  return [
    { text: NO_GEOMEAN_FIGURE, style: ["bold"] },
    { text: NO_STABLE_METRICS, style: ["dim"] },
  ];
}

/**
 * The geomean of one candidate column: the aggregate, then how many metrics
 * stand behind it.
 */
export function geomeanColumnCell(
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

export function styleGlyphAndDelta(cell: string, options: GlyphDeltaOptions): string {
  const styled = styleWithin(cell, getGlyph(options.shown), options.style);
  return options.delta === ""
    ? styled
    : styleWithin(styled, options.delta, options.style, options.deltaSearch);
}

/** Width the metric-name column needs: the widest label any of its rows carries. */
export function computeMetricColumnWidth(headerLabel: string, labelLengths: number[]): number {
  return computeColumnWidth(headerLabel.length, labelLengths, METRIC_COLUMN_MIN);
}

/** A metric's name cell inside a section: its short name, indented under its group. */
export function indentedSectionLabel(shortName: string, group: string | undefined): string {
  const label = sectionLabel(shortName, group);
  return group === undefined ? label : `${GROUP_INDENT}${label}`;
}

/**
 * A section's title: the kind, and — where none of its metrics gates — the tag
 * saying so and the config key that decided it.
 */
export function sectionAnnotation<Metric>(
  section: SectionPlan<Metric>,
  configKinds: ConfigKinds | undefined,
): string | undefined {
  if (section.hasGating) return undefined;
  return formatLabel(informationalTag(section.kind, configKinds), ["dim"]);
}

/** Every metric row a section holds, its group blocks flattened back into row order. */
export function sectionMetrics<Metric>(section: SectionPlan<Metric>): Metric[] {
  return section.blocks.flatMap((block) =>
    block.type === "group" ? block.metrics : [block.metric],
  );
}

/** The lines one section's blocks produce: groups, standalone metrics, and sub-geomeans. */
export function planBlocks<Metric, Cell>(
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
 */
export function planBody<Metric, Cell>(
  layout: SectionLayout<Metric>,
  rows: AggregateRows<Metric, Cell> | undefined,
  annotation: (section: SectionPlan<Metric>) => string | undefined,
): BodyLine<Metric, Cell>[] {
  if (layout.sections.length <= 1) {
    const body: BodyLine<Metric, Cell>[] = [];
    const section = layout.sections[0];
    if (section !== undefined) {
      const tag = annotation(section);
      if (tag !== undefined) body.push({ type: "title", text: tag });
    }
    body.push(
      { type: "header" },
      { type: "rule" },
      ...layout.ordered.map((row) => ({ type: "metric" as const, row })),
    );
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
export function aggregateLabelLengths<Metric, Cell>(
  body: readonly BodyLine<Metric, Cell>[],
): number[] {
  return body.flatMap((line) =>
    line.type === "group" || line.type === "aggregate" ? [line.label.length] : [],
  );
}

export function widestHeaderLabel<Metric, Cell>(body: readonly BodyLine<Metric, Cell>[]): string {
  const labels = body.flatMap((line) =>
    line.type === "header" ? [line.title ?? METRIC_COLUMN_HEADER] : [],
  );
  return labels.reduce(
    (widest, label) => (label.length > widest.length ? label : widest),
    METRIC_COLUMN_HEADER,
  );
}

/** One dash span per column width, joined by `separator`. */
export function formatHorizontalLine(widths: readonly number[], separator: string): string {
  return widths.map((width) => "─".repeat(width)).join(separator);
}

/** The rule row separating a table's header (or body) from what follows. */
export function formatTableRule(widths: readonly number[]): string {
  return formatHorizontalLine(widths, "┼");
}

/**
 * The line opening a section — the top of its box, meeting each column at a
 * top-T junction so the border joins the separators of the header below it.
 */
export function formatTableBorder(widths: readonly number[]): string {
  return formatHorizontalLine(widths, "┬");
}

export function formatRow(row: Row, widths: Widths, styleCell?: CellStyler): string {
  return formatTableLine(
    [row[0], alignRight(row[1], widths[1]), alignRight(row[2], widths[2]), alignLeft(row[3])],
    widths,
    styleCell,
  );
}

/** Style the verdict cell of a row, leaving every other cell as it was padded. */
export function styleVerdictCell(style: (cell: string) => string): CellStyler {
  return (cell, index) => (index === VERDICT_COLUMN ? style(cell) : cell);
}

/**
 * Style a row's own label where it sits in the padded label cell, falling
 * through to `rest` for every other cell.
 */
export function styleLabelCell(
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
 */
export function styleHeaderCell(
  label: string,
  hasTitle: boolean,
  styleVariantCells: CellStyler,
): CellStyler {
  return hasTitle ? styleLabelCell(label, ["bold"], styleVariantCells) : styleVariantCells;
}

/**
 * One line of a table body rendered to text.
 */
// fallow-ignore-next-line complexity
export function renderBodyLine<Metric, Cell>(
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

export { formatLabel, computeColumnWidth, formatTableLine, VARIANT_NAME_STYLE };
export type { DisplayClass };
