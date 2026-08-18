import type { GeomeanResult } from "../verdict/verdict.js";
import {
  candidateCellParts,
  baselineCellParts,
  type CellStyler,
  formatVerdictDelta,
  geomeanScopeLabel,
  GEOMEAN_LABEL,
  type MetricCellParts,
  shownClass,
  styleWithin,
  VERDICT_STYLES,
} from "./format.js";
import {
  flatGeomeanOf,
  groupGeomeanOf,
  kindGeomeanOf,
  planSections,
  type SectionLayout,
} from "./sections.js";
import {
  AGGREGATE_LABEL_STYLE,
  type AggregateColumnCell,
  aggregateLabelLengths,
  alignLeft,
  alignRight,
  type BodyLine,
  type BodyLineRenderers,
  CANDIDATE_COLUMN_OFFSET,
  computeColumnWidth,
  computeMetricColumnWidth,
  type DisplayClass,
  formatTableBorder,
  formatTableLine,
  formatTableRule,
  geomeanColumnCell,
  GROUP_LABEL_STYLE,
  indentedSectionLabel,
  joinValueCell,
  joinVerdictCell,
  type MetricRowBase,
  planBody,
  renderBodyLine,
  sectionAnnotation,
  styleGlyphAndDelta,
  styleHeaderCell,
  styleLabelCell,
  styleSpans,
  VALUE_COLUMN_MIN,
  VARIANT_NAME_STYLE,
  valueWidths,
  type ValueWidths,
  verdictParts,
  type VerdictParts,
  verdictWidths,
  type VerdictWidths,
  widestHeaderLabel,
} from "./text-table-core.js";
import type { CandidateComparison, ComparisonResult } from "./types.js";

// ---------------------------------------------------------------------------
// Multi-candidate-only types
// ---------------------------------------------------------------------------

interface CandidateCell {
  readonly value: MetricCellParts;
  readonly verdict: VerdictParts | undefined;
  readonly delta: string;
  readonly outcome: DisplayClass | undefined;
  text: string;
}

interface ComparisonRow extends MetricRowBase {
  readonly baseline: MetricCellParts;
  baselineCell: string;
  readonly candidates: readonly CandidateCell[];
}

interface CandidateColumn {
  readonly header: string;
  readonly cells: CandidateCell[];
}

interface ComparisonGrid {
  readonly layout: SectionLayout<ComparisonRow>;
  readonly columns: readonly CandidateColumn[];
}

// ---------------------------------------------------------------------------
// Multi-candidate helpers
// ---------------------------------------------------------------------------

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

function joinCandidateCell(
  cell: CandidateCell,
  values: ValueWidths,
  verdicts: VerdictWidths,
): string {
  const value = joinValueCell(cell.value, values);
  if (cell.verdict === undefined) return value;
  return `${value}  ${joinVerdictCell(cell.verdict, verdicts)}`;
}

function buildComparisonBody(
  result: ComparisonResult,
  layout: SectionLayout<ComparisonRow>,
): BodyLine<ComparisonRow, readonly AggregateColumnCell[]>[] {
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

  return planBody<ComparisonRow, readonly AggregateColumnCell[]>(
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
}

interface ComparisonWidthContext {
  baselineHeader: string;
  layout: SectionLayout<ComparisonRow>;
  columns: readonly CandidateColumn[];
  body: BodyLine<ComparisonRow, readonly AggregateColumnCell[]>[];
  sectioned: boolean;
}

function computeComparisonWidths(ctx: ComparisonWidthContext): number[] {
  const { baselineHeader, layout, columns, body, sectioned } = ctx;
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
  return [metricWidth, baselineWidth, ...candidateWidths];
}

interface ComparisonRenderContext {
  baselineHeader: string;
  columns: readonly CandidateColumn[];
  widths: number[];
  baselineWidth: number;
  sectioned: boolean;
}

function buildComparisonRenderers(
  ctx: ComparisonRenderContext,
): BodyLineRenderers<ComparisonRow, readonly AggregateColumnCell[]> {
  const { baselineHeader, columns, widths, baselineWidth, sectioned } = ctx;
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

  const headers = columns.map((column) => column.header);
  const blankCandidateCells = columns.map(() => "");
  const styleVariantCells: CellStyler = (cell, columnIndex) => {
    if (columnIndex === 0) return cell;
    if (columnIndex === 1) return styleWithin(cell, baselineHeader, VARIANT_NAME_STYLE);
    const column = columns[columnIndex - CANDIDATE_COLUMN_OFFSET];
    return column === undefined ? cell : styleWithin(cell, column.header, VARIANT_NAME_STYLE);
  };

  return {
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
          return styleGlyphAndDelta(cell, {
            shown: outcome,
            delta: candidateCell.delta,
            style: VERDICT_STYLES[outcome],
          });
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
}

// ---------------------------------------------------------------------------
// renderComparisonTable
// ---------------------------------------------------------------------------

/** Render a multi-candidate comparison grid (one baseline vs. two or more candidates). */
export function renderComparisonTable(result: ComparisonResult): string[] {
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

  const body = buildComparisonBody(result, layout);
  const widths = computeComparisonWidths({ baselineHeader, layout, columns, body, sectioned });
  const baselineWidth = widths[1];
  if (baselineWidth === undefined) throw new Error("missing baseline width");

  const rule = formatTableRule(widths);
  const border = formatTableBorder(widths);
  const renderers = buildComparisonRenderers({
    baselineHeader,
    columns,
    widths,
    baselineWidth,
    sectioned,
  });

  return body.map((bodyLine) => renderBodyLine(bodyLine, rule, border, renderers));
}
