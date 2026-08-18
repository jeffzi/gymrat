import type { GeomeanResult, MetricVerdict } from "../verdict/verdict.js";
import {
  baselineCellParts,
  candidateCellParts,
  type CellStyler,
  displayClass,
  type DisplayClass,
  formatVerdictDelta,
  geomeanLabel,
  geomeanParts,
  geomeanValueStyle,
  type MetricCellParts,
  NO_GEOMEAN_FIGURE,
  NO_STABLE_METRICS,
  QUIET_VERDICTS,
  scopedGeomeanLabel,
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
  aggregateLabelLengths,
  type AggregateRow,
  bandField,
  type BodyLine,
  type BodyLineRenderers,
  computeColumnWidth,
  computeMetricColumnWidth,
  formatRow,
  formatTableBorder,
  formatTableRule,
  GROUP_LABEL_STYLE,
  indentedSectionLabel,
  joinValueCell,
  joinVerdictCell,
  type MetricRowBase,
  planBody,
  renderBodyLine,
  type Row,
  sectionAnnotation,
  type StyledSpan,
  styleGlyphAndDelta,
  styleHeaderCell,
  styleLabelCell,
  styleSpans,
  styleVerdictCell,
  VALUE_COLUMN_MIN,
  VARIANT_NAME_STYLE,
  valueWidths,
  VERDICT_COLUMN,
  VERDICT_COLUMN_MIN,
  verdictParts,
  type VerdictParts,
  verdictWidths,
  type VerdictWidths,
  widestHeaderLabel,
  type Widths,
} from "./text-table-core.js";
import type { CandidateComparison, ComparisonResult } from "./types.js";

// ---------------------------------------------------------------------------
// Single-candidate-only types
// ---------------------------------------------------------------------------

interface MetricRow {
  readonly cells: Row;
  readonly verdict: MetricVerdict | undefined;
  readonly band: string;
}

interface MeasuredRow extends MetricRowBase {
  readonly baseline: MetricCellParts;
  readonly candidate: MetricCellParts;
  readonly verdict: MetricVerdict | undefined;
  readonly parts: VerdictParts | undefined;
}

const GEOMEAN_GLYPH_SLOT = " ";

interface AggregateCell {
  readonly parts: VerdictParts;
  readonly spans: readonly StyledSpan[];
}

// ---------------------------------------------------------------------------
// Single-candidate helpers
// ---------------------------------------------------------------------------

function geomeanVerdictCell(
  geomean: GeomeanResult,
  outcomes: ReadonlyArray<DisplayClass | undefined>,
): AggregateCell {
  const parts = geomeanParts(geomean);
  if (parts === null) {
    return {
      parts: { glyph: NO_GEOMEAN_FIGURE, delta: "", word: NO_STABLE_METRICS, band: "", pairs: "" },
      spans: [
        { text: NO_GEOMEAN_FIGURE, style: ["bold"] },
        { text: NO_STABLE_METRICS, style: ["dim"] },
      ],
    };
  }
  return {
    parts: { glyph: GEOMEAN_GLYPH_SLOT, delta: parts.delta, word: "", band: parts.band, pairs: "" },
    spans: [{ text: parts.delta, style: geomeanValueStyle(geomean, outcomes) }],
  };
}

function aggregateSpans(cell: AggregateCell, widths: VerdictWidths): readonly StyledSpan[] {
  const band = bandField(cell.parts.band, widths.band);
  return band === "" ? cell.spans : [...cell.spans, { text: band, style: ["dim"] }];
}

function measuredOutcomes(metrics: readonly MeasuredRow[]): (DisplayClass | undefined)[] {
  return metrics.map((row) => shownClass(row.verdict));
}

function formatMetricRow(row: MetricRow, widths: Widths): string {
  if (row.verdict === undefined) return formatRow(row.cells, widths);

  const outcome = displayClass(row.verdict);
  const delta = QUIET_VERDICTS.has(outcome) ? formatVerdictDelta(row.verdict) : "";
  return formatRow(
    row.cells,
    widths,
    styleVerdictCell((cell) => {
      const styled = styleGlyphAndDelta(cell, {
        shown: outcome,
        delta,
        style: VERDICT_STYLES[outcome],
      });
      return row.band === "" ? styled : styleWithin(styled, row.band, ["dim"]);
    }),
  );
}

// ---------------------------------------------------------------------------
// renderTable decomposition
// ---------------------------------------------------------------------------

function buildTableBody(
  result: ComparisonResult,
  candidate: CandidateComparison,
  candidateIndex: number,
): {
  layout: SectionLayout<MeasuredRow>;
  body: BodyLine<MeasuredRow, AggregateCell>[];
} {
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

  return { layout, body };
}

function computeTableWidths(
  headers: Row,
  body: BodyLine<MeasuredRow, AggregateCell>[],
  cellsByRow: Map<MeasuredRow, MetricRow>,
  verdictFields: VerdictWidths,
): Widths {
  const rows = [...cellsByRow.values()];
  const aggregateParts = body.flatMap((line) =>
    line.type === "aggregate" ? [line.cell.parts] : [],
  );
  const valueColumnWidth = (index: 1 | 2) =>
    computeColumnWidth(
      headers[index].length,
      rows.map((row) => row.cells[index].length),
      VALUE_COLUMN_MIN,
    );
  return [
    computeMetricColumnWidth(widestHeaderLabel(body), [
      ...rows.map((row) => row.cells[0].length),
      ...aggregateLabelLengths(body),
    ]),
    valueColumnWidth(1),
    valueColumnWidth(2),
    computeColumnWidth(
      headers[3].length,
      [
        ...rows.map((row) => row.cells[3].length),
        ...aggregateParts.map((parts) => joinVerdictCell(parts, verdictFields).length),
      ],
      VERDICT_COLUMN_MIN,
    ),
  ];
}

interface TableRenderContext {
  headers: Row;
  baseline: string;
  widths: Widths;
  cellsByRow: Map<MeasuredRow, MetricRow>;
  verdictFields: VerdictWidths;
}

function buildTableRenderers(
  ctx: TableRenderContext,
): BodyLineRenderers<MeasuredRow, AggregateCell> {
  const { headers, baseline, widths, cellsByRow, verdictFields } = ctx;
  const styleVariantCells: CellStyler = (cell, index) => {
    if (index === 1 || index === 2) {
      return styleWithin(cell, headers[index], VARIANT_NAME_STYLE);
    }
    if (index === VERDICT_COLUMN) {
      return styleWithin(cell, baseline, VARIANT_NAME_STYLE, { last: true });
    }
    return cell;
  };

  return {
    header: (label, hasTitle) => {
      const headerRow: Row = [label, headers[1], headers[2], headers[3]];
      return formatRow(headerRow, widths, styleHeaderCell(label, hasTitle, styleVariantCells));
    },
    group: (label) =>
      formatRow([label, "", "", ""], widths, styleLabelCell(label, GROUP_LABEL_STYLE)),
    metric: (row) => {
      const cells = cellsByRow.get(row);
      if (cells === undefined) throw new Error(`no cells for metric row ${row.name}`);
      return formatMetricRow(cells, widths);
    },
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
}

// ---------------------------------------------------------------------------
// renderTable
// ---------------------------------------------------------------------------

/** Render a two-revision comparison table (one baseline vs. one candidate). */
export function renderTable(
  result: ComparisonResult,
  candidate: CandidateComparison,
  candidateIndex: number,
): string[] {
  const baseline = result.baselineLabel;
  const headers: Row = ["metric", baseline, candidate.label, `vs ${baseline}`];
  const { layout, body } = buildTableBody(result, candidate, candidateIndex);
  const sectioned = layout.sections.length > 1;

  const baselineFields = valueWidths(layout.ordered.map((row) => row.baseline));
  const candidateFields = valueWidths(layout.ordered.map((row) => row.candidate));

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

  const cellsByRow = new Map(layout.ordered.map((row) => [row, toMetricRow(row)]));
  const widths = computeTableWidths(headers, body, cellsByRow, verdictFields);

  const rule = formatTableRule(widths);
  const border = formatTableBorder(widths);
  const renderers = buildTableRenderers({ headers, baseline, widths, cellsByRow, verdictFields });

  return body.map((bodyLine) => renderBodyLine(bodyLine, rule, border, renderers));
}
