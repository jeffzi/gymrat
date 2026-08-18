import { assertNever } from "../errors.js";
import {
  type CellStyler,
  formatMetricCellParts,
  formatTableLine,
  type MetricCellParts,
  styleWithin,
} from "./format.js";
import { planSections } from "./sections.js";
import {
  aggregateLabelLengths,
  alignRight,
  type BodyLineRenderers,
  computeColumnWidth,
  computeMetricColumnWidth,
  formatTableBorder,
  formatTableRule,
  GROUP_LABEL_STYLE,
  indentedSectionLabel,
  joinValueCell,
  type NamedRow,
  planBody,
  renderBodyLine,
  sectionAnnotation,
  styleHeaderCell,
  styleLabelCell,
  VALUE_COLUMN_MIN,
  valueWidths,
  VARIANT_NAME_STYLE,
  widestHeaderLabel,
} from "./text-table-core.js";
import type { MeasurementResult } from "./types.js";

type MeasureRow = readonly [metric: string, value: string];
type MeasureWidths = readonly [number, number];

interface MeasureMetricRow extends NamedRow {
  readonly value: MetricCellParts;
}

function formatMeasureRow(row: MeasureRow, widths: MeasureWidths, styleCell?: CellStyler): string {
  return formatTableLine([row[0], alignRight(row[1], widths[1])], widths, styleCell);
}

/** Render a single-revision measurement table with median and spread for each metric. */
export function renderMeasureTable(result: MeasurementResult, label: string): string[] {
  const layout = planSections(result.metrics, (name, group, metric): MeasureMetricRow => ({
    name,
    label: indentedSectionLabel(metric.meta.shortName, group),
    value: formatMetricCellParts(metric.median, metric.spread, metric.meta.unit),
  }));
  const sectioned = layout.sections.length > 1;

  const valueFields = valueWidths(layout.ordered.map((row) => row.value));
  const body = planBody<MeasureMetricRow, never>(layout, undefined, (section) =>
    sectionAnnotation(section, result.configKinds),
  );

  const toMeasureRow = (row: MeasureMetricRow): MeasureRow => [
    sectioned ? row.label : row.name,
    joinValueCell(row.value, valueFields),
  ];

  const cellsByRow = new Map(layout.ordered.map((row) => [row, toMeasureRow(row)]));
  const rows = [...cellsByRow.values()];
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
    metric: (row) => {
      const cells = cellsByRow.get(row);
      if (cells === undefined) throw new Error(`no cells for metric row ${row.name}`);
      return formatMeasureRow(cells, widths);
    },
    aggregate: ({ cell }) => assertNever(cell),
  };

  return body.map((bodyLine) => renderBodyLine(bodyLine, rule, border, renderers));
}
