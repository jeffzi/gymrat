import { assertNever } from "../errors.js";
import type { WorktreeRemovalFailure } from "../targets.js";
import type { GeomeanExclusion, GeomeanResult, Method, MetricVerdict } from "../verdict/verdict.js";
import {
  type CellStyler,
  computeColumnWidth,
  countVerdicts,
  formatDelta,
  formatHintLabel,
  formatLabel,
  formatNoiseBand,
  formatPairCount,
  formatPValue,
  formatSpread,
  formatTableLine,
  formatValue,
  formatVerdictDelta,
  getGlyph,
  selectHighlights,
  styleGlyph,
  styleWithin,
  VERDICT_GLOSSES,
} from "./format.js";
import type { ComparisonResult, MetricComparisons } from "./types.js";

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
 * The verdicts whose rows recede.
 *
 * A metric that sat within the noise, or was too jittery to judge, is the row a
 * reader skips; dimming it whole leaves the rows that did move to carry the
 * table.
 */
const QUIET_VERDICTS: ReadonlySet<MetricVerdict["verdict"]> = new Set(["no-signal", "unstable"]);

const GEOMEAN_LABEL = "geomean (gating metrics)";

/**
 * Width the highlights block right-aligns its deltas in — the length of a
 * `±NN.N%` percentage.
 *
 * A delta wider than this overflows rather than truncating, so the one word that
 * can stand in for a delta, `unstable`, simply starts two columns further left.
 */
const HIGHLIGHT_DELTA_WIDTH = 6;

const SAMPLES_HINT = "re-run with --samples 6 or more for statistical verdicts";

/**
 * The hint closes the method footer, well below the table the report's colors
 * are there to organize, so it stays plain in a colored report too. Routing the
 * label through `formatHintLabel` anyway keeps it on one code path with the
 * styled hints `formatCliError` prints to stderr.
 */
const HINT_LABEL = formatHintLabel(false);

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

function formatMetricCell(median?: number, spread?: number, unit?: "ns" | "bytes"): string {
  if (median === undefined) return "";
  return `${formatValue(median, unit)}${formatSpread(spread)}`;
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
  return [getGlyph(verdict.verdict), delta, band, pairs].filter((part) => part !== "").join("  ");
}

/** How many gating metrics the geomean left out, and on what grounds. */
function formatExclusions(excluded: readonly GeomeanExclusion[]): string {
  if (excluded.length === 0) return "";
  const reasons = [...new Set(excluded.map((exclusion) => exclusion.reason))];
  return `${excluded.length} excluded: ${reasons.join(", ")}`;
}

/** The geomean's verdict cell, and the aggregate inside it that the row exists for. */
interface GeomeanCell {
  readonly figure: string;
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
  if (geomean.n === 0) return { figure: "—", text: "—  no stable gating metrics" };

  const stable = `${geomean.n} stable metric${geomean.n === 1 ? "" : "s"}`;
  const exclusions = formatExclusions(geomean.excluded);
  const provenance = exclusions === "" ? stable : `${stable} · ${exclusions}`;
  const figure = formatDelta(geomean.value);
  return { figure, text: `${figure}  ${provenance}` };
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
    styleVerdictCell((cell) => styleGlyph(cell, outcome, useColor)),
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
function renderTable(result: ComparisonResult, useColor: boolean): string[] {
  const [baseline, candidate] = result.labels;
  const headers: Row = ["metric", baseline, candidate, `vs ${baseline}`];

  const rows: MetricRow[] = Object.entries(result.metrics).map(([name, metric]) => ({
    cells: [
      name,
      formatMetricCell(metric.medianA, metric.spreadA, metric.meta.unit),
      formatMetricCell(metric.medianB, metric.spreadB, metric.meta.unit),
      formatVerdictCell(metric.verdict, result.samples),
    ],
    verdict: metric.verdict,
  }));

  const widths: Widths = [
    computeColumnWidth(
      headers[0].length,
      [...rows.map((row) => row.cells[0].length), GEOMEAN_LABEL.length],
      METRIC_COLUMN_MIN,
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

  const rule = widths.map((width) => "─".repeat(width)).join("┼");
  const geomeanCell = formatGeomeanCell(result.geomean);
  const geomean: Row = [GEOMEAN_LABEL, baseline, candidate, geomeanCell.text];

  return [
    formatLabel(formatRow(headers, widths), ["bold"], useColor),
    rule,
    ...rows.map((row) => formatMetricRow(row.cells, widths, row.verdict, useColor)),
    rule,
    formatRow(
      geomean,
      widths,
      styleVerdictCell((cell) => styleWithin(cell, geomeanCell.figure, ["bold"], useColor)),
    ),
  ];
}

/** One line tallying every verdict class the run produced. */
function renderSummary(metrics: MetricComparisons): string {
  const counts = countVerdicts(metrics);
  return [
    `${getGlyph("improved")} ${counts.improved} ${VERDICT_GLOSSES.improved}`,
    `${getGlyph("regressed")} ${counts.regressed} ${VERDICT_GLOSSES.regressed}`,
    `${getGlyph("unstable")} ${counts.unstable} ${VERDICT_GLOSSES.unstable}`,
    `${getGlyph("no-signal")} ${counts.noSignal} ${VERDICT_GLOSSES["no-signal"]}`,
  ].join("   ");
}

/** The statistic backing a verdict, in the terms of the method that produced it. */
function formatEvidence(verdict: MetricVerdict): string {
  switch (verdict.method) {
    case "signed-rank":
      return formatPValue(verdict.p);
    case "band":
      return `band ${formatNoiseBand(verdict.band)}`;
    case "exact":
      return "(exact)";
    /* v8 ignore next -- exhaustive switch; compile-time guard via assertNever */
    default:
      return assertNever(verdict);
  }
}

/**
 * The metrics worth a second look, loudest first, with the evidence behind each.
 *
 * Empty when nothing moved: a heading over an empty list reads as a rendering
 * bug, and a run that changed nothing has nothing to highlight.
 */
/** Gap between the longest highlighted metric name and the delta that follows it. */
const HIGHLIGHT_NAME_GUTTER = 2;

function renderHighlights(metrics: MetricComparisons): string[] {
  const highlights = selectHighlights(metrics);
  if (highlights.length === 0) return [];

  const nameWidth =
    Math.max(...highlights.map((highlight) => highlight.name.length)) + HIGHLIGHT_NAME_GUTTER;

  const lines = ["highlights"];
  for (const { name, metric } of highlights) {
    const verdict = metric.verdict;
    const delta = formatVerdictDelta(verdict);
    lines.push(
      `  ${getGlyph(verdict.verdict)} ${name.padEnd(nameWidth)}${delta.padStart(
        HIGHLIGHT_DELTA_WIDTH,
      )}  ${formatEvidence(verdict)}`,
    );
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
function renderLegend(baseline: string): string {
  const glosses = [
    `${getGlyph("improved")} ${VERDICT_GLOSSES.improved}`,
    `${getGlyph("regressed")} ${VERDICT_GLOSSES.regressed}`,
    `${getGlyph("unstable")} ${VERDICT_GLOSSES.unstable}`,
    `${getGlyph("no-signal")} ${VERDICT_GLOSSES["no-signal"]}`,
  ].join(" · ");
  return `legend: ${glosses} — candidates are judged against ${baseline}`;
}

/** The pair counts behind every verdict a given method decided. */
function pairCounts(metrics: MetricComparisons, method: Method): number[] {
  const counts: number[] = [];
  for (const { verdict } of Object.values(metrics)) {
    if (verdict?.method === method) counts.push(verdict.n);
  }
  return counts;
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
function renderMethodFooter(result: ComparisonResult): string[] {
  const signedRank = pairCounts(result.metrics, "signed-rank");
  const band = pairCounts(result.metrics, "band");
  const lines: string[] = [];

  if (signedRank.length > 0) {
    const pairs = formatPairCount(Math.min(...signedRank));
    lines.push(`verdicts: Wilcoxon signed-rank on pairs (${pairs} ≥ 6) · ~ = no signal at α=0.05`);
  }
  if (band.length > 0) {
    const pairs = formatPairCount(Math.max(...band));
    lines.push(
      `noise band ±(half-range × K) — ${pairs} below signed-rank floor (6 pairs)`,
      `${HINT_LABEL} ${SAMPLES_HINT}`,
    );
  }

  return lines;
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
 * Render a comparison as the plain-text report the CLI prints.
 *
 * The run header and the metric table come first, then everything that
 * interprets them — verdict counts, highlights, legend, method, and any cleanup
 * failure. Terminals anchor on the last lines they printed, so the summary sits
 * below the table it summarizes rather than above it.
 *
 * Whether the report is styled is the caller's decision, not this renderer's: it
 * never reads `isTTY` or `NO_COLOR`, so the same result renders the same bytes
 * wherever it runs. Color is off unless asked for, which keeps a piped or
 * captured report plain by default.
 */
export function renderReport(result: ComparisonResult, useColor = false): string {
  const [baseline, candidate] = result.labels;
  const lines = [
    `gymrat compare · ${baseline} ↔ ${candidate} · ${result.samples} paired samples · adapter: ${result.adapter}`,
    ...renderTable(result, useColor),
    "",
    renderSummary(result.metrics),
  ];

  const highlights = renderHighlights(result.metrics);
  if (highlights.length > 0) {
    lines.push("", ...highlights);
  }

  lines.push(
    "",
    renderLegend(baseline),
    ...renderMethodFooter(result),
    ...renderWorktreeFooter(result),
  );

  return lines.join("\n");
}
