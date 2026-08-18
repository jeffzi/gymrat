import type { WorktreeRemovalFailure } from "../targets.js";
import {
  displayClass,
  formatDelta,
  formatEvidence,
  formatHintLabel,
  formatLabel,
  formatVariantName,
  formatVerdictDelta,
  GATED_GEOMEAN_LABEL,
  getGlyph,
  hasUnstableHighlight,
  type HighlightBlock,
  highlightLabel,
  pluralize,
  selectHighlights,
  type Style,
  styleWithin,
  truncateLabels,
  UNSTABLE_FUTILITY_NOTE,
  verdictSummaryParts,
  VERDICT_STYLES,
  footerLines,
  withColor,
  withDisplayLabels,
} from "./format.js";
import { spansManyKinds } from "./sections.js";
import { renderMeasureTable } from "./text-measure.js";
import { renderComparisonTable } from "./text-multi.js";
import { renderTable } from "./text-single.js";
import { styleGlyphAndDelta, styleSpans } from "./text-table-core.js";
import type {
  CandidateComparison,
  ComparisonResult,
  FailOnCondition,
  MeasurementResult,
  MetricComparisons,
  ReportOptions,
  WorktreeCleanupOutcome,
} from "./types.js";

// ---------------------------------------------------------------------------
// Header
// ---------------------------------------------------------------------------

/** The `·` separator in the report header, dimmed in colored mode. */
const HEADER_SEPARATOR = "·";

/** Join a run header's parts with the dimmed `·` separator every report header shares. */
function joinHeaderParts(parts: readonly string[]): string {
  return parts.join(` ${formatLabel(HEADER_SEPARATOR, ["dim"])} `);
}

// ---------------------------------------------------------------------------
// Highlights
// ---------------------------------------------------------------------------

/** One line tallying every verdict class one candidate earned. */
function renderSummary(metrics: MetricComparisons, candidateIndex: number): string {
  return verdictSummaryParts(metrics, candidateIndex).join("   ");
}

/** Gap between the longest highlighted metric name and the delta that follows it. */
const HIGHLIGHT_NAME_GUTTER = 2;

/**
 * Width the highlights block right-aligns its deltas in — the length of a
 * `±NN.N%` percentage.
 */
const HIGHLIGHT_DELTA_WIDTH = 6;

/** The heading a candidate's non-empty highlights list opens with. */
const HIGHLIGHTS_HEADING = "highlights";

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
    let styled = styleGlyphAndDelta(plain, {
      shown,
      delta,
      style,
      deltaSearch: { last: true },
    });
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

function gateTripLines(
  candidate: CandidateComparison,
  conditions: readonly FailOnCondition[],
): string[] {
  const thresholds = conditions.flatMap((condition) =>
    condition.kind === "geomean" ? [condition.pct] : [],
  );

  return candidate.kinds.flatMap((kind) => {
    const geomean = kind.gatedGeomean;
    if (geomean === undefined || geomean.n === 0) return [];

    const delta = formatDelta(geomean.value);
    const style: Style = VERDICT_STYLES.regressed;
    return thresholds
      .filter((pct) => geomean.value >= pct)
      .map((pct) => {
        const plain = `  ${GATE_TRIP_GLYPH} ${kind.kind} ${GATED_GEOMEAN_LABEL} ${delta} exceeded --fail-on geomean:${pct}`;
        return styleSpans(plain, [
          { text: GATE_TRIP_GLYPH, style },
          { text: delta, style },
        ]);
      });
  });
}

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

function renderSummaries(result: ComparisonResult): string[] {
  const labelWidth = Math.max(...result.candidates.map((candidate) => candidate.label.length));
  return result.candidates.map((candidate, index) => {
    const paddedLabel = candidate.label.padEnd(labelWidth);
    const styledLabel = formatLabel(paddedLabel, ["bold"]);
    return `${styledLabel}  ${renderSummary(result.metrics, index)}`;
  });
}

// ---------------------------------------------------------------------------
// Footers
// ---------------------------------------------------------------------------

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

function renderMethodFooter(result: ComparisonResult, verbose: boolean): string[] {
  return footerLines(result.metrics, verbose, (hint) => `${formatHintLabel()} ${hint}`);
}

/**
 * Collapse a git diagnostic onto one line.
 *
 * git routinely emits several lines for one failure — a `warning:` line before the
 * `fatal:` line, plus indented continuations.
 */
function toSingleLine(text: string): string {
  return text.replace(/\s+/g, " ").trim();
}

/** Format worktree removal failures and prune errors into indented diagnostic lines. */
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

function renderWorktreeFooter(result: WorktreeCleanupOutcome): string[] {
  const details = formatCleanupFailures(result.worktreesLeftBehind, result.worktreePruneError);
  if (details.length === 0) return [];

  return [
    `${pluralize(result.worktreesRemoved, "worktree")} removed · ${result.worktreesLeftBehind.length} left behind`,
    ...details,
  ];
}

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

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/** Pluralize the "paired sample" label used in comparison report headers. */
export function pairedSamples(samples: number): string {
  return pluralize(samples, "paired sample");
}

function compareHeader(display: ComparisonResult): string {
  const candidateNames = display.candidates
    .map((candidate) => formatVariantName(candidate.label))
    .join(", ");
  return joinHeaderParts([
    formatLabel("gymrat compare", ["bold"]),
    `baseline ${formatVariantName(display.baselineLabel)} ↔ ${candidateNames}`,
    pairedSamples(display.samples),
    `adapter: ${display.adapter}`,
  ]);
}

export function renderReport(result: ComparisonResult, options: ReportOptions = {}): string {
  return withColor(options.color, () => {
    const display = withDisplayLabels(result);
    const lines = [options.header ?? compareHeader(display)];

    const conditions = options.failOn ?? [];
    if (display.candidates.length > 1) {
      lines.push(...renderComparison(display, conditions));
    } else {
      const candidate = display.candidates[0];
      if (candidate === undefined) throw new Error("expected at least one candidate");
      lines.push(...renderCandidate(display, candidate, 0, conditions));
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

export function renderMeasureReport(
  result: MeasurementResult,
  options: ReportOptions = {},
): string {
  return withColor(options.color, () => {
    const label = truncateLabels([result.label])[0] ?? result.label;
    const header = joinHeaderParts([
      formatLabel("gymrat measure", ["bold"]),
      formatVariantName(label),
      pluralize(result.samples, "sample"),
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
