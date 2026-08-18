import { styleText } from "node:util";

import type { DisplayClass } from "./format.js";
import type { ComparisonResult } from "./types.js";

// ---------------------------------------------------------------------------
// Table layout
// ---------------------------------------------------------------------------

/**
 * Width a column needs to hold its header and every cell, never below `minWidth`.
 *
 * The result includes two columns of gutter beyond the widest content, so
 * callers padding cells to this width get a visible gap before the next
 * column's separator for free.
 */
export function computeColumnWidth(
  headerLength: number,
  contentLengths: number[],
  minWidth: number,
): number {
  const maxContent = Math.max(headerLength, ...contentLengths);
  return Math.max(maxContent + 2, minWidth);
}

/** How a cell renders once it has been padded to its column width. */
export type CellStyler = (cell: string, index: number) => string;

/**
 * Pad each cell to its column width and join them with the column separator.
 *
 * `styleCell` runs on the finished line, after the padding and the trailing
 * trim: every width in the table is measured on plain text, and an ANSI escape
 * introduced before padding would be counted as visible width, shifting every
 * column to its right.
 */
export function formatTableLine(
  cells: readonly string[],
  widths: readonly number[],
  styleCell?: CellStyler,
): string {
  const padded = widths.map((width, i) => (cells[i] ?? "").padEnd(width));
  // Only the trailing run is cut: leading padding is the first column's width,
  // which a row opening on an empty cell needs to stay under the header.
  const line = padded.join("│").trimEnd();
  if (styleCell === undefined) return line;
  return line.split("│").map(styleCell).join("│");
}

// ---------------------------------------------------------------------------
// Styling and color
// ---------------------------------------------------------------------------

/** The style-tag union {@link styleText} accepts — the type every renderer's style constants are typed against. */
export type Style = Parameters<typeof styleText>[0];

/**
 * Apply `style` to `label` via `styleText`.
 *
 * `stream` lets callers target `styleText`'s TTY/color auto-detection at a
 * specific stream instead of the default `process.stdout`. Either way,
 * `styleText` returns the bare label when the environment suppresses styling
 * (`NO_COLOR`, a non-TTY stream).
 */
export function formatLabel(label: string, style: Style, stream?: NodeJS.WriteStream): string {
  return stream !== undefined ? styleText(style, label, { stream }) : styleText(style, label);
}

/**
 * Replace every `` `...` `` span in `text` with its content styled yellow.
 *
 * Backticks are stripped regardless of whether color is active, so callers can
 * embed backtick-marked command names in user-facing strings and let the render
 * layer decide how to present them. When `stream` is provided, `styleText`
 * targets that stream's TTY/color detection (same semantics as {@link formatLabel}).
 */
export function highlightInlineCode(text: string, stream?: NodeJS.WriteStream): string {
  return text.replace(/`([^`]+)`/g, (_match, code: string) => formatLabel(code, "yellow", stream));
}

/** Set `name`, or unset it when `value` is `undefined` — assigning `undefined` would store the string. */
function setEnvVar(name: string, value: string | undefined): void {
  if (value === undefined) {
    delete process.env[name];
  } else {
    process.env[name] = value;
  }
}

/**
 * Run `fn` with the environment `styleText` reads pinned to `color`.
 *
 * `styleText` consults `FORCE_COLOR`, `NO_COLOR` and the stream's TTY-ness on
 * every call, so an explicit color choice can only reach it through the
 * environment. `undefined` leaves both variables alone — no choice was made, and
 * auto-detection is what `styleText` already does. Either variable is restored
 * afterwards, so a render cannot leak its choice into whatever runs next.
 */
export function withColor<T>(color: boolean | undefined, fn: () => T): T {
  if (color === undefined) return fn();

  const force = process.env.FORCE_COLOR;
  const no = process.env.NO_COLOR;
  try {
    setEnvVar("FORCE_COLOR", color ? "1" : undefined);
    setEnvVar("NO_COLOR", color ? undefined : "1");
    return fn();
  } finally {
    setEnvVar("FORCE_COLOR", force);
    setEnvVar("NO_COLOR", no);
  }
}

// ---------------------------------------------------------------------------
// Label truncation
// ---------------------------------------------------------------------------

/** U+2026, one character wide — three periods would cost two more columns. */
const ELLIPSIS = "…";

/** Locale-independent: UAX #29 cluster boundaries are the same everywhere. */
const GRAPHEME_SEGMENTER = new Intl.Segmenter(undefined, { granularity: "grapheme" });

/**
 * `text` split into the characters a reader perceives.
 *
 * A single perceived character can span several code points — a family emoji
 * joins four of them with zero-width joiners, a waving hand carries a skin-tone
 * modifier — and each code point outside the basic plane spans two UTF-16
 * units. Slicing by either unit can cut inside a cluster, and a fragment of one
 * is not the character it came from: it renders as a different glyph, or as a
 * replacement box, which is worth less than the column it costs.
 */
function graphemes(text: string): string[] {
  return [...GRAPHEME_SEGMENTER.segment(text)].map((segment) => segment.segment);
}

/**
 * `text` fitted into `maxWidth` characters, keeping both of its ends.
 *
 * The ends carry the identity: branch names that share a prefix differ in their
 * tail, and a progress line names its step at the front and its target at the
 * back. Cutting from the middle keeps both, where a plain slice would drop
 * whichever end runs past the budget.
 *
 * Text already inside the budget comes back untouched, so widening the budget
 * can never lengthen the result. The budget is counted and spent in whole
 * grapheme clusters, so a cluster is either kept entire or elided entire.
 */
export function shortenLabel(text: string, maxWidth: number): string {
  if (maxWidth <= 0) return "";

  const clusters = graphemes(text);
  if (clusters.length <= maxWidth) return text;

  const kept = maxWidth - ELLIPSIS.length;
  if (kept <= 0) return ELLIPSIS;

  const head = Math.ceil(kept / 2);
  const tail = kept - head;
  // Index the tail from the front: `slice(-0)` would return the whole array.
  const start = clusters.slice(0, head).join("");
  const end = clusters.slice(clusters.length - tail).join("");
  return `${start}${ELLIPSIS}${end}`;
}

/**
 * Widest a variant label prints, ellipsis included.
 *
 * A branch name is free to be as long as git allows, but every column it heads
 * is sized from it, so an unbounded one pushes the figures the report exists to
 * show off the right edge of the terminal.
 */
const LABEL_DISPLAY_WIDTH = 20;

/**
 * Every variant label under the name the report prints for it.
 *
 * Labels are shortened as a set rather than one at a time because the ends are
 * what tell sibling branches apart: `feature/experiment-one-fastpath` and
 * `feature/exploration-two-fastpath` share both of theirs, so the narrowest
 * width that keeps them distinct is the one worth spending. The width grows
 * until the displayed names are as distinct as the labels were — two candidates
 * named identically stay that way, which is the run's own doing, not the
 * display's.
 *
 * Truncation is display-only: `label=` parsing, the config, and the JSON
 * renderer all keep the full label.
 */
export function truncateLabels(labels: readonly string[]): string[] {
  const longest = Math.max(0, ...labels.map((label) => label.length));
  const distinct = new Set(labels).size;
  for (let maxWidth = LABEL_DISPLAY_WIDTH; maxWidth < longest; maxWidth++) {
    const shortened = labels.map((label) => shortenLabel(label, maxWidth));
    if (new Set(shortened).size === distinct) return shortened;
  }
  return [...labels];
}

/**
 * `result` with every variant label replaced by the name the report prints.
 *
 * Every renderer reads its labels off this copy, so a label is shortened once
 * per report and prints the same way in the header, the column it heads, the
 * geomean row and the legend.
 */
export function withDisplayLabels(result: ComparisonResult): ComparisonResult {
  const [baseline, ...candidates] = truncateLabels([
    result.baselineLabel,
    ...result.candidates.map((candidate) => candidate.label),
  ]);
  return {
    ...result,
    baselineLabel: baseline ?? result.baselineLabel,
    candidates: result.candidates.map((candidate, index) => ({
      ...candidate,
      label: candidates[index] ?? candidate.label,
    })),
  };
}

// ---------------------------------------------------------------------------
// Variant and verdict styling
// ---------------------------------------------------------------------------

/** The style a variant name wears where the report names it as a name. */
export const VARIANT_NAME_STYLE: Style = ["bold", "underline"];

/** A variant name, styled — for the unpadded prose of the run header. */
export function formatVariantName(label: string, stream?: NodeJS.WriteStream): string {
  return formatLabel(label, VARIANT_NAME_STYLE, stream);
}

/** The style the word `Hint` wears — the only part of the label the underline reaches. */
const HINT_WORD_STYLE: Style = ["yellow", "underline"];

/** The style the label's colon wears — colored with the word, never underlined. */
const HINT_COLON_STYLE: Style = ["yellow"];

/**
 * The `Hint:` label every hint line opens with, styled by `styleText` auto-detection.
 *
 * Word and colon are styled as two spans so the underline stops at the word: an
 * underscore running under a colon reads as punctuation of its own.
 */
export function formatHintLabel(stream?: NodeJS.WriteStream): string {
  return formatLabel("Hint", HINT_WORD_STYLE, stream) + formatLabel(":", HINT_COLON_STYLE, stream);
}

/**
 * The color each display class wears wherever the report states a verdict.
 *
 * Every style here is worn by the verdict itself — a glyph, a delta, a tally —
 * never by the row or the values around it, so a class that recedes has to say
 * so in its own color: within noise and inconclusive dim, identical reads cyan
 * for "measured the same", and unstable keeps its amber warning.
 */
export const VERDICT_STYLES: Record<DisplayClass, Style> = {
  improved: ["green"],
  regressed: ["red"],
  unstable: ["yellow"],
  identical: ["cyan"],
  "within-noise": ["dim"],
  inconclusive: ["dim"],
};

/** Which occurrence of the marker `styleWithin` reaches for. */
export interface StyleWithinOptions {
  /**
   * Style the last occurrence rather than the first.
   *
   * Set it wherever the marker can also appear in the prose introducing it — a
   * one-letter variant name inside a `vs ` prefix, say.
   */
  last?: boolean;
}

/**
 * Style `marker` where it sits inside an already-padded cell.
 *
 * Styling a cell before it is padded is the alignment bug this exists to
 * prevent: `padEnd` counts an ANSI escape as visible width, so a styled cell is
 * padded short and every column after it slides left.
 *
 * The splice is positional rather than a `String.replace`, because a marker is
 * user data — a branch name, a metric name — and `replace` reads `$&`, `` $` ``,
 * `$'` and `$<n>` in its replacement argument as patterns, splicing the
 * surrounding cell text into the styled span. A marker the cell does not
 * contain leaves the cell alone.
 */
export function styleWithin(
  cell: string,
  marker: string,
  style: Style,
  options: StyleWithinOptions = {},
): string {
  const index = options.last === true ? cell.lastIndexOf(marker) : cell.indexOf(marker);
  if (index === -1) {
    return cell;
  }
  return cell.slice(0, index) + formatLabel(marker, style) + cell.slice(index + marker.length);
}
