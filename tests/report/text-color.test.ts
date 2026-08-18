import { stripVTControlCharacters as stripAnsi } from "node:util";

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { renderMeasureReport, renderReport } from "../../src/report/text.js";
import type { ComparisonResult, MeasurementResult } from "../../src/report/types.js";
import {
  bandMetric,
  createCandidate,
  createComparisonResult,
  exactMetric,
  otherKind,
  signedRankMetric,
} from "../fixtures/comparison-result.js";
import {
  createMeasurementResult,
  measuredMetric,
  twoKindMeasurement,
} from "../fixtures/measurement-result.js";

/** Character offsets of every occurrence of `glyph` in a rendered table line. */
function offsetsOf(line: string, glyph: string): number[] {
  const offsets: number[] = [];
  for (let i = line.indexOf(glyph); i !== -1; i = line.indexOf(glyph, i + 1)) {
    offsets.push(i);
  }
  return offsets;
}

/**
 * Character offsets of every column separator in a rendered table line.
 *
 * Two lines whose separators sit at the same offsets have aligned columns.
 */
function separatorOffsets(line: string): number[] {
  return offsetsOf(line, "│");
}

/** The cells of a rendered table line, padding included. */
function cellsOf(line: string): string[] {
  return line.split("│");
}

/** The last cell of a rendered table line — the delta column. */
function deltaCellOf(line: string): string {
  const cell = cellsOf(line).at(-1);
  if (cell === undefined) {
    throw new Error(`no cells in line: ${JSON.stringify(line)}`);
  }
  return cell;
}

/** The single rendered line starting with `prefix`, or a failure naming the report. */
function lineStartingWith(report: string, prefix: string): string {
  const line = report.split("\n").find((candidate) => candidate.startsWith(prefix));
  if (line === undefined) {
    throw new Error(`no line starting with ${prefix} in report:\n${report}`);
  }
  return line;
}

/**
 * The first rendered line containing `needle`, or a failure naming the report.
 *
 * A colored line starts with escape codes rather than its text, so the color
 * tests match on content instead of a prefix.
 */
function lineContaining(report: string, needle: string): string {
  const line = report.split("\n").find((candidate) => candidate.includes(needle));
  if (line === undefined) {
    throw new Error(`no line containing ${needle} in report:\n${report}`);
  }
  return line;
}

/** Matches a line dimmed end-to-end: opens with SGR 2, closes with SGR 22. */
const DIMMED_LINE = /^\x1b\[2m.*\x1b\[22m$/;

/**
 * The SGR parameters opened immediately before `marker` in `line`.
 *
 * Only the unbroken run of escape sequences touching the marker counts, so a
 * style opened at the start of the line does not leak into the result.
 *
 * Pass `last` to read the trailing occurrence instead of the leading one, for
 * a marker that repeats within the line.
 */
function stylesAt(line: string, marker: string, options: { last?: boolean } = {}): string[] {
  const index = options.last === true ? line.lastIndexOf(marker) : line.indexOf(marker);
  if (index === -1) {
    throw new Error(`no ${marker} in line: ${JSON.stringify(line)}`);
  }
  const opened = /((?:\x1b\[\d+m)*)$/.exec(line.slice(0, index))?.[1] ?? "";
  return [...opened.matchAll(/\x1b\[(\d+)m/g)].map((match) => match[1] ?? "");
}

/** For each SGR parameter that closes a style, the parameters it closes. */
const SGR_CLOSERS: Readonly<Record<string, RegExp>> = {
  "0": /^\d+$/,
  "22": /^[12]$/,
  "23": /^3$/,
  "24": /^4$/,
  "39": /^(?:3[0-7]|9[0-7])$/,
  "49": /^(?:4[0-7]|10[0-7])$/,
};

/**
 * The SGR parameters still open at each column separator of `line`.
 *
 * A separator that inherits its row's style reports that style here; one left
 * in the terminal's default color reports nothing.
 */
function separatorStyles(line: string): string[][] {
  let open: string[] = [];
  const styles: string[][] = [];
  for (const token of line.matchAll(/\x1b\[(\d+)m|│/g)) {
    const parameter = token[1];
    if (parameter === undefined) {
      styles.push(open);
      continue;
    }
    const closes = SGR_CLOSERS[parameter];
    open = closes === undefined ? [...open, parameter] : open.filter((p) => !closes.test(p));
  }
  return styles;
}

/** Every rendered table row of a report, styling stripped, in report order. */
function tableRows(report: string): string[] {
  return report
    .split("\n")
    .map((line) => stripAnsi(line))
    .filter((line) => line.includes("│"));
}

/**
 * One entry per report line, coarse enough to read as a layout.
 *
 * A table row collapses to its first cell, a column rule collapses to a marker,
 * and every other line stays as its plain text. A section's top border joins its
 * columns with top-T junctions rather than the crossings of a rule, so it gets
 * its own marker.
 */
function tableShape(report: string): string[] {
  return report.split("\n").map((line) => {
    const bare = stripAnsi(line);
    if (/^─+┼/.test(bare)) {
      return "<rule>";
    }
    if (/^[─┬]+$/.test(bare)) {
      return "<border>";
    }
    if (!bare.includes("│")) {
      return bare.trimEnd();
    }
    return cellsOf(bare)[0]?.trim() ?? "";
  });
}

/** The table region of a report: everything down to the last table row. */
function tableRegion(report: string): string[] {
  const shape = tableShape(report);
  const lines = report.split("\n");
  const last = lines.reduce(
    (found, line, index) => (stripAnsi(line).includes("│") ? index : found),
    -1,
  );
  if (last === -1) {
    throw new Error(`no table rows in report:\n${report}`);
  }
  return shape.slice(0, last + 1);
}

/** The lines of the `highlights` block, its heading excluded. */
function highlightLines(report: string): string[] {
  const lines = report.split("\n");
  const start = lines.findIndex((line) => stripAnsi(line) === "highlights");
  if (start === -1) {
    return [];
  }
  const rest = lines.slice(start + 1);
  const end = rest.indexOf("");
  return end === -1 ? rest : rest.slice(0, end);
}

beforeEach(() => {
  vi.stubEnv("NO_COLOR", "1");
  vi.stubEnv("FORCE_COLOR", undefined);
});

afterEach(() => {
  vi.unstubAllEnvs();
});

describe("renderReport", () => {
  describe("when rendering with color", () => {
    beforeEach(() => {
      vi.stubEnv("FORCE_COLOR", "1");
    });

    /** A run whose rows cover every verdict class, plus a geomean figure. */
    function colorfulResult(): ComparisonResult {
      return createComparisonResult({
        metrics: {
          "faster/time": signedRankMetric({ verdict: "improved", delta: -17.5, unit: "ns" }),
          "slower/time": signedRankMetric({ verdict: "regressed", delta: 2.4, unit: "ns" }),
          "flat/time": signedRankMetric({ verdict: "no-signal", delta: 0.3, unit: "ns" }),
          "tied/heap": bandMetric({ verdict: "no-signal", delta: -0.5, n: 10, usableN: 0 }),
          "single-pair/time": bandMetric({ delta: -0.4, noisePct: 0.5, n: 1, unit: "ns" }),
          "jittery/time": signedRankMetric({ verdict: "unstable", delta: -50, noisePct: 30 }),
        },
        candidates: [
          createCandidate({
            kinds: [
              otherKind(-5.8, 3, {
                excluded: [{ metric: "jittery/time", reason: "unstable" }],
              }),
            ],
          }),
        ],
        worktreesRemoved: 1,
        worktreesLeftBehind: [{ dir: "/tmp/gymrat-abc", error: "is locked" }],
        worktreePruneError: "fatal: not a git repository",
      });
    }

    it("leaves the report unstyled when NO_COLOR is set", () => {
      vi.stubEnv("FORCE_COLOR", undefined);
      vi.stubEnv("NO_COLOR", "1");

      const output = renderReport(colorfulResult());

      expect(output).not.toContain("\x1b[");
    });

    /**
     * The guard against styling a cell before padding it.
     *
     * `padEnd` counts escape codes as characters, so a renderer that styles
     * first and pads second pads short and slides every column to the right of
     * it out of line. Stripping the styles back off has to leave every
     * separator where the header put it.
     */
    it("pads on the plain text, so the colored columns line up once the styles are stripped", () => {
      const bare = stripAnsi(renderReport(colorfulResult()));
      const headerOffsets = separatorOffsets(lineStartingWith(bare, "metric"));

      expect
        .soft(separatorOffsets(lineStartingWith(bare, "faster/time")))
        .toStrictEqual(headerOffsets);
      expect(separatorOffsets(lineStartingWith(bare, "geomean"))).toStrictEqual(headerOffsets);
    });

    it("measures the verdict sub-fields on the plain text, so the bands stack once stripped", () => {
      const result = createComparisonResult({
        metrics: {
          "improved/time": signedRankMetric({
            verdict: "improved",
            delta: -12.4,
            noisePct: 2.5,
            unit: "ns",
          }),
          "flat/time": signedRankMetric({
            verdict: "no-signal",
            delta: 0.4,
            noisePct: 100,
            unit: "ns",
          }),
        },
      });

      const bare = stripAnsi(renderReport(result));

      expect
        .soft(cellsOf(lineStartingWith(bare, "improved/time")).at(-1)?.trim())
        .toBe("✓  -12.4%  ±  2.5%");
      expect(cellsOf(lineStartingWith(bare, "flat/time")).at(-1)?.trim()).toBe(
        "~   +0.4%  ±100.0%",
      );
    });

    it.each([
      { verdict: "improved", metric: "faster/time", glyph: "✓", color: "green", code: "32" },
      { verdict: "regressed", metric: "slower/time", glyph: "✗", color: "red", code: "31" },
      { verdict: "unstable", metric: "jittery/time", glyph: "≈", color: "yellow", code: "33" },
      { verdict: "identical", metric: "tied/heap", glyph: "=", color: "cyan", code: "36" },
      { verdict: "within noise", metric: "flat/time", glyph: "~", color: "dim", code: "2" },
      { verdict: "inconclusive", metric: "single-pair/time", glyph: "?", color: "dim", code: "2" },
    ])("paints the $verdict verdict $color on its row", ({ metric, glyph, code }) => {
      const row = lineContaining(renderReport(colorfulResult()), metric);

      expect(stylesAt(row, glyph)).toContain(code);
    });

    it.each([
      { verdict: "within noise", metric: "flat/time" },
      { verdict: "identical", metric: "tied/heap" },
      { verdict: "inconclusive", metric: "single-pair/time" },
      { verdict: "unstable", metric: "jittery/time" },
    ])("leaves the name and value cells of a $verdict row unstyled", ({ metric }) => {
      const row = lineContaining(renderReport(colorfulResult()), metric);

      // Every cell but the last: the metric name and the two value columns.
      expect(cellsOf(row).slice(0, -1).join("│")).not.toContain("\x1b[");
    });

    it.each([
      { verdict: "improved", metric: "faster/time" },
      { verdict: "regressed", metric: "slower/time" },
    ])("leaves the $verdict row without an end-to-end dim", ({ metric }) => {
      const row = lineContaining(renderReport(colorfulResult()), metric);

      expect(row).not.toMatch(DIMMED_LINE);
    });

    it("emboldens the geomean figure", () => {
      const report = renderReport(colorfulResult());

      expect(stylesAt(lineContaining(report, "geomean"), "-5.8%")).toContain("1");
    });

    it.each([
      { position: "run header", anchor: "gymrat compare" },
      { position: "column header", anchor: "metric  " },
    ])("emboldens and underlines the variant names in the $position", ({ anchor }) => {
      const line = lineContaining(renderReport(colorfulResult()), anchor);

      expect.soft(stylesAt(line, "main")).toStrictEqual(["1", "4"]);
      expect(stylesAt(line, "perf/faster-decode")).toStrictEqual(["1", "4"]);
    });

    it("emboldens and underlines the baseline name inside the delta column header", () => {
      const header = lineContaining(renderReport(colorfulResult()), "metric  ");

      const deltaCell = deltaCellOf(header);

      expect(stylesAt(deltaCell, "main")).toStrictEqual(["1", "4"]);
    });

    // A short branch name can hide inside the "vs " prose that introduces it,
    // so the styled span has to be the name after the prefix, not the prefix.
    it.each([{ baseline: "v" }, { baseline: "s" }, { baseline: "vs" }])(
      "emboldens the '$baseline' baseline after the vs prefix, leaving the prefix plain",
      ({ baseline }) => {
        const result = createComparisonResult({
          baselineLabel: baseline,
          metrics: { "a/time": signedRankMetric({ verdict: "improved", delta: -10, unit: "ns" }) },
        });

        const deltaCell = deltaCellOf(lineContaining(renderReport(result), "metric  "));

        expect(deltaCell).toContain(`vs \x1b[1m\x1b[4m${baseline}\x1b[24m\x1b[22m`);
      },
    );

    it("leaves the rest of the column header row unstyled", () => {
      const header = lineContaining(renderReport(colorfulResult()), "metric  ");

      expect.soft(header).not.toMatch(/^\x1b\[1m/);
      expect(stylesAt(header, "metric")).toStrictEqual([]);
    });

    it("emboldens 'gymrat compare' in the report header", () => {
      const header = lineContaining(renderReport(colorfulResult()), "gymrat compare");

      expect(stylesAt(header, "gymrat compare")).toContain("1");
    });

    it("dims each · separator in the report header", () => {
      const header = lineContaining(renderReport(colorfulResult()), "gymrat compare");

      expect(stylesAt(header, "·")).toContain("2");
    });

    /**
     * The guard against dimming separators by rewriting the finished header.
     *
     * A `·` is legal in a branch name and in an adapter name, so a pass that
     * replaces every `·` in the styled line splices dim codes into the middle of
     * a variant name's own style span.
     */
    it("leaves a · inside a variant name out of the separator dimming", () => {
      const result = createComparisonResult({
        baselineLabel: "main·1",
        candidates: [createCandidate({ label: "perf·2" })],
      });

      const header = lineContaining(renderReport(result), "gymrat compare");

      // cspell:disable-next-line — ANSI escape digits abut the branch name
      expect.soft(header).toContain("\x1b[1m\x1b[4mmain·1\x1b[24m\x1b[22m");
      // cspell:disable-next-line
      expect(header).toContain("\x1b[1m\x1b[4mperf·2\x1b[24m\x1b[22m");
    });

    it("leaves a · inside the adapter name out of the separator dimming", () => {
      const result = createComparisonResult({ adapter: "metric·lines" });

      const header = lineContaining(renderReport(result), "gymrat compare");

      expect(header).toContain("adapter: metric·lines");
    });

    it.each([
      { label: "improved", glyph: "✓", code: "32", color: "green" },
      { label: "regressed", glyph: "✗", code: "31", color: "red" },
      { label: "unstable", glyph: "≈", code: "33", color: "yellow" },
    ])("styles the non-zero $label tally $color in the verdict summary", ({ glyph, code }) => {
      const summary = lineContaining(renderReport(colorfulResult()), "improved");

      expect(stylesAt(summary, glyph)).toContain(code);
    });

    it.each([{ glyph: "✗" }, { glyph: "=" }])(
      "dims the zero-count $glyph segment in the verdict summary",
      ({ glyph }) => {
        const result = createComparisonResult({
          metrics: {
            "faster/time": signedRankMetric({ verdict: "improved", delta: -10, unit: "ns" }),
          },
        });
        const summary = lineContaining(renderReport(result), "improved");

        expect(stylesAt(summary, glyph)).toContain("2");
      },
    );

    it("paints the non-zero identical tally cyan in the verdict summary", () => {
      const result = createComparisonResult({
        metrics: {
          "tied/heap": bandMetric({ verdict: "no-signal", delta: -0.5, n: 10, usableN: 0 }),
        },
      });
      const summary = lineContaining(renderReport(result), "identical");

      expect(stylesAt(summary, "=")).toContain("36");
    });

    it("dims the within-noise segment regardless of count in the verdict summary", () => {
      const summary = lineContaining(renderReport(colorfulResult()), "within noise");

      expect(stylesAt(summary, "~")).toContain("2");
    });

    it("emboldens the highlights heading", () => {
      const lines = renderReport(colorfulResult()).split("\n");
      const heading = lines.find((line) => stripAnsi(line) === "highlights");

      expect(heading).toMatch(/^\x1b\[1m/);
    });

    it("styles the improved highlight glyph and delta green", () => {
      const result = createComparisonResult({
        metrics: {
          "faster/time": signedRankMetric({ verdict: "improved", delta: -17.5, unit: "ns" }),
        },
      });
      const highlights = highlightLines(renderReport(result));
      const entry = highlights[0];
      if (entry === undefined) throw new Error("no highlight entry");

      expect.soft(stylesAt(entry, "✓")).toContain("32");
      expect(stylesAt(entry, "-17.5%")).toContain("32");
    });

    it("styles the regressed highlight glyph and delta red", () => {
      const result = createComparisonResult({
        metrics: {
          "slower/time": signedRankMetric({ verdict: "regressed", delta: 2.2, unit: "ns" }),
        },
      });
      const highlights = highlightLines(renderReport(result));
      const entry = highlights[0];
      if (entry === undefined) throw new Error("no highlight entry");

      expect.soft(stylesAt(entry, "✗")).toContain("31");
      expect(stylesAt(entry, "+2.2%")).toContain("31");
    });

    it("styles the unstable highlight glyph and word yellow", () => {
      const result = createComparisonResult({
        metrics: {
          "jittery/time": bandMetric({ verdict: "unstable", delta: 5, noisePct: 30 }),
        },
      });
      const highlights = highlightLines(renderReport(result));
      const entry = highlights[0];
      if (entry === undefined) throw new Error("no highlight entry");

      expect.soft(stylesAt(entry, "≈")).toContain("33");
      expect(stylesAt(entry, "unstable")).toContain("33");
    });

    it("styles the verdict word rather than a metric name that spells it too", () => {
      const result = createComparisonResult({
        metrics: {
          "unstable-parse/time": bandMetric({ verdict: "unstable", delta: 5, noisePct: 30 }),
        },
      });
      const entry = highlightLines(renderReport(result))[0];
      if (entry === undefined) throw new Error("no highlight entry");

      expect.soft(stripAnsi(entry).trim()).toBe("≈ unstable-parse/time  unstable  noise ±30.0%");
      expect(stylesAt(entry, "unstable", { last: true })).toContain("33");
    });

    it("dims the evidence suffixes in highlight entries", () => {
      const result = createComparisonResult({
        metrics: {
          "cheaper/heap": exactMetric({ delta: -7.9 }),
          "jittery/time": bandMetric({ verdict: "unstable", delta: 5, noisePct: 30 }),
        },
      });
      const highlights = highlightLines(renderReport(result));
      const exactEntry = highlights.find((line) => line.includes("cheaper/heap"));
      const unstableEntry = highlights.find((line) => line.includes("jittery/time"));
      if (exactEntry === undefined) throw new Error("cheaper/heap entry not found");
      if (unstableEntry === undefined) throw new Error("jittery/time entry not found");

      expect.soft(stylesAt(exactEntry, "(exact)")).toContain("2");
      expect(stylesAt(unstableEntry, "noise")).toContain("2");
    });

    it("dims the futility note closing the highlights", () => {
      const result = createComparisonResult({
        metrics: {
          "jittery/time": bandMetric({ verdict: "unstable", delta: 5, noisePct: 30 }),
        },
      });
      const note = lineContaining(renderReport(result), "won't stabilize");

      expect(stylesAt(note, "unstable metrics")).toContain("2");
    });

    it("dims the verdict method description", () => {
      const result = createComparisonResult({
        metrics: {
          "a/time": signedRankMetric({ verdict: "improved", delta: -10, unit: "ns" }),
        },
      });
      const method = lineContaining(renderReport(result, { verbose: true }), "Wilcoxon");

      expect(method).toMatch(DIMMED_LINE);
    });

    /** A single no-signal metric that fell back to the band, so the hint and its footer render. */
    function bandFallbackResult(): ComparisonResult {
      return createComparisonResult({
        metrics: { "a/time": bandMetric({ verdict: "no-signal", delta: -5 }) },
      });
    }

    it("dims the noise-band description", () => {
      const band = lineContaining(
        renderReport(bandFallbackResult(), { verbose: true }),
        "noise band",
      );

      expect(band).toMatch(DIMMED_LINE);
    });

    it("styles the Hint word yellow and underlined", () => {
      const hint = lineContaining(renderReport(bandFallbackResult()), "Hint");

      expect.soft(stylesAt(hint, "Hint")).toContain("33");
      expect(stylesAt(hint, "Hint")).toContain("4");
    });

    it("styles the hint label colon yellow without underlining it", () => {
      const hint = lineContaining(renderReport(bandFallbackResult()), "Hint");

      expect.soft(stylesAt(hint, ":")).toContain("33");
      expect(stylesAt(hint, ":")).not.toContain("4");
    });

    it("renders the hint sentence text plain in colored mode", () => {
      const hint = lineContaining(renderReport(bandFallbackResult()), "Hint");
      const afterLabel = hint.slice(hint.indexOf("re-run"));

      expect(afterLabel).not.toContain("\x1b[2m");
    });

    it("renders the hint line entirely plain when color is off", () => {
      vi.stubEnv("FORCE_COLOR", undefined);
      vi.stubEnv("NO_COLOR", "1");

      const hint = lineContaining(renderReport(bandFallbackResult()), "Hint:");

      expect(hint).not.toContain("\x1b[");
    });

    it("dims the band annotation on improved and regressed rows", () => {
      const report = renderReport(colorfulResult());

      expect.soft(stylesAt(lineContaining(report, "faster/time"), "±2.5%")).toContain("2");
      expect(stylesAt(lineContaining(report, "slower/time"), "±2.5%")).toContain("2");
    });

    it("places the verdict color on the delta, not the noise band, when both share a digit sequence", () => {
      const result = createComparisonResult({
        metrics: {
          "collision/time": bandMetric({
            delta: 0,
            noisePct: 10,
            n: 10,
            usableN: 0,
          }),
        },
      });

      const row = lineContaining(renderReport(result), "collision/time");

      // The delta "0.0%" and the noise band "±10.0%" both contain "0.0%".
      // The identical verdict color (cyan/36) must wrap the delta, and
      // dim (2) the band — not the other way around.
      expect.soft(stylesAt(row, "0.0%")).toContain("36");
      expect(stylesAt(row, "±10.0%")).toContain("2");
    });

    // The band behind a single pair is the noise floor constant, so a styled
    // report has no more business printing it than a plain one does.
    it("leaves the floor band off an inconclusive row", () => {
      const row = lineContaining(renderReport(colorfulResult()), "single-pair/time");

      expect(stripAnsi(cellsOf(row).at(-1) ?? "")).not.toContain("±");
    });

    describe("when the color option overrides the environment", () => {
      it("leaves the report unstyled when color is false, despite FORCE_COLOR", () => {
        const output = renderReport(colorfulResult(), { color: false });

        expect(output).not.toContain("\x1b[");
      });

      it("styles the report when color is true, despite NO_COLOR", () => {
        vi.stubEnv("FORCE_COLOR", undefined);
        vi.stubEnv("NO_COLOR", "1");

        const output = renderReport(colorfulResult(), { color: true });

        expect(output).toContain("\x1b[");
      });
    });
  });
});

describe("renderMeasureReport", () => {
  describe("when rendering the run header", () => {
    it("names the target, the sample count and the adapter", () => {
      const result = createMeasurementResult({
        label: "experiment",
        samples: 10,
        adapter: "mitata",
      });

      const output = renderMeasureReport(result);

      expect(output).toContain("gymrat measure · experiment · 10 samples · adapter: mitata");
    });

    it.each([
      { samples: 1, expected: "· 1 sample ·" },
      { samples: 2, expected: "· 2 samples ·" },
    ])("matches the sample noun to a count of $samples", ({ samples, expected }) => {
      const output = renderMeasureReport(createMeasurementResult({ samples }));

      expect(output).toContain(expected);
    });
  });

  describe("when every metric shares one kind", () => {
    /** A flat single-kind run of two metrics measured in nanoseconds. */
    function flatMeasurement(): MeasurementResult {
      return createMeasurementResult({
        metrics: {
          "decode/time": measuredMetric({ median: 100, spread: 1, unit: "ns" }),
          "encode/time": measuredMetric({ median: 2048, spread: 2, unit: "ns" }),
        },
      });
    }

    it("draws one flat table, headed by the metric column and the target", () => {
      expect(tableRegion(renderMeasureReport(flatMeasurement()))).toStrictEqual([
        "gymrat measure · main · 10 samples · adapter: mitata",
        "metric",
        "<rule>",
        "decode/time",
        "encode/time",
      ]);
    });

    it("labels the value column with the target's own label", () => {
      const headerLine = lineStartingWith(renderMeasureReport(flatMeasurement()), "metric");

      expect(cellsOf(headerLine).map((cell) => cell.trim())).toStrictEqual(["metric", "main"]);
    });

    it("states each metric's median in its own unit", () => {
      const rows = tableRows(renderMeasureReport(flatMeasurement())).slice(1);

      expect(rows.map((row) => cellsOf(row).map((cell) => cell.trim()))).toStrictEqual([
        ["decode/time", "100ns ± 1%"],
        ["encode/time", "2.0µs ± 2%"],
      ]);
    });
  });

  describe("when rendering a metric row", () => {
    it.each([
      { case: "states the spread behind the median", spread: 1, expected: "100ns ± 1%" },
      { case: "drops the ± when nothing measured a spread", spread: undefined, expected: "100ns" },
    ])("$case", ({ spread, expected }) => {
      const result = createMeasurementResult({
        metrics: { "decode/time": measuredMetric({ median: 100, spread, unit: "ns" }) },
      });

      const row = lineStartingWith(renderMeasureReport(result), "decode/time");

      expect(cellsOf(row).at(-1)?.trim()).toBe(expected);
    });
  });

  describe("when there is nothing to compare against", () => {
    it("carries no delta, verdict, geomean or highlight anywhere in the report", () => {
      const output = stripAnsi(renderMeasureReport(twoKindMeasurement()));

      expect.soft(output).not.toContain("geomean");
      expect.soft(output).not.toContain("highlights");
      expect.soft(output).not.toContain("vs ");
      expect(output).not.toMatch(/[✓✗≈~?]/);
    });
  });

  describe("when the run spans more than one metric kind", () => {
    it("gives each kind its own titled section, closed by no aggregate at all", () => {
      expect(tableRegion(renderMeasureReport(twoKindMeasurement()))).toStrictEqual([
        "gymrat measure · main · 10 samples · adapter: mitata",
        "",
        "<border>",
        "time",
        "<rule>",
        "entity",
        "alive_check",
        "spawn",
        "",
        "warmup",
        "",
        "informational — gating off (config: kinds.memory.gating = false)",
        "<border>",
        "memory",
        "<rule>",
        "encode",
      ]);
    });

    it.each([
      { placement: "indented under its group, stripped of the group prefix", row: "  alive_check" },
      { placement: "at the margin under its bare short name", row: "warmup" },
    ])("names a metric row $placement", ({ row }) => {
      const line = lineStartingWith(renderMeasureReport(twoKindMeasurement()), row);

      expect(cellsOf(line)[0]?.trimEnd()).toBe(row);
    });

    it.each([
      {
        source: "the kind-level config entry",
        makeResult: () => twoKindMeasurement({ configKinds: { memory: { gating: false } } }),
        expected: "informational — gating off (config: kinds.memory.gating = false)",
      },
      {
        source: "per-metric overrides alone",
        makeResult: (): MeasurementResult => {
          const { configKinds: _, ...rest } = twoKindMeasurement();
          return rest;
        },
        expected: "informational — gating off",
      },
    ])("credits $source for a non-gating kind's informational tag", ({ makeResult, expected }) => {
      const report = renderMeasureReport(makeResult());

      expect(lineContaining(report, "informational")).toBe(expected);
    });

    it("lines every section's columns up with the first section's header", () => {
      const report = renderMeasureReport(twoKindMeasurement());
      const offsets = separatorOffsets(lineStartingWith(report, "time"));

      expect.soft(separatorOffsets(lineStartingWith(report, "memory"))).toStrictEqual(offsets);
      expect.soft(separatorOffsets(lineStartingWith(report, "entity "))).toStrictEqual(offsets);
      expect(separatorOffsets(lineStartingWith(report, "  alive_check"))).toStrictEqual(offsets);
    });

    it("states every kind's medians in that kind's own unit", () => {
      const report = renderMeasureReport(twoKindMeasurement());

      expect.soft(cellsOf(lineStartingWith(report, "  spawn")).at(-1)?.trim()).toBe("104ns ± 1%");
      expect(cellsOf(lineStartingWith(report, "encode")).at(-1)?.trim()).toBe("93B ± 1%");
    });
  });

  describe("when reporting worktree cleanup", () => {
    it("says nothing at all when the run left nothing behind", () => {
      const output = renderMeasureReport(createMeasurementResult({ worktreesRemoved: 0 }));

      expect(output).not.toContain("worktree");
    });

    it("closes the report with the left-behind worktrees and the prune failure", () => {
      const result = createMeasurementResult({
        worktreesRemoved: 0,
        worktreesLeftBehind: [{ dir: "/tmp/gymrat-abc", error: "is locked" }],
        worktreePruneError: "fatal: not a git repository",
      });

      const lines = renderMeasureReport(result).split("\n");

      expect.soft(lines.at(-3)).toContain("0 worktrees removed · 1 left behind");
      expect.soft(lines.at(-2)).toBe("  left behind: /tmp/gymrat-abc (is locked)");
      expect(lines.at(-1)).toBe("  worktree prune failed: fatal: not a git repository");
    });
  });

  /**
   * Byte-level pins on the whole rendered measure report.
   *
   * Same rationale as the comparison-report goldens: substring and offset
   * assertions let column widths drift silently, so the snapshots catch any
   * user-visible output change that those tests wouldn't.
   */
  describe("when rendering a whole report", () => {
    it("matches the recorded bytes for a two-kind run", async () => {
      await expect(renderMeasureReport(twoKindMeasurement())).toMatchFileSnapshot(
        "../fixtures/measure-two-kind.golden.txt",
      );
    });

    it("matches the recorded bytes with ANSI color", async () => {
      vi.stubEnv("FORCE_COLOR", "1");

      await expect(renderMeasureReport(twoKindMeasurement())).toMatchFileSnapshot(
        "../fixtures/measure-two-kind-color.golden.txt",
      );
    });

    it("matches the recorded bytes when the cleanup footer is present", async () => {
      const result = twoKindMeasurement({
        worktreesRemoved: 1,
        worktreesLeftBehind: [{ dir: "/tmp/gymrat-abc", error: "is locked" }],
      });

      await expect(renderMeasureReport(result)).toMatchFileSnapshot(
        "../fixtures/measure-cleanup-footer.golden.txt",
      );
    });

    it("matches the recorded bytes for the cleanup footer with ANSI color", async () => {
      vi.stubEnv("FORCE_COLOR", "1");

      const result = twoKindMeasurement({
        worktreesRemoved: 1,
        worktreesLeftBehind: [{ dir: "/tmp/gymrat-abc", error: "is locked" }],
      });

      await expect(renderMeasureReport(result)).toMatchFileSnapshot(
        "../fixtures/measure-cleanup-footer-color.golden.txt",
      );
    });
  });

  describe("when rendering with color", () => {
    beforeEach(() => {
      vi.stubEnv("FORCE_COLOR", "1");
    });

    it("names the target in the same style the comparison header gives a variant", () => {
      const report = renderMeasureReport(twoKindMeasurement());

      expect(stylesAt(lineContaining(report, "gymrat measure"), "main")).toStrictEqual(["1", "4"]);
    });

    it("emboldens the kind name in the section header and dims the informational tag", () => {
      const report = renderMeasureReport(twoKindMeasurement());
      const header = report
        .split("\n")
        .find((line) => line.includes("│") && stripAnsi(line).trimStart().startsWith("memory"));
      if (header === undefined) {
        throw new Error(`no memory header in report:\n${report}`);
      }

      expect.soft(stylesAt(header, "memory")).toStrictEqual(["1"]);
      expect(stylesAt(lineContaining(report, "informational"), "informational")).toStrictEqual([
        "2",
      ]);
    });

    it("leaves every column separator in the default color, whatever style its row carries", () => {
      const rows = renderMeasureReport(twoKindMeasurement())
        .split("\n")
        .filter((line) => line.includes("│"));

      const inherited = rows.filter((row) =>
        separatorStyles(row).some((styles) => styles.length > 0),
      );

      expect(inherited).toStrictEqual([]);
    });

    it("leaves the report unstyled when NO_COLOR is set", () => {
      vi.stubEnv("FORCE_COLOR", undefined);
      vi.stubEnv("NO_COLOR", "1");

      expect(renderMeasureReport(twoKindMeasurement())).not.toContain("\x1b[");
    });

    it.each([
      { setting: "off", color: false, styled: false },
      { setting: "on", color: true, styled: true },
    ])("overrides the environment when the color option is $setting", ({ color, styled }) => {
      vi.stubEnv("FORCE_COLOR", undefined);
      vi.stubEnv("NO_COLOR", "1");

      const output = renderMeasureReport(twoKindMeasurement(), { color });

      expect(output.includes("\x1b[")).toBe(styled);
    });
  });
});
