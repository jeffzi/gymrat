import { stripVTControlCharacters as stripAnsi } from "node:util";

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { renderMeasureReport, renderReport } from "../../src/report/text.js";
import type { ComparisonResult, MeasurementResult, ReportOptions } from "../../src/report/types.js";
import {
  bandMetric,
  bandVerdict,
  createCandidate,
  createComparisonResult,
  exactMetric,
  exactVerdict,
  geomeanOf,
  groupedComparison,
  kindMetric,
  memoryKind,
  metricMeta,
  multiCandidateResult,
  nWayKindMetric,
  nWayMetric,
  otherKind,
  signedRankMetric,
  signedRankVerdict,
  singleSampleResult,
  timeKind,
  twoKindMetrics,
  twoKindResult,
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

/**
 * The part of `cell` ahead of its verdict `glyph` — the value and its padding.
 *
 * A candidate column packs a value and a verdict into one cell, so this is the
 * half that carries no verdict of its own and must stay unstyled. The escape
 * sequences opening the verdict's own style sit right in front of the glyph, so
 * they are trimmed off the tail rather than counted against the value.
 */
function valuePartOf(cell: string, glyph: string): string {
  const index = cell.indexOf(glyph);
  if (index === -1) {
    throw new Error(`no ${glyph} in cell: ${JSON.stringify(cell)}`);
  }
  return cell.slice(0, index).replace(/(?:\x1b\[\d+m)*$/, "");
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

/** The last rendered table row of a report — the row the table closes on. */
function lastTableRow(report: string): string {
  const row = tableRows(report).at(-1);
  if (row === undefined) {
    throw new Error(`no table rows in report:\n${report}`);
  }
  return row;
}

/**
 * The rule under a table header, drawn to the full width of the columns.
 *
 * Every row of the table is laid out from the same widths, so the rule is the
 * width a correctly sized row cannot exceed.
 */
function columnRule(report: string): string {
  const rule = report.split("\n").find((line) => /^─+┼/.test(stripAnsi(line)));
  if (rule === undefined) {
    throw new Error(`no column rule in report:\n${report}`);
  }
  return stripAnsi(rule);
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
  afterEach(() => {
    vi.unstubAllEnvs();
  });

  describe("when rendering the run header", () => {
    it("names the baseline's role, both variants, the sample count and the adapter", () => {
      const result = createComparisonResult({
        baselineLabel: "main",
        candidates: [createCandidate({ label: "experiment" })],
        samples: 10,
        adapter: "mitata",
      });

      const output = renderReport(result);

      expect(output).toContain(
        "gymrat compare · baseline main ↔ experiment · 10 paired samples · adapter: mitata",
      );
    });

    it.each([
      { samples: 1, expected: "· 1 paired sample ·" },
      { samples: 2, expected: "· 2 paired samples ·" },
    ])("matches the sample noun to a count of $samples", ({ samples, expected }) => {
      const output = renderReport(createComparisonResult({ samples }));

      expect(output).toContain(expected);
    });

    it("lists every candidate against the one baseline", () => {
      const result = createComparisonResult({
        baselineLabel: "main",
        candidates: [
          createCandidate({ label: "candidate-a" }),
          createCandidate({ label: "candidate-b" }),
        ],
      });

      const output = renderReport(result);

      expect(output).toContain("gymrat compare · baseline main ↔ candidate-a, candidate-b ·");
    });

    it("opens with a header override instead of the compare header, when one is given", () => {
      const result = createComparisonResult();

      const output = renderReport(result, { header: "iteration 3 · experiment vs baseline" });

      expect.soft(stripAnsi(output).split("\n")[0]).toBe("iteration 3 · experiment vs baseline");
      expect(stripAnsi(output)).not.toContain("gymrat compare");
    });
  });

  describe("when rendering the table header", () => {
    it("labels the value columns with the two target labels and the delta with the baseline", () => {
      const result = createComparisonResult();
      const output = renderReport(result);

      const headerLine = lineStartingWith(output, "metric");

      expect
        .soft(cellsOf(headerLine).map((cell) => cell.trim()))
        .toStrictEqual(["metric", "main", "perf/faster-decode", "vs main"]);
      expect.soft(output).toContain("gymrat compare");
      expect(output).toContain("geomean");
    });
  });

  describe("when a variant label overflows the display width", () => {
    it("truncates the label wherever it prints, leaving metric names whole", () => {
      const result = createComparisonResult({
        baselineLabel: "main",
        candidates: [createCandidate({ label: "feature/entity-spawn-fastpath" })],
        metrics: {
          "decode/an-extremely-long-metric-name/time": signedRankMetric({
            verdict: "improved",
            delta: -10,
            unit: "ns",
          }),
        },
      });

      const output = renderReport(result);

      expect.soft(output).not.toContain("feature/entity-spawn-fastpath");
      expect.soft(output).toContain("feature/en…-fastpath");
      expect(output).toContain("decode/an-extremely-long-metric-name/time");
    });
  });

  describe("when rendering a metric row", () => {
    it.each([
      {
        desc: "pairs the improved glyph with the delta and the noise band",
        verdict: "improved" as const,
        delta: -17.5,
        noisePct: 2.5,
        expected: "✓  -17.5%  ±2.5%",
      },
      {
        desc: "prints the word unstable alone, with no band trailing it",
        verdict: "unstable" as const,
        delta: -50,
        noisePct: 30,
        expected: "≈  unstable",
      },
    ])("$desc", ({ verdict, delta, noisePct, expected }) => {
      const result = createComparisonResult({
        metrics: { "decode/time": signedRankMetric({ verdict, delta, noisePct, unit: "ns" }) },
      });

      const row = lineStartingWith(renderReport(result), "decode/time");

      expect(cellsOf(row).at(-1)?.trim()).toBe(expected);
    });

    it("drops the per-row pair count and p-value now the footer carries them", () => {
      const result = createComparisonResult({
        metrics: { "decode/time": signedRankMetric({ verdict: "improved", delta: -10, p: 0.002 }) },
      });

      const row = lineStartingWith(renderReport(result), "decode/time");

      expect.soft(row).not.toContain("n=");
      expect(row).not.toContain("p=");
    });

    it("shows only the measured side of a one-sided metric, with no verdict", () => {
      const result = createComparisonResult({
        metrics: {
          "old-only/time": {
            baselineMedian: 2048,
            baselineSpread: 2,
            candidates: [{}],
            meta: metricMeta("old-only/time", { gating: false, unit: "ns" }),
          },
        },
      });

      const row = lineStartingWith(renderReport(result), "old-only/time");

      expect.soft(row).toContain("2.0µs ± 2%");
      // A trailing separator means the candidate and verdict cells were trimmed away.
      expect(row.endsWith("│")).toBe(true);
    });

    it("keeps the glyph when the delta is undefined arithmetic", () => {
      const result = createComparisonResult({
        metrics: {
          "nan-delta/count": {
            baselineMedian: 0,
            candidates: [
              {
                median: 120,
                verdict: exactVerdict({ delta: Number.NaN }),
              },
            ],
            meta: metricMeta("nan-delta/count", { exact: true }),
          },
        },
      });

      const row = lineStartingWith(renderReport(result), "nan-delta/count");

      expect(cellsOf(row).at(-1)?.trim()).toBe("~");
    });

    it("states a spread wider than the median in absolute units", () => {
      const result = createComparisonResult({
        metrics: {
          "jittery/heap": signedRankMetric({
            verdict: "unstable",
            delta: 5,
            baselineMedian: 5,
            baselineSpread: 7620,
            unit: "bytes",
          }),
        },
      });

      const row = lineStartingWith(renderReport(result), "jittery/heap");

      expect(cellsOf(row)[1]?.trim()).toBe("5B ± 381B");
    });
  });

  describe("when a metric was paired on fewer rounds than the run", () => {
    it.each([
      {
        method: "signed-rank",
        metric: signedRankMetric({ verdict: "improved", delta: -10, n: 8 }),
        expected: "✓  -10.0%  ±2.5%  n=8",
      },
      {
        method: "band",
        metric: bandMetric({ verdict: "improved", delta: -5, n: 4 }),
        expected: "✓  -5.0%  ±2.5%  n=4",
      },
      {
        method: "exact",
        metric: exactMetric({ delta: -7.9, n: 6, unit: "ns" }),
        expected: "✓  -7.9%  n=6",
      },
    ])("annotates its $method verdict with the pair count behind it", ({ metric, expected }) => {
      const result = createComparisonResult({
        samples: 10,
        metrics: { "decode/time": metric },
      });

      const row = lineStartingWith(renderReport(result), "decode/time");

      expect(cellsOf(row).at(-1)?.trim()).toBe(expected);
    });
  });

  describe("when rendering columns of differing widths", () => {
    it("right-aligns the value cells and keeps the metric names left-aligned", () => {
      const result = createComparisonResult({
        metrics: {
          short: signedRankMetric({
            verdict: "improved",
            delta: -50,
            baselineMedian: 914,
            unit: "ns",
          }),
          "very-long-metric-name": signedRankMetric({
            verdict: "improved",
            delta: -10,
            baselineMedian: 49152,
            unit: "bytes",
          }),
        },
      });

      const report = renderReport(result);
      const shortCell = cellsOf(lineStartingWith(report, "short"))[1] ?? "";
      const longCell = cellsOf(lineStartingWith(report, "very-long-metric-name"))[1] ?? "";

      // Right-aligned cells end at the same column; left-aligned ones would not.
      expect.soft(shortCell.trimEnd()).toHaveLength(longCell.trimEnd().length);
      expect.soft(shortCell.trim()).toBe("914ns ± 1%");
      expect(longCell.trim()).toBe("49.2KB ± 1%");
    });

    it("lines the column separators up across header, metric rows and geomean", () => {
      const result = createComparisonResult({
        metrics: {
          "a/time": signedRankMetric({ verdict: "improved", delta: -5, unit: "ns" }),
          "a-much-longer-metric/time": signedRankMetric({
            verdict: "improved",
            delta: -5,
            baselineMedian: 100000,
            unit: "ns",
          }),
        },
        candidates: [
          createCandidate({
            kinds: [otherKind(-5, 2)],
          }),
        ],
      });

      const report = renderReport(result);
      const headerOffsets = separatorOffsets(lineStartingWith(report, "metric"));

      expect
        .soft(separatorOffsets(lineStartingWith(report, "a/time")))
        .toStrictEqual(headerOffsets);
      expect
        .soft(separatorOffsets(lineStartingWith(report, "a-much-longer-metric/time")))
        .toStrictEqual(headerOffsets);
      expect(separatorOffsets(lineStartingWith(report, "geomean"))).toStrictEqual(headerOffsets);
    });
  });

  describe("when aligning the value columns", () => {
    /** A run whose two metrics differ in how wide their value sub-fields print. */
    function valueResult(metrics: ComparisonResult["metrics"]): ComparisonResult {
      return createComparisonResult({ metrics });
    }

    it.each([
      {
        desc: "percentage spreads",
        metrics: {
          "first/metric": signedRankMetric({
            verdict: "improved",
            delta: -10,
            baselineMedian: 162000,
            baselineSpread: 9,
            unit: "ns" as const,
          }),
          "second/metric": signedRankMetric({
            verdict: "improved",
            delta: -10,
            baselineMedian: 29200,
            baselineSpread: 12,
            unit: "ns" as const,
          }),
        },
        first: "162.0µs ±  9%",
        second: " 29.2µs ± 12%",
      },
      {
        desc: "an absolute spread beside a percentage one",
        metrics: {
          "first/metric": signedRankMetric({
            verdict: "improved",
            delta: -10,
            baselineMedian: 5,
            baselineSpread: 7620,
            unit: "bytes" as const,
          }),
          "second/metric": signedRankMetric({
            verdict: "improved",
            delta: -10,
            baselineMedian: 49152,
            baselineSpread: 1,
            unit: "bytes" as const,
          }),
        },
        first: "    5B ± 381B",
        second: "49.2KB ±   1%",
      },
    ])("stacks the magnitude, the ± and the spread of $desc", ({ metrics, first, second }) => {
      const report = renderReport(valueResult(metrics));

      const firstCell = cellsOf(lineStartingWith(report, "first/metric"))[1] ?? "";
      const secondCell = cellsOf(lineStartingWith(report, "second/metric"))[1] ?? "";

      expect.soft(firstCell).toContain(first);
      expect.soft(secondCell).toContain(second);
      expect(firstCell.indexOf("±")).toBe(secondCell.indexOf("±"));
    });

    it("keeps a magnitude with no spread of its own in the magnitude field", () => {
      const report = renderReport(
        valueResult({
          "first/metric": signedRankMetric({
            verdict: "improved",
            delta: -10,
            baselineMedian: 2048,
            baselineSpread: 2,
            unit: "ns",
          }),
          "second/metric": {
            baselineMedian: 120,
            candidates: [{ median: 120, verdict: exactVerdict() }],
            meta: metricMeta("second/metric", { exact: true }),
          },
        }),
      );

      const firstCell = cellsOf(lineStartingWith(report, "first/metric"))[1] ?? "";
      const secondCell = cellsOf(lineStartingWith(report, "second/metric"))[1] ?? "";

      // "2.0µs" and "120" end at the same offset; the ± field stays blank below.
      expect(firstCell.indexOf("2.0µs") + "2.0µs".length).toBe(
        secondCell.indexOf("120") + "120".length,
      );
    });
  });

  describe("when aligning the verdict column", () => {
    it("right-aligns every delta, the unsigned zero included, and pins the band's ±", () => {
      const result = createComparisonResult({
        metrics: {
          "regressed/time": signedRankMetric({
            verdict: "regressed",
            delta: 0.4,
            noisePct: 2.5,
            unit: "ns",
          }),
          "flat/time": signedRankMetric({
            verdict: "no-signal",
            delta: 0,
            noisePct: 100,
            unit: "ns",
          }),
          "improved/time": signedRankMetric({
            verdict: "improved",
            delta: -12.4,
            noisePct: 30,
            unit: "ns",
          }),
        },
      });

      const report = renderReport(result);
      const verdictOf = (metric: string): string | undefined =>
        cellsOf(lineStartingWith(report, metric)).at(-1)?.trim();

      expect.soft(verdictOf("regressed/time")).toBe("✗   +0.4%  ±  2.5%");
      expect.soft(verdictOf("flat/time")).toBe("~    0.0%  ±100.0%");
      expect(verdictOf("improved/time")).toBe("✓  -12.4%  ± 30.0%");
    });

    it("seats the word unstable in the delta slot without widening it for the other rows", () => {
      const result = createComparisonResult({
        metrics: {
          "improved/time": signedRankMetric({ verdict: "improved", delta: -12.4, unit: "ns" }),
          "jittery/time": signedRankMetric({
            verdict: "unstable",
            delta: -50,
            noisePct: 30,
            unit: "ns",
          }),
        },
      });

      const report = renderReport(result);

      expect
        .soft(cellsOf(lineStartingWith(report, "improved/time")).at(-1)?.trim())
        .toBe("✓  -12.4%  ±2.5%");
      expect(cellsOf(lineStartingWith(report, "jittery/time")).at(-1)?.trim()).toBe("≈  unstable");
    });
  });

  describe("when aligning the sub-fields of candidate columns", () => {
    /**
     * Two candidates whose columns differ in how wide their sub-fields print.
     *
     * Each column carries its own widest magnitude, spread and delta, so a
     * renderer that measured the table as a whole would pad one from the other.
     */
    function twoColumnResult(): ComparisonResult {
      return createComparisonResult({
        candidates: [
          createCandidate({ label: "candidate-a" }),
          createCandidate({ label: "candidate-b" }),
        ],
        metrics: {
          "decode/time": {
            baselineMedian: 1735,
            baselineSpread: 1,
            candidates: [
              {
                median: 1425,
                spread: 1,
                verdict: signedRankVerdict({ verdict: "improved", delta: -10, p: 0.002 }),
              },
              {
                median: 1698,
                spread: 2,
                verdict: signedRankVerdict({
                  verdict: "unstable",
                  delta: -2.1,
                  p: 0.32,
                  noisePct: 30,
                  noiseAbs: 30,
                }),
              },
            ],
            meta: metricMeta("decode/time", { unit: "ns" }),
          },
          "encode/time": {
            baselineMedian: 914,
            baselineSpread: 1,
            candidates: [
              {
                median: 934,
                spread: 1,
                verdict: signedRankVerdict({ verdict: "regressed", delta: 2.2, p: 0.002 }),
              },
              {
                median: 1200000,
                spread: 12,
                verdict: signedRankVerdict({ delta: -2.1, p: 0.32 }),
              },
            ],
            meta: metricMeta("encode/time", { unit: "ns" }),
          },
        },
      });
    }

    it.each([{ color: false }, { color: true }])(
      "stacks the value and verdict sub-fields within each candidate column (color: $color)",
      ({ color }) => {
        if (color) {
          vi.stubEnv("FORCE_COLOR", "1");
        }

        const report = stripAnsi(renderReport(twoColumnResult()));

        const decode = cellsOf(lineStartingWith(report, "decode/time")).map((cell) => cell.trim());
        const encode = cellsOf(lineStartingWith(report, "encode/time")).map((cell) => cell.trim());

        expect
          .soft(decode.slice(2))
          .toStrictEqual(["1.4µs ± 1%  ✓  -10.0%", "1.7µs ±  2%  ≈  unstable"]);
        expect(encode.slice(2)).toStrictEqual(["934ns ± 1%  ✗   +2.2%", "1.2ms ± 12%  ~  -2.1%"]);
      },
    );
  });

  describe("when rendering the geomean row", () => {
    it.each([
      { n: 4, expectedLabel: "geomean (4 stable metrics)" },
      { n: 1, expectedLabel: "geomean (1 stable metric)" },
    ])("renders the label as '$expectedLabel' when n=$n", ({ n, expectedLabel }) => {
      const result = createComparisonResult({
        metrics: { "a/time": signedRankMetric({ verdict: "improved", delta: -6 }) },
        candidates: [
          createCandidate({
            kinds: [otherKind(-5.8, n)],
          }),
        ],
      });

      const row = lineStartingWith(renderReport(result), "geomean");

      expect(cellsOf(row)[0]?.trim()).toBe(expectedLabel);
    });

    it("aligns its delta with the delta column above", () => {
      const result = createComparisonResult({
        metrics: { "a/time": signedRankMetric({ verdict: "improved", delta: -17.9 }) },
        candidates: [
          createCandidate({
            kinds: [otherKind(-6, 1)],
          }),
        ],
      });

      const report = renderReport(result);
      const metricCell = cellsOf(lineStartingWith(report, "a/time")).at(-1) ?? "";
      const geomeanCell = cellsOf(lineStartingWith(report, "geomean")).at(-1) ?? "";

      // Right-aligned deltas end at the same offset within the verdict column.
      expect(geomeanCell.indexOf("-6.0%") + "-6.0%".length).toBe(
        metricCell.indexOf("-17.9%") + "-17.9%".length,
      );
    });

    it("leaves the excluded metrics to the verdict summary", () => {
      const result = createComparisonResult({
        metrics: { "a/time": signedRankMetric({ verdict: "improved", delta: -6 }) },
        candidates: [
          createCandidate({
            kinds: [
              otherKind(0, 1, {
                excluded: [{ metric: "nan-delta/count", reason: "undefined-ratio" }],
              }),
            ],
          }),
        ],
      });

      const row = lineStartingWith(renderReport(result), "geomean");

      expect(row).not.toContain("excluded");
    });

    it("reports no stable metrics when every one was excluded", () => {
      const result = createComparisonResult({
        metrics: { "jittery/time": signedRankMetric({ verdict: "unstable", delta: -50 }) },
        candidates: [
          createCandidate({
            kinds: [
              otherKind(Number.NaN, 0, {
                excluded: [{ metric: "jittery/time", reason: "unstable" }],
              }),
            ],
          }),
        ],
      });

      const row = lineStartingWith(renderReport(result), "geomean");

      expect(cellsOf(row).map((cell) => cell.trim())).toStrictEqual([
        "geomean",
        "",
        "",
        "—  no stable metrics",
      ]);
    });
  });

  describe("when an aggregate carries a noise band", () => {
    it.each([
      { level: "group", label: "geomean · entity", expected: "-3.1%  ±1.5%" },
      { level: "kind", label: "geomean · time", expected: "-3.2%  ±2.0%" },
    ])("states the $level geomean's band behind its delta", ({ label, expected }) => {
      const row = lineStartingWith(renderReport(twoKindResult()), label);

      expect(deltaCellOf(row).trim()).toBe(expected);
    });

    it("states the flat table's geomean band behind its delta", () => {
      const result = createComparisonResult({
        metrics: { "faster/time": signedRankMetric({ verdict: "improved", delta: -17.5 }) },
        candidates: [createCandidate({ kinds: [otherKind(-5.8, 1, { band: 1.2 })] })],
      });

      const row = lineStartingWith(renderReport(result), "geomean");

      expect(deltaCellOf(row).trim()).toBe("-5.8%  ±1.2%");
    });

    it("prints the delta alone when the aggregate has no band of its own", () => {
      const row = lineStartingWith(renderReport(twoKindResult()), "geomean · memory");

      expect(deltaCellOf(row).trim()).toBe("-7.0%");
    });

    it("widens the verdict column for a band no metric row of its own carries", () => {
      const result = createComparisonResult({
        samples: 1,
        metrics: { "decode/time": bandMetric({ delta: -0.4, noisePct: 0.5, n: 1, unit: "ns" }) },
        candidates: [createCandidate({ kinds: [otherKind(-0.1, 1, { band: 0.5 })] })],
      });

      const report = renderReport(result);
      const row = lineStartingWith(report, "geomean");

      // A single pair leaves the metric row inconclusive and without a band, so
      // the propagated aggregate band is the only band the column has to hold.
      expect.soft(deltaCellOf(row).trim()).toBe("-0.1%  ±0.5%");
      expect(stripAnsi(row).trimEnd().length).toBeLessThanOrEqual(columnRule(report).length);
    });

    it("lines the aggregate band up with the metric rows' band column", () => {
      const report = renderReport(twoKindResult());

      expect(deltaCellOf(lineStartingWith(report, "geomean · time")).indexOf("±")).toBe(
        deltaCellOf(lineStartingWith(report, "  alive_check")).indexOf("±"),
      );
    });

    it("dims the aggregate band, as it dims the bands on the metric rows", () => {
      vi.stubEnv("FORCE_COLOR", "1");

      const line = lineContaining(renderReport(twoKindResult()), "geomean · time");

      expect(stylesAt(line, "±2.0%")).toContain("2");
    });

    it("leaves the compact multi-candidate aggregate cells band-free", () => {
      const result = createComparisonResult({
        candidates: [
          createCandidate({ label: "candidate-a", kinds: [otherKind(-10, 1, { band: 3 })] }),
          createCandidate({ label: "candidate-b", kinds: [otherKind(4, 1, { band: 3 })] }),
        ],
        metrics: {
          "decode/time": nWayMetric([
            { verdict: "improved", delta: -10, median: 90 },
            { verdict: "regressed", delta: 4, median: 104 },
          ]),
        },
      });

      const row = lineStartingWith(renderReport(result), "geomean");

      expect(cellsOf(row).map((cell) => cell.trim())).toStrictEqual([
        "geomean",
        "",
        "-10.0% · 1 stable metric",
        "+4.0% · 1 stable metric",
      ]);
    });
  });

  describe("when closing the table", () => {
    it("ends on the geomean row", () => {
      const result = createComparisonResult({
        metrics: { "a/time": signedRankMetric({ verdict: "improved", delta: -6 }) },
      });

      const row = lastTableRow(renderReport(result));

      expect(cellsOf(row)[0]?.trim()).toBe("geomean (10 stable metrics)");
    });
  });

  describe("when summarizing the verdicts", () => {
    it("counts every verdict class on one line below the table", () => {
      const result = createComparisonResult({
        metrics: {
          "faster/time": signedRankMetric({ verdict: "improved", delta: -10 }),
          "also-faster/time": signedRankMetric({ verdict: "improved", delta: -5 }),
          "slower/time": signedRankMetric({ verdict: "regressed", delta: 8 }),
          "jittery/time": signedRankMetric({ verdict: "unstable", delta: 5, noisePct: 300 }),
          "flat/time": signedRankMetric({ verdict: "no-signal", delta: 0.2 }),
        },
      });

      const report = renderReport(result);

      expect(lineStartingWith(report, "✓ 2 improved")).toBe(
        "✓ 2 improved   ✗ 1 regressed   ≈ 1 unstable   = 0 identical   ~ 1 within noise   ? 0 inconclusive",
      );
    });
  });

  describe("when ties starved the signed-rank test", () => {
    /** A run whose `tied/heap` metric moved too little to break any pair apart. */
    function identicalResult(): ComparisonResult {
      return createComparisonResult({
        metrics: {
          "faster/time": signedRankMetric({ verdict: "improved", delta: -10, unit: "ns" }),
          "tied/heap": bandMetric({ verdict: "no-signal", delta: -0.5, n: 10, usableN: 0 }),
        },
      });
    }

    it("marks the row identical rather than within noise", () => {
      const row = lineStartingWith(renderReport(identicalResult()), "tied/heap");

      // The row's delta is padded to the width of faster/time's -10.0%.
      expect(cellsOf(row).at(-1)?.trim()).toBe("=   -0.5%  ±2.5%");
    });

    it("tallies it apart from the metrics that are merely within noise", () => {
      const report = renderReport(identicalResult());

      expect(lineStartingWith(report, "✓ 1 improved")).toBe(
        "✓ 1 improved   ✗ 0 regressed   ≈ 0 unstable   = 1 identical   ~ 0 within noise   ? 0 inconclusive",
      );
    });

    it("leaves it out of the highlights", () => {
      const highlights = highlightLines(renderReport(identicalResult())).map((line) => line.trim());

      expect(highlights).toStrictEqual(["✓ faster/time  -10.0%"]);
    });

    it("says nothing more about it in the footer", () => {
      expect(renderReport(identicalResult())).not.toContain("close-to-identical");
    });

    it("marks the candidate cell identical in an N-way row", () => {
      const result = createComparisonResult({
        candidates: [
          createCandidate({ label: "candidate-a" }),
          createCandidate({ label: "candidate-b" }),
        ],
        metrics: {
          "tied/time": {
            baselineMedian: 100,
            baselineSpread: 1,
            candidates: [
              {
                median: 100,
                spread: 1,
                verdict: bandVerdict({ usableN: 0 }),
              },
              {
                median: 90,
                spread: 1,
                verdict: signedRankVerdict({ verdict: "improved", delta: -10, p: 0.002 }),
              },
            ],
            meta: metricMeta("tied/time", { unit: "ns" }),
          },
        },
      });

      const row = lineStartingWith(renderReport(result), "tied/time");

      expect(cellsOf(row).map((cell) => cell.trim())).toStrictEqual([
        "tied/time",
        "100ns ± 1%",
        "100ns ± 1%  =  -0.5%",
        "90ns ± 1%  ✓  -10.0%",
      ]);
    });
  });

  describe("when every verdict rests on a single pair", () => {
    it("marks the row inconclusive and drops the floor band behind it", () => {
      const row = lineStartingWith(renderReport(singleSampleResult()), "decode/time");

      expect(cellsOf(row).at(-1)?.trim()).toBe("?  -0.4%");
    });

    it("tallies the metrics in their own bucket rather than within noise", () => {
      const report = renderReport(singleSampleResult());

      expect(lineStartingWith(report, "✓ 0 improved")).toBe(
        "✓ 0 improved   ✗ 0 regressed   ≈ 0 unstable   = 0 identical   ~ 0 within noise   ? 2 inconclusive",
      );
    });

    it("still hints at the longer run that would settle the question", () => {
      expect(renderReport(singleSampleResult())).toContain(
        "Hint: re-run with --samples 6 or more for statistical verdicts",
      );
    });
  });

  describe("when rendering the highlights block", () => {
    it("carries the glyph, the delta and the evidence for each highlighted metric", () => {
      const result = createComparisonResult({
        metrics: {
          "slower/time": signedRankMetric({ verdict: "regressed", delta: 2.2, p: 0.002 }),
          "cheaper/heap": exactMetric({ delta: -7.9 }),
          "jittery/time": bandMetric({ verdict: "unstable", delta: 5, noisePct: 30 }),
        },
      });

      const highlights = highlightLines(renderReport(result)).map((line) => line.trim());

      expect(highlights).toStrictEqual([
        "✗ slower/time    +2.2%",
        "✓ cheaper/heap   -7.9%  (exact)",
        "≈ jittery/time  unstable  noise ±30.0%",
        "unstable metrics won't stabilize with more samples",
      ]);
    });

    it("states the noise in absolute units once it outgrows the median", () => {
      const result = createComparisonResult({
        metrics: {
          "jittery/heap": signedRankMetric({
            verdict: "unstable",
            delta: 5,
            baselineMedian: 5,
            noisePct: 7620,
            noiseAbs: 381,
            unit: "bytes",
          }),
        },
      });

      const highlights = highlightLines(renderReport(result)).map((line) => line.trim());

      expect(highlights[0]).toBe("≈ jittery/heap  unstable  ±381B noise on a 5B median");
    });

    it("omits the block when nothing improved, regressed or was unstable", () => {
      const result = createComparisonResult({
        metrics: { "flat/time": signedRankMetric({ verdict: "no-signal", delta: 0.2 }) },
      });

      const output = renderReport(result);

      expect(output).not.toContain("highlights");
    });
  });

  describe("when a --fail-on geomean condition would trip", () => {
    /** A two-kind run whose gating `time` kind regressed past a 2% threshold. */
    function trippingResult(): ComparisonResult {
      return twoKindResult({
        candidates: [
          createCandidate({
            kinds: [
              timeKind({ geomean: geomeanOf(3.1, 3), gatedGeomean: geomeanOf(3.1, 3) }),
              memoryKind(),
            ],
          }),
        ],
      });
    }

    it("echoes the tripped kind, its geomean and the condition below the metric highlights", () => {
      const highlights = highlightLines(
        renderReport(trippingResult(), { failOn: [{ kind: "geomean", pct: 2 }] }),
      ).map((line) => line.trim());

      expect(highlights).toStrictEqual([
        "✗ time · entity.spawn         +4.0%",
        "✓ time · entity.alive_check  -10.0%",
        "✓ memory · encode             -7.0%",
        "⚑ time geomean +3.1% exceeded --fail-on geomean:2",
      ]);
    });

    it.each<{ when: string; options: ReportOptions }>([
      { when: "no conditions reached the renderer", options: {} },
      {
        when: "the threshold sits beyond the gated geomean",
        options: { failOn: [{ kind: "geomean", pct: 10 }] },
      },
      {
        when: "only the regressed condition was given",
        options: { failOn: [{ kind: "regressed" }] },
      },
    ])("says nothing about a gate when $when", ({ options }) => {
      const report = renderReport(trippingResult(), options);

      expect(report).not.toContain("⚑");
    });

    it("says nothing about a gate for an informational kind past the threshold", () => {
      const result = twoKindResult({
        candidates: [
          createCandidate({
            kinds: [timeKind(), otherKind(9, 1, {}, { hasGating: false, gatedGeomean: undefined })],
          }),
        ],
      });

      const report = renderReport(result, { failOn: [{ kind: "geomean", pct: 2 }] });

      expect(report).not.toContain("⚑");
    });

    it.each([
      {
        when: "the overall geomean trips but the gated one does not",
        geomean: 5,
        gated: 1,
        expected: [],
      },
      {
        when: "the gated geomean trips but the overall one does not",
        geomean: 1,
        gated: 5,
        expected: ["⚑ time geomean +5.0% exceeded --fail-on geomean:2"],
      },
    ])("judges the gate on the gated geomean when $when", ({ geomean, gated, expected }) => {
      const result = twoKindResult({
        candidates: [
          createCandidate({
            kinds: [
              timeKind({ geomean: geomeanOf(geomean, 3), gatedGeomean: geomeanOf(gated, 3) }),
              memoryKind(),
            ],
          }),
        ],
      });

      const highlights = highlightLines(
        renderReport(result, { failOn: [{ kind: "geomean", pct: 2 }] }),
      ).map((line) => line.trim());

      expect(highlights.filter((line) => line.startsWith("⚑"))).toStrictEqual(expected);
    });

    it("flags only the candidates whose own gated geomean exceeded the threshold", () => {
      const highlights = highlightLines(
        renderReport(groupedComparison(), { failOn: [{ kind: "geomean", pct: 2 }] }),
      );

      expect(highlights).toStrictEqual([
        "  candidate-a",
        "    ✓ time · entity.alive_check  -10.0%",
        "    ✓ memory · encode             -7.0%",
        "  candidate-b",
        "    ✗ time · entity.alive_check   +4.0%",
        "    ✓ memory · encode             -2.0%",
        "    ⚑ time geomean +4.0% exceeded --fail-on geomean:2",
      ]);
    });

    it("paints the gate-trip glyph and delta red, as the regression they are", () => {
      vi.stubEnv("FORCE_COLOR", "1");

      const line = lineContaining(
        renderReport(trippingResult(), { failOn: [{ kind: "geomean", pct: 2 }] }),
        "⚑",
      );

      expect.soft(stylesAt(line, "⚑")).toContain("31");
      expect(stylesAt(line, "+3.1%")).toContain("31");
    });
  });

  describe("when closing the report", () => {
    it.each([
      { mode: "plain", options: {} },
      { mode: "verbose", options: { verbose: true } },
    ])(
      "spells out no legend in $mode mode, leaving the summary line to do that job",
      ({ options }) => {
        const result = createComparisonResult({
          metrics: { "a/time": signedRankMetric({ verdict: "improved", delta: -10 }) },
        });

        const output = renderReport(result, options);

        expect.soft(output).not.toContain("legend:");
        expect(output).not.toContain("candidates are judged against");
      },
    );
  });

  describe("when the report is not verbose", () => {
    it("stops after the highlights, naming no verdict method", () => {
      const result = createComparisonResult({
        metrics: {
          "a/time": signedRankMetric({ verdict: "improved", delta: -10 }),
          "b/time": bandMetric({ verdict: "improved", delta: -5, n: 10, usableN: 3 }),
        },
      });

      const output = renderReport(result);

      expect.soft(output).not.toContain("Wilcoxon");
      expect(output).not.toContain("noise band");
    });

    it("still hints at more samples when a metric ran short of pairs", () => {
      const result = createComparisonResult({
        metrics: { "a/time": bandMetric({ verdict: "no-signal", delta: -5, n: 4 }) },
      });

      const output = renderReport(result);

      expect.soft(output).not.toContain("noise band");
      expect(output).toContain("Hint: re-run with --samples 6 or more for statistical verdicts");
    });

    it("keeps the worktree footer below the hint", () => {
      const result = createComparisonResult({
        metrics: { "a/time": bandMetric({ verdict: "no-signal", delta: -5, n: 4 }) },
        worktreesRemoved: 1,
        worktreesLeftBehind: [],
      });

      const lines = renderReport(result).split("\n");

      expect.soft(lines.at(-2)).toContain("Hint:");
      expect(lines.at(-1)).toBe("1 worktree removed · 0 left behind");
    });
  });

  describe("when naming the verdict method", () => {
    it("names the signed-rank test once any metric used it", () => {
      const result = createComparisonResult({
        metrics: {
          "a/time": signedRankMetric({ verdict: "improved", delta: -10 }),
          "b/time": bandMetric({ verdict: "improved", delta: -5 }),
        },
      });

      const output = renderReport(result, { verbose: true });

      expect.soft(output).toContain("Wilcoxon signed-rank");
      expect(output).toContain("n=10 ≥ 6");
    });

    it("names the noise band and hints at more samples when it is the only method", () => {
      const result = createComparisonResult({
        metrics: { "a/time": bandMetric({ verdict: "no-signal", delta: -5 }) },
      });

      const output = renderReport(result, { verbose: true });

      expect.soft(output).toContain("noise band ±(half-range × K)");
      expect.soft(output).toContain("below signed-rank floor (6 pairs)");
      expect.soft(output).not.toContain("Wilcoxon");
      expect(output).toContain("Hint: re-run with --samples 6 or more for statistical verdicts");
    });

    it("names no method, and drops the hint, when every metric was exact", () => {
      const result = createComparisonResult({
        metrics: { "a/heap": exactMetric({ delta: -7.9 }) },
      });

      const output = renderReport(result, { verbose: true });

      expect.soft(output).not.toContain("Wilcoxon");
      expect.soft(output).not.toContain("noise band");
      expect(output).not.toContain("Hint:");
    });

    it("drops the hint when the signed-rank test carried the run", () => {
      const result = createComparisonResult({
        metrics: { "a/time": signedRankMetric({ verdict: "improved", delta: -10 }) },
      });

      const output = renderReport(result, { verbose: true });

      expect(output).not.toContain("Hint:");
    });

    it("gives each band fallback the phrasing its own cause earned", () => {
      const result = createComparisonResult({
        metrics: {
          "short/time": bandMetric({ verdict: "no-signal", delta: 1, n: 4 }),
          "tied/heap": bandMetric({ verdict: "no-signal", delta: -0.5, n: 10, usableN: 3 }),
        },
      });

      const bandLines = renderReport(result, { verbose: true })
        .split("\n")
        .filter((line) => line.startsWith("noise band"));

      expect(bandLines).toStrictEqual([
        "noise band ±(half-range × K) — n=4 below signed-rank floor (6 pairs)",
        "noise band ±(half-range × K) — ties left n=3 usable pairs (6 needed)",
      ]);
    });
  });

  describe("when metrics used different verdict methods", () => {
    /**
     * A run whose metrics genuinely disagree on method.
     *
     * `decode/time` paired on 10 of the 12 rounds — enough for the signed-rank
     * test — while `encode/time` paired on 4 and fell back to the noise band.
     */
    function mixedMethodResult(): ComparisonResult {
      return createComparisonResult({
        samples: 12,
        metrics: {
          "decode/time": signedRankMetric({ verdict: "improved", delta: -10, n: 10 }),
          "encode/time": bandMetric({ verdict: "no-signal", delta: 1, n: 4 }),
        },
      });
    }

    it("names each method present, with the pair counts that chose it", () => {
      const report = renderReport(mixedMethodResult(), { verbose: true });

      const signedRankLine = lineStartingWith(report, "verdicts:");
      const bandLine = lineStartingWith(report, "noise band");

      expect
        .soft(signedRankLine)
        .toBe("verdicts: Wilcoxon signed-rank on pairs (n=10 ≥ 6) · ~ = no signal at α=0.05");
      expect
        .soft(bandLine)
        .toBe("noise band ±(half-range × K) — n=4 below signed-rank floor (6 pairs)");
      expect(report.indexOf(signedRankLine)).toBeLessThan(report.indexOf(bandLine));
    });

    it("hints at more samples once any metric fell back to the band", () => {
      const output = renderReport(mixedMethodResult());

      expect(output).toContain("Hint: re-run with --samples 6 or more for statistical verdicts");
    });
  });

  describe("when reporting worktree cleanup", () => {
    it("says nothing at all when the run left nothing behind", () => {
      const result = createComparisonResult({
        worktreesRemoved: 0,
        worktreesLeftBehind: [],
      });

      const output = renderReport(result);

      expect.soft(output).not.toContain("worktree");
      expect(output).not.toContain("left behind");
    });

    it.each([
      { removed: 1, leftBehind: [], expected: "1 worktree removed · 0 left behind" },
      {
        removed: 2,
        leftBehind: [
          { dir: "/tmp/gymrat-a", error: "locked" },
          { dir: "/tmp/gymrat-b", error: "locked" },
        ],
        expected: "2 worktrees removed · 2 left behind",
      },
    ])("reports both counts as '$expected'", ({ removed, leftBehind, expected }) => {
      const result = createComparisonResult({
        worktreesRemoved: removed,
        worktreesLeftBehind: leftBehind,
      });

      const output = renderReport(result);

      expect(output).toContain(expected);
    });

    it("names each left-behind worktree directory with git's reason", () => {
      const result = createComparisonResult({
        worktreesRemoved: 1,
        worktreesLeftBehind: [
          { dir: "/tmp/gymrat-abc", error: "contains modified or untracked files" },
          { dir: "/tmp/gymrat-def", error: "is locked" },
        ],
      });

      const output = renderReport(result);

      expect
        .soft(output)
        .toContain("left behind: /tmp/gymrat-abc (contains modified or untracked files)");
      expect(output).toContain("left behind: /tmp/gymrat-def (is locked)");
    });

    it("reports the prune failure with git's reason even when nothing else went wrong", () => {
      const result = createComparisonResult({
        worktreePruneError: "fatal: not a git repository",
      });

      const output = renderReport(result);

      expect(output).toContain("worktree prune failed: fatal: not a git repository");
    });

    it("keeps a left-behind entry on one line when git's reason spans several lines", () => {
      const result = createComparisonResult({
        worktreesRemoved: 1,
        worktreesLeftBehind: [
          {
            dir: "/tmp/gymrat-abc",
            error: "warning: could not open directory\n  fatal: '/tmp/gymrat-abc' is locked",
          },
        ],
      });

      const detailLines = renderReport(result)
        .split("\n")
        .filter((line) => line.includes("/tmp/gymrat-abc"));

      expect(detailLines).toStrictEqual([
        "  left behind: /tmp/gymrat-abc (warning: could not open directory fatal: '/tmp/gymrat-abc' is locked)",
      ]);
    });

    it("keeps the prune failure on one line when git's reason spans several lines", () => {
      const result = createComparisonResult({
        worktreePruneError: "warning: unable to unlink\n  fatal: not a git repository",
      });

      const pruneLines = renderReport(result)
        .split("\n")
        .filter((line) => line.includes("prune failed"));

      expect(pruneLines).toStrictEqual([
        "  worktree prune failed: warning: unable to unlink fatal: not a git repository",
      ]);
    });

    it("closes the report with the left-behind worktrees and the prune failure", () => {
      const result = createComparisonResult({
        worktreesRemoved: 0,
        worktreesLeftBehind: [{ dir: "/tmp/gymrat-abc", error: "is locked" }],
        worktreePruneError: "fatal: not a git repository",
      });

      const lines = renderReport(result).split("\n");

      expect.soft(lines.at(-3)).toContain("0 worktrees removed · 1 left behind");
      expect.soft(lines.at(-2)).toBe("  left behind: /tmp/gymrat-abc (is locked)");
      expect(lines.at(-1)).toBe("  worktree prune failed: fatal: not a git repository");
    });
  });

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
      const entry = highlights[0]!;

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
      const entry = highlights[0]!;

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
      const entry = highlights[0]!;

      expect.soft(stylesAt(entry, "≈")).toContain("33");
      expect(stylesAt(entry, "unstable")).toContain("33");
    });

    it("styles the verdict word rather than a metric name that spells it too", () => {
      const result = createComparisonResult({
        metrics: {
          "unstable-parse/time": bandMetric({ verdict: "unstable", delta: 5, noisePct: 30 }),
        },
      });
      const entry = highlightLines(renderReport(result))[0]!;

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
      const exactEntry = highlights.find((line) => line.includes("cheaper/heap"))!;
      const unstableEntry = highlights.find((line) => line.includes("jittery/time"))!;

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

  describe("when ordering the report sections", () => {
    /** A two-metric run whose only footer content is the signed-rank method line. */
    function orderedResult(): ComparisonResult {
      return createComparisonResult({
        baselineLabel: "main",
        metrics: {
          "metric1/time": signedRankMetric({ verdict: "improved", delta: -10, unit: "ns" }),
          "metric2/time": signedRankMetric({
            verdict: "no-signal",
            delta: 2,
            gating: false,
            unit: "ns",
          }),
        },
        candidates: [
          createCandidate({
            label: "faster",
            kinds: [otherKind(-5, 1)],
          }),
        ],
      });
    }

    it("emits table, summary and highlights in that order, and closes there", () => {
      const lines = renderReport(orderedResult()).split("\n");

      // Each section's content is asserted by its own test; this pins the order.
      expect.soft(lines[0]).toContain("gymrat compare · baseline main ↔ faster");
      expect.soft(lines[1]).toMatch(/^metric\s+│/);
      expect.soft(lines[2]).toMatch(/^─+┼/);
      expect.soft(lines[3]).toContain("metric1/time");
      expect.soft(lines[4]).toContain("metric2/time");
      expect.soft(lines[5]).toMatch(/^─+┼/);
      expect.soft(lines[6]).toContain("geomean");
      expect.soft(lines[7]).toBe("");
      expect.soft(lines[8]).toContain("✓ 1 improved");
      expect.soft(lines[9]).toBe("");
      expect.soft(lines[10]).toBe("highlights");
      expect.soft(lines[11]).toContain("metric1/time");
      expect(lines).toHaveLength(12);
    });

    it("adds the method block below a blank line when verbose", () => {
      const lines = renderReport(orderedResult(), { verbose: true }).split("\n");

      expect.soft(lines[11]).toContain("metric1/time");
      expect.soft(lines[12]).toBe("");
      expect.soft(lines[13]).toContain("Wilcoxon signed-rank");
      expect(lines).toHaveLength(14);
    });
  });

  describe("when rendering more than one candidate", () => {
    it("heads one column per candidate with that candidate's name alone", () => {
      const headerLine = lineStartingWith(renderReport(multiCandidateResult()), "metric");

      expect(cellsOf(headerLine).map((cell) => cell.trim())).toStrictEqual([
        "metric",
        "main",
        "candidate-a",
        "candidate-b",
        "candidate-c",
      ]);
    });

    it("keeps the baseline figure and pairs each candidate's own figure with its verdict", () => {
      const row = lineStartingWith(renderReport(multiCandidateResult()), "decode/time");

      expect(cellsOf(row).map((cell) => cell.trim())).toStrictEqual([
        "decode/time",
        "100ns ± 1%",
        "90ns ± 1%  ✓  -10.0%",
        "104ns ± 1%  ✗  +4.0%",
        "150ns ± 3%  ≈  unstable",
      ]);
    });

    it("carries one geomean figure per candidate column, each with its own count", () => {
      const row = lineStartingWith(renderReport(multiCandidateResult()), "geomean");

      expect(cellsOf(row).map((cell) => cell.trim())).toStrictEqual([
        "geomean",
        "",
        "-10.0% · 1 stable metric",
        "+4.0% · 1 stable metric",
        "0.0% · 1 stable metric",
      ]);
    });

    it("closes the table on the geomean row", () => {
      const row = lastTableRow(renderReport(multiCandidateResult()));

      expect(cellsOf(row)[0]?.trim()).toBe("geomean");
    });

    it("lines the column separators up across header, metric rows and geomean", () => {
      const report = renderReport(multiCandidateResult());
      const headerOffsets = separatorOffsets(lineStartingWith(report, "metric"));

      expect
        .soft(separatorOffsets(lineStartingWith(report, "decode/time")))
        .toStrictEqual(headerOffsets);
      expect(separatorOffsets(lineStartingWith(report, "geomean"))).toStrictEqual(headerOffsets);
    });

    it("summarizes each candidate on its own line, behind that candidate's label", () => {
      const summaries = renderReport(multiCandidateResult())
        .split("\n")
        .filter((line) => /✓ \d+ improved/.test(line));

      expect(summaries).toStrictEqual([
        "candidate-a  ✓ 1 improved   ✗ 0 regressed   ≈ 0 unstable   = 0 identical   ~ 0 within noise   ? 0 inconclusive",
        "candidate-b  ✓ 0 improved   ✗ 1 regressed   ≈ 0 unstable   = 0 identical   ~ 0 within noise   ? 0 inconclusive",
        "candidate-c  ✓ 0 improved   ✗ 0 regressed   ≈ 1 unstable   = 0 identical   ~ 0 within noise   ? 0 inconclusive",
      ]);
    });

    it("groups the highlights into one subsection per candidate", () => {
      const highlights = highlightLines(renderReport(multiCandidateResult()));

      expect(highlights).toStrictEqual([
        "  candidate-a",
        "    ✓ decode/time  -10.0%",
        "  candidate-b",
        "    ✗ decode/time   +4.0%",
        "  candidate-c",
        "    ≈ decode/time  unstable  noise ±30.0%",
        "  unstable metrics won't stabilize with more samples",
      ]);
    });

    it("drops the highlights section when no candidate has anything to highlight", () => {
      const result = createComparisonResult({
        candidates: [
          createCandidate({ label: "candidate-a" }),
          createCandidate({ label: "candidate-b" }),
        ],
        metrics: {
          "decode/time": nWayMetric([
            { verdict: "no-signal", delta: 0.4, median: 100 },
            { verdict: "no-signal", delta: -0.3, median: 100 },
          ]),
        },
      });

      const report = renderReport(result);

      expect.soft(report).not.toContain("highlights");
      expect(report).toContain("candidate-a  ✓ 0 improved");
    });

    /**
     * A run whose two rows differ in whether any candidate moved at all.
     *
     * `mixed/time` moved for one candidate and stayed flat for the other, which
     * is exactly the row a per-candidate dimming rule would wrongly recede.
     */
    function dimmingResult(): ComparisonResult {
      return createComparisonResult({
        candidates: [
          createCandidate({ label: "candidate-a" }),
          createCandidate({ label: "candidate-b" }),
        ],
        metrics: {
          "flat/time": nWayMetric([
            { verdict: "no-signal", delta: 0.3, median: 100 },
            { verdict: "unstable", delta: -50, median: 50 },
          ]),
          "mixed/time": nWayMetric([
            { verdict: "no-signal", delta: 0.3, median: 100 },
            { verdict: "improved", delta: -17.5, median: 83 },
          ]),
        },
      });
    }

    describe("when color styling is applied", () => {
      beforeEach(() => {
        vi.stubEnv("FORCE_COLOR", "1");
      });

      it.each([{ row: "flat/time" }, { row: "mixed/time" }])(
        "leaves the $row row without an end-to-end dim, quiet cells and all",
        ({ row }) => {
          expect(lineContaining(renderReport(dimmingResult()), row)).not.toMatch(DIMMED_LINE);
        },
      );

      it("styles each cell's verdict by that cell's own verdict on an all-quiet row", () => {
        const row = lineContaining(renderReport(dimmingResult()), "flat/time");

        expect.soft(stylesAt(row, "~")).toContain("2");
        expect(stylesAt(row, "≈")).toContain("33");
      });

      it("leaves the name and the values on an all-quiet row unstyled", () => {
        const cells = cellsOf(lineContaining(renderReport(dimmingResult()), "flat/time"));

        expect.soft(cells.slice(0, 2).join("│")).not.toContain("\x1b[");
        expect.soft(valuePartOf(cells[2] ?? "", "~")).not.toContain("\x1b[");
        expect(valuePartOf(cells[3] ?? "", "≈")).not.toContain("\x1b[");
      });

      it.each([
        { verdict: "improved", column: 2, glyph: "✓" },
        { verdict: "regressed", column: 3, glyph: "✗" },
        { verdict: "unstable", column: 4, glyph: "≈" },
      ])(
        "leaves the $verdict cell's value plain beside its neighbors' verdicts",
        ({ column, glyph }) => {
          const row = lineContaining(renderReport(multiCandidateResult()), "decode/time");

          expect(valuePartOf(cellsOf(row)[column] ?? "", glyph)).not.toContain("\x1b[");
        },
      );

      it("paints an unstable N-way cell's verdict amber, as an unstable row is painted", () => {
        const row = lineContaining(renderReport(multiCandidateResult()), "decode/time");

        expect.soft(stylesAt(row, "≈")).toContain("33");
        expect(stylesAt(row, "unstable")).toContain("33");
      });

      it("pads on the plain text, so the colored columns line up once the styles are stripped", () => {
        const bare = stripAnsi(renderReport(multiCandidateResult()));
        const headerOffsets = separatorOffsets(lineStartingWith(bare, "metric"));

        expect
          .soft(separatorOffsets(lineStartingWith(bare, "decode/time")))
          .toStrictEqual(headerOffsets);
        expect(separatorOffsets(lineStartingWith(bare, "geomean"))).toStrictEqual(headerOffsets);
      });

      it("emboldens the candidate label in N-way summary lines when color is on", () => {
        const report = renderReport(multiCandidateResult());
        const summaries = report.split("\n").filter((line) => /✓ \d+ improved/.test(line));

        expect(summaries).toHaveLength(3);
        expect.soft(stylesAt(summaries[0]!, "candidate-a")).toContain("1");
        expect.soft(stylesAt(summaries[1]!, "candidate-b")).toContain("1");
        expect(stylesAt(summaries[2]!, "candidate-c")).toContain("1");
      });

      it("emboldens the candidate sublabels in N-way highlights when color is on", () => {
        const highlights = highlightLines(renderReport(multiCandidateResult()));
        const sublabels = highlights.filter((line) => {
          const stripped = stripAnsi(line).trim();
          return ["candidate-a", "candidate-b", "candidate-c"].includes(stripped);
        });

        expect(sublabels).toHaveLength(3);
        for (const sublabel of sublabels) {
          expect(sublabel).toContain("\x1b[1m");
        }
      });

      it("colors glyph and delta together in N-way improved and regressed cells", () => {
        const report = renderReport(multiCandidateResult());
        const row = lineContaining(report, "decode/time");

        expect.soft(stylesAt(row, "✓")).toContain("32");
        expect.soft(stylesAt(row, "-10.0%")).toContain("32");
        expect.soft(stylesAt(row, "✗")).toContain("31");
        expect(stylesAt(row, "+4.0%")).toContain("31");
      });

      it("dims the quiet candidate segment on a bright N-way row", () => {
        const report = renderReport(dimmingResult());
        const row = lineContaining(report, "mixed/time");

        expect.soft(stylesAt(row, "~")).toContain("2");
        expect(stylesAt(row, "+0.3%")).toContain("2");
      });

      it("dims the provenance in N-way geomean cells", () => {
        const report = renderReport(multiCandidateResult());
        const geomean = lineContaining(report, "geomean");

        expect(stylesAt(geomean, "1 stable metric")).toContain("2");
      });
    });
  });

  describe("when the run spans more than one metric kind", () => {
    it("gives each kind its own titled section, closed by that kind's geomean", () => {
      expect(tableRegion(renderReport(twoKindResult()))).toStrictEqual([
        "gymrat compare · baseline main ↔ perf/faster-decode · 10 paired samples · adapter: mitata",
        "",
        "<border>",
        "time",
        "<rule>",
        "entity",
        "alive_check",
        "spawn",
        "geomean · entity (2)",
        "",
        "warmup",
        "<rule>",
        "geomean · time (3)",
        "",
        "informational — gating off (config: kinds.memory.gating = false)",
        "<border>",
        "memory",
        "<rule>",
        "encode",
        "<rule>",
        "geomean · memory (1)",
      ]);
    });

    it("spans a section's top border across the full table width", () => {
      const report = stripAnsi(renderReport(twoKindResult()));
      const lines = report.split("\n");
      const header = lineStartingWith(report, "time");
      const headerIndex = lines.indexOf(header);
      const border = lines[headerIndex - 1];
      const rule = lines[headerIndex + 1];
      if (border === undefined || rule === undefined) {
        throw new Error(`no border or rule around section header in report:\n${report}`);
      }

      expect.soft(border).not.toContain("┼");
      expect(border).toHaveLength(rule.length);
    });

    it("joins a section's top border to the header's columns", () => {
      const report = stripAnsi(renderReport(twoKindResult()));
      const lines = report.split("\n");
      const header = lineStartingWith(report, "time");
      const headerIndex = lines.indexOf(header);
      const border = lines[headerIndex - 1];
      if (border === undefined) {
        throw new Error(`no border above section header in report:\n${report}`);
      }

      expect(offsetsOf(border, "┬")).toStrictEqual(separatorOffsets(header));
    });

    it("lines every section's columns up with the first section's header", () => {
      const report = renderReport(twoKindResult());
      const bare = stripAnsi(report);
      const timeHeader = lineStartingWith(bare, "time");
      const memoryHeader = lineStartingWith(bare, "memory");
      const offsets = separatorOffsets(timeHeader);

      expect.soft(separatorOffsets(memoryHeader)).toStrictEqual(offsets);
      expect
        .soft(separatorOffsets(lineStartingWith(report, "  alive_check")))
        .toStrictEqual(offsets);
      expect.soft(separatorOffsets(lineStartingWith(report, "entity "))).toStrictEqual(offsets);
      expect(separatorOffsets(lineStartingWith(report, "geomean · memory"))).toStrictEqual(offsets);
    });

    it.each([
      {
        source: "the kind-level config entry",
        configKinds: { memory: { gating: false } },
        expected: "informational — gating off (config: kinds.memory.gating = false)",
      },
      {
        source: "per-metric overrides alone",
        configKinds: undefined,
        expected: "informational — gating off",
      },
    ])("credits $source for a non-gating kind's informational tag", ({ configKinds, expected }) => {
      const report = renderReport(twoKindResult({ configKinds }));

      expect(lineContaining(report, "informational")).toBe(expected);
    });

    it.each([
      { placement: "indented under its group, stripped of the group prefix", row: "  alive_check" },
      { placement: "at the margin under its bare short name", row: "warmup" },
    ])("names a metric row $placement", ({ row }) => {
      const line = lineStartingWith(renderReport(twoKindResult()), row);

      expect(cellsOf(line)[0]?.trimEnd()).toBe(row);
    });

    it("names each highlight by kind and short metric, padded to keep the deltas aligned", () => {
      const highlights = highlightLines(renderReport(twoKindResult())).map((line) => line.trim());

      expect(highlights).toStrictEqual([
        "✗ time · entity.spawn         +4.0%",
        "✓ time · entity.alive_check  -10.0%",
        "✓ memory · encode             -7.0%",
      ]);
    });

    it("prefixes the kind inside every candidate's highlight subsection", () => {
      const result = createComparisonResult({
        metrics: {
          "entity.alive_check/time": nWayKindMetric({
            kind: "time",
            shortName: "entity.alive_check",
            candidates: [
              { verdict: "improved", delta: -10, median: 90 },
              { verdict: "regressed", delta: 4, median: 104 },
            ],
          }),
          "encode/heap": nWayKindMetric({
            kind: "memory",
            shortName: "encode",
            gating: false,
            candidates: [
              { verdict: "improved", delta: -7, median: 93 },
              { verdict: "improved", delta: -2, median: 98 },
            ],
          }),
        },
        candidates: [
          createCandidate({
            label: "candidate-a",
            kinds: [timeKind({ geomean: geomeanOf(-10, 1), groups: [] }), memoryKind()],
          }),
          createCandidate({
            label: "candidate-b",
            kinds: [
              timeKind({ geomean: geomeanOf(4, 1), groups: [] }),
              memoryKind({ geomean: geomeanOf(-2, 1) }),
            ],
          }),
        ],
        configKinds: { memory: { gating: false } },
      });

      expect(highlightLines(renderReport(result))).toStrictEqual([
        "  candidate-a",
        "    ✓ time · entity.alive_check  -10.0%",
        "    ✓ memory · encode             -7.0%",
        "  candidate-b",
        "    ✗ time · entity.alive_check   +4.0%",
        "    ✓ memory · encode             -2.0%",
      ]);
    });

    it("counts the excluded metrics into a geomean label's provenance", () => {
      const result = twoKindResult({
        candidates: [
          createCandidate({
            kinds: [
              timeKind({
                geomean: geomeanOf(-3.2, 2, {
                  excluded: [{ metric: "warmup/time", reason: "unstable" }],
                }),
              }),
              memoryKind(),
            ],
          }),
        ],
      });

      const row = lineStartingWith(renderReport(result), "geomean · time");

      expect(cellsOf(row)[0]?.trim()).toBe("geomean · time (2/3)");
    });

    it.each([
      {
        gating: "one kind gates",
        makeResult: (): ComparisonResult => twoKindResult(),
      },
      {
        gating: "several kinds gate",
        makeResult: (): ComparisonResult =>
          twoKindResult({
            metrics: twoKindMetrics({ memoryGates: true }),
            candidates: [
              createCandidate({
                kinds: [
                  timeKind(),
                  memoryKind({ hasGating: true, gatedGeomean: geomeanOf(6.1, 1) }),
                ],
              }),
            ],
            configKinds: undefined,
          }),
      },
      {
        gating: "no kind gates",
        makeResult: (): ComparisonResult =>
          twoKindResult({
            metrics: twoKindMetrics({ timeGates: false }),
            candidates: [
              createCandidate({
                kinds: [timeKind({ hasGating: false, gatedGeomean: undefined }), memoryKind()],
              }),
            ],
          }),
      },
    ])(
      "closes the table on the last kind's geomean, with no gated row, when $gating",
      ({ makeResult }) => {
        const report = renderReport(makeResult());

        expect.soft(tableRegion(report).at(-1)).toBe("geomean · memory (1)");
        expect(report).not.toContain("geomean · gated");
      },
    );

    it("carries one figure per candidate column on every aggregate row", () => {
      const report = renderReport(groupedComparison());
      const cellsAt = (label: string): string[] =>
        cellsOf(lineStartingWith(report, label)).map((cell) => cell.trim());

      expect
        .soft(cellsAt("geomean · entity"))
        .toStrictEqual([
          "geomean · entity",
          "",
          "-10.0% · 1 stable metric",
          "+4.0% · 1 stable metric",
        ]);
      expect
        .soft(cellsAt("geomean · time"))
        .toStrictEqual([
          "geomean · time",
          "",
          "-10.0% · 1 stable metric",
          "+4.0% · 1 stable metric",
        ]);
      expect(tableRegion(report).at(-1)).toBe("geomean · memory");
    });

    describe("when color styling is applied", () => {
      beforeEach(() => {
        vi.stubEnv("FORCE_COLOR", "1");
      });

      it.each([
        { row: "sub-geomean", label: "geomean · entity", value: "-3.1%" },
        { row: "kind geomean", label: "geomean · time", value: "-3.2%" },
      ])("paints an improving $row value green once it clears its band", ({ label, value }) => {
        const line = lineContaining(renderReport(twoKindResult()), label);

        expect(stylesAt(line, value)).toStrictEqual(["1", "32"]);
      });

      it("emboldens the kind name in the section header and dims the informational tag", () => {
        const report = renderReport(twoKindResult());
        const header = report
          .split("\n")
          .find((line) => line.includes("│") && stripAnsi(line).trimStart().startsWith("memory"));
        if (header === undefined) {
          throw new Error("no memory header in report");
        }
        const tag = lineContaining(report, "informational");

        expect.soft(stylesAt(header, "memory")).toStrictEqual(["1"]);
        expect(stylesAt(tag, "informational")).toStrictEqual(["2"]);
      });

      it("leaves every column separator in the default color, whatever style its row carries", () => {
        const rows = renderReport(twoKindResult())
          .split("\n")
          .filter((line) => line.includes("│"));

        const inherited = rows.filter((row) =>
          separatorStyles(row).some((styles) => styles.length > 0),
        );

        expect(inherited).toStrictEqual([]);
      });
    });
  });

  describe("when every metric shares one kind", () => {
    /** A single gating kind whose two metrics share a group. */
    function oneKindResult(overrides: Partial<ComparisonResult> = {}): ComparisonResult {
      return createComparisonResult({
        metrics: {
          "entity.alive_check/time": kindMetric({
            kind: "time",
            shortName: "entity.alive_check",
            verdict: "improved",
            delta: -10,
          }),
          "entity.spawn/time": kindMetric({
            kind: "time",
            shortName: "entity.spawn",
            verdict: "regressed",
            delta: 4,
          }),
        },
        candidates: [
          createCandidate({
            kinds: [
              {
                kind: "time",
                hasGating: true,
                geomean: geomeanOf(-3.2, 2),
                groups: [{ group: "entity", geomean: geomeanOf(-3.2, 2) }],
                gatedGeomean: geomeanOf(-3.2, 2),
              },
            ],
          }),
        ],
        ...overrides,
      });
    }

    it("keeps the flat layout, full metric names and one geomean row", () => {
      expect(tableRegion(renderReport(oneKindResult()))).toStrictEqual([
        "gymrat compare · baseline main ↔ perf/faster-decode · 10 paired samples · adapter: mitata",
        "metric",
        "<rule>",
        "entity.alive_check/time",
        "entity.spawn/time",
        "<rule>",
        "geomean (2 stable metrics)",
      ]);
    });

    it("reports no stable metrics when that kind does not gate", () => {
      const result = createComparisonResult({
        metrics: {
          "warmup/time": kindMetric({
            kind: "time",
            shortName: "warmup",
            verdict: "improved",
            delta: -10,
            gating: false,
          }),
        },
        candidates: [
          createCandidate({
            kinds: [{ kind: "time", hasGating: false, geomean: geomeanOf(-10, 1), groups: [] }],
          }),
        ],
      });

      const row = lineStartingWith(renderReport(result), "geomean");

      expect(cellsOf(row).map((cell) => cell.trim())).toStrictEqual([
        "geomean",
        "",
        "",
        "—  no stable metrics",
      ]);
    });

    it("paints the flat geomean value by its own band, color on", () => {
      vi.stubEnv("FORCE_COLOR", "1");
      const result = oneKindResult({
        candidates: [
          createCandidate({
            kinds: [
              {
                kind: "time",
                hasGating: true,
                geomean: geomeanOf(-3.2, 2),
                groups: [],
                gatedGeomean: geomeanOf(-3.2, 2, { band: 1 }),
              },
            ],
          }),
        ],
      });

      const line = lineContaining(renderReport(result), "geomean");

      expect(stylesAt(line, "-3.2%")).toStrictEqual(["1", "32"]);
    });
  });

  describe("when labelling the group and geomean rows", () => {
    /** A single-candidate run of one kind, whose table closes on a flat geomean row. */
    function flatResult(): ComparisonResult {
      return createComparisonResult({
        metrics: { "faster/time": signedRankMetric({ verdict: "improved", delta: -17.5 }) },
      });
    }

    describe("when color is on", () => {
      beforeEach(() => {
        vi.stubEnv("FORCE_COLOR", "1");
      });

      it.each([
        { table: "single-candidate", makeResult: twoKindResult },
        { table: "multi-candidate", makeResult: groupedComparison },
      ])("paints the group sub-header blue in the $table table", ({ makeResult }) => {
        const row = lineContaining(renderReport(makeResult()), "entity");

        expect(stylesAt(row, "entity")).toStrictEqual(["34"]);
      });

      it.each([
        {
          level: "group",
          table: "single-candidate",
          label: "geomean · entity",
          makeResult: twoKindResult,
        },
        {
          level: "kind",
          table: "single-candidate",
          label: "geomean · time",
          makeResult: twoKindResult,
        },
        { level: "flat", table: "single-candidate", label: "geomean", makeResult: flatResult },
        {
          level: "group",
          table: "multi-candidate",
          label: "geomean · entity",
          makeResult: groupedComparison,
        },
        {
          level: "kind",
          table: "multi-candidate",
          label: "geomean · time",
          makeResult: groupedComparison,
        },
        {
          level: "flat",
          table: "multi-candidate",
          label: "geomean",
          makeResult: multiCandidateResult,
        },
      ])("emboldens the $level geomean label in the $table table", ({ label, makeResult }) => {
        const row = lineContaining(renderReport(makeResult()), label);

        expect(stylesAt(row, label)).toStrictEqual(["1"]);
      });
    });
  });

  describe("when every metric behind a geomean landed within noise", () => {
    /**
     * A two-kind run whose every metric landed within noise.
     *
     * Each geomean figure sits far outside its own band, so a rule that reads
     * the band alone would paint all of them green.
     */
    function quietTwoKindResult(): ComparisonResult {
      return createComparisonResult({
        metrics: {
          "entity.alive_check/time": kindMetric({
            kind: "time",
            shortName: "entity.alive_check",
            verdict: "no-signal",
            delta: -9,
          }),
          "entity.spawn/time": kindMetric({
            kind: "time",
            shortName: "entity.spawn",
            verdict: "no-signal",
            delta: -8,
          }),
          "encode/heap": kindMetric({
            kind: "memory",
            shortName: "encode",
            verdict: "no-signal",
            delta: -7,
            gating: false,
            unit: "bytes",
          }),
        },
        candidates: [
          createCandidate({
            kinds: [
              timeKind({
                geomean: geomeanOf(-8.5, 2),
                groups: [{ group: "entity", geomean: geomeanOf(-8.6, 2) }],
                gatedGeomean: geomeanOf(-8.5, 2),
              }),
              memoryKind({ geomean: geomeanOf(-7, 1) }),
            ],
          }),
        ],
        configKinds: { memory: { gating: false } },
      });
    }

    describe("when color is on", () => {
      beforeEach(() => {
        vi.stubEnv("FORCE_COLOR", "1");
      });

      it.each([
        { level: "group", label: "geomean · entity", value: "-8.6%" },
        { level: "kind", label: "geomean · time", value: "-8.5%" },
      ])("leaves the $level geomean value emboldened and uncolored", ({ label, value }) => {
        const line = lineContaining(renderReport(quietTwoKindResult()), label);

        expect(stylesAt(line, value)).toStrictEqual(["1"]);
      });

      it("leaves the flat geomean value emboldened and uncolored", () => {
        const result = createComparisonResult({
          metrics: { "faster/time": signedRankMetric({ verdict: "no-signal", delta: -0.5 }) },
        });

        const line = lineContaining(renderReport(result), "geomean");

        expect(stylesAt(line, "-5.8%")).toStrictEqual(["1"]);
      });

      it("judges each candidate column by that column's own verdicts", () => {
        const result = createComparisonResult({
          metrics: {
            "entity.alive_check/time": nWayKindMetric({
              kind: "time",
              shortName: "entity.alive_check",
              candidates: [
                { verdict: "no-signal", delta: -9, median: 91 },
                { verdict: "improved", delta: -12, median: 88 },
              ],
            }),
            "encode/heap": nWayKindMetric({
              kind: "memory",
              shortName: "encode",
              gating: false,
              candidates: [
                { verdict: "no-signal", delta: -1, median: 99 },
                { verdict: "improved", delta: -2, median: 98 },
              ],
            }),
          },
          candidates: [
            createCandidate({
              label: "candidate-a",
              kinds: [
                timeKind({
                  geomean: geomeanOf(-9, 1),
                  groups: [{ group: "entity", geomean: geomeanOf(-9, 1) }],
                  gatedGeomean: geomeanOf(-9, 1),
                }),
                memoryKind({ geomean: geomeanOf(-1, 1) }),
              ],
            }),
            createCandidate({
              label: "candidate-b",
              kinds: [
                timeKind({
                  geomean: geomeanOf(-12, 1),
                  groups: [{ group: "entity", geomean: geomeanOf(-12, 1) }],
                  gatedGeomean: geomeanOf(-12, 1),
                }),
                memoryKind({ geomean: geomeanOf(-2, 1) }),
              ],
            }),
          ],
          configKinds: { memory: { gating: false } },
        });

        const line = lineContaining(renderReport(result), "geomean · time");

        expect.soft(stylesAt(line, "-9.0%")).toStrictEqual(["1"]);
        expect(stylesAt(line, "-12.0%")).toStrictEqual(["1", "32"]);
      });
    });
  });

  /**
   * Byte-level pins on the whole rendered report.
   *
   * Every other assertion in this file checks a substring or a separator offset,
   * which leaves column widths and inter-cell spacing free to drift while the
   * suite stays green. The report is the product's interface, so that drift is a
   * user-visible output change — these two snapshots are what makes it fail a run
   * instead of shipping.
   *
   * A diff here is never cosmetic. Re-record only when changing the report is the
   * intended change; a refactor that moves these bytes has changed behavior,
   * whatever its intent.
   */
  describe("when rendering a whole report", () => {
    it("matches the recorded bytes for a representative run", async () => {
      // The recorded bytes are plain: an ambient FORCE_COLOR in the caller's
      // shell would otherwise style them and fail a run that changed nothing.
      vi.stubEnv("FORCE_COLOR", undefined);
      vi.stubEnv("NO_COLOR", "1");

      const result = createComparisonResult({
        metrics: {
          "decode/text=digits/time": {
            baselineMedian: 1735,
            baselineSpread: 1,
            candidates: [
              {
                median: 1425,
                spread: 1,
                verdict: signedRankVerdict({ verdict: "improved", delta: -17.9, p: 0.002 }),
              },
            ],
            meta: metricMeta("decode/text=digits/time", { unit: "ns" }),
          },
          "decode/text=words/time": {
            baselineMedian: 3065,
            baselineSpread: 1,
            candidates: [
              {
                median: 3093,
                spread: 3,
                verdict: signedRankVerdict({ delta: 0.9, p: 0.49 }),
              },
            ],
            meta: metricMeta("decode/text=words/time", { unit: "ns" }),
          },
          "encode/time": {
            baselineMedian: 914,
            baselineSpread: 1,
            candidates: [
              {
                median: 934,
                spread: 1,
                verdict: signedRankVerdict({ verdict: "regressed", delta: 2.2, p: 0.002 }),
              },
            ],
            meta: metricMeta("encode/time", { unit: "ns" }),
          },
          "encode/heap": {
            baselineMedian: 49152,
            baselineSpread: 0,
            candidates: [
              {
                median: 45261,
                spread: 0,
                verdict: exactVerdict({ verdict: "improved", delta: -7.9 }),
              },
            ],
            meta: metricMeta("encode/heap", { exact: true, unit: "bytes" }),
          },
        },
        candidates: [
          createCandidate({
            kinds: [otherKind(-6, 4)],
          }),
        ],
      });

      await expect(renderReport(result)).toMatchFileSnapshot(
        "../fixtures/report-representative.golden.txt",
      );
    });

    function degenerateResult(): ComparisonResult {
      return createComparisonResult({
        samples: 4,
        adapter: "metric-lines",
        metrics: {
          "zero-median/time": {
            baselineMedian: 0,
            candidates: [
              {
                median: 0,
                verdict: exactVerdict({ n: 4 }),
              },
            ],
            meta: metricMeta("zero-median/time", { exact: true, unit: "ns" }),
          },
          "nan-delta/count": {
            baselineMedian: 0,
            candidates: [
              {
                median: 120,
                verdict: exactVerdict({ delta: Number.NaN, n: 4 }),
              },
            ],
            meta: metricMeta("nan-delta/count", { exact: true }),
          },
          "old-side-only/time": {
            baselineMedian: 2048,
            baselineSpread: 2,
            candidates: [{}],
            meta: metricMeta("old-side-only/time", { unit: "ns" }),
          },
          "throughput/ops": {
            baselineMedian: 1200,
            baselineSpread: 5,
            candidates: [
              {
                median: 1560,
                spread: 4,
                verdict: bandVerdict({ verdict: "improved", delta: 30, n: 4, usableN: 4 }),
              },
            ],
            meta: metricMeta("throughput/ops", { direction: "higher", gating: false }),
          },
        },
        candidates: [
          createCandidate({
            kinds: [
              otherKind(0, 1, {
                excluded: [{ metric: "nan-delta/count", reason: "undefined-ratio" }],
              }),
            ],
          }),
        ],
        worktreesRemoved: 1,
        worktreesLeftBehind: [{ dir: "/tmp/gymrat-abc123", error: "contains modified files" }],
        worktreePruneError: "could not lock config file",
      });
    }

    it("matches the recorded bytes for degenerate inputs and a dirty cleanup", async () => {
      // The recorded bytes are plain: an ambient FORCE_COLOR in the caller's
      // shell would otherwise style them and fail a run that changed nothing.
      vi.stubEnv("FORCE_COLOR", undefined);
      vi.stubEnv("NO_COLOR", "1");

      await expect(renderReport(degenerateResult())).toMatchFileSnapshot(
        "../fixtures/report-degenerate.golden.txt",
      );
    });

    function twoCandidateResult(): ComparisonResult {
      return createComparisonResult({
        candidates: [
          createCandidate({
            label: "perf/simd-decode",
            // Both geomeans sit inside their noise band, so the colored twin of
            // this golden pins the styling of a geomean that stays there.
            kinds: [otherKind(-12.4, 3, { band: 30 })],
          }),
          createCandidate({
            label: "perf/lut-decode",
            kinds: [
              otherKind(1.2, 2, {
                excluded: [{ metric: "encode/time", reason: "unstable" }],
                band: 30,
              }),
            ],
          }),
        ],
        metrics: {
          "decode/text=digits/time": {
            baselineMedian: 1735,
            baselineSpread: 1,
            candidates: [
              {
                median: 1425,
                spread: 1,
                verdict: signedRankVerdict({ verdict: "improved", delta: -17.9, p: 0.002 }),
              },
              {
                median: 1698,
                spread: 2,
                verdict: signedRankVerdict({ delta: -2.1, p: 0.32 }),
              },
            ],
            meta: metricMeta("decode/text=digits/time", { unit: "ns" }),
          },
          "encode/time": {
            baselineMedian: 914,
            baselineSpread: 1,
            candidates: [
              {
                median: 934,
                spread: 1,
                verdict: signedRankVerdict({ verdict: "regressed", delta: 2.2, p: 0.002 }),
              },
              {
                median: 1200,
                spread: 12,
                // The band method only runs below six pairs, so this metric was
                // dropped on most rounds — which also pins the n= annotation in
                // an N-way cell.
                verdict: bandVerdict({
                  verdict: "unstable",
                  delta: 31.3,
                  n: 4,
                  usableN: 4,
                  band: 30,
                  noisePct: 30,
                  noiseAbs: 30,
                }),
              },
            ],
            meta: metricMeta("encode/time", { unit: "ns" }),
          },
          "encode/heap": {
            baselineMedian: 49152,
            baselineSpread: 0,
            // The second candidate never reported this metric, so its cell stays empty.
            candidates: [
              {
                median: 45261,
                spread: 0,
                verdict: exactVerdict({ verdict: "improved", delta: -7.9 }),
              },
              {},
            ],
            meta: metricMeta("encode/heap", { exact: true, unit: "bytes" }),
          },
        },
      });
    }

    it("matches the recorded bytes for a verbose run with two candidates", async () => {
      // The recorded bytes are plain: an ambient FORCE_COLOR in the caller's
      // shell would otherwise style them and fail a run that changed nothing.
      vi.stubEnv("FORCE_COLOR", undefined);
      vi.stubEnv("NO_COLOR", "1");

      await expect(renderReport(twoCandidateResult(), { verbose: true })).toMatchFileSnapshot(
        "../fixtures/report-two-candidates.golden.txt",
      );
    });

    it("matches the recorded bytes for a run split into kind sections", async () => {
      // The recorded bytes are plain: an ambient FORCE_COLOR in the caller's
      // shell would otherwise style them and fail a run that changed nothing.
      vi.stubEnv("FORCE_COLOR", undefined);
      vi.stubEnv("NO_COLOR", "1");

      await expect(renderReport(twoKindResult())).toMatchFileSnapshot(
        "../fixtures/report-sectioned.golden.txt",
      );
    });

    it("matches the recorded bytes for a run of one paired sample", async () => {
      // The recorded bytes are plain: an ambient FORCE_COLOR in the caller's
      // shell would otherwise style them and fail a run that changed nothing.
      vi.stubEnv("FORCE_COLOR", undefined);
      vi.stubEnv("NO_COLOR", "1");

      await expect(renderReport(singleSampleResult())).toMatchFileSnapshot(
        "../fixtures/report-single-sample.golden.txt",
      );
    });

    it("matches the recorded bytes for a colored run of one paired sample", async () => {
      vi.stubEnv("FORCE_COLOR", "1");

      await expect(renderReport(singleSampleResult())).toMatchFileSnapshot(
        "../fixtures/report-single-sample-color.golden.txt",
      );
    });

    it("matches the recorded bytes for a representative colored run", async () => {
      vi.stubEnv("FORCE_COLOR", "1");

      const result = createComparisonResult({
        metrics: {
          "faster/time": signedRankMetric({ verdict: "improved", delta: -17.5, unit: "ns" }),
          "slower/time": signedRankMetric({ verdict: "regressed", delta: 2.4, unit: "ns" }),
          "flat/time": signedRankMetric({ verdict: "no-signal", delta: 0.3, unit: "ns" }),
          "jittery/time": signedRankMetric({ verdict: "unstable", delta: -50, noisePct: 30 }),
        },
        candidates: [
          createCandidate({
            // Inside its noise band, so this golden pins the styling of a
            // geomean that never crosses it.
            kinds: [
              otherKind(-5.8, 3, {
                excluded: [{ metric: "jittery/time", reason: "unstable" }],
                band: 30,
              }),
            ],
          }),
        ],
        worktreesRemoved: 1,
        worktreesLeftBehind: [{ dir: "/tmp/gymrat-abc", error: "is locked" }],
        worktreePruneError: "fatal: not a git repository",
      });

      await expect(renderReport(result)).toMatchFileSnapshot(
        "../fixtures/report-representative-color.golden.txt",
      );
    });

    it("matches the recorded bytes for a verbose degenerate colored run", async () => {
      vi.stubEnv("FORCE_COLOR", "1");

      await expect(renderReport(degenerateResult(), { verbose: true })).toMatchFileSnapshot(
        "../fixtures/report-degenerate-color.golden.txt",
      );
    });

    it("matches the recorded bytes for a verbose two-candidate colored run", async () => {
      vi.stubEnv("FORCE_COLOR", "1");

      await expect(renderReport(twoCandidateResult(), { verbose: true })).toMatchFileSnapshot(
        "../fixtures/report-two-candidates-color.golden.txt",
      );
    });

    it("matches the recorded bytes for a colored run split into kind sections", async () => {
      vi.stubEnv("FORCE_COLOR", "1");

      await expect(renderReport(twoKindResult())).toMatchFileSnapshot(
        "../fixtures/report-sectioned-color.golden.txt",
      );
    });
  });
});

describe("renderMeasureReport", () => {
  afterEach(() => {
    vi.unstubAllEnvs();
  });

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
        configKinds: { memory: { gating: false } },
        expected: "informational — gating off (config: kinds.memory.gating = false)",
      },
      {
        source: "per-metric overrides alone",
        configKinds: undefined,
        expected: "informational — gating off",
      },
    ])("credits $source for a non-gating kind's informational tag", ({ configKinds, expected }) => {
      const report = renderMeasureReport(twoKindMeasurement({ configKinds }));

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
