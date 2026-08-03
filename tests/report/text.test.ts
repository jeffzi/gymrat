import { stripVTControlCharacters as stripAnsi } from "node:util";

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { renderReport } from "../../src/report/text.js";
import type { ComparisonResult } from "../../src/report/types.js";
import {
  createCandidate,
  createComparisonResult,
  metricMeta,
  multiCandidateResult,
  signedRankMetric,
  bandMetric,
  exactMetric,
  nWayMetric,
} from "../fixtures/comparison-result.js";

/**
 * Character offsets of every column separator in a rendered table line.
 *
 * Two lines whose separators sit at the same offsets have aligned columns.
 */
function separatorOffsets(line: string): number[] {
  const offsets: number[] = [];
  for (let i = line.indexOf("│"); i !== -1; i = line.indexOf("│", i + 1)) {
    offsets.push(i);
  }
  return offsets;
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
 */
function stylesAt(line: string, marker: string): string[] {
  const index = line.indexOf(marker);
  if (index === -1) {
    throw new Error(`no ${marker} in line: ${JSON.stringify(line)}`);
  }
  const opened = /((?:\x1b\[\d+m)*)$/.exec(line.slice(0, index))?.[1] ?? "";
  return [...opened.matchAll(/\x1b\[(\d+)m/g)].map((match) => match[1] ?? "");
}

/**
 * The row echoing the header labels at the foot of the table.
 *
 * It is the line below the geomean row, which is the last row carrying data.
 */
function echoRow(report: string): string {
  const lines = report.split("\n");
  const geomean = lines.findIndex((line) => stripAnsi(line).startsWith("geomean"));
  const echo = geomean === -1 ? undefined : lines[geomean + 1];
  if (echo === undefined) {
    throw new Error(`no row below the geomean row in report:\n${report}`);
  }
  return echo;
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
                verdict: { verdict: "no-signal", method: "exact", delta: Number.NaN, n: 10 },
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
        candidates: [createCandidate({ geomean: { value: -5, n: 2, excluded: [] } })],
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
            candidates: [
              { median: 120, verdict: { verdict: "no-signal", method: "exact", delta: 0, n: 10 } },
            ],
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
                verdict: {
                  verdict: "improved",
                  method: "signed-rank",
                  delta: -10,
                  n: 10,
                  p: 0.002,
                  noisePct: 2.5,
                  noiseAbs: 2.5,
                },
              },
              {
                median: 1698,
                spread: 2,
                verdict: {
                  verdict: "unstable",
                  method: "signed-rank",
                  delta: -2.1,
                  n: 10,
                  p: 0.32,
                  noisePct: 30,
                  noiseAbs: 30,
                },
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
                verdict: {
                  verdict: "regressed",
                  method: "signed-rank",
                  delta: 2.2,
                  n: 10,
                  p: 0.002,
                  noisePct: 2.5,
                  noiseAbs: 2.5,
                },
              },
              {
                median: 1200000,
                spread: 12,
                verdict: {
                  verdict: "no-signal",
                  method: "signed-rank",
                  delta: -2.1,
                  n: 10,
                  p: 0.32,
                  noisePct: 2.5,
                  noiseAbs: 2.5,
                },
              },
            ],
            meta: metricMeta("encode/time", { unit: "ns" }),
          },
        },
      });
    }

    it("stacks the value and verdict sub-fields within each candidate column", () => {
      const report = renderReport(twoColumnResult());

      const decode = cellsOf(lineStartingWith(report, "decode/time")).map((cell) => cell.trim());
      const encode = cellsOf(lineStartingWith(report, "encode/time")).map((cell) => cell.trim());

      expect
        .soft(decode.slice(2))
        .toStrictEqual(["1.4µs ± 1%  ✓  -10.0%", "1.7µs ±  2%  ≈  unstable"]);
      expect(encode.slice(2)).toStrictEqual(["934ns ± 1%  ✗   +2.2%", "1.2ms ± 12%  ~  -2.1%"]);
    });

    it("measures the candidate sub-fields on the plain text, so colored cells stack the same", () => {
      vi.stubEnv("FORCE_COLOR", "1");

      const bare = stripAnsi(renderReport(twoColumnResult()));

      const decode = cellsOf(lineStartingWith(bare, "decode/time")).map((cell) => cell.trim());
      const encode = cellsOf(lineStartingWith(bare, "encode/time")).map((cell) => cell.trim());

      expect
        .soft(decode.slice(2))
        .toStrictEqual(["1.4µs ± 1%  ✓  -10.0%", "1.7µs ±  2%  ≈  unstable"]);
      expect(encode.slice(2)).toStrictEqual(["934ns ± 1%  ✗   +2.2%", "1.2ms ± 12%  ~  -2.1%"]);
    });
  });

  describe("when rendering the geomean row", () => {
    it("reduces to a counted label and the delta alone", () => {
      const result = createComparisonResult({
        metrics: { "a/time": signedRankMetric({ verdict: "improved", delta: -6 }) },
        candidates: [createCandidate({ geomean: { value: -5.8, n: 4, excluded: [] } })],
      });

      const row = lineStartingWith(renderReport(result), "geomean");

      expect(cellsOf(row).map((cell) => cell.trim())).toStrictEqual([
        "geomean (4 stable metrics)",
        "",
        "",
        "-5.8%",
      ]);
    });

    it("counts a lone stable metric in the singular", () => {
      const result = createComparisonResult({
        metrics: { "a/time": signedRankMetric({ verdict: "improved", delta: -6 }) },
        candidates: [createCandidate({ geomean: { value: -5.8, n: 1, excluded: [] } })],
      });

      const row = lineStartingWith(renderReport(result), "geomean");

      expect(cellsOf(row)[0]?.trim()).toBe("geomean (1 stable metric)");
    });

    it("aligns its delta with the delta column above", () => {
      const result = createComparisonResult({
        metrics: { "a/time": signedRankMetric({ verdict: "improved", delta: -17.9 }) },
        candidates: [createCandidate({ geomean: { value: -6, n: 1, excluded: [] } })],
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
            geomean: {
              value: 0,
              n: 1,
              excluded: [{ metric: "nan-delta/count", reason: "undefined-ratio" }],
            },
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
            geomean: {
              value: Number.NaN,
              n: 0,
              excluded: [{ metric: "jittery/time", reason: "unstable" }],
            },
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

  describe("when closing the table with an echo of the header", () => {
    it("repeats the header labels on the line below the geomean row", () => {
      const result = createComparisonResult({
        metrics: { "a/time": signedRankMetric({ verdict: "improved", delta: -6 }) },
      });

      const echo = echoRow(renderReport(result));

      expect(cellsOf(echo).map((cell) => cell.trim())).toStrictEqual([
        "",
        "main",
        "perf/faster-decode",
        "vs main",
      ]);
    });

    it("lines its columns up with the header", () => {
      const result = createComparisonResult({
        metrics: {
          "a-much-longer-metric/time": signedRankMetric({ verdict: "improved", delta: -6 }),
        },
      });

      const report = renderReport(result);

      expect(separatorOffsets(echoRow(report))).toStrictEqual(
        separatorOffsets(lineStartingWith(report, "metric")),
      );
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
        "✓ 2 improved   ✗ 1 regressed   ≈ 1 unstable   = 0 identical   ~ 1 within noise",
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
        "✓ 1 improved   ✗ 0 regressed   ≈ 0 unstable   = 1 identical   ~ 0 within noise",
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
                verdict: {
                  verdict: "no-signal",
                  method: "band",
                  delta: -0.5,
                  n: 10,
                  usableN: 0,
                  band: 2.5,
                  noisePct: 2.5,
                  noiseAbs: 2.5,
                },
              },
              {
                median: 90,
                spread: 1,
                verdict: {
                  verdict: "improved",
                  method: "signed-rank",
                  delta: -10,
                  n: 10,
                  p: 0.002,
                  noisePct: 2.5,
                  noiseAbs: 2.5,
                },
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
          "jittery/time": signedRankMetric({ verdict: "unstable", delta: -50, noisePct: 30 }),
        },
        candidates: [
          createCandidate({
            geomean: {
              value: -5.8,
              n: 3,
              excluded: [{ metric: "jittery/time", reason: "unstable" }],
            },
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
    ])("paints the $verdict verdict $color on its row", ({ metric, glyph, code }) => {
      const row = lineContaining(renderReport(colorfulResult()), metric);

      expect(stylesAt(row, glyph)).toContain(code);
    });

    it.each([
      { verdict: "within noise", metric: "flat/time" },
      { verdict: "identical", metric: "tied/heap" },
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

    it("dims the echo row closing the table", () => {
      expect(echoRow(renderReport(colorfulResult()))).toMatch(DIMMED_LINE);
    });

    it("keeps the echoed columns aligned with the header once the styles are stripped", () => {
      const bare = stripAnsi(renderReport(colorfulResult()));

      expect(separatorOffsets(echoRow(bare))).toStrictEqual(
        separatorOffsets(lineStartingWith(bare, "metric")),
      );
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
          createCandidate({ label: "faster", geomean: { value: -5, n: 1, excluded: [] } }),
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
      expect.soft(lines[7]).toContain("faster");
      expect.soft(lines[8]).toBe("");
      expect.soft(lines[9]).toContain("✓ 1 improved");
      expect.soft(lines[10]).toBe("");
      expect.soft(lines[11]).toBe("highlights");
      expect.soft(lines[12]).toContain("metric1/time");
      expect(lines).toHaveLength(13);
    });

    it("adds the method block below a blank line when verbose", () => {
      const lines = renderReport(orderedResult(), { verbose: true }).split("\n");

      expect.soft(lines[12]).toContain("metric1/time");
      expect.soft(lines[13]).toBe("");
      expect.soft(lines[14]).toContain("Wilcoxon signed-rank");
      expect(lines).toHaveLength(15);
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

    it("echoes one header label per column below the geomean row", () => {
      const echo = echoRow(renderReport(multiCandidateResult()));

      expect(cellsOf(echo).map((cell) => cell.trim())).toStrictEqual([
        "",
        "main",
        "candidate-a",
        "candidate-b",
        "candidate-c",
      ]);
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
        "candidate-a  ✓ 1 improved   ✗ 0 regressed   ≈ 0 unstable   = 0 identical   ~ 0 within noise",
        "candidate-b  ✓ 0 improved   ✗ 1 regressed   ≈ 0 unstable   = 0 identical   ~ 0 within noise",
        "candidate-c  ✓ 0 improved   ✗ 0 regressed   ≈ 1 unstable   = 0 identical   ~ 0 within noise",
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

    describe("color styling", () => {
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

      it("dims the echo row closing an N-way table", () => {
        expect(echoRow(renderReport(multiCandidateResult()))).toMatch(DIMMED_LINE);
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
      const result = createComparisonResult({
        metrics: {
          "decode/text=digits/time": {
            baselineMedian: 1735,
            baselineSpread: 1,
            candidates: [
              {
                median: 1425,
                spread: 1,
                verdict: {
                  verdict: "improved",
                  method: "signed-rank",
                  delta: -17.9,
                  n: 10,
                  p: 0.002,
                  noisePct: 2.5,
                  noiseAbs: 2.5,
                },
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
                verdict: {
                  verdict: "no-signal",
                  method: "signed-rank",
                  delta: 0.9,
                  n: 10,
                  p: 0.49,
                  noisePct: 2.5,
                  noiseAbs: 2.5,
                },
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
                verdict: {
                  verdict: "regressed",
                  method: "signed-rank",
                  delta: 2.2,
                  n: 10,
                  p: 0.002,
                  noisePct: 2.5,
                  noiseAbs: 2.5,
                },
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
                verdict: { verdict: "improved", method: "exact", delta: -7.9, n: 10 },
              },
            ],
            meta: metricMeta("encode/heap", { exact: true, unit: "bytes" }),
          },
        },
        candidates: [createCandidate({ geomean: { value: -6, n: 4, excluded: [] } })],
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
            baselineSpread: 0,
            candidates: [
              {
                median: 0,
                spread: 0,
                verdict: { verdict: "no-signal", method: "exact", delta: 0, n: 4 },
              },
            ],
            meta: metricMeta("zero-median/time", { exact: true, unit: "ns" }),
          },
          "nan-delta/count": {
            baselineMedian: 0,
            candidates: [
              {
                median: 120,
                verdict: { verdict: "no-signal", method: "exact", delta: Number.NaN, n: 4 },
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
                verdict: {
                  verdict: "improved",
                  method: "band",
                  delta: 30,
                  n: 4,
                  usableN: 4,
                  band: 2.5,
                  noisePct: 2.5,
                  noiseAbs: 2.5,
                },
              },
            ],
            meta: metricMeta("throughput/ops", { direction: "higher", gating: false }),
          },
        },
        candidates: [
          createCandidate({
            geomean: {
              value: 0,
              n: 1,
              excluded: [{ metric: "nan-delta/count", reason: "undefined-ratio" }],
            },
          }),
        ],
        worktreesRemoved: 1,
        worktreesLeftBehind: [{ dir: "/tmp/gymrat-abc123", error: "contains modified files" }],
        worktreePruneError: "could not lock config file",
      });
    }

    it("matches the recorded bytes for degenerate inputs and a dirty cleanup", async () => {
      await expect(renderReport(degenerateResult())).toMatchFileSnapshot(
        "../fixtures/report-degenerate.golden.txt",
      );
    });

    function twoCandidateResult(): ComparisonResult {
      return createComparisonResult({
        candidates: [
          createCandidate({
            label: "perf/simd-decode",
            geomean: { value: -12.4, n: 3, excluded: [] },
          }),
          createCandidate({
            label: "perf/lut-decode",
            geomean: {
              value: 1.2,
              n: 2,
              excluded: [{ metric: "encode/time", reason: "unstable" }],
            },
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
                verdict: {
                  verdict: "improved",
                  method: "signed-rank",
                  delta: -17.9,
                  n: 10,
                  p: 0.002,
                  noisePct: 2.5,
                  noiseAbs: 2.5,
                },
              },
              {
                median: 1698,
                spread: 2,
                verdict: {
                  verdict: "no-signal",
                  method: "signed-rank",
                  delta: -2.1,
                  n: 10,
                  p: 0.32,
                  noisePct: 2.5,
                  noiseAbs: 2.5,
                },
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
                verdict: {
                  verdict: "regressed",
                  method: "signed-rank",
                  delta: 2.2,
                  n: 10,
                  p: 0.002,
                  noisePct: 2.5,
                  noiseAbs: 2.5,
                },
              },
              {
                median: 1200,
                spread: 12,
                // The band method only runs below six pairs, so this metric was
                // dropped on most rounds — which also pins the n= annotation in
                // an N-way cell.
                verdict: {
                  verdict: "unstable",
                  method: "band",
                  delta: 31.3,
                  n: 4,
                  usableN: 4,
                  band: 30,
                  noisePct: 30,
                  noiseAbs: 30,
                },
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
                verdict: { verdict: "improved", method: "exact", delta: -7.9, n: 10 },
              },
              {},
            ],
            meta: metricMeta("encode/heap", { exact: true, unit: "bytes" }),
          },
        },
      });
    }

    it("matches the recorded bytes for a verbose run with two candidates", async () => {
      await expect(renderReport(twoCandidateResult(), { verbose: true })).toMatchFileSnapshot(
        "../fixtures/report-two-candidates.golden.txt",
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
            geomean: {
              value: -5.8,
              n: 3,
              excluded: [{ metric: "jittery/time", reason: "unstable" }],
            },
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
  });
});
