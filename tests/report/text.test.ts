import { stripVTControlCharacters as stripAnsi } from "node:util";

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { renderReport } from "../../src/report/text.js";
import type { ComparisonResult } from "../../src/report/types.js";
import {
  bandMetric,
  createCandidate,
  createComparisonResult,
  exactMetric,
  exactVerdict,
  metricMeta,
  nWayMetric,
  otherKind,
  signedRankMetric,
  signedRankVerdict,
  twoKindResult,
} from "../fixtures/comparison-result.js";

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

beforeEach(() => {
  vi.stubEnv("NO_COLOR", "1");
  vi.stubEnv("FORCE_COLOR", undefined);
});

afterEach(() => {
  vi.unstubAllEnvs();
});

describe("renderReport", () => {
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
});
