import { describe, it, expect } from "vitest";

import { renderReport } from "../../src/report/text.js";
import type { ComparisonResult } from "../../src/report/types.js";
import type { ApproximateVerdictValue } from "../../src/verdict/verdict.js";
import { createComparisonResult } from "../fixtures/comparison-result.js";

type Metrics = ComparisonResult["metrics"];
type MetricEntry = Metrics[string];

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

/** `text` with every ANSI style sequence removed. */
function stripAnsi(text: string): string {
  return text.replace(/\x1b\[\d+m/g, "");
}

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

/** The lines of the `highlights` block, its heading excluded. */
function highlightLines(report: string): string[] {
  const lines = report.split("\n");
  const start = lines.indexOf("highlights");
  if (start === -1) {
    return [];
  }
  const rest = lines.slice(start + 1);
  const end = rest.indexOf("");
  return end === -1 ? rest : rest.slice(0, end);
}

/** A two-sided metric whose verdict came from the signed-rank method. */
function signedRankMetric(options: {
  verdict: ApproximateVerdictValue;
  delta: number;
  medianA?: number;
  medianB?: number;
  p?: number;
  noisePct?: number;
  unit?: "ns" | "bytes";
  gating?: boolean;
  n?: number;
}): MetricEntry {
  const {
    verdict,
    delta,
    medianA = 100,
    medianB = medianA * (1 + delta / 100),
    p = 0.01,
    noisePct = 2.5,
    unit,
    gating = true,
    n = 10,
  } = options;
  return {
    medianA,
    medianB,
    spreadA: 1,
    spreadB: 1,
    verdict: { verdict, method: "signed-rank", delta, n, p, noisePct },
    meta: { direction: "lower", gating, exact: false, unit },
  };
}

/** A two-sided metric whose verdict fell back to the noise band. */
function bandMetric(options: {
  verdict: ApproximateVerdictValue;
  delta: number;
  noisePct?: number;
  n?: number;
}): MetricEntry {
  const { verdict, delta, noisePct = 2.5, n = 4 } = options;
  return {
    medianA: 100,
    medianB: 100 + delta,
    spreadA: 5,
    spreadB: 4,
    verdict: { verdict, method: "band", delta, n, band: noisePct, noisePct },
    meta: { direction: "lower", gating: true, exact: false },
  };
}

/** A counted metric, compared exactly rather than statistically. */
function exactMetric(options: {
  delta: number;
  medianA?: number;
  medianB?: number;
  n?: number;
  unit?: "ns" | "bytes";
}): MetricEntry {
  const {
    delta,
    medianA = 1000,
    medianB = 1000 * (1 + delta / 100),
    n = 10,
    unit = "bytes",
  } = options;
  return {
    medianA,
    medianB,
    verdict: { verdict: delta < 0 ? "improved" : "regressed", method: "exact", delta, n },
    meta: { direction: "lower", gating: true, exact: true, unit },
  };
}

describe("renderReport", () => {
  describe("when rendering the run header", () => {
    it("names both branches, the sample count and the adapter", () => {
      const result = createComparisonResult({
        labels: ["main", "experiment"],
        samples: 10,
        adapter: "mitata",
      });

      const output = renderReport(result);

      expect(output).toContain(
        "gymrat compare · main ↔ experiment · 10 paired samples · adapter: mitata",
      );
    });
  });

  describe("when rendering the table header", () => {
    it("labels the value columns with the two target labels and the delta with the baseline", () => {
      const result = createComparisonResult();

      const headerLine = lineStartingWith(renderReport(result), "metric");

      expect(cellsOf(headerLine).map((cell) => cell.trim())).toStrictEqual([
        "metric",
        "main",
        "perf/faster-decode",
        "vs main",
      ]);
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
        desc: "prints the word unstable in place of a delta the noise swamped",
        verdict: "unstable" as const,
        delta: -50,
        noisePct: 30,
        expected: "≈  unstable  ±30.0%",
      },
    ])("$desc", ({ verdict, delta, noisePct, expected }) => {
      const result = createComparisonResult({
        metrics: { "decode/time": signedRankMetric({ verdict, delta, noisePct, unit: "ns" }) },
      });

      const row = lineStartingWith(renderReport(result), "decode/time");

      expect(row).toContain(expected);
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
            medianA: 2048,
            spreadA: 2,
            meta: { direction: "lower", gating: false, exact: false, unit: "ns" },
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
            medianA: 0,
            medianB: 120,
            verdict: { verdict: "no-signal", method: "exact", delta: Number.NaN, n: 10 },
            meta: { direction: "lower", gating: true, exact: true },
          },
        },
      });

      const row = lineStartingWith(renderReport(result), "nan-delta/count");

      expect(cellsOf(row).at(-1)?.trim()).toBe("~");
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
          short: signedRankMetric({ verdict: "improved", delta: -50, medianA: 914, unit: "ns" }),
          "very-long-metric-name": signedRankMetric({
            verdict: "improved",
            delta: -10,
            medianA: 49152,
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
            medianA: 100000,
            unit: "ns",
          }),
        },
        geomean: { value: -5, n: 2, excluded: [] },
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

  describe("when rendering the geomean row", () => {
    it("repeats both labels in its value cells and counts the metrics behind the figure", () => {
      const result = createComparisonResult({
        metrics: { "a/time": signedRankMetric({ verdict: "improved", delta: -6 }) },
        geomean: { value: -5.8, n: 4, excluded: [] },
      });

      const row = lineStartingWith(renderReport(result), "geomean");

      expect(cellsOf(row).map((cell) => cell.trim())).toStrictEqual([
        "geomean (gating metrics)",
        "main",
        "perf/faster-decode",
        "-5.8%  4 stable metrics",
      ]);
    });

    it("names how many metrics were excluded and why", () => {
      const result = createComparisonResult({
        metrics: { "a/time": signedRankMetric({ verdict: "improved", delta: -6 }) },
        geomean: {
          value: 0,
          n: 1,
          excluded: [{ metric: "nan-delta/count", reason: "undefined-ratio" }],
        },
      });

      const row = lineStartingWith(renderReport(result), "geomean");

      expect(row).toContain("0.0%  1 stable metric · 1 excluded: undefined-ratio");
    });

    it("reports no stable gating metrics when every one was excluded", () => {
      const result = createComparisonResult({
        metrics: { "jittery/time": signedRankMetric({ verdict: "unstable", delta: -50 }) },
        geomean: {
          value: Number.NaN,
          n: 0,
          excluded: [{ metric: "jittery/time", reason: "unstable" }],
        },
      });

      const row = lineStartingWith(renderReport(result), "geomean");

      expect(cellsOf(row).at(-1)?.trim()).toBe("—  no stable gating metrics");
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
        "✓ 2 improved   ✗ 1 regressed   ≈ 1 unstable   ~ 1 within noise",
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
        "✗ slower/time    +2.2%  p=0.002",
        "✓ cheaper/heap   -7.9%  (exact)",
        "≈ jittery/time  unstable  band ±30.0%",
      ]);
    });

    it("omits the block when nothing improved, regressed or was unstable", () => {
      const result = createComparisonResult({
        metrics: { "flat/time": signedRankMetric({ verdict: "no-signal", delta: 0.2 }) },
      });

      const output = renderReport(result);

      expect(output).not.toContain("highlights");
    });
  });

  describe("when rendering the legend", () => {
    it("explains every glyph and names the baseline", () => {
      const result = createComparisonResult({
        metrics: { "a/time": signedRankMetric({ verdict: "improved", delta: -10 }) },
      });

      const legend = lineStartingWith(renderReport(result), "legend:");

      expect.soft(legend).toContain("✓ improved");
      expect.soft(legend).toContain("✗ regressed");
      expect.soft(legend).toContain("≈ unstable");
      expect.soft(legend).toContain("~ within noise");
      expect(legend).toContain("candidates are judged against main");
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

      const output = renderReport(result);

      expect.soft(output).toContain("Wilcoxon signed-rank");
      expect(output).toContain("n=10 ≥ 6");
    });

    it("names the noise band and hints at more samples when it is the only method", () => {
      const result = createComparisonResult({
        metrics: { "a/time": bandMetric({ verdict: "no-signal", delta: -5 }) },
      });

      const output = renderReport(result);

      expect.soft(output).toContain("noise band ±(half-range × K)");
      expect.soft(output).toContain("below signed-rank floor (6 pairs)");
      expect.soft(output).not.toContain("Wilcoxon");
      expect(output).toContain("Hint: re-run with --samples 6 or more for statistical verdicts");
    });

    it("names no method, and drops the hint, when every metric was exact", () => {
      const result = createComparisonResult({
        metrics: { "a/heap": exactMetric({ delta: -7.9 }) },
      });

      const output = renderReport(result);

      expect.soft(output).not.toContain("Wilcoxon");
      expect.soft(output).not.toContain("noise band");
      expect(output).not.toContain("Hint:");
    });

    it("drops the hint when the signed-rank test carried the run", () => {
      const result = createComparisonResult({
        metrics: { "a/time": signedRankMetric({ verdict: "improved", delta: -10 }) },
      });

      const output = renderReport(result);

      expect(output).not.toContain("Hint:");
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
      const report = renderReport(mixedMethodResult());

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

  describe("when rendering a run with no metrics", () => {
    it("still renders the header, the table, the geomean row and the legend", () => {
      const result = createComparisonResult({ metrics: {} });

      const output = renderReport(result);

      expect.soft(output).toContain("gymrat compare");
      expect.soft(output).toContain("metric");
      expect.soft(output).toContain("geomean");
      expect(output).toContain("legend:");
    });
  });

  describe("when rendering with color", () => {
    /** A run whose rows cover every verdict class, plus a geomean figure. */
    function colorfulResult(): ComparisonResult {
      return createComparisonResult({
        metrics: {
          "faster/time": signedRankMetric({ verdict: "improved", delta: -17.5, unit: "ns" }),
          "slower/time": signedRankMetric({ verdict: "regressed", delta: 2.4, unit: "ns" }),
          "flat/time": signedRankMetric({ verdict: "no-signal", delta: 0.3, unit: "ns" }),
          "jittery/time": signedRankMetric({ verdict: "unstable", delta: -50, noisePct: 30 }),
        },
        geomean: {
          value: -5.8,
          n: 3,
          excluded: [{ metric: "jittery/time", reason: "unstable" }],
        },
        worktreesRemoved: 1,
        worktreesLeftBehind: [{ dir: "/tmp/gymrat-abc", error: "is locked" }],
        worktreePruneError: "fatal: not a git repository",
      });
    }

    it("leaves the report unstyled when the caller asks for no color", () => {
      const output = renderReport(colorfulResult());

      expect(output).not.toContain("\x1b[");
    });

    /**
     * The guard against styling a cell before padding it.
     *
     * `padEnd` counts escape codes as characters, so a renderer that styles
     * first and pads second pads short and slides every column to the right of
     * it out of line. Stripping the styles back off has to land on the
     * uncolored report, byte for byte.
     */
    it("pads on the plain text, so stripping the styles restores the uncolored report", () => {
      const result = colorfulResult();

      const colored = renderReport(result, true);

      expect.soft(colored).toContain("\x1b[");
      expect(stripAnsi(colored)).toBe(renderReport(result, false));
    });

    it.each([
      { verdict: "improved", metric: "faster/time", glyph: "✓", color: "green", code: "32" },
      { verdict: "regressed", metric: "slower/time", glyph: "✗", color: "red", code: "31" },
      { verdict: "unstable", metric: "jittery/time", glyph: "≈", color: "amber", code: "33" },
    ])("paints the $verdict glyph $color", ({ metric, glyph, code }) => {
      const row = lineContaining(renderReport(colorfulResult(), true), metric);

      expect(stylesAt(row, glyph)).toContain(code);
    });

    it("paints the no-signal glyph with no color of its own", () => {
      const row = lineContaining(renderReport(colorfulResult(), true), "flat/time");

      expect(stylesAt(row, "~")).toStrictEqual([]);
    });

    it.each([
      { verdict: "within noise", metric: "flat/time" },
      { verdict: "unstable", metric: "jittery/time" },
    ])("dims the whole $verdict row", ({ metric }) => {
      const row = lineContaining(renderReport(colorfulResult(), true), metric);

      expect(row).toMatch(/^\x1b\[2m.*\x1b\[22m$/);
    });

    it.each([
      { verdict: "improved", metric: "faster/time" },
      { verdict: "regressed", metric: "slower/time" },
    ])("leaves the $verdict row at full brightness", ({ metric }) => {
      const row = lineContaining(renderReport(colorfulResult(), true), metric);

      expect(row).not.toContain("\x1b[2m");
    });

    it("emboldens the header row and the geomean figure", () => {
      const report = renderReport(colorfulResult(), true);

      expect.soft(lineContaining(report, "vs main")).toMatch(/^\x1b\[1m.*\x1b\[22m$/);
      expect(stylesAt(lineContaining(report, "geomean"), "-5.8%")).toContain("1");
    });
  });

  describe("when rendering a full report", () => {
    it("emits table, summary, highlights, legend and method in that order", () => {
      const result = createComparisonResult({
        labels: ["main", "faster"],
        metrics: {
          "metric1/time": signedRankMetric({ verdict: "improved", delta: -10, unit: "ns" }),
          "metric2/time": signedRankMetric({
            verdict: "no-signal",
            delta: 2,
            gating: false,
            unit: "ns",
          }),
        },
        geomean: { value: -5, n: 1, excluded: [] },
      });

      const lines = renderReport(result).split("\n");

      // Each section's content is asserted by its own test; this pins the order.
      expect.soft(lines[0]).toContain("gymrat compare · main ↔ faster");
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
      expect.soft(lines[12]).toBe("");
      expect.soft(lines[13]).toContain("legend:");
      expect.soft(lines[14]).toContain("Wilcoxon signed-rank");
      expect(lines).toHaveLength(15);
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
            medianA: 1735,
            medianB: 1425,
            spreadA: 1,
            spreadB: 1,
            verdict: {
              verdict: "improved",
              method: "signed-rank",
              delta: -17.9,
              n: 10,
              p: 0.002,
              noisePct: 2.5,
            },
            meta: { direction: "lower", gating: true, exact: false, unit: "ns" },
          },
          "decode/text=words/time": {
            medianA: 3065,
            medianB: 3093,
            spreadA: 1,
            spreadB: 3,
            verdict: {
              verdict: "no-signal",
              method: "signed-rank",
              delta: 0.9,
              n: 10,
              p: 0.49,
              noisePct: 2.5,
            },
            meta: { direction: "lower", gating: true, exact: false, unit: "ns" },
          },
          "encode/time": {
            medianA: 914,
            medianB: 934,
            spreadA: 1,
            spreadB: 1,
            verdict: {
              verdict: "regressed",
              method: "signed-rank",
              delta: 2.2,
              n: 10,
              p: 0.002,
              noisePct: 2.5,
            },
            meta: { direction: "lower", gating: true, exact: false, unit: "ns" },
          },
          "encode/heap": {
            medianA: 49152,
            medianB: 45261,
            spreadA: 0,
            spreadB: 0,
            verdict: { verdict: "improved", method: "exact", delta: -7.9, n: 10 },
            meta: { direction: "lower", gating: true, exact: true, unit: "bytes" },
          },
        },
        geomean: { value: -6, n: 4, excluded: [] },
      });

      await expect(renderReport(result)).toMatchFileSnapshot(
        "../fixtures/report-representative.golden.txt",
      );
    });

    it("matches the recorded bytes for degenerate inputs and a dirty cleanup", async () => {
      const result = createComparisonResult({
        samples: 4,
        adapter: "metric-lines",
        metrics: {
          "zero-median/time": {
            medianA: 0,
            medianB: 0,
            spreadA: 0,
            spreadB: 0,
            verdict: { verdict: "no-signal", method: "exact", delta: 0, n: 4 },
            meta: { direction: "lower", gating: true, exact: true, unit: "ns" },
          },
          "nan-delta/count": {
            medianA: 0,
            medianB: 120,
            verdict: { verdict: "no-signal", method: "exact", delta: Number.NaN, n: 4 },
            meta: { direction: "lower", gating: true, exact: true },
          },
          "old-side-only/time": {
            medianA: 2048,
            spreadA: 2,
            meta: { direction: "lower", gating: true, exact: false, unit: "ns" },
          },
          "throughput/ops": {
            medianA: 1200,
            medianB: 1560,
            spreadA: 5,
            spreadB: 4,
            verdict: {
              verdict: "improved",
              method: "band",
              delta: 30,
              n: 4,
              band: 2.5,
              noisePct: 2.5,
            },
            meta: { direction: "higher", gating: false, exact: false },
          },
        },
        geomean: {
          value: 0,
          n: 1,
          excluded: [{ metric: "nan-delta/count", reason: "undefined-ratio" }],
        },
        worktreesRemoved: 1,
        worktreesLeftBehind: [{ dir: "/tmp/gymrat-abc123", error: "contains modified files" }],
        worktreePruneError: "could not lock config file",
      });

      await expect(renderReport(result)).toMatchFileSnapshot(
        "../fixtures/report-degenerate.golden.txt",
      );
    });
  });
});
