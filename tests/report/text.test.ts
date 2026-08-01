import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { renderReport } from "../../src/report/text.js";
import type { ComparisonResult } from "../../src/report/types.js";
import {
  createCandidate,
  createComparisonResult,
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
    it("names both branches, the sample count and the adapter", () => {
      const result = createComparisonResult({
        baselineLabel: "main",
        candidates: [createCandidate({ label: "experiment" })],
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
      const output = renderReport(result);

      const headerLine = lineStartingWith(output, "metric");

      expect
        .soft(cellsOf(headerLine).map((cell) => cell.trim()))
        .toStrictEqual(["metric", "main", "perf/faster-decode", "vs main"]);
      expect.soft(output).toContain("gymrat compare");
      expect.soft(output).toContain("geomean");
      expect(output).toContain("legend:");
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
            baselineMedian: 2048,
            baselineSpread: 2,
            candidates: [{}],
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
            baselineMedian: 0,
            candidates: [
              {
                median: 120,
                verdict: { verdict: "no-signal", method: "exact", delta: Number.NaN, n: 10 },
              },
            ],
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

  describe("when rendering the geomean row", () => {
    it("repeats both labels in its value cells and counts the metrics behind the figure", () => {
      const result = createComparisonResult({
        metrics: { "a/time": signedRankMetric({ verdict: "improved", delta: -6 }) },
        candidates: [createCandidate({ geomean: { value: -5.8, n: 4, excluded: [] } })],
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

      expect(row).toContain("0.0%  1 stable metric · 1 excluded: undefined-ratio");
    });

    it("reports no stable gating metrics when every one was excluded", () => {
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
          "tied/heap": bandMetric({ verdict: "no-signal", delta: -0.5, n: 10, usableN: 3 }),
        },
      });
    }

    it("marks the row identical rather than within noise", () => {
      const row = lineStartingWith(renderReport(identicalResult()), "tied/heap");

      expect(cellsOf(row).at(-1)?.trim()).toBe("=  -0.5%  ±2.5%");
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
                  usableN: 3,
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
            meta: { direction: "lower", gating: true, exact: false, unit: "ns" },
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
     * it out of line. Stripping the styles back off has to land on the
     * uncolored report, byte for byte.
     */
    it("pads on the plain text, so stripping the styles restores the uncolored report", () => {
      const result = colorfulResult();

      vi.stubEnv("FORCE_COLOR", undefined);
      vi.stubEnv("NO_COLOR", "1");
      const plain = renderReport(result);

      vi.stubEnv("NO_COLOR", undefined);
      vi.stubEnv("FORCE_COLOR", "1");
      const colored = renderReport(result);

      expect.soft(colored).toContain("\x1b[");
      expect(stripAnsi(colored)).toBe(plain);
    });

    it.each([
      { verdict: "improved", metric: "faster/time", glyph: "✓", color: "green", code: "32" },
      { verdict: "regressed", metric: "slower/time", glyph: "✗", color: "red", code: "31" },
      { verdict: "unstable", metric: "jittery/time", glyph: "≈", color: "yellow", code: "33" },
    ])("paints the $verdict glyph $color", ({ metric, glyph, code }) => {
      const row = lineContaining(renderReport(colorfulResult()), metric);

      expect(stylesAt(row, glyph)).toContain(code);
    });

    it("paints the no-signal glyph with no color of its own", () => {
      const row = lineContaining(renderReport(colorfulResult()), "flat/time");

      expect(stylesAt(row, "~")).toStrictEqual([]);
    });

    it.each([
      { verdict: "within noise", metric: "flat/time" },
      { verdict: "unstable", metric: "jittery/time" },
    ])("dims the whole $verdict row", ({ metric }) => {
      const row = lineContaining(renderReport(colorfulResult()), metric);

      expect(row).toMatch(DIMMED_LINE);
    });

    it.each([
      { verdict: "improved", metric: "faster/time" },
      { verdict: "regressed", metric: "slower/time" },
    ])("leaves the $verdict row at full brightness", ({ metric }) => {
      const row = lineContaining(renderReport(colorfulResult()), metric);

      expect(row).not.toMatch(DIMMED_LINE);
    });

    it("emboldens the header row and the geomean figure", () => {
      const report = renderReport(colorfulResult());

      expect.soft(lineContaining(report, "vs main")).toMatch(/^\x1b\[1m.*\x1b\[22m$/);
      expect(stylesAt(lineContaining(report, "geomean"), "-5.8%")).toContain("1");
    });

    it("emboldens 'gymrat compare' in the report header", () => {
      const header = lineContaining(renderReport(colorfulResult()), "gymrat compare");

      expect(stylesAt(header, "gymrat compare")).toContain("1");
    });

    it("dims each · separator in the report header", () => {
      const header = lineContaining(renderReport(colorfulResult()), "gymrat compare");

      expect(stylesAt(header, "·")).toContain("2");
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
          "tied/heap": bandMetric({ verdict: "no-signal", delta: -0.5, n: 10, usableN: 3 }),
        },
      });
      const summary = lineContaining(renderReport(result), "identical");

      expect(stylesAt(summary, "=")).toContain("36");
    });

    it("paints the identical glyph on its row with no color of its own", () => {
      const result = createComparisonResult({
        metrics: {
          "tied/heap": bandMetric({ verdict: "no-signal", delta: -0.5, n: 10, usableN: 3 }),
        },
      });
      const row = lineContaining(renderReport(result), "tied/heap");

      expect(stylesAt(row, "=")).toStrictEqual([]);
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

    it("dims the legend line overall", () => {
      const legend = lineContaining(renderReport(colorfulResult()), "legend:");

      expect(legend).toMatch(DIMMED_LINE);
    });

    it.each([
      { glyph: "✓", code: "32", color: "green" },
      { glyph: "✗", code: "31", color: "red" },
      { glyph: "≈", code: "33", color: "yellow" },
    ])("colors the legend $glyph glyph $color inside the dim line", ({ glyph, code }) => {
      const legend = lineContaining(renderReport(colorfulResult()), "legend:");

      expect(stylesAt(legend, glyph)).toContain(code);
    });

    it("leaves the legend ~ glyph uncolored", () => {
      const legend = lineContaining(renderReport(colorfulResult()), "legend:");

      expect(stylesAt(legend, "~")).toStrictEqual([]);
    });

    it("dims the verdict method description", () => {
      const result = createComparisonResult({
        metrics: {
          "a/time": signedRankMetric({ verdict: "improved", delta: -10, unit: "ns" }),
        },
      });
      const method = lineContaining(renderReport(result), "Wilcoxon");

      expect(method).toMatch(DIMMED_LINE);
    });

    it("dims the noise-band description", () => {
      const result = createComparisonResult({
        metrics: { "a/time": bandMetric({ verdict: "no-signal", delta: -5 }) },
      });
      const band = lineContaining(renderReport(result), "noise band");

      expect(band).toMatch(DIMMED_LINE);
    });

    it("styles the Hint: label yellow and underlined", () => {
      const result = createComparisonResult({
        metrics: { "a/time": bandMetric({ verdict: "no-signal", delta: -5 }) },
      });
      const hint = lineContaining(renderReport(result), "Hint:");

      expect.soft(stylesAt(hint, "Hint:")).toContain("33");
      expect(stylesAt(hint, "Hint:")).toContain("4");
    });

    it("renders the hint sentence text plain in colored mode", () => {
      const result = createComparisonResult({
        metrics: { "a/time": bandMetric({ verdict: "no-signal", delta: -5 }) },
      });
      const hint = lineContaining(renderReport(result), "Hint:");
      const afterLabel = hint.slice(hint.indexOf("re-run"));

      expect(afterLabel).not.toContain("\x1b[2m");
    });

    it("renders the hint line entirely plain when color is off", () => {
      vi.stubEnv("FORCE_COLOR", undefined);
      vi.stubEnv("NO_COLOR", "1");

      const result = createComparisonResult({
        metrics: { "a/time": bandMetric({ verdict: "no-signal", delta: -5 }) },
      });
      const hint = lineContaining(renderReport(result), "Hint:");

      expect(hint).not.toContain("\x1b[");
    });

    it("dims the band annotation on improved and regressed rows", () => {
      const report = renderReport(colorfulResult());

      expect.soft(stylesAt(lineContaining(report, "faster/time"), "±2.5%")).toContain("2");
      expect(stylesAt(lineContaining(report, "slower/time"), "±2.5%")).toContain("2");
    });

    it("dims the repeated labels and provenance in the geomean row", () => {
      const report = renderReport(colorfulResult());
      const geomean = lineContaining(report, "geomean");

      expect.soft(stylesAt(geomean, "main")).toContain("2");
      expect.soft(stylesAt(geomean, "perf/faster-decode")).toContain("2");
      expect(stylesAt(geomean, "3 stable")).toContain("2");
    });
  });

  describe("when ordering the report sections", () => {
    it("emits table, summary, highlights, legend and method in that order", () => {
      const result = createComparisonResult({
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

  describe("when rendering more than one candidate", () => {
    /**
     * One baseline and three candidates whose verdicts on the same metric disagree.
     *
     * Every candidate was judged against the same baseline samples, so a renderer
     * that reused one candidate's verdict for the next would collapse the three
     * columns, summaries and highlight subsections into one.
     */
    function multiCandidateResult(): ComparisonResult {
      return createComparisonResult({
        baselineLabel: "main",
        candidates: [
          createCandidate({ label: "candidate-a", geomean: { value: -10, n: 1, excluded: [] } }),
          createCandidate({ label: "candidate-b", geomean: { value: 4, n: 1, excluded: [] } }),
          createCandidate({ label: "candidate-c", geomean: { value: 0, n: 1, excluded: [] } }),
        ],
        metrics: {
          "decode/time": {
            baselineMedian: 100,
            baselineSpread: 1,
            candidates: [
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
              {
                median: 104,
                spread: 1,
                verdict: {
                  verdict: "regressed",
                  method: "signed-rank",
                  delta: 4,
                  n: 10,
                  p: 0.002,
                  noisePct: 2.5,
                  noiseAbs: 2.5,
                },
              },
              {
                median: 150,
                spread: 3,
                verdict: {
                  verdict: "unstable",
                  method: "band",
                  delta: 50,
                  n: 10,
                  usableN: 3,
                  band: 30,
                  noisePct: 30,
                  noiseAbs: 30,
                },
              },
            ],
            meta: { direction: "lower", gating: true, exact: false, unit: "ns" },
          },
        },
      });
    }

    it("heads one column per candidate with its comparison against the baseline", () => {
      const headerLine = lineStartingWith(renderReport(multiCandidateResult()), "metric");

      expect(cellsOf(headerLine).map((cell) => cell.trim())).toStrictEqual([
        "metric",
        "main",
        "candidate-a vs main",
        "candidate-b vs main",
        "candidate-c vs main",
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

    it("carries one geomean figure per candidate column and repeats the baseline label", () => {
      const row = lineStartingWith(renderReport(multiCandidateResult()), "geomean");

      expect(cellsOf(row).map((cell) => cell.trim())).toStrictEqual([
        "geomean (gating metrics)",
        "main",
        "-10.0%  1 stable metric",
        "+4.0%  1 stable metric",
        "0.0%  1 stable metric",
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

      it("dims a row only once every candidate on it stayed flat or unstable", () => {
        const report = renderReport(dimmingResult());

        expect.soft(lineContaining(report, "flat/time")).toMatch(DIMMED_LINE);
        expect(lineContaining(report, "mixed/time")).not.toMatch(DIMMED_LINE);
      });

      it("pads on the plain text, so stripping the styles restores the uncolored report", () => {
        const result = multiCandidateResult();

        vi.stubEnv("FORCE_COLOR", undefined);
        vi.stubEnv("NO_COLOR", "1");
        const plain = renderReport(result);

        vi.stubEnv("NO_COLOR", undefined);
        vi.stubEnv("FORCE_COLOR", "1");
        const colored = renderReport(result);

        expect.soft(colored).toContain("\x1b[");
        expect(stripAnsi(colored)).toBe(plain);
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

      it("dims the baseline label and provenance in N-way geomean cells", () => {
        const report = renderReport(multiCandidateResult());
        const geomean = lineContaining(report, "geomean");

        expect.soft(stylesAt(geomean, "main")).toContain("2");
        expect(stylesAt(geomean, "1 stable metric")).toContain("2");
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
            meta: { direction: "lower", gating: true, exact: false, unit: "ns" },
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
            meta: { direction: "lower", gating: true, exact: false, unit: "ns" },
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
            meta: { direction: "lower", gating: true, exact: false, unit: "ns" },
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
            meta: { direction: "lower", gating: true, exact: true, unit: "bytes" },
          },
        },
        candidates: [createCandidate({ geomean: { value: -6, n: 4, excluded: [] } })],
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
            baselineMedian: 0,
            baselineSpread: 0,
            candidates: [
              {
                median: 0,
                spread: 0,
                verdict: { verdict: "no-signal", method: "exact", delta: 0, n: 4 },
              },
            ],
            meta: { direction: "lower", gating: true, exact: true, unit: "ns" },
          },
          "nan-delta/count": {
            baselineMedian: 0,
            candidates: [
              {
                median: 120,
                verdict: { verdict: "no-signal", method: "exact", delta: Number.NaN, n: 4 },
              },
            ],
            meta: { direction: "lower", gating: true, exact: true },
          },
          "old-side-only/time": {
            baselineMedian: 2048,
            baselineSpread: 2,
            candidates: [{}],
            meta: { direction: "lower", gating: true, exact: false, unit: "ns" },
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
            meta: { direction: "higher", gating: false, exact: false },
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

      await expect(renderReport(result)).toMatchFileSnapshot(
        "../fixtures/report-degenerate.golden.txt",
      );
    });

    it("matches the recorded bytes for a run with two candidates", async () => {
      const result = createComparisonResult({
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
            meta: { direction: "lower", gating: true, exact: false, unit: "ns" },
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
            meta: { direction: "lower", gating: true, exact: false, unit: "ns" },
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
            meta: { direction: "lower", gating: true, exact: true, unit: "bytes" },
          },
        },
      });

      await expect(renderReport(result)).toMatchFileSnapshot(
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

    it("matches the recorded bytes for a degenerate colored run", async () => {
      vi.stubEnv("FORCE_COLOR", "1");

      const result = createComparisonResult({
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
            meta: { direction: "lower", gating: true, exact: true, unit: "ns" },
          },
          "nan-delta/count": {
            baselineMedian: 0,
            candidates: [
              {
                median: 120,
                verdict: { verdict: "no-signal", method: "exact", delta: Number.NaN, n: 4 },
              },
            ],
            meta: { direction: "lower", gating: true, exact: true },
          },
          "old-side-only/time": {
            baselineMedian: 2048,
            baselineSpread: 2,
            candidates: [{}],
            meta: { direction: "lower", gating: true, exact: false, unit: "ns" },
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
            meta: { direction: "higher", gating: false, exact: false },
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

      await expect(renderReport(result)).toMatchFileSnapshot(
        "../fixtures/report-degenerate-color.golden.txt",
      );
    });

    it("matches the recorded bytes for a two-candidate colored run", async () => {
      vi.stubEnv("FORCE_COLOR", "1");

      const result = createComparisonResult({
        baselineLabel: "main",
        candidates: [
          createCandidate({ label: "candidate-a", geomean: { value: -10, n: 1, excluded: [] } }),
          createCandidate({ label: "candidate-b", geomean: { value: 4, n: 1, excluded: [] } }),
          createCandidate({ label: "candidate-c", geomean: { value: 0, n: 1, excluded: [] } }),
        ],
        metrics: {
          "decode/time": {
            baselineMedian: 100,
            baselineSpread: 1,
            candidates: [
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
              {
                median: 104,
                spread: 1,
                verdict: {
                  verdict: "regressed",
                  method: "signed-rank",
                  delta: 4,
                  n: 10,
                  p: 0.002,
                  noisePct: 2.5,
                  noiseAbs: 2.5,
                },
              },
              {
                median: 150,
                spread: 3,
                verdict: {
                  verdict: "unstable",
                  method: "band",
                  delta: 50,
                  n: 10,
                  usableN: 3,
                  band: 30,
                  noisePct: 30,
                  noiseAbs: 30,
                },
              },
            ],
            meta: { direction: "lower", gating: true, exact: false, unit: "ns" },
          },
        },
      });

      await expect(renderReport(result)).toMatchFileSnapshot(
        "../fixtures/report-two-candidates-color.golden.txt",
      );
    });
  });
});
