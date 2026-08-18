import { stripVTControlCharacters as stripAnsi } from "node:util";

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { renderReport } from "../../src/report/text.js";
import type { ComparisonResult, ReportOptions } from "../../src/report/types.js";
import type { KindAggregate } from "../../src/verdict/aggregate.js";
import {
  bandMetric,
  bandVerdict,
  createCandidate,
  createComparisonResult,
  exactMetric,
  geomeanOf,
  groupedComparison,
  memoryKind,
  metricMeta,
  otherKind,
  signedRankMetric,
  signedRankVerdict,
  singleSampleResult,
  timeKind,
  twoKindResult,
} from "../fixtures/comparison-result.js";

function withoutGatedGeomean(kind: KindAggregate): KindAggregate {
  const { gatedGeomean: _, ...rest } = kind;
  return rest;
}

/** Character offsets of every occurrence of `glyph` in a rendered table line. */
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
/** Every rendered table row of a report, styling stripped, in report order. */
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
        "⚑ time gated geomean +3.1% exceeded --fail-on geomean:2",
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
            kinds: [timeKind(), withoutGatedGeomean(otherKind(9, 1))],
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
        expected: ["⚑ time gated geomean +5.0% exceeded --fail-on geomean:2"],
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
        "    ⚑ time gated geomean +4.0% exceeded --fail-on geomean:2",
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
        worktreesLeftBehind: [{ dir: "/tmp/gymrat-abc", error: "is locked" }],
      });

      const lines = renderReport(result).split("\n");

      expect.soft(lines.at(-3)).toContain("Hint:");
      expect.soft(lines.at(-2)).toBe("1 worktree removed · 1 left behind");
      expect(lines.at(-1)).toBe("  left behind: /tmp/gymrat-abc (is locked)");
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

    it("suppresses the footer when every worktree was removed cleanly", () => {
      const result = createComparisonResult({
        worktreesRemoved: 3,
        worktreesLeftBehind: [],
      });

      const output = renderReport(result);

      expect.soft(output).not.toContain("worktree");
      expect(output).not.toContain("left behind");
    });

    it("still renders the footer when leftover worktrees need attention", () => {
      const result = createComparisonResult({
        worktreesRemoved: 2,
        worktreesLeftBehind: [{ dir: "/tmp/gymrat-abc", error: "is locked" }],
      });

      const output = renderReport(result);

      expect.soft(output).toContain("2 worktrees removed · 1 left behind");
      expect(output).toContain("left behind: /tmp/gymrat-abc (is locked)");
    });

    it("still renders the footer when only the prune step failed", () => {
      const result = createComparisonResult({
        worktreesRemoved: 3,
        worktreesLeftBehind: [],
        worktreePruneError: "fatal: not a git repository",
      });

      const output = renderReport(result);

      expect(output).toContain("worktree prune failed: fatal: not a git repository");
    });

    it.each([
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
});
