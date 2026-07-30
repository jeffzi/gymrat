import { describe, it, expect } from "vitest";

import { renderReport } from "../../src/report/text.js";
import { createComparisonResult } from "../fixtures/comparison-result.js";

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

describe("renderReport", () => {
  describe("when rendering with signed-rank method metrics", () => {
    it("includes header line with branch names, sample count, and adapter", () => {
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

    it("includes column headers with metric, old, new, and delta labels", () => {
      const result = createComparisonResult();

      const output = renderReport(result);

      expect(output).toContain("metric");
      expect(output).toContain("old (main)");
      expect(output).toContain("new (perf/faster-decode)");
      expect(output).toContain("vs old");
    });

    it("includes separator line with box-drawing characters", () => {
      const result = createComparisonResult();

      const output = renderReport(result);

      expect(output).toContain("──────────────");
      expect(output).toContain("│");
      expect(output).toContain("┼");
    });

    it.each([
      {
        verdict: "improved" as const,
        metricName: "decode/text=digits time",
        expected: ["✓", "-17.5%", "p=0.002"],
        medianA: 1.726,
        medianB: 1.423,
        deltaVal: -17.5,
        pVal: 0.002,
      },
      {
        verdict: "regressed" as const,
        metricName: "encode time",
        expected: ["✗", "+2.4%"],
        medianA: 912,
        medianB: 934,
        deltaVal: 2.4,
        pVal: 0.014,
      },
      {
        verdict: "no-signal" as const,
        metricName: "decode/text=words time",
        expected: ["~", "p=0.62"],
        medianA: 3.07,
        medianB: 3.081,
        deltaVal: 0.36,
        pVal: 0.62,
      },
    ])(
      "renders metric row with $verdict verdict and glyph",
      ({ verdict, metricName, expected, medianA, medianB, deltaVal, pVal }) => {
        const result = createComparisonResult({
          metrics: {
            [metricName]: {
              medianA,
              medianB,
              spreadA: 1,
              spreadB: 2,
              verdict: {
                verdict,
                method: "signed-rank",
                delta: deltaVal,
                n: 10,
                p: pVal,
                noisePct: 2.5,
              },
              meta: {
                direction: "lower",
                gating: true,
                exact: false,
                unit: "ns",
              },
            },
          },
        });

        const lines = renderReport(result).split("\n");

        // Assert on the metric's own row: the footer legend also contains "~",
        // so a whole-output toContain would pass regardless of the glyph rendered.
        const metricRow = lines.find((line) => line.startsWith(metricName));
        expect(metricRow).toBeDefined();
        for (const s of expected) {
          expect(metricRow!).toContain(s);
        }
      },
    );
  });

  describe("when rendering with band method", () => {
    it("shows band annotation instead of p-value", () => {
      const result = createComparisonResult({
        metrics: {
          "response-time": {
            medianA: 100,
            medianB: 95,
            spreadA: 5,
            spreadB: 4,
            verdict: {
              verdict: "improved",
              method: "band",
              delta: -5.0,
              n: 4,
              band: 2.5,
              noisePct: 2.5,
            },
            meta: {
              direction: "lower",
              gating: true,
              exact: false,
            },
          },
        },
      });

      const output = renderReport(result);

      expect(output).toContain("band ±2.5%");
      expect(output).toContain("n=4");
    });

    it("renders a sub-percent band at its own precision", () => {
      const result = createComparisonResult({
        metrics: {
          "band-metric": {
            medianA: 1000,
            medianB: 950,
            spreadA: 1,
            spreadB: 1,
            verdict: {
              verdict: "improved",
              method: "band",
              delta: -5.0,
              n: 4,
              band: 0.5,
              noisePct: 0.5,
            },
            meta: {
              direction: "lower",
              gating: true,
              exact: false,
            },
          },
        },
      });

      const output = renderReport(result);

      expect(output).toContain("band ±0.5%");
    });
  });

  describe("when rendering with exact method", () => {
    it("shows the exact annotation in place of a p-value or band", () => {
      const result = createComparisonResult({
        metrics: {
          "decode heap_bytes": {
            medianA: 48000,
            medianB: 44200,
            spreadA: undefined,
            spreadB: undefined,
            verdict: {
              verdict: "improved",
              method: "exact",
              delta: -7.9,
              n: 10,
            },
            meta: {
              direction: "lower",
              gating: true,
              exact: true,
              unit: "bytes",
            },
          },
        },
      });

      const output = renderReport(result);

      expect(output).toContain("(exact)");
    });

    it("omits spread when method is exact", () => {
      const result = createComparisonResult({
        metrics: {
          "memory-usage": {
            medianA: 1000000,
            medianB: 950000,
            spreadA: undefined,
            spreadB: undefined,
            verdict: {
              verdict: "improved",
              method: "exact",
              delta: -5.0,
              n: 10,
            },
            meta: {
              direction: "lower",
              gating: true,
              exact: true,
            },
          },
        },
      });

      const output = renderReport(result);

      expect(output).not.toContain("±");
    });
  });

  describe("when rendering one-sided metrics", () => {
    it.each([
      { side: "new", medianA: undefined, medianB: 42, expectedValue: "42" },
      { side: "old", medianA: 100, medianB: undefined, expectedValue: "100" },
    ])(
      "shows the $side value and renders no verdict glyph",
      ({ side, medianA, medianB, expectedValue }) => {
        const result = createComparisonResult({
          metrics: {
            [`${side}-only-metric`]: {
              medianA,
              medianB,
              spreadA: undefined,
              spreadB: undefined,
              verdict: undefined,
              meta: {
                direction: "lower",
                gating: false,
                exact: true,
              },
            },
          },
        });

        const output = renderReport(result);

        expect(output).toContain(`${side}-only-metric`);
        expect(output).toContain(expectedValue);
        expect(output).not.toContain("✓");
        expect(output).not.toContain("✗");
        expect(output).not.toContain("~");
      },
    );
  });

  describe("when rendering geomean row", () => {
    it("includes geomean row after separator with delta percentage", () => {
      const result = createComparisonResult({
        metrics: {
          metric1: {
            medianA: 100,
            medianB: 94,
            spreadA: 1,
            spreadB: 2,
            verdict: {
              verdict: "improved",
              method: "signed-rank",
              delta: -6.0,
              n: 10,
              p: 0.01,
              noisePct: 2.5,
            },
            meta: {
              direction: "lower",
              gating: true,
              exact: false,
            },
          },
        },
        geomean: {
          value: -5.8,
          n: 10,
          excluded: [],
        },
      });

      const output = renderReport(result);

      expect(output).toContain("geomean");
      expect(output).toContain("-5.8%");
    });

    it("shows geomean with gating metrics label", () => {
      const result = createComparisonResult({
        geomean: {
          value: -3.2,
          n: 10,
          excluded: [],
        },
      });

      const output = renderReport(result);

      expect(output).toContain("geomean (gating metrics)");
    });
  });

  describe("when rendering verdict method footer", () => {
    it("includes signed-rank footer when signed-rank method is used", () => {
      const result = createComparisonResult({
        metrics: {
          metric1: {
            medianA: 100,
            medianB: 90,
            spreadA: 1,
            spreadB: 2,
            verdict: {
              verdict: "improved",
              method: "signed-rank",
              delta: -10.0,
              n: 10,
              p: 0.001,
              noisePct: 2.5,
            },
            meta: {
              direction: "lower",
              gating: true,
              exact: false,
            },
          },
        },
      });

      const output = renderReport(result);

      expect(output).toContain("Wilcoxon signed-rank");
      expect(output).toContain("n=10 ≥ 6");
      expect(output).toContain("~ = no signal at α=0.05");
    });

    it("mentions band method footer when band method is used", () => {
      const result = createComparisonResult({
        metrics: {
          metric1: {
            medianA: 100,
            medianB: 95,
            spreadA: 5,
            spreadB: 4,
            verdict: {
              verdict: "improved",
              method: "band",
              delta: -5.0,
              n: 4,
              band: 2.5,
              noisePct: 2.5,
            },
            meta: {
              direction: "lower",
              gating: true,
              exact: false,
            },
          },
        },
      });

      const output = renderReport(result);

      expect(output).toContain("noise band ±(half-range × K)");
      expect(output).toContain("n=4");
      expect(output).toContain("below signed-rank floor (6 pairs)");
    });
  });

  describe("when rendering worktree footer", () => {
    it.each([
      { removed: 0, leftBehind: [], expected: "0 worktrees removed · 0 left behind" },
      { removed: 1, leftBehind: [], expected: "1 worktree removed · 0 left behind" },
      {
        removed: 2,
        leftBehind: [
          { dir: "/tmp/gymrat-a", error: "locked" },
          { dir: "/tmp/gymrat-b", error: "locked" },
          { dir: "/tmp/gymrat-c", error: "locked" },
        ],
        expected: "2 worktrees removed · 3 left behind",
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

      expect(output).toContain(
        "left behind: /tmp/gymrat-abc (contains modified or untracked files)",
      );
      expect(output).toContain("left behind: /tmp/gymrat-def (is locked)");
    });

    it("reports the prune failure with git's reason", () => {
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

    it("reports left-behind worktrees and the prune failure together", () => {
      const result = createComparisonResult({
        worktreesRemoved: 0,
        worktreesLeftBehind: [{ dir: "/tmp/gymrat-abc", error: "is locked" }],
        worktreePruneError: "fatal: not a git repository",
      });

      const lines = renderReport(result).split("\n");

      expect(lines.at(-3)).toContain("0 worktrees removed · 1 left behind");
      expect(lines.at(-2)).toBe("  left behind: /tmp/gymrat-abc (is locked)");
      expect(lines.at(-1)).toBe("  worktree prune failed: fatal: not a git repository");
    });
  });

  describe("when rendering full report", () => {
    it("emits sections in order: header, table, geomean, method footer, worktree footer", () => {
      const result = createComparisonResult({
        labels: ["main", "faster"],
        samples: 10,
        adapter: "mitata",
        metrics: {
          metric1: {
            medianA: 1000,
            medianB: 900,
            spreadA: 2,
            spreadB: 3,
            verdict: {
              verdict: "improved",
              method: "signed-rank",
              delta: -10.0,
              n: 10,
              p: 0.005,
              noisePct: 2.5,
            },
            meta: {
              direction: "lower",
              gating: true,
              exact: false,
            },
          },
          metric2: {
            medianA: 500,
            medianB: 510,
            spreadA: 1,
            spreadB: 1,
            verdict: {
              verdict: "no-signal",
              method: "signed-rank",
              delta: 2.0,
              n: 10,
              p: 0.5,
              noisePct: 2.5,
            },
            meta: {
              direction: "lower",
              gating: false,
              exact: false,
            },
          },
        },
        geomean: {
          value: -5.0,
          n: 10,
          excluded: [],
        },
        worktreesRemoved: 0,
        worktreesLeftBehind: [],
      });

      const lines = renderReport(result).split("\n");

      // Each section's content is asserted by its own test; this pins the order.
      expect(lines[0]).toContain("gymrat compare · main ↔ faster");
      expect(lines[1]).toContain("old (main)");
      expect(lines[2]).toMatch(/^─+┼/);
      expect(lines[3]).toContain("metric1");
      expect(lines[4]).toContain("metric2");
      expect(lines[5]).toMatch(/^─+┼/);
      expect(lines[6]).toContain("geomean");
      expect(lines[7]).toContain("Wilcoxon signed-rank");
      expect(lines[8]).toContain("worktrees removed");
      expect(lines).toHaveLength(9);
    });
  });

  describe("when delta is zero", () => {
    it("still renders a delta cell rather than leaving it blank", () => {
      const result = createComparisonResult({
        metrics: {
          latency: {
            medianA: 100,
            medianB: 100,
            spreadA: 1,
            spreadB: 1,
            verdict: {
              verdict: "no-signal",
              method: "signed-rank",
              delta: 0,
              n: 10,
              p: 1.0,
              noisePct: 2.5,
            },
            meta: {
              direction: "lower",
              gating: true,
              exact: false,
            },
          },
        },
      });

      const lines = renderReport(result).split("\n");

      // The footer legend also contains "~", so the glyph is asserted on the row itself.
      const metricRow = lines.find((line) => line.startsWith("latency"));
      expect(metricRow).toBeDefined();
      expect(metricRow!).toContain("~");
      expect(metricRow!).toContain("0.0%");
    });
  });

  describe("when metric names have different lengths", () => {
    it("pads cells so the column separators line up", () => {
      const result = createComparisonResult({
        metrics: {
          short: {
            medianA: 1,
            medianB: 2,
            spreadA: 1,
            spreadB: 1,
            verdict: {
              verdict: "improved",
              method: "signed-rank",
              delta: -50.0,
              n: 10,
              p: 0.01,
              noisePct: 2.5,
            },
            meta: {
              direction: "lower",
              gating: true,
              exact: false,
            },
          },
          "very-long-metric-name": {
            medianA: 100000,
            medianB: 90000,
            spreadA: 1,
            spreadB: 1,
            verdict: {
              verdict: "improved",
              method: "signed-rank",
              delta: -10.0,
              n: 10,
              p: 0.01,
              noisePct: 2.5,
            },
            meta: {
              direction: "lower",
              gating: true,
              exact: false,
            },
          },
        },
      });

      const output = renderReport(result);
      const lines = output.split("\n");
      const headerLine = lines.find((line) => line.includes("old (main)"))!;
      const shortLine = lines.find((line) => line.startsWith("short"))!;
      const longLine = lines.find((line) => line.startsWith("very-long-metric-name"))!;

      expect(separatorOffsets(shortLine)).toStrictEqual(separatorOffsets(headerLine));
      expect(separatorOffsets(longLine)).toStrictEqual(separatorOffsets(headerLine));
    });

    it("aligns the geomean row separators with the table when metric names are short", () => {
      const result = createComparisonResult({
        metrics: {
          "a/time": {
            medianA: 100,
            medianB: 95,
            spreadA: 1,
            spreadB: 1,
            verdict: {
              verdict: "improved",
              method: "signed-rank",
              delta: -5,
              n: 10,
              p: 0.01,
              noisePct: 2.5,
            },
            meta: { direction: "lower", gating: true, exact: false, unit: "ns" },
          },
        },
        geomean: { value: -5, n: 1, excluded: [] },
      });

      const lines = renderReport(result).split("\n");

      const headerLine = lines.find((line) => line.includes("old (main)"));
      const geomeanLine = lines.find((line) => line.includes("geomean"));
      expect(headerLine).toBeDefined();
      expect(geomeanLine).toBeDefined();
      expect(separatorOffsets(geomeanLine!)).toStrictEqual(separatorOffsets(headerLine!));
    });
  });

  describe("when rendering with mixed methods", () => {
    it("uses signed-rank footer when multiple metrics include signed-rank", () => {
      const result = createComparisonResult({
        metrics: {
          metric1: {
            medianA: 100,
            medianB: 90,
            spreadA: 1,
            spreadB: 2,
            verdict: {
              verdict: "improved",
              method: "signed-rank",
              delta: -10.0,
              n: 10,
              p: 0.001,
              noisePct: 2.5,
            },
            meta: {
              direction: "lower",
              gating: true,
              exact: false,
            },
          },
          metric2: {
            medianA: 1000,
            medianB: 950,
            spreadA: 5,
            spreadB: 4,
            verdict: {
              verdict: "improved",
              method: "band",
              delta: -5.0,
              n: 4,
              band: 2.5,
              noisePct: 2.5,
            },
            meta: {
              direction: "lower",
              gating: true,
              exact: false,
            },
          },
        },
      });

      const output = renderReport(result);

      expect(output).toContain("Wilcoxon signed-rank");
    });

    it("uses band footer when only band methods are used (no signed-rank)", () => {
      const result = createComparisonResult({
        metrics: {
          metric1: {
            medianA: 100,
            medianB: 95,
            spreadA: 5,
            spreadB: 4,
            verdict: {
              verdict: "no-signal",
              method: "band",
              delta: -5.0,
              n: 4,
              band: 2.5,
              noisePct: 2.5,
            },
            meta: {
              direction: "lower",
              gating: true,
              exact: false,
            },
          },
          metric2: {
            medianA: 500,
            medianB: 480,
            spreadA: 3,
            spreadB: 5,
            verdict: {
              verdict: "no-signal",
              method: "band",
              delta: -4.0,
              n: 5,
              band: 1.8,
              noisePct: 1.8,
            },
            meta: {
              direction: "lower",
              gating: true,
              exact: false,
            },
          },
        },
      });

      const output = renderReport(result);

      expect(output).toContain("noise band ±(half-range × K)");
      expect(output).not.toContain("Wilcoxon");
    });
  });

  describe("when only exact methods are used", () => {
    it("omits both the signed-rank and band footers", () => {
      const result = createComparisonResult({
        metrics: {
          metric1: {
            medianA: 100,
            medianB: 95,
            spreadA: undefined,
            spreadB: undefined,
            verdict: {
              verdict: "improved",
              method: "exact",
              delta: -5.0,
              n: 10,
            },
            meta: {
              direction: "lower",
              gating: true,
              exact: true,
            },
          },
          metric2: {
            medianA: 200,
            medianB: 180,
            spreadA: undefined,
            spreadB: undefined,
            verdict: {
              verdict: "improved",
              method: "exact",
              delta: -10.0,
              n: 10,
            },
            meta: {
              direction: "lower",
              gating: true,
              exact: true,
            },
          },
        },
      });

      const output = renderReport(result);

      expect(output).not.toContain("Wilcoxon");
      expect(output).not.toContain("noise band");
      expect(output).toContain("worktrees removed");
    });
  });

  describe("when there are no metrics", () => {
    it("still renders the header, geomean row and footers", () => {
      const result = createComparisonResult({
        metrics: {},
      });

      const output = renderReport(result);

      expect(output).toContain("gymrat compare");
      expect(output).toContain("metric");
      expect(output).toContain("geomean");
      expect(output).toContain("worktrees removed");
    });
  });

  describe("when the table mixes one-sided and two-sided metrics", () => {
    it("renders a row for each", () => {
      const result = createComparisonResult({
        metrics: {
          "one-sided": {
            medianA: 100,
            medianB: undefined,
            spreadA: 1,
            spreadB: undefined,
            verdict: undefined,
            meta: {
              direction: "lower",
              gating: false,
              exact: false,
            },
          },
          "two-sided": {
            medianA: 200,
            medianB: 180,
            spreadA: 2,
            spreadB: 3,
            verdict: {
              verdict: "improved",
              method: "exact",
              delta: -10.0,
              n: 10,
            },
            meta: {
              direction: "lower",
              gating: true,
              exact: true,
            },
          },
        },
      });

      const lines = renderReport(result).split("\n");
      const oneSidedRow = lines.find((line) => line.startsWith("one-sided"))!;
      const twoSidedRow = lines.find((line) => line.startsWith("two-sided"))!;

      // A trailing separator means the delta cell was empty and got trimmed away.
      expect(oneSidedRow).toContain("100 ± 1%");
      expect(oneSidedRow.endsWith("│")).toBe(true);
      expect(twoSidedRow).toContain("(exact)");
    });
  });

  describe("when delta is NaN", () => {
    it("renders the glyph and annotation without a delta percentage", () => {
      const result = createComparisonResult({
        metrics: {
          "nan-delta": {
            medianA: 0,
            medianB: 100,
            spreadA: 1,
            spreadB: 1,
            verdict: {
              verdict: "no-signal",
              method: "exact",
              delta: Number.NaN,
              n: 10,
            },
            meta: {
              direction: "lower",
              gating: true,
              exact: true,
            },
          },
        },
      });

      const output = renderReport(result);

      expect(output).toContain("nan-delta");
      expect(output).toContain("~");
      expect(output).toContain("(exact)");
      // The line should have the glyph and annotation but no delta value like "-50%"
      const lines = output.split("\n");
      const nanDeltaLine = lines.find((line) => line.includes("nan-delta"));
      expect(nanDeltaLine).toBeDefined();
      expect(nanDeltaLine!).toMatch(/~\s+\(exact\)/);
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
