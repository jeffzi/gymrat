import { describe, expect, it } from "vitest";

import { wilcoxonSignedRank } from "../../src/verdict/wilcoxon.js";

describe("wilcoxonSignedRank", () => {
  describe("when input is degenerate", () => {
    const cases: { desc: string; x: number[]; y: number[]; n: number }[] = [
      { desc: "an empty array", x: [], y: [], n: 0 },
      {
        desc: "all-zero diffs",
        x: [5, 10, 15],
        y: [5, 10, 15],
        n: 0,
      },
      { desc: "a single pair", x: [1], y: [2], n: 1 },
    ];

    it.each(cases)("returns p = 1 and n = $n for $desc", ({ x, y, n }) => {
      expect(wilcoxonSignedRank(x, y)).toStrictEqual({ p: 1, n });
    });
  });

  describe("when the input reaches the underlying test", () => {
    it("returns the p-value computed for the pairs", () => {
      const result = wilcoxonSignedRank([10, 12, 14, 16, 18, 20], [9, 10, 13, 14, 15, 17]);

      expect(result.p).toBeCloseTo(0.0345, 4);
      expect(result.n).toBe(6);
    });

    it("clamps a p-value the exact branch inflates above one", () => {
      // Diffs -1, +2, +3, -4 split the signed ranks evenly (W+ = W- = 5), so the
      // exact two-sided p-value is doubled past 1 before clamping.
      const result = wilcoxonSignedRank([11, 12, 13, 10], [12, 10, 10, 14]);

      expect(result).toStrictEqual({ p: 1, n: 4 });
    });

    it("excludes zero-difference pairs from n", () => {
      const result = wilcoxonSignedRank(
        [5, 5, 10, 12, 14, 16, 18, 20],
        [5, 5, 9, 10, 13, 14, 15, 17],
      );

      expect(result.n).toBe(6);
    });
  });
});
