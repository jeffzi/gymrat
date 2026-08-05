import { describe, expect, it } from "vitest";

import { wilcoxonSignedRank } from "../../src/verdict/wilcoxon.js";

describe("wilcoxonSignedRank", () => {
  describe("when input is degenerate", () => {
    const cases: { desc: string; pairs: [number, number][]; n: number }[] = [
      { desc: "an empty array", pairs: [], n: 0 },
      {
        desc: "all-zero diffs",
        pairs: [
          [5, 5],
          [10, 10],
          [15, 15],
        ],
        n: 0,
      },
      { desc: "a single pair", pairs: [[1, 2]], n: 1 },
    ];

    it.each(cases)("returns p = 1 and n = $n for $desc", ({ pairs, n }) => {
      expect(wilcoxonSignedRank(pairs)).toStrictEqual({ p: 1, n });
    });
  });

  describe("when the input reaches the underlying test", () => {
    it("returns the p-value computed for the pairs", () => {
      const result = wilcoxonSignedRank([
        [10, 9],
        [12, 10],
        [14, 13],
        [16, 14],
        [18, 15],
        [20, 17],
      ]);

      expect(result.p).toBeCloseTo(0.0345, 4);
      expect(result.n).toBe(6);
    });

    it("excludes zero-difference pairs from n", () => {
      const result = wilcoxonSignedRank([
        [5, 5],
        [5, 5],
        [10, 9],
        [12, 10],
        [14, 13],
        [16, 14],
        [18, 15],
        [20, 17],
      ]);

      expect(result.n).toBe(6);
    });
  });
});
