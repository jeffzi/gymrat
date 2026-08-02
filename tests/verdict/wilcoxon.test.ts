import { describe, expect, it } from "vitest";

import { wilcoxonSignedRank } from "../../src/verdict/wilcoxon.js";

describe("wilcoxonSignedRank", () => {
  describe("when input is degenerate", () => {
    it("returns p = 1 and n = 0 for empty array", () => {
      expect(wilcoxonSignedRank([])).toStrictEqual({ p: 1, n: 0 });
    });

    it("returns p = 1 and n = 0 when all diffs are zero", () => {
      const result = wilcoxonSignedRank([
        [5, 5],
        [10, 10],
        [15, 15],
      ]);
      expect(result).toStrictEqual({ p: 1, n: 0 });
    });

    it("returns p = 1 and n = 1 for single pair", () => {
      const result = wilcoxonSignedRank([[1, 2]]);
      expect(result).toStrictEqual({ p: 1, n: 1 });
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
