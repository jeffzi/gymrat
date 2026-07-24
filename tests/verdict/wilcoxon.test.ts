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
});
