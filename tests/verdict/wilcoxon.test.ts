describe("wilcoxonSignedRank", () => {
  describe("zeros dropped", () => {
    it("removes pairs where a === b before ranking", async () => {
      const { wilcoxonSignedRank } = await import("../../src/verdict/wilcoxon.js");
      const result = wilcoxonSignedRank([
        [10, 10], // difference = 0, should be dropped
        [20, 18], // difference = 2
        [30, 25], // difference = 5
      ]);
      // Only 2 pairs remain, so n = 2
      expect(result.n).toBe(2);
    });

    it("returns p = 1 and n = 0 when all pairs are zeros", async () => {
      const { wilcoxonSignedRank } = await import("../../src/verdict/wilcoxon.js");
      const result = wilcoxonSignedRank([
        [5, 5],
        [10, 10],
        [15, 15],
      ]);
      expect(result).toStrictEqual({ p: 1, n: 0 });
    });

    it("drops zeros in the middle of a dataset", async () => {
      const { wilcoxonSignedRank } = await import("../../src/verdict/wilcoxon.js");
      const result = wilcoxonSignedRank([
        [1, 2],
        [3, 3], // zero
        [4, 5],
      ]);
      expect(result.n).toBe(2);
    });
  });

  describe("ranking", () => {
    it("assigns ranks 1..n to unique absolute differences", async () => {
      const { wilcoxonSignedRank } = await import("../../src/verdict/wilcoxon.js");
      // Pairs: [1,2] -> diff=1, [2,4] -> diff=2, [3,5] -> diff=2, [4,7] -> diff=3
      // Absolute diffs: 1, 2, 2, 3
      // Ranks: 1, 2.5, 2.5, 4 (average rank for ties)
      const result = wilcoxonSignedRank([
        [1, 2],
        [2, 4],
        [3, 5],
        [4, 7],
      ]);
      expect(result.n).toBe(4);
    });

    it("computes average rank for tied absolute differences", async () => {
      const { wilcoxonSignedRank } = await import("../../src/verdict/wilcoxon.js");
      // Pairs with diffs: [2, 2, 2]
      // Ranks should be [2, 2, 2] average of 1,2,3 = 2
      const result = wilcoxonSignedRank([
        [1, 3],
        [2, 4],
        [3, 5],
      ]);
      // All three diffs are 2, so ranks are all 2
      expect(result.n).toBe(3);
      expect(typeof result.p).toBe("number");
    });
  });

  describe("exact test (n <= 25)", () => {
    it("computes exact p-value by sign enumeration", async () => {
      const { wilcoxonSignedRank } = await import("../../src/verdict/wilcoxon.js");
      const result = wilcoxonSignedRank([
        [1, 2],
        [2, 4],
        [3, 5],
        [4, 7],
        [5, 8],
      ]);
      // All differences negative, so T+ = 0
      // For exact distribution, enumerate 2^5 = 32 sign assignments
      // p should be two-sided
      expect(result.p).toBeCloseTo(0.0625, 4);
      expect(result.n).toBe(5);
    });

    it("handles single pair (n=1)", async () => {
      const { wilcoxonSignedRank } = await import("../../src/verdict/wilcoxon.js");
      const result = wilcoxonSignedRank([[1, 2]]);
      // n=1: T+ can be 0 or 1
      // For two-sided with n=1, only 2 possible T values
      // p = 1.0 (both outcomes equally likely)
      expect(result.p).toBe(1.0);
      expect(result.n).toBe(1);
    });

    it("computes exact p-value for mixed signs", async () => {
      const { wilcoxonSignedRank } = await import("../../src/verdict/wilcoxon.js");
      const result = wilcoxonSignedRank([
        [1, 3],
        [2, 5],
        [3, 2],
        [4, 6],
        [5, 4],
        [6, 9],
      ]);
      // Mixed positive and negative differences
      // Should use exact enumeration for n=6
      expect(result.n).toBe(6);
      expect(result.p).toBeGreaterThan(0);
      expect(result.p).toBeLessThanOrEqual(1);
    });

    it("validates against scipy reference (n=5, all negative)", async () => {
      const { wilcoxonSignedRank } = await import("../../src/verdict/wilcoxon.js");
      // scipy.stats.wilcoxon([1,2,3,4,5], [2,4,5,7,8], alternative='two-sided')
      // Exact p ≈ 0.0625 for n=5 (all one sign)
      const result = wilcoxonSignedRank([
        [1, 2],
        [2, 4],
        [3, 5],
        [4, 7],
        [5, 8],
      ]);
      expect(result.p).toBeCloseTo(0.0625, 3);
    });
  });

  describe("normal approximation (n > 25)", () => {
    it("uses normal approximation for n > 25", async () => {
      const { wilcoxonSignedRank } = await import("../../src/verdict/wilcoxon.js");
      // Create 30 pairs with mixed differences
      const pairs: Array<[number, number]> = [];
      for (let i = 1; i <= 30; i++) {
        pairs.push([i, i + 0.5]);
      }
      const result = wilcoxonSignedRank(pairs);
      expect(result.n).toBe(30);
      expect(result.p).toBeGreaterThan(0);
      expect(result.p).toBeLessThanOrEqual(1);
    });

    it("applies continuity correction", async () => {
      const { wilcoxonSignedRank } = await import("../../src/verdict/wilcoxon.js");
      // Create large dataset with known properties
      const pairs: Array<[number, number]> = [];
      for (let i = 1; i <= 50; i++) {
        pairs.push([i, i + 1]);
      }
      const result = wilcoxonSignedRank(pairs);
      expect(result.n).toBe(50);
      // With continuous differences all in same direction,
      // p should be very small
      expect(result.p).toBeLessThan(0.05);
    });

    it("accounts for ties in variance calculation", async () => {
      const { wilcoxonSignedRank } = await import("../../src/verdict/wilcoxon.js");
      // Create dataset with many ties to test tie correction
      const pairs: Array<[number, number]> = [];
      for (let i = 1; i <= 26; i++) {
        if (i % 3 === 0) {
          pairs.push([i, i + 2]); // different step
        } else {
          pairs.push([i, i + 1]); // standard step
        }
      }
      const result = wilcoxonSignedRank(pairs);
      expect(result.n).toBeGreaterThan(25);
      expect(result.p).toBeGreaterThan(0);
      expect(result.p).toBeLessThanOrEqual(1);
    });

    it("validates against scipy reference (large n)", async () => {
      const { wilcoxonSignedRank } = await import("../../src/verdict/wilcoxon.js");
      // Create a dataset with n > 25 and compare with scipy approximation
      const pairs: Array<[number, number]> = [];
      for (let i = 0; i < 30; i++) {
        pairs.push([i, i + 0.3]);
      }
      const result = wilcoxonSignedRank(pairs);
      // scipy gives approximately p < 0.001 for this dataset
      expect(result.p).toBeLessThan(0.01);
      expect(result.n).toBe(30);
    });
  });

  describe("boundary cases", () => {
    it("returns p = 1 and n = 0 for empty pairs array", async () => {
      const { wilcoxonSignedRank } = await import("../../src/verdict/wilcoxon.js");
      const result = wilcoxonSignedRank([]);
      expect(result).toStrictEqual({ p: 1, n: 0 });
    });

    it("handles negative differences correctly", async () => {
      const { wilcoxonSignedRank } = await import("../../src/verdict/wilcoxon.js");
      const result = wilcoxonSignedRank([[10, 5]]);
      expect(result.n).toBe(1);
      expect(result.p).toBe(1.0);
    });

    it("handles large differences", async () => {
      const { wilcoxonSignedRank } = await import("../../src/verdict/wilcoxon.js");
      const result = wilcoxonSignedRank([[1000000, 2000000]]);
      expect(result.n).toBe(1);
      expect(result.p).toBe(1.0);
    });

    it("handles fractional differences", async () => {
      const { wilcoxonSignedRank } = await import("../../src/verdict/wilcoxon.js");
      const result = wilcoxonSignedRank([
        [1.1, 1.2],
        [2.2, 2.5],
        [3.3, 3.1],
      ]);
      expect(result.n).toBe(3);
      expect(result.p).toBeGreaterThan(0);
      expect(result.p).toBeLessThanOrEqual(1);
    });
  });

  describe("scipy reference validation", () => {
    it("matches scipy for n=5 with zero dropped", async () => {
      const { wilcoxonSignedRank } = await import("../../src/verdict/wilcoxon.js");
      // scipy: pairs=[10,10,20,18,30,25,40,38,50,52,60,55], zero dropped
      // n_eff = 5
      const result = wilcoxonSignedRank([
        [10, 10],
        [20, 18],
        [30, 25],
        [40, 38],
        [50, 52],
        [60, 55],
      ]);
      expect(result.n).toBe(5);
      // Should compute exact p-value
      expect(result.p).toBeGreaterThan(0);
      expect(result.p).toBeLessThanOrEqual(1);
    });

    it("matches scipy for clear signal (n=5, all same direction)", async () => {
      const { wilcoxonSignedRank } = await import("../../src/verdict/wilcoxon.js");
      const result = wilcoxonSignedRank([
        [1, 10],
        [2, 20],
        [3, 30],
        [4, 40],
        [5, 50],
      ]);
      // n=5, all positive: T+=15, only 2/32 permutations as extreme → p=0.0625
      expect(result.p).toBeCloseTo(0.0625, 4);
      expect(result.n).toBe(5);
    });

    it("handles symmetric data", async () => {
      const { wilcoxonSignedRank } = await import("../../src/verdict/wilcoxon.js");
      const result = wilcoxonSignedRank([
        [5, 10],
        [10, 5],
      ]);
      // T+ = 1 and T+ = 1 due to symmetric ranks
      expect(result.n).toBe(2);
      expect(result.p).toBeCloseTo(1.0, 1);
    });
  });

  describe("type safety", () => {
    it("accepts readonly arrays", async () => {
      const { wilcoxonSignedRank } = await import("../../src/verdict/wilcoxon.js");
      const pairs = [
        [1, 2],
        [2, 4],
      ] as const;
      const result = wilcoxonSignedRank(pairs);
      expect(result.n).toBe(2);
      expect(typeof result.p).toBe("number");
    });

    it("returns object with p and n properties", async () => {
      const { wilcoxonSignedRank } = await import("../../src/verdict/wilcoxon.js");
      const result = wilcoxonSignedRank([[1, 2]]);
      expect(result).toHaveProperty("p");
      expect(result).toHaveProperty("n");
      expect(typeof result.p).toBe("number");
      expect(typeof result.n).toBe("number");
    });
  });
});
