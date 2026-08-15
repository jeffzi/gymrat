import { describe, expect, it } from "vitest";

import { computeMetricStats } from "../../src/sampling.js";

describe("computeMetricStats", () => {
  describe("when the median is denormal", () => {
    it("returns no spread instead of Infinity", () => {
      // Arrange: the median of [0, 5e-324, 1] is 5e-324 (denormal).
      // halfRange is (1 - 0) / 2 = 0.5, so the ratio 0.5 / 5e-324 overflows
      // to Infinity — which is not a meaningful spread.
      const values = [0, 5e-324, 1];

      // Act
      const stats = computeMetricStats(values);

      // Assert
      expect(stats.spread).toBeUndefined();
    });
  });
});
