import { describe, it, expect } from "vitest";

import type { MetricVerdict } from "../../src/verdict/verdict.js";
import { computeVerdicts } from "../../src/verdict/verdict.js";

function getVerdict(result: Record<string, MetricVerdict>, key: string): MetricVerdict {
  const v = result[key];
  if (!v) throw new Error(`Expected verdict for "${key}" but it was missing`);
  return v;
}

describe("computeVerdicts", () => {
  describe("verdict record shape", () => {
    it("returns MetricVerdict with required fields: verdict, method, delta, n", () => {
      const result = computeVerdicts([{ metric: 100 }], [{ metric: 95 }], {
        metric: { direction: "lower", gating: true, exact: true },
      });

      const verdict = getVerdict(result, "metric");
      expect(verdict).toHaveProperty("verdict");
      expect(verdict).toHaveProperty("method");
      expect(verdict).toHaveProperty("delta");
      expect(verdict).toHaveProperty("n");
    });

    it("includes p field only for signed-rank method", () => {
      const result = computeVerdicts([{ metric: 100 }], [{ metric: 95 }], {
        metric: { direction: "lower", gating: true, exact: true },
      });

      const verdict = getVerdict(result, "metric");
      expect(verdict.method).toBe("exact");
      expect("p" in verdict).toBe(false);
    });

    it("includes band field only for band method", () => {
      const result = computeVerdicts([{ metric: 100 }], [{ metric: 95 }], {
        metric: { direction: "lower", gating: true, exact: true },
      });

      const verdict = getVerdict(result, "metric");
      expect(verdict.method).toBe("exact");
      expect("band" in verdict).toBe(false);
    });

    it("marks verdict as tri-state: improved, regressed, or no-signal", () => {
      const result = computeVerdicts([{ metric: 100 }], [{ metric: 90 }], {
        metric: { direction: "lower", gating: true, exact: true },
      });

      expect(getVerdict(result, "metric").verdict).toBe("improved");
    });
  });

  describe("delta computation", () => {
    it("computes delta% = 100 × (median(B) − median(A)) / median(A)", () => {
      const result = computeVerdicts([{ metric: 100 }], [{ metric: 110 }], {
        metric: { direction: "lower", gating: true, exact: true },
      });

      expect(getVerdict(result, "metric").delta).toBe(10);
    });

    it("computes negative delta when second sample is lower", () => {
      const result = computeVerdicts([{ metric: 100 }], [{ metric: 95 }], {
        metric: { direction: "lower", gating: true, exact: true },
      });

      expect(getVerdict(result, "metric").delta).toBeCloseTo(-5, 5);
    });

    it("computes delta from per-run medians", () => {
      const result = computeVerdicts(
        [{ metric: 90 }, { metric: 100 }, { metric: 110 }],
        [{ metric: 85 }, { metric: 95 }, { metric: 105 }],
        { metric: { direction: "lower", gating: true, exact: true } },
      );

      // median(A) = 100, median(B) = 95 → (95 - 100) / 100 * 100 = -5
      expect(getVerdict(result, "metric").delta).toBeCloseTo(-5, 5);
    });

    it("always reports delta even under no-signal verdict", () => {
      const result = computeVerdicts([{ metric: 100 }], [{ metric: 100 }], {
        metric: { direction: "lower", gating: true, exact: true },
      });

      const v = getVerdict(result, "metric");
      expect(v.verdict).toBe("no-signal");
      expect(v.delta).toBe(0);
    });
  });

  describe("exact path behavior", () => {
    it("marks any difference in medians as signal when exact: true", () => {
      const result = computeVerdicts([{ metric: 100 }], [{ metric: 100.01 }], {
        metric: { direction: "lower", gating: true, exact: true },
      });

      expect(getVerdict(result, "metric").verdict).not.toBe("no-signal");
    });

    it("marks equal medians as no-signal", () => {
      const result = computeVerdicts([{ metric: 100 }], [{ metric: 100 }], {
        metric: { direction: "lower", gating: true, exact: true },
      });

      expect(getVerdict(result, "metric").verdict).toBe("no-signal");
    });

    it("uses exact method for exact-flagged metrics", () => {
      const result = computeVerdicts([{ metric: 100 }], [{ metric: 95 }], {
        metric: { direction: "lower", gating: true, exact: true },
      });

      expect(getVerdict(result, "metric").method).toBe("exact");
    });

    it("works with single sample (n=1) for exact path", () => {
      const result = computeVerdicts([{ metric: 100 }], [{ metric: 95 }], {
        metric: { direction: "lower", gating: true, exact: true },
      });

      expect(getVerdict(result, "metric").n).toBe(1);
    });
  });

  describe("direction-awareness", () => {
    it('treats lower delta as improved when direction: "lower"', () => {
      const result = computeVerdicts([{ metric: 100 }], [{ metric: 95 }], {
        metric: { direction: "lower", gating: true, exact: true },
      });

      const v = getVerdict(result, "metric");
      expect(v.verdict).toBe("improved");
      expect(v.delta).toBeCloseTo(-5, 5);
    });

    it('treats higher delta as improved when direction: "higher"', () => {
      const result = computeVerdicts([{ metric: 100 }], [{ metric: 105 }], {
        metric: { direction: "higher", gating: true, exact: true },
      });

      const v = getVerdict(result, "metric");
      expect(v.verdict).toBe("improved");
      expect(v.delta).toBeCloseTo(5, 5);
    });

    it('treats higher delta as regressed when direction: "lower"', () => {
      const result = computeVerdicts([{ metric: 100 }], [{ metric: 105 }], {
        metric: { direction: "lower", gating: true, exact: true },
      });

      const v = getVerdict(result, "metric");
      expect(v.verdict).toBe("regressed");
      expect(v.delta).toBeCloseTo(5, 5);
    });

    it('treats lower delta as regressed when direction: "higher"', () => {
      const result = computeVerdicts([{ metric: 100 }], [{ metric: 95 }], {
        metric: { direction: "higher", gating: true, exact: true },
      });

      const v = getVerdict(result, "metric");
      expect(v.verdict).toBe("regressed");
      expect(v.delta).toBeCloseTo(-5, 5);
    });
  });

  describe("pairing and filtering", () => {
    it("pairs samplesA[i] with samplesB[i] by index", () => {
      const result = computeVerdicts(
        [{ metric: 90 }, { metric: 110 }],
        [{ metric: 85 }, { metric: 105 }],
        { metric: { direction: "lower", gating: true, exact: true } },
      );

      expect(getVerdict(result, "metric").n).toBe(2);
    });

    it("drops windows where metric is missing from samplesA", () => {
      const result = computeVerdicts([{ metric: 100 }, {}], [{ metric: 95 }, { metric: 90 }], {
        metric: { direction: "lower", gating: true, exact: true },
      });

      expect(getVerdict(result, "metric").n).toBe(1);
    });

    it("drops windows where metric is missing from samplesB", () => {
      const result = computeVerdicts([{ metric: 100 }, { metric: 110 }], [{ metric: 95 }, {}], {
        metric: { direction: "lower", gating: true, exact: true },
      });

      expect(getVerdict(result, "metric").n).toBe(1);
    });

    it("skips metrics present on only one side across all windows", () => {
      const result = computeVerdicts([{ metricA: 100 }], [{ metricB: 95 }], {
        metricA: { direction: "lower", gating: true, exact: true },
        metricB: { direction: "lower", gating: true, exact: true },
      });

      expect(result).toStrictEqual({});
    });

    it("keeps metrics with at least one paired window", () => {
      const result = computeVerdicts(
        [{ metric: 100 }, { metric: 90 }],
        [{ metric: 95 }, { metric: 85 }],
        { metric: { direction: "lower", gating: true, exact: true } },
      );

      expect(result).toHaveProperty("metric");
    });
  });

  describe("multiple metrics", () => {
    it("returns verdicts for all metrics with paired samples", () => {
      const result = computeVerdicts([{ a: 100, b: 50 }], [{ a: 95, b: 45 }], {
        a: { direction: "lower", gating: true, exact: true },
        b: { direction: "lower", gating: true, exact: true },
      });

      expect(result).toHaveProperty("a");
      expect(result).toHaveProperty("b");
    });

    it("includes gating flag in metadata (for future use)", () => {
      const result = computeVerdicts([{ a: 100, b: 50 }], [{ a: 95, b: 45 }], {
        a: { direction: "lower", gating: true, exact: true },
        b: { direction: "lower", gating: false, exact: true },
      });

      expect(result).toHaveProperty("a");
      expect(result).toHaveProperty("b");
    });

    it("respects exact flag per metric", () => {
      const result = computeVerdicts(
        [{ exactMetric: 100, otherMetric: 50 }],
        [{ exactMetric: 100.001, otherMetric: 45 }],
        {
          exactMetric: { direction: "lower", gating: true, exact: true },
          otherMetric: { direction: "lower", gating: true, exact: false },
        },
      );

      const v = getVerdict(result, "exactMetric");
      expect(v.verdict).not.toBe("no-signal");
      expect(v.method).toBe("exact");
    });
  });

  describe("edge cases", () => {
    it("handles zero as a valid metric value", () => {
      const result = computeVerdicts([{ metric: 0 }], [{ metric: 0 }], {
        metric: { direction: "lower", gating: true, exact: true },
      });

      expect(getVerdict(result, "metric").delta).toBe(0);
    });

    it("produces NaN delta when medianA is zero and medianB is not", () => {
      const result = computeVerdicts([{ metric: 0 }], [{ metric: 5 }], {
        metric: { direction: "lower", gating: true, exact: true },
      });

      expect(getVerdict(result, "metric").delta).toBeNaN();
    });

    it("handles negative metric values", () => {
      const result = computeVerdicts([{ metric: -100 }], [{ metric: -95 }], {
        metric: { direction: "lower", gating: true, exact: true },
      });

      // (-95 - (-100)) / (-100) * 100 = -5
      expect(getVerdict(result, "metric").delta).toBeCloseTo(-5, 5);
    });

    it("returns empty object when no metrics have paired samples", () => {
      const result = computeVerdicts([{ a: 100 }], [{ b: 95 }], {
        a: { direction: "lower", gating: true, exact: true },
        b: { direction: "lower", gating: true, exact: true },
      });

      expect(result).toStrictEqual({});
    });

    it("handles many paired windows correctly", () => {
      const samplesA = Array.from({ length: 100 }, () => ({ metric: 100 }));
      const samplesB = Array.from({ length: 100 }, () => ({ metric: 95 }));

      const result = computeVerdicts(samplesA, samplesB, {
        metric: { direction: "lower", gating: true, exact: true },
      });

      const v = getVerdict(result, "metric");
      expect(v.n).toBe(100);
      expect(v.delta).toBeCloseTo(-5, 5);
    });
  });
});
