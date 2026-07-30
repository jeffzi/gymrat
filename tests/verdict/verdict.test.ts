import { describe, it, expect } from "vitest";

import type { BandVerdict, MetricVerdict, SignedRankVerdict } from "../../src/verdict/verdict.js";
import { computeGeomean, computeVerdicts } from "../../src/verdict/verdict.js";

function getVerdict(result: Record<string, MetricVerdict>, key: string): MetricVerdict {
  const verdict = result[key];
  if (!verdict) throw new Error(`Expected verdict for "${key}" but it was missing`);
  return verdict;
}

/**
 * Fetch a verdict that must have come from the signed-rank path, so `p` is readable.
 *
 * `MetricVerdict` is discriminated on `method`, and `expect(v.method).toBe(...)`
 * does not narrow the binding — asserting the method and reading the statistic
 * has to happen in one step.
 */
function getSignedRankVerdict(
  result: Record<string, MetricVerdict>,
  key: string,
): SignedRankVerdict {
  const verdict = getVerdict(result, key);
  if (verdict.method !== "signed-rank") {
    throw new Error(`Expected a signed-rank verdict for "${key}" but got "${verdict.method}"`);
  }
  return verdict;
}

/**
 * Fetch a verdict that must have come from the band path, so `band` is readable.
 */
function getBandVerdict(result: Record<string, MetricVerdict>, key: string): BandVerdict {
  const verdict = getVerdict(result, key);
  if (verdict.method !== "band") {
    throw new Error(`Expected a band verdict for "${key}" but got "${verdict.method}"`);
  }
  return verdict;
}

/**
 * Fetch a verdict that must have come from a non-exact path, so `noisePct` is readable.
 */
function getNoisyVerdict(
  result: Record<string, MetricVerdict>,
  key: string,
): SignedRankVerdict | BandVerdict {
  const verdict = getVerdict(result, key);
  if (verdict.method === "exact") {
    throw new Error(`Expected a non-exact verdict for "${key}" but got "exact"`);
  }
  return verdict;
}

const METRIC_EXACT_LOWER = { metric: { direction: "lower" as const, gating: true, exact: true } };
const METRIC_EXACT_HIGHER = { metric: { direction: "higher" as const, gating: true, exact: true } };

const METRIC_APPROX_LOWER = { metric: { direction: "lower" as const, gating: true, exact: false } };
const METRIC_APPROX_HIGHER = {
  metric: { direction: "higher" as const, gating: true, exact: false },
};

function createSamples(n: number, value: number): Array<{ metric: number }> {
  return Array.from({ length: n }, () => ({ metric: value }));
}

/**
 * Six paired windows noisy enough to reach noisePct = 30, staying on the signed-rank path.
 *
 * pairedA: median = 100, halfRange = 20 → 0.2; pairedB: median = 50, halfRange = 10 → 0.2
 * noisePct = max(1.5 × 100 × 0.2, 0.5) = 30
 * All six diffs are non-zero and negative → n = 6, p = 0.03125 → improved
 */
const NOISY_SIGNED_RANK_SAMPLES = {
  samplesA: [
    { metric: 80 },
    { metric: 90 },
    { metric: 100 },
    { metric: 100 },
    { metric: 110 },
    { metric: 120 },
  ],
  samplesB: [
    { metric: 40 },
    { metric: 45 },
    { metric: 50 },
    { metric: 50 },
    { metric: 55 },
    { metric: 60 },
  ],
};

/**
 * Two paired windows noisy enough to reach noisePct = 30, staying on the band path.
 *
 * pairedA: median = 100, halfRange = 20 → 0.2; pairedB: median = 10, halfRange = 2 → 0.2
 * noisePct = band = max(1.5 × 100 × 0.2, 0.5) = 30
 * delta = (10 − 100) / 100 × 100 = −90, and |−90| > 30 → improved
 */
const NOISY_BAND_SAMPLES = {
  samplesA: [{ metric: 80 }, { metric: 120 }],
  samplesB: [{ metric: 8 }, { metric: 12 }],
};

describe("computeVerdicts", () => {
  describe("verdict record shape", () => {
    it("carries exactly verdict, method, delta and n for the exact method", () => {
      const result = computeVerdicts([{ metric: 100 }], [{ metric: 95 }], METRIC_EXACT_LOWER);

      // toStrictEqual also proves p and band are absent for this method.
      expect(getVerdict(result, "metric")).toStrictEqual({
        verdict: "improved",
        method: "exact",
        delta: -5,
        n: 1,
      });
    });
  });

  describe("delta computation", () => {
    it("computes delta% = 100 × (median(B) − median(A)) / median(A)", () => {
      const result = computeVerdicts([{ metric: 100 }], [{ metric: 110 }], METRIC_EXACT_LOWER);

      expect(getVerdict(result, "metric").delta).toBe(10);
    });

    it("computes negative delta when second sample is lower", () => {
      const result = computeVerdicts([{ metric: 100 }], [{ metric: 95 }], METRIC_EXACT_LOWER);

      expect(getVerdict(result, "metric").delta).toBeCloseTo(-5, 5);
    });

    it("computes delta from per-run medians", () => {
      const result = computeVerdicts(
        [{ metric: 90 }, { metric: 100 }, { metric: 110 }],
        [{ metric: 85 }, { metric: 95 }, { metric: 105 }],
        METRIC_EXACT_LOWER,
      );

      // median(A) = 100, median(B) = 95 → (95 - 100) / 100 * 100 = -5
      expect(getVerdict(result, "metric").delta).toBeCloseTo(-5, 5);
    });

    it("always reports delta even under no-signal verdict", () => {
      const result = computeVerdicts([{ metric: 100 }], [{ metric: 100 }], METRIC_EXACT_LOWER);

      const verdict = getVerdict(result, "metric");
      expect(verdict.verdict).toBe("no-signal");
      expect(verdict.delta).toBe(0);
    });
  });

  describe("exact path behavior", () => {
    it("marks any difference in medians as signal when exact: true", () => {
      const result = computeVerdicts([{ metric: 100 }], [{ metric: 100.01 }], METRIC_EXACT_LOWER);

      expect(getVerdict(result, "metric").verdict).not.toBe("no-signal");
    });

    it("marks equal medians as no-signal", () => {
      const result = computeVerdicts([{ metric: 100 }], [{ metric: 100 }], METRIC_EXACT_LOWER);

      expect(getVerdict(result, "metric").verdict).toBe("no-signal");
    });
  });

  describe("direction-awareness", () => {
    it.each([
      {
        direction: "lower",
        samplesB: [{ metric: 95 }],
        expectedDelta: -5,
        expectedVerdict: "improved",
      },
      {
        direction: "higher",
        samplesB: [{ metric: 105 }],
        expectedDelta: 5,
        expectedVerdict: "improved",
      },
      {
        direction: "lower",
        samplesB: [{ metric: 105 }],
        expectedDelta: 5,
        expectedVerdict: "regressed",
      },
      {
        direction: "higher",
        samplesB: [{ metric: 95 }],
        expectedDelta: -5,
        expectedVerdict: "regressed",
      },
    ])(
      "treats delta as $expectedVerdict when direction: $direction and delta is $expectedDelta",
      ({ direction, samplesB, expectedDelta, expectedVerdict }) => {
        const config = direction === "lower" ? METRIC_EXACT_LOWER : METRIC_EXACT_HIGHER;
        const result = computeVerdicts([{ metric: 100 }], samplesB, config);

        const verdict = getVerdict(result, "metric");
        expect(verdict.verdict).toBe(expectedVerdict);
        expect(verdict.delta).toBeCloseTo(expectedDelta, 5);
      },
    );
  });

  describe("pairing and filtering", () => {
    it("pairs samplesA[i] with samplesB[i] by index", () => {
      const result = computeVerdicts(
        [{ metric: 90 }, { metric: 110 }],
        [{ metric: 85 }, { metric: 105 }],
        METRIC_EXACT_LOWER,
      );

      expect(getVerdict(result, "metric").n).toBe(2);
    });

    it("drops windows where metric is missing from samplesA", () => {
      const result = computeVerdicts(
        [{ metric: 100 }, {}],
        [{ metric: 95 }, { metric: 90 }],
        METRIC_EXACT_LOWER,
      );

      expect(getVerdict(result, "metric").n).toBe(1);
    });

    it("drops windows where metric is missing from samplesB", () => {
      const result = computeVerdicts(
        [{ metric: 100 }, { metric: 110 }],
        [{ metric: 95 }, {}],
        METRIC_EXACT_LOWER,
      );

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
        METRIC_EXACT_LOWER,
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

    it("respects exact flag per metric", () => {
      const result = computeVerdicts(
        [{ exactMetric: 100, otherMetric: 50 }],
        [{ exactMetric: 100.001, otherMetric: 45 }],
        {
          exactMetric: { direction: "lower", gating: true, exact: true },
          otherMetric: { direction: "lower", gating: true, exact: false },
        },
      );

      const verdict = getVerdict(result, "exactMetric");
      expect(verdict.verdict).not.toBe("no-signal");
      expect(verdict.method).toBe("exact");
    });
  });

  describe("edge cases", () => {
    it("handles zero as a valid metric value", () => {
      const result = computeVerdicts([{ metric: 0 }], [{ metric: 0 }], METRIC_EXACT_LOWER);

      expect(getVerdict(result, "metric").delta).toBe(0);
    });

    it("produces NaN delta when medianA is zero and medianB is not", () => {
      const result = computeVerdicts([{ metric: 0 }], [{ metric: 5 }], METRIC_EXACT_LOWER);

      expect(getVerdict(result, "metric").delta).toBeNaN();
    });

    it("handles negative metric values", () => {
      const result = computeVerdicts([{ metric: -100 }], [{ metric: -95 }], METRIC_EXACT_LOWER);

      // (-95 - (-100)) / (-100) * 100 = -5
      expect(getVerdict(result, "metric").delta).toBeCloseTo(-5, 5);
    });

    it("handles many paired windows correctly", () => {
      const samplesA = createSamples(100, 100);
      const samplesB = createSamples(100, 95);

      const result = computeVerdicts(samplesA, samplesB, METRIC_EXACT_LOWER);

      const verdict = getVerdict(result, "metric");
      expect(verdict.n).toBe(100);
      expect(verdict.delta).toBeCloseTo(-5, 5);
    });
  });

  describe("signed-rank method", () => {
    it("uses signed-rank method when exact: false and n >= 6", () => {
      // Create 6 paired samples with a consistent difference
      const samplesA = createSamples(6, 100);
      const samplesB = createSamples(6, 95);

      const result = computeVerdicts(samplesA, samplesB, METRIC_APPROX_LOWER);

      const verdict = getVerdict(result, "metric");
      expect(verdict.method).toBe("signed-rank");
    });

    it("includes p field in MetricVerdict for signed-rank method", () => {
      const samplesA = createSamples(6, 100);
      const samplesB = createSamples(6, 95);

      const result = computeVerdicts(samplesA, samplesB, METRIC_APPROX_LOWER);

      const verdict = getSignedRankVerdict(result, "metric");
      expect("p" in verdict).toBe(true);
      expect(typeof verdict.p).toBe("number");
    });

    it("returns no-signal when p >= 0.05", () => {
      // Symmetric diffs cancel out: -1,+1,-2,+2,-3,+3 → all non-zero, n=6, p=1.0
      const samplesA = createSamples(6, 100);
      const samplesB = [
        { metric: 99 },
        { metric: 101 },
        { metric: 98 },
        { metric: 102 },
        { metric: 97 },
        { metric: 103 },
      ];

      const result = computeVerdicts(samplesA, samplesB, METRIC_APPROX_LOWER);

      const verdict = getSignedRankVerdict(result, "metric");
      expect(verdict.p).toBeGreaterThanOrEqual(0.05);
      expect(verdict.verdict).toBe("no-signal");
    });

    it("falls back to band method when wilcoxon result.n < 6 after dropping zero diffs", () => {
      // 6 samples, but 3 have zero difference → wilcoxon sees n=3 < 6, falls back to band
      const samplesA = createSamples(6, 100);
      const samplesB = [
        { metric: 100 },
        { metric: 100 },
        { metric: 100 },
        { metric: 95 },
        { metric: 90 },
        { metric: 105 },
      ];

      const result = computeVerdicts(samplesA, samplesB, METRIC_APPROX_LOWER);

      const verdict = getVerdict(result, "metric");
      expect(verdict.method).toBe("band");
      expect("band" in verdict).toBe(true);
      expect("p" in verdict).toBe(false);
    });
  });

  describe("band method", () => {
    it("uses band method when exact: false and n < 6", () => {
      const samplesA = createSamples(2, 100);
      const samplesB = createSamples(2, 95);

      const result = computeVerdicts(samplesA, samplesB, METRIC_APPROX_LOWER);

      const verdict = getVerdict(result, "metric");
      expect(verdict.method).toBe("band");
    });

    it("includes band field in MetricVerdict for band method", () => {
      const samplesA = createSamples(2, 100);
      const samplesB = createSamples(2, 95);

      const result = computeVerdicts(samplesA, samplesB, METRIC_APPROX_LOWER);

      const verdict = getBandVerdict(result, "metric");
      expect("band" in verdict).toBe(true);
      expect(typeof verdict.band).toBe("number");
    });

    it("signals when |delta%| > band%", () => {
      // samplesA: [100, 110] → median=105, halfRange=5
      // samplesB: [30, 50] → median=40, halfRange=10
      // max(5/105, 10/40) = 0.25
      // band = max(1.5 * 100 * 0.25, 0.5) = 37.5%
      // delta = (40 - 105) / 105 * 100 ≈ -61.9%
      // |61.9| > 37.5 → signal
      const samplesA = [{ metric: 100 }, { metric: 110 }];
      const samplesB = [{ metric: 30 }, { metric: 50 }];

      const result = computeVerdicts(samplesA, samplesB, METRIC_APPROX_LOWER);

      const verdict = getVerdict(result, "metric");
      expect(verdict.method).toBe("band");
      expect(verdict.verdict).toBe("improved");
    });

    it("returns no-signal when |delta%| <= band%", () => {
      // samplesA: [100, 100] → median=100, halfRange=0
      // samplesB: [101, 99] → median=100, halfRange=1
      // max(0/100, 1/100) = 0.01
      // band = max(1.5 * 100 * 0.01, 0.5) = 1.5%
      // delta = 0%
      // |0| <= 1.5 → no-signal
      const samplesA = createSamples(2, 100);
      const samplesB = [{ metric: 101 }, { metric: 99 }];

      const result = computeVerdicts(samplesA, samplesB, METRIC_APPROX_LOWER);

      const verdict = getVerdict(result, "metric");
      expect(verdict.method).toBe("band");
      expect(verdict.verdict).toBe("no-signal");
    });

    it("uses max(K × 100 × max(halfRange(A)/median(A), halfRange(B)/median(B)), floor%)", () => {
      // samplesA: [80, 120] → median = 100, range = 40, halfRange = 20
      // samplesB: [90, 110] → median = 100, range = 20, halfRange = 10
      // max(20/100, 10/100) = 0.2
      // band = max(1.5 * 100 * 0.2, 0.5) = max(30, 0.5) = 30%
      const samplesA = [{ metric: 80 }, { metric: 120 }];
      const samplesB = [{ metric: 90 }, { metric: 110 }];

      const result = computeVerdicts(samplesA, samplesB, METRIC_APPROX_LOWER);

      const verdict = getBandVerdict(result, "metric");
      expect(verdict.band).toBeCloseTo(30, 1);
      // The band a delta is judged against is the metric's own noise, so a band
      // verdict reports the same number twice.
      expect(verdict.noisePct).toBeCloseTo(30, 1);
    });

    it("applies floor% (0.5%) when K × 100 × spread < floor", () => {
      // Create stable metrics where K × 100 × spread < 0.5%
      // samplesA: [100, 100] → median = 100, halfRange = 0
      // samplesB: [100, 100.1] → median = 100, halfRange = 0.05
      // max(0/100, 0.05/100) = 0.0005
      // band = max(1.5 * 100 * 0.0005, 0.5) = max(0.075, 0.5) = 0.5%
      const samplesA = createSamples(2, 100);
      const samplesB = [{ metric: 100 }, { metric: 100.1 }];

      const result = computeVerdicts(samplesA, samplesB, METRIC_APPROX_LOWER);

      const verdict = getBandVerdict(result, "metric");
      expect(verdict.band).toBeCloseTo(0.5, 1);
    });

    it("treats spread as 0 when median is 0 (band = floor)", () => {
      // When medianA or medianB is 0, halfRange/median is undefined
      // Should treat spread as 0 and use floor
      // samplesA: [-5, 5] → median = 0, halfRange = 5
      // samplesB: [-10, 10] → median = 0, halfRange = 10
      // When dividing by 0, treat spread as 0
      // band = max(1.5 * 100 * 0, 0.5) = 0.5%
      const samplesA = [{ metric: -5 }, { metric: 5 }];
      const samplesB = [{ metric: -10 }, { metric: 10 }];

      const result = computeVerdicts(samplesA, samplesB, METRIC_APPROX_LOWER);

      const verdict = getBandVerdict(result, "metric");
      expect(verdict.band).toBeCloseTo(0.5, 1);
    });
  });

  describe("signed-rank direction variants", () => {
    it.each([
      { direction: "lower", samplesAValue: 100, samplesBValue: 95, expectedVerdict: "improved" },
      { direction: "lower", samplesAValue: 100, samplesBValue: 105, expectedVerdict: "regressed" },
      { direction: "higher", samplesAValue: 50, samplesBValue: 100, expectedVerdict: "improved" },
      { direction: "higher", samplesAValue: 100, samplesBValue: 50, expectedVerdict: "regressed" },
    ])(
      "with direction: $direction, $expectedVerdict when delta = $samplesBValue - $samplesAValue",
      ({ direction, samplesAValue, samplesBValue, expectedVerdict }) => {
        const config = direction === "higher" ? METRIC_APPROX_HIGHER : METRIC_APPROX_LOWER;
        const samplesA = createSamples(6, samplesAValue);
        const samplesB = createSamples(6, samplesBValue);

        const result = computeVerdicts(samplesA, samplesB, config);

        const verdict = getVerdict(result, "metric");
        expect(verdict.method).toBe("signed-rank");
        expect(verdict.verdict).toBe(expectedVerdict);
      },
    );
  });

  describe("band direction variants", () => {
    it.each([
      { samplesAValue: 50, samplesBValue: 100, direction: "higher", expectedVerdict: "improved" },
      { samplesAValue: 100, samplesBValue: 50, direction: "higher", expectedVerdict: "regressed" },
    ])(
      "with direction: $direction, $expectedVerdict when delta = $samplesBValue - $samplesAValue",
      ({ samplesAValue, samplesBValue, direction, expectedVerdict }) => {
        const config = direction === "higher" ? METRIC_APPROX_HIGHER : METRIC_APPROX_LOWER;
        const samplesA = createSamples(2, samplesAValue);
        const samplesB = createSamples(2, samplesBValue);

        const result = computeVerdicts(samplesA, samplesB, config);

        const verdict = getVerdict(result, "metric");
        expect(verdict.method).toBe("band");
        expect(verdict.verdict).toBe(expectedVerdict);
      },
    );

    it("computes band from high-spread samples", () => {
      // samplesA: [0.1, 0.1] → median=0.1, halfRange=0
      // samplesB: [0.05, 0.15] → median=0.1, halfRange=0.05
      // max(0/0.1, 0.05/0.1) = 0.5
      // band = max(1.5 * 100 * 0.5, 0.5) = 75%
      const samplesA = [{ metric: 0.1 }, { metric: 0.1 }];
      const samplesB = [{ metric: 0.05 }, { metric: 0.15 }];

      const result = computeVerdicts(samplesA, samplesB, METRIC_APPROX_LOWER);

      const verdict = getBandVerdict(result, "metric");
      expect(verdict.band).toBeCloseTo(75, 0);
      expect(verdict.verdict).toBe("no-signal");
    });

    it("both samples have zero median → band = floor (0.5)", () => {
      const samplesA = createSamples(2, 0);
      const samplesB = createSamples(2, 0);

      const result = computeVerdicts(samplesA, samplesB, METRIC_APPROX_LOWER);

      const verdict = getBandVerdict(result, "metric");
      expect(verdict.band).toBeCloseTo(0.5, 1);
    });
  });

  describe("when the metric is measured by a non-exact method", () => {
    it("carries noisePct from the noise-band formula on a signed-rank verdict", () => {
      // pairedA: [80, 90, 100, 100, 110, 120] → median = 100, halfRange = 20 → 0.2
      // pairedB: [95, 95, 95, 105, 105, 105] → median = 100, halfRange = 5 → 0.05
      // diffs are all non-zero, so n = 6 keeps this on the signed-rank path
      // noisePct = max(1.5 * 100 * max(0.2, 0.05), 0.5) = 30
      const samplesA = [
        { metric: 80 },
        { metric: 90 },
        { metric: 100 },
        { metric: 100 },
        { metric: 110 },
        { metric: 120 },
      ];
      const samplesB = [
        { metric: 95 },
        { metric: 95 },
        { metric: 95 },
        { metric: 105 },
        { metric: 105 },
        { metric: 105 },
      ];

      const result = computeVerdicts(samplesA, samplesB, METRIC_APPROX_LOWER);

      const verdict = getNoisyVerdict(result, "metric");
      expect(verdict.method).toBe("signed-rank");
      expect(verdict.noisePct).toBeCloseTo(30, 5);
    });

    it("treats a zero median as zero spread when computing noisePct under signed-rank", () => {
      // pairedA: median = 0, halfRange = 5 → ratio undefined, contributes 0
      // pairedB: median = 0, halfRange = 10 → ratio undefined, contributes 0
      // diffs are all non-zero, so n = 6 keeps this on the signed-rank path
      // noisePct = max(1.5 * 100 * 0, 0.5) = 0.5
      const samplesA = [
        { metric: -5 },
        { metric: -3 },
        { metric: -1 },
        { metric: 1 },
        { metric: 3 },
        { metric: 5 },
      ];
      const samplesB = [
        { metric: -10 },
        { metric: -6 },
        { metric: -2 },
        { metric: 2 },
        { metric: 6 },
        { metric: 10 },
      ];

      const result = computeVerdicts(samplesA, samplesB, METRIC_APPROX_LOWER);

      const verdict = getSignedRankVerdict(result, "metric");
      expect(verdict.noisePct).toBeCloseTo(0.5, 5);
    });
  });

  describe("when noisePct exceeds the unstable threshold", () => {
    it.each([
      { method: "signed-rank", unstableNoisePct: 20, expected: "unstable" },
      { method: "signed-rank", unstableNoisePct: 30, expected: "improved" },
      { method: "band", unstableNoisePct: 20, expected: "unstable" },
      { method: "band", unstableNoisePct: 30, expected: "improved" },
    ])(
      "$method with noisePct 30 is $expected when unstableNoisePct is $unstableNoisePct",
      ({ method, unstableNoisePct, expected }) => {
        const { samplesA, samplesB } =
          method === "signed-rank" ? NOISY_SIGNED_RANK_SAMPLES : NOISY_BAND_SAMPLES;

        const result = computeVerdicts(samplesA, samplesB, METRIC_APPROX_LOWER, unstableNoisePct);

        const verdict = getVerdict(result, "metric");
        expect(verdict.method).toBe(method);
        expect(verdict.verdict).toBe(expected);
      },
    );

    it.each([
      // pairedB: [10, 150, 410] → median = 150, halfRange = 200 → 4/3
      // noisePct = max(1.5 * 100 * 4/3, 0.5) = 200 → exactly at the default, so not unstable
      // delta = (150 - 100) / 100 * 100 = 50, |50| <= band (200) → no-signal
      { samplesB: [{ metric: 10 }, { metric: 150 }, { metric: 410 }], expected: "no-signal" },
      // pairedB: [10, 150, 412] → median = 150, halfRange = 201 → 1.34
      // noisePct = max(1.5 * 100 * 1.34, 0.5) = 201 → just past the default
      { samplesB: [{ metric: 10 }, { metric: 150 }, { metric: 412 }], expected: "unstable" },
    ])("defaults the unstable threshold to 200: $expected", ({ samplesB, expected }) => {
      // pairedA: [100, 100, 100] → median = 100, halfRange = 0, so B drives the noise band
      const samplesA = createSamples(3, 100);

      const result = computeVerdicts(samplesA, samplesB, METRIC_APPROX_LOWER);

      expect(getVerdict(result, "metric").verdict).toBe(expected);
    });

    it("never reports unstable for an exact metric, however wide its spread", () => {
      // pairedA: [1, 100, 10000] → median = 100, halfRange = 4999.5 → spread 49.995
      // pairedB: [1, 50, 10000] → median = 50, halfRange = 4999.5 → spread 99.99 drives
      //   a ~15000% band; delta = -50%
      const samplesA = [{ metric: 1 }, { metric: 100 }, { metric: 10_000 }];
      const samplesB = [{ metric: 1 }, { metric: 50 }, { metric: 10_000 }];

      const result = computeVerdicts(samplesA, samplesB, METRIC_EXACT_LOWER, 1);

      const verdict = getVerdict(result, "metric");
      expect(verdict.method).toBe("exact");
      expect(verdict.verdict).toBe("improved");
    });
  });
});

describe("computeGeomean", () => {
  describe("empty and exclusion cases", () => {
    it("returns { value: 0, n: 0, excluded: [] } when no metrics in verdicts", () => {
      const result = computeGeomean({}, {});
      expect(result).toStrictEqual({ value: 0, n: 0, excluded: [] });
    });

    it("returns { value: 0, n: 0, excluded: [] } when no gating metrics", () => {
      const verdicts = {
        metric1: {
          verdict: "improved" as const,
          method: "exact" as const,
          delta: -5,
          n: 1,
        },
      };
      const metricMeta = {
        metric1: { direction: "lower" as const, gating: false, exact: true },
      };
      const result = computeGeomean(verdicts, metricMeta);
      expect(result).toStrictEqual({ value: 0, n: 0, excluded: [] });
    });

    it.each([
      { direction: "lower" as const, delta: Number.NaN, reason: "undefined-ratio" },
      // 1 + (-150/100) = -0.5 ≤ 0
      { direction: "lower" as const, delta: -150, reason: "infinite-rho" },
      // boundary: ρ = 1 + delta/100 = 0 exactly
      { direction: "lower" as const, delta: -100, reason: "infinite-rho" },
      // ρ = 1 / (1 + delta/100) = 1/0 = Infinity
      { direction: "higher" as const, delta: -100, reason: "infinite-rho" },
    ])(
      "excludes the sole gating metric as $reason (direction: $direction, delta: $delta)",
      ({ direction, delta, reason }) => {
        const verdicts = {
          metric1: { verdict: "regressed" as const, method: "exact" as const, delta, n: 1 },
        };
        const metricMeta = {
          metric1: { direction, gating: true, exact: true },
        };

        const result = computeGeomean(verdicts, metricMeta);

        expect(result.value).toBe(0);
        expect(result.n).toBe(0);
        expect(result.excluded).toStrictEqual([{ metric: "metric1", reason }]);
      },
    );
  });

  describe("single gating metric", () => {
    it("returns geomean = ρ when one gating metric with direction: lower", () => {
      const verdicts = {
        metric1: {
          verdict: "improved" as const,
          method: "exact" as const,
          delta: -5, // ρ = 1 + (-5/100) = 0.95
          n: 1,
        },
      };
      const metricMeta = {
        metric1: { direction: "lower" as const, gating: true, exact: true },
      };
      const result = computeGeomean(verdicts, metricMeta);
      expect(result.n).toBe(1);
      expect(result.excluded).toStrictEqual([]);
      // value = (0.95 - 1) * 100 = -5
      expect(result.value).toBeCloseTo(-5, 5);
    });

    it("returns geomean = 1/ρ when one gating metric with direction: higher", () => {
      const verdicts = {
        metric1: {
          verdict: "improved" as const,
          method: "exact" as const,
          delta: 5, // ρ_higher = 1 / (1 + 5/100) = 1 / 1.05 ≈ 0.952
          n: 1,
        },
      };
      const metricMeta = {
        metric1: { direction: "higher" as const, gating: true, exact: true },
      };
      const result = computeGeomean(verdicts, metricMeta);
      expect(result.n).toBe(1);
      expect(result.excluded).toStrictEqual([]);
      // value = (1/1.05 - 1) * 100 ≈ -4.76
      expect(result.value).toBeCloseTo(-4.76, 1);
    });

    it("returns value: 0 when single gating metric with zero delta", () => {
      const verdicts = {
        metric1: {
          verdict: "no-signal" as const,
          method: "exact" as const,
          delta: 0,
          n: 1,
        },
      };
      const metricMeta = {
        metric1: { direction: "lower" as const, gating: true, exact: true },
      };
      const result = computeGeomean(verdicts, metricMeta);
      expect(result.n).toBe(1);
      expect(result.value).toBeCloseTo(0, 5);
    });
  });

  describe("multiple gating metrics", () => {
    it("computes geomean of multiple direction: lower metrics", () => {
      // metric1: delta = -10 → ρ₁ = 0.9
      // metric2: delta = -5 → ρ₂ = 0.95
      // geomean = (0.9 × 0.95)^(1/2) = 0.8550^(1/2) ≈ 0.9246
      // value = (0.9246 - 1) * 100 ≈ -7.54
      const verdicts = {
        metric1: {
          verdict: "improved" as const,
          method: "exact" as const,
          delta: -10,
          n: 1,
        },
        metric2: {
          verdict: "improved" as const,
          method: "exact" as const,
          delta: -5,
          n: 1,
        },
      };
      const metricMeta = {
        metric1: { direction: "lower" as const, gating: true, exact: true },
        metric2: { direction: "lower" as const, gating: true, exact: true },
      };
      const result = computeGeomean(verdicts, metricMeta);
      expect(result.n).toBe(2);
      expect(result.excluded).toStrictEqual([]);
      expect(result.value).toBeCloseTo(-7.54, 1);
    });

    it("respects direction per metric in geomean calculation", () => {
      // metric1: lower, delta = -10 → ρ₁ = 0.9
      // metric2: higher, delta = 10 → ρ₂ = 1 / 1.1 ≈ 0.909
      // geomean = (0.9 × 0.909)^(1/2) ≈ 0.9045
      // value ≈ -9.55
      const verdicts = {
        metric1: {
          verdict: "improved" as const,
          method: "exact" as const,
          delta: -10,
          n: 1,
        },
        metric2: {
          verdict: "improved" as const,
          method: "exact" as const,
          delta: 10,
          n: 1,
        },
      };
      const metricMeta = {
        metric1: { direction: "lower" as const, gating: true, exact: true },
        metric2: { direction: "higher" as const, gating: true, exact: true },
      };
      const result = computeGeomean(verdicts, metricMeta);
      expect(result.n).toBe(2);
      expect(result.excluded).toStrictEqual([]);
      expect(result.value).toBeCloseTo(-9.55, 0);
    });

    it("excludes non-gating metrics from geomean calculation", () => {
      // Only metric1 is gating
      const verdicts = {
        metric1: {
          verdict: "improved" as const,
          method: "exact" as const,
          delta: -5,
          n: 1,
        },
        metric2: {
          verdict: "improved" as const,
          method: "exact" as const,
          delta: -10,
          n: 1,
        },
      };
      const metricMeta = {
        metric1: { direction: "lower" as const, gating: true, exact: true },
        metric2: { direction: "lower" as const, gating: false, exact: true },
      };
      const result = computeGeomean(verdicts, metricMeta);
      expect(result.n).toBe(1);
      expect(result.excluded).toStrictEqual([]);
      // geomean = 0.95, value ≈ -5
      expect(result.value).toBeCloseTo(-5, 5);
    });

    it.each([
      { badDelta: Number.NaN, reason: "undefined-ratio" },
      // ρ = -0.5 ≤ 0
      { badDelta: -150, reason: "infinite-rho" },
    ])(
      "excludes metric1 as $reason, keeps metric2's ratio in the geomean",
      ({ badDelta, reason }) => {
        // metric2: delta = -5 → ρ = 0.95, the only ratio left in the geomean
        const verdicts = {
          metric1: {
            verdict: "regressed" as const,
            method: "exact" as const,
            delta: badDelta,
            n: 1,
          },
          metric2: { verdict: "improved" as const, method: "exact" as const, delta: -5, n: 1 },
        };
        const metricMeta = {
          metric1: { direction: "lower" as const, gating: true, exact: true },
          metric2: { direction: "lower" as const, gating: true, exact: true },
        };

        const result = computeGeomean(verdicts, metricMeta);

        expect(result.n).toBe(1);
        expect(result.excluded).toStrictEqual([{ metric: "metric1", reason }]);
        expect(result.value).toBeCloseTo(-5, 5);
      },
    );

    it("returns value: 0, n: 0 when all gating metrics are excluded", () => {
      const verdicts = {
        metric1: {
          verdict: "regressed" as const,
          method: "exact" as const,
          delta: -150,
          n: 1,
        },
        metric2: {
          verdict: "no-signal" as const,
          method: "exact" as const,
          delta: Number.NaN,
          n: 1,
        },
      };
      const metricMeta = {
        metric1: { direction: "lower" as const, gating: true, exact: true },
        metric2: { direction: "lower" as const, gating: true, exact: true },
      };
      const result = computeGeomean(verdicts, metricMeta);
      expect(result.value).toBe(0);
      expect(result.n).toBe(0);
      expect(result.excluded).toStrictEqual([
        { metric: "metric1", reason: "infinite-rho" },
        { metric: "metric2", reason: "undefined-ratio" },
      ]);
    });
  });

  describe("edge cases and special values", () => {
    it("handles large positive delta for direction: lower", () => {
      // delta = 100 → ρ = 2.0, value = 100
      const verdicts = {
        metric1: {
          verdict: "regressed" as const,
          method: "exact" as const,
          delta: 100,
          n: 1,
        },
      };
      const metricMeta = {
        metric1: { direction: "lower" as const, gating: true, exact: true },
      };
      const result = computeGeomean(verdicts, metricMeta);
      expect(result.value).toBeCloseTo(100, 1);
    });

    it("handles negative delta for direction: higher (regression)", () => {
      // delta = -10 → ρ = 1 / 0.9 ≈ 1.111, value ≈ 11.1
      const verdicts = {
        metric1: {
          verdict: "regressed" as const,
          method: "exact" as const,
          delta: -10,
          n: 1,
        },
      };
      const metricMeta = {
        metric1: { direction: "higher" as const, gating: true, exact: true },
      };
      const result = computeGeomean(verdicts, metricMeta);
      expect(result.value).toBeCloseTo(11.11, 1);
    });

    it("computes n correctly with mix of included/excluded metrics", () => {
      const verdicts = {
        included1: {
          verdict: "improved" as const,
          method: "exact" as const,
          delta: -5,
          n: 1,
        },
        included2: {
          verdict: "improved" as const,
          method: "exact" as const,
          delta: -10,
          n: 1,
        },
        excluded1: {
          verdict: "no-signal" as const,
          method: "exact" as const,
          delta: Number.NaN,
          n: 1,
        },
      };
      const metricMeta = {
        included1: { direction: "lower" as const, gating: true, exact: true },
        included2: { direction: "lower" as const, gating: true, exact: true },
        excluded1: { direction: "lower" as const, gating: true, exact: true },
      };
      const result = computeGeomean(verdicts, metricMeta);
      expect(result.n).toBe(2);
      expect(result.excluded).toStrictEqual([{ metric: "excluded1", reason: "undefined-ratio" }]);
    });
  });

  describe("when a gating metric is unstable", () => {
    it("excludes an unstable gating metric even though its ρ is valid", () => {
      // noisy: delta = -50 → ρ = 0.5 would be usable, but the verdict is unjudgeable
      // stable: delta = -5 → ρ = 0.95, the only ratio left in the geomean
      const verdicts = {
        noisy: {
          verdict: "unstable" as const,
          method: "band" as const,
          delta: -50,
          n: 4,
          band: 250,
          noisePct: 250,
        },
        stable: {
          verdict: "improved" as const,
          method: "exact" as const,
          delta: -5,
          n: 4,
        },
      };
      const metricMeta = {
        noisy: { direction: "lower" as const, gating: true, exact: false },
        stable: { direction: "lower" as const, gating: true, exact: true },
      };

      const result = computeGeomean(verdicts, metricMeta);

      expect(result.n).toBe(1);
      expect(result.excluded).toStrictEqual([{ metric: "noisy", reason: "unstable" }]);
      expect(result.value).toBeCloseTo(-5, 5);
    });

    it("reports unstable rather than undefined-ratio when the delta is also NaN", () => {
      const verdicts = {
        noisy: {
          verdict: "unstable" as const,
          method: "band" as const,
          delta: Number.NaN,
          n: 4,
          band: 300,
          noisePct: 300,
        },
      };
      const metricMeta = {
        noisy: { direction: "lower" as const, gating: true, exact: false },
      };

      const result = computeGeomean(verdicts, metricMeta);

      expect(result).toStrictEqual({
        value: 0,
        n: 0,
        excluded: [{ metric: "noisy", reason: "unstable" }],
      });
    });
  });
});
