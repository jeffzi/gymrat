import { describe, it, expect } from "vitest";

import type { BandVerdict, MetricVerdict, SignedRankVerdict } from "../../src/verdict/verdict.js";
import { computeVerdicts } from "../../src/verdict/verdict.js";
import { metricRecord } from "../fixtures/metrics.js";

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
 * Fetch a verdict that must have come from the band path.
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
const METRIC_BYTES_LOWER = {
  metric: { direction: "lower" as const, gating: true, exact: false, unit: "bytes" as const },
};
const METRIC_BYTES_HIGHER = {
  metric: { direction: "higher" as const, gating: true, exact: false, unit: "bytes" as const },
};
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

      expect(result).toStrictEqual(metricRecord({}));
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

  describe("when a metric is named after an Object.prototype member", () => {
    // A plain object literal would read `__proto__` as its prototype rather than
    // as a metric name, so the key reaches the samples through a variable.
    const PROTO = "__proto__";

    it("keeps the verdict for a metric named __proto__", () => {
      const result = computeVerdicts([{ [PROTO]: 100 }], [{ [PROTO]: 95 }], {
        [PROTO]: { direction: "lower" as const, gating: true, exact: true },
      });

      // Read through entries: the verdict has to be an own key of the record,
      // not something a prototype hands back.
      expect(Object.entries(result)).toStrictEqual([
        [PROTO, { verdict: "improved", method: "exact", delta: -5, n: 1 }],
      ]);
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

    it.each([
      {
        movement: "rises toward zero",
        medianA: -100,
        medianB: -95,
        // (−95 − (−100)) / |−100| × 100 = 5
        expectedDelta: 5,
        expectedVerdict: "regressed",
      },
      {
        movement: "falls further below zero",
        medianA: -1,
        medianB: -2,
        // (−2 − (−1)) / |−1| × 100 = −100
        expectedDelta: -100,
        expectedVerdict: "improved",
      },
    ])(
      "signs the delta by the way a negative value moved when it $movement",
      ({ medianA, medianB, expectedDelta, expectedVerdict }) => {
        // Normalizing by the magnitude of the baseline median keeps the delta's sign
        // tied to the direction the value actually moved, so `direction: lower` still
        // reads a drop as an improvement below zero.
        const result = computeVerdicts(
          [{ metric: medianA }],
          [{ metric: medianB }],
          METRIC_EXACT_LOWER,
        );

        const verdict = getVerdict(result, "metric");
        expect(verdict.delta).toBeCloseTo(expectedDelta, 5);
        expect(verdict.verdict).toBe(expectedVerdict);
      },
    );

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
      // n = 6 is the signed-rank method's activation threshold; fewer samples fall back to the band method.
      const samplesA = createSamples(6, 100);
      const samplesB = createSamples(6, 95);

      const result = computeVerdicts(samplesA, samplesB, METRIC_APPROX_LOWER);

      const verdict = getVerdict(result, "metric");
      expect(verdict.method).toBe("signed-rank");
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

    it("returns no-signal when the delta is NaN", () => {
      // Every baseline window is 0, so the delta has no magnitude to normalize against
      // and the metric offers nothing to compare — even though the six identical,
      // non-zero diffs make the signed-rank test itself significant.
      const samplesA = createSamples(6, 0);
      const samplesB = createSamples(6, 5);

      const result = computeVerdicts(samplesA, samplesB, METRIC_APPROX_LOWER);

      const verdict = getSignedRankVerdict(result, "metric");
      expect(verdict.p).toBeLessThan(0.05);
      expect(verdict.delta).toBeNaN();
      expect(verdict.verdict).toBe("no-signal");
    });

    it.each([
      { desc: "direction lower", meta: METRIC_APPROX_LOWER },
      { desc: "direction higher", meta: METRIC_APPROX_HIGHER },
    ])("returns no-signal when delta is zero ($desc)", ({ meta }) => {
      // 8 pairs whose medians are equal (both 100) but whose individual values
      // differ consistently — all 6 non-zero diffs are positive, so
      // p = 0.03125 < 0.05. Without a zero-delta guard the function would
      // classify this as "regressed" for either direction.
      const samplesA = [
        { metric: 80 },
        { metric: 90 },
        { metric: 95 },
        { metric: 100 },
        { metric: 100 },
        { metric: 105 },
        { metric: 110 },
        { metric: 120 },
      ];
      const samplesB = [
        { metric: 81 },
        { metric: 91 },
        { metric: 96 },
        { metric: 100 },
        { metric: 100 },
        { metric: 106 },
        { metric: 111 },
        { metric: 121 },
      ];

      const result = computeVerdicts(samplesA, samplesB, meta);

      const verdict = getSignedRankVerdict(result, "metric");
      expect(verdict.p).toBeLessThan(0.05);
      expect(verdict.delta).toBe(0);
      expect(verdict.verdict).toBe("no-signal");
    });

    it("keeps a significant verdict when the delta is smaller than the noise band", () => {
      // The p-value already accounts for the windows' spread, so the K × spread
      // band is not a second gate on the signed-rank path: every window moved
      // down by 5%, which is significant even though the band is 30%.
      // pairedA: [80, 90, 100, 100, 110, 120] → median = 100, halfRange = 20 → 0.2
      // pairedB: every value 5% lower → median = 95, halfRange = 19 → 0.2
      // noisePct = max(1.5 * 100 * 0.2, 0.5) = 30, delta = -5
      const samplesA = [
        { metric: 80 },
        { metric: 90 },
        { metric: 100 },
        { metric: 100 },
        { metric: 110 },
        { metric: 120 },
      ];
      const samplesB = [
        { metric: 76 },
        { metric: 85.5 },
        { metric: 95 },
        { metric: 95 },
        { metric: 104.5 },
        { metric: 114 },
      ];

      const result = computeVerdicts(samplesA, samplesB, METRIC_APPROX_LOWER);

      const verdict = getSignedRankVerdict(result, "metric");
      expect.soft(verdict.p).toBeLessThan(0.05);
      expect.soft(verdict.delta).toBeCloseTo(-5, 5);
      expect.soft(verdict.noisePct).toBeCloseTo(30, 5);
      expect(verdict.verdict).toBe("improved");
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
      expect(verdict.noisePct).toBeCloseTo(0.5, 1);
    });

    it.each([
      {
        cause: "tied pairs",
        // 6 pairs, but 3 of them differ by zero → the Wilcoxon test can use only 3
        samplesB: [
          { metric: 100 },
          { metric: 100 },
          { metric: 100 },
          { metric: 95 },
          { metric: 90 },
          { metric: 105 },
        ],
        expectedN: 6,
        expectedUsableN: 3,
      },
      {
        cause: "too few samples",
        // 3 pairs is already short of the Wilcoxon minimum, and one of them is tied
        samplesB: [{ metric: 100 }, { metric: 95 }, { metric: 90 }],
        expectedN: 3,
        expectedUsableN: 2,
      },
    ])(
      "reports usableN as the non-zero-difference pairs behind a $cause fallback",
      ({ samplesB, expectedN, expectedUsableN }) => {
        const samplesA = createSamples(samplesB.length, 100);

        const result = computeVerdicts(samplesA, samplesB, METRIC_APPROX_LOWER);

        const verdict = getBandVerdict(result, "metric");
        expect(verdict.n).toBe(expectedN);
        expect(verdict.usableN).toBe(expectedUsableN);
      },
    );

    it.each([
      { pairs: 1, expected: "no-signal" },
      { pairs: 2, expected: "improved" },
    ])(
      "reports $expected for a −50% delta measured from $pairs paired window(s)",
      ({ pairs, expected }) => {
        // A single window has no observable spread, so the band collapses to the floor
        // and any delta would look definitive. Two windows give a usable half-range.
        const samplesA = createSamples(pairs, 100);
        const samplesB = createSamples(pairs, 50);

        const result = computeVerdicts(samplesA, samplesB, METRIC_APPROX_LOWER);

        const verdict = getVerdict(result, "metric");
        expect(verdict.method).toBe("band");
        expect(verdict.verdict).toBe(expected);
      },
    );

    it("measures the band against the magnitude of a negative median", () => {
      // samplesA: [-60, -40] → median = -50, halfRange = 10 → 10 / |−50| = 0.2
      // samplesB: [-50, -50] → median = -50, halfRange = 0 → 0
      // band = max(1.5 * 100 * 0.2, 0.5) = 30
      const samplesA = [{ metric: -60 }, { metric: -40 }];
      const samplesB = [{ metric: -50 }, { metric: -50 }];

      const result = computeVerdicts(samplesA, samplesB, METRIC_APPROX_LOWER);

      const verdict = getBandVerdict(result, "metric");
      expect(verdict.noisePct).toBeCloseTo(30, 5);
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
      expect(verdict.noisePct).toBeCloseTo(0.5, 1);
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
      expect(verdict.noisePct).toBeCloseTo(75, 0);
      expect(verdict.verdict).toBe("no-signal");
    });

    it("both samples have zero median → band = floor (0.5)", () => {
      const samplesA = createSamples(2, 0);
      const samplesB = createSamples(2, 0);

      const result = computeVerdicts(samplesA, samplesB, METRIC_APPROX_LOWER);

      const verdict = getBandVerdict(result, "metric");
      expect(verdict.noisePct).toBeCloseTo(0.5, 1);
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

    it("measures noisePct against the magnitude of a negative median under signed-rank", () => {
      // pairedA: [-60, -55, -50, -50, -45, -40] → median = -50, halfRange = 10 → 10 / |−50| = 0.2
      // pairedB: every value 1 lower → median = -51, halfRange = 10 → 10 / |−51| ≈ 0.196
      // every diff is non-zero, so n = 6 keeps this on the signed-rank path
      // noisePct = max(1.5 * 100 * 0.2, 0.5) = 30
      const samplesA = [
        { metric: -60 },
        { metric: -55 },
        { metric: -50 },
        { metric: -50 },
        { metric: -45 },
        { metric: -40 },
      ];
      const samplesB = [
        { metric: -61 },
        { metric: -56 },
        { metric: -51 },
        { metric: -51 },
        { metric: -46 },
        { metric: -41 },
      ];

      const result = computeVerdicts(samplesA, samplesB, METRIC_APPROX_LOWER);

      const verdict = getSignedRankVerdict(result, "metric");
      expect(verdict.noisePct).toBeCloseTo(30, 5);
    });

    it.each([
      {
        method: "signed-rank",
        // No tied difference among the 6 pairs, so the verdict stays on the signed-rank path
        valuesA: [180, 190, 200, 200, 210, 220],
        valuesB: [90, 95, 100, 100, 105, 110],
      },
      {
        method: "band",
        valuesA: [180, 220],
        valuesB: [90, 110],
      },
    ])(
      "carries noiseAbs as K × the widest half-range on a $method verdict",
      ({ method, valuesA, valuesB }) => {
        // halfRange(A) = 20 around median 200, halfRange(B) = 10 around median 100
        // noisePct = max(1.5 * 100 * max(0.1, 0.1), 0.5) = 15
        // noiseAbs = 1.5 * max(20, 10) = 30, in the metric's own raw unit
        const result = computeVerdicts(
          valuesA.map((metric) => ({ metric })),
          valuesB.map((metric) => ({ metric })),
          METRIC_APPROX_LOWER,
        );

        const verdict = getNoisyVerdict(result, "metric");
        expect(verdict.method).toBe(method);
        expect(verdict.noisePct).toBeCloseTo(15, 5);
        expect(verdict.noiseAbs).toBeCloseTo(30, 5);
      },
    );
  });

  describe("when the metric is measured in whole bytes", () => {
    it.each([{ pairs: 2 }, { pairs: 5 }])(
      "reads a 4B against 3B move as no-signal from $pairs paired windows",
      ({ pairs }) => {
        // Both sides are perfectly stable, so nothing but the one-byte resolution
        // sets the band: max(100 / 4, 100 / 3) = 33.3%, and the −25% delta is a
        // single byte of quantization rather than a measured improvement.
        const result = computeVerdicts(
          createSamples(pairs, 4),
          createSamples(pairs, 3),
          METRIC_BYTES_LOWER,
        );

        const verdict = getBandVerdict(result, "metric");
        expect.soft(verdict.delta).toBeCloseTo(-25, 5);
        expect.soft(verdict.noisePct).toBeCloseTo(100 / 3, 5);
        expect(verdict.verdict).toBe("no-signal");
      },
    );

    it.each([
      { direction: "lower", meta: METRIC_BYTES_LOWER },
      { direction: "higher", meta: METRIC_BYTES_HIGHER },
    ])(
      "reads a 4B against 3B move as no-signal from 6 paired windows (direction: $direction)",
      ({ meta }) => {
        // Six pairs put the metric on the signed-rank path, where six identical
        // −1B diffs are significant (p < 0.05). The move is still one byte of
        // resolution, so the same 33.3% floor that silences it at five pairs has
        // to silence it here — a verdict cannot start depending on window count.
        const result = computeVerdicts(createSamples(6, 4), createSamples(6, 3), meta);

        const verdict = getSignedRankVerdict(result, "metric");
        expect.soft(verdict.p).toBeLessThan(0.05);
        expect.soft(verdict.delta).toBeCloseTo(-25, 5);
        expect.soft(verdict.noisePct).toBeCloseTo(100 / 3, 5);
        expect(verdict.verdict).toBe("no-signal");
      },
    );

    it.each([
      { direction: "lower", meta: METRIC_BYTES_LOWER, expected: "improved" },
      { direction: "higher", meta: METRIC_BYTES_HIGHER, expected: "regressed" },
    ])(
      "reports $expected for a 100B against 75B move from 6 paired windows (direction: $direction)",
      ({ meta, expected }) => {
        // One byte against 75B is a 1.33% floor, so a −25% move clears it by far
        // and the signed-rank verdict stands.
        const result = computeVerdicts(createSamples(6, 100), createSamples(6, 75), meta);

        const verdict = getSignedRankVerdict(result, "metric");
        expect.soft(verdict.delta).toBeCloseTo(-25, 5);
        expect.soft(verdict.noisePct).toBeCloseTo(100 / 75, 5);
        expect(verdict.verdict).toBe(expected);
      },
    );

    it.each([
      {
        unit: "ns",
        meta: {
          metric: { direction: "lower" as const, gating: true, exact: false, unit: "ns" as const },
        },
      },
      { unit: "none", meta: METRIC_APPROX_LOWER },
    ])(
      "reports improved for a 4 against 3 move from 6 paired windows when unit is $unit",
      ({ meta }) => {
        // Nothing quantizes these values to whole units, so no resolution floor
        // applies and the −25% move keeps its signed-rank verdict.
        const result = computeVerdicts(createSamples(6, 4), createSamples(6, 3), meta);

        const verdict = getSignedRankVerdict(result, "metric");
        expect.soft(verdict.noisePct).toBeCloseTo(0.5, 5);
        expect(verdict.verdict).toBe("improved");
      },
    );

    it("leaves a zero median out of the byte floor instead of dividing by it", () => {
      // medianB = 0 contributes nothing, so medianA alone sets the floor at 100 / 4 = 25%
      // — a per-side term, not 100 / min(medians), which would be infinite here.
      const result = computeVerdicts(createSamples(2, 4), createSamples(2, 0), METRIC_BYTES_LOWER);

      const verdict = getBandVerdict(result, "metric");
      expect.soft(verdict.noisePct).toBeCloseTo(25, 5);
      // delta = −100%, still well clear of the 25% band
      expect(verdict.verdict).toBe("improved");
    });

    it.each([
      // halfRange = 50 around 1_000_050 → 1.5 × 100 × 5e-5 = 0.0075%, under the floor
      { spread: "stable", valuesB: [1_000_000, 1_000_100], expectedBand: 0.5 },
      // halfRange = 200_000 around 1_000_000 → 1.5 × 100 × 0.2 = 30%
      { spread: "wide", valuesB: [800_000, 1_200_000], expectedBand: 30 },
    ])(
      "keeps the band at $expectedBand% for a $spread megabyte-scale metric",
      ({ valuesB, expectedBand }) => {
        // One byte against a megabyte is 0.0001% — negligible beside both the 0.5%
        // floor and any measured spread.
        const result = computeVerdicts(
          createSamples(2, 1_000_000),
          valuesB.map((metric) => ({ metric })),
          METRIC_BYTES_LOWER,
        );

        expect(getBandVerdict(result, "metric").noisePct).toBeCloseTo(expectedBand, 5);
      },
    );

    it.each([
      {
        unit: "ns",
        meta: {
          direction: "lower" as const,
          gating: true,
          exact: false,
          unit: "ns" as const,
        },
      },
      {
        unit: "none",
        meta: { direction: "lower" as const, gating: true, exact: false },
      },
    ])("keeps the 0.5% floor for a 4 against 3 move when unit is $unit", ({ meta }) => {
      // Averaged values are not quantized to whole units, so the same numbers that
      // read as one byte of noise stay a genuine −25% improvement here.
      const result = computeVerdicts(createSamples(2, 4), createSamples(2, 3), { metric: meta });

      const verdict = getBandVerdict(result, "metric");
      expect.soft(verdict.noisePct).toBeCloseTo(0.5, 5);
      expect(verdict.verdict).toBe("improved");
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
