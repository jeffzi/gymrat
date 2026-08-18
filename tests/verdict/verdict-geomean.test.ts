import { describe, it, expect } from "vitest";

import type { MetricMetadata, MetricVerdict } from "../../src/verdict/verdict.js";
import { computeGeomean, computeVerdicts } from "../../src/verdict/verdict.js";
import { buildInputs } from "../fixtures/verdict-inputs.js";

const METRIC_BYTES_LOWER = {
  metric: { direction: "lower" as const, gating: true, exact: false, unit: "bytes" as const },
};

function createSamples(n: number, value: number): Array<{ metric: number }> {
  return Array.from({ length: n }, () => ({ metric: value }));
}

/**
 * Gating verdicts keyed `metric1`…`metricN`, one per entry of `noise`.
 *
 * A number becomes a band verdict carrying that `noisePct`; `null` becomes an
 * exact verdict, which carries no noise figure at all. Every verdict is
 * judgeable and has a usable ratio, so the geomean includes all of them.
 */
function gatingVerdictsWithNoise(noise: readonly (number | null)[]): {
  verdicts: Record<string, MetricVerdict>;
  metricMeta: Record<string, MetricMetadata>;
} {
  const verdicts: Record<string, MetricVerdict> = {};
  const metricMeta: Record<string, MetricMetadata> = {};

  noise.forEach((noisePct, index) => {
    const key = `metric${index + 1}`;
    verdicts[key] =
      noisePct === null
        ? { verdict: "improved", method: "exact", delta: -50, n: 4 }
        : {
            verdict: "improved",
            method: "band",
            delta: -50,
            n: 4,
            usableN: 4,
            noisePct,
            noiseAbs: noisePct / 2,
          };
    metricMeta[key] = { direction: "lower", gating: true, exact: noisePct === null };
  });

  return { verdicts, metricMeta };
}

/** Verdicts/metricMeta for a single gating metric with the given direction and delta. */
function singleGatingMetric(
  direction: "lower" | "higher",
  delta: number,
): { verdicts: Record<string, MetricVerdict>; metricMeta: Record<string, MetricMetadata> } {
  return buildInputs([{ name: "metric1", direction, delta }]);
}

describe("computeGeomean", () => {
  describe("empty and exclusion cases", () => {
    it("returns a zeroed result when no metrics in verdicts", () => {
      const result = computeGeomean({}, {});
      expect(result).toStrictEqual({ value: 0, n: 0, excluded: [], band: 0 });
    });

    it("aggregates a non-gating metric like any other, leaving subset choice to the caller", () => {
      // Which metrics belong in a geomean is decided before the call — the
      // aggregate layer hands over exactly the subset it wants averaged.
      const { verdicts, metricMeta } = buildInputs([{ name: "metric1", delta: -5, gating: false }]);

      const result = computeGeomean(verdicts, metricMeta);
      expect(result.n).toBe(1);
      expect(result.value).toBeCloseTo(-5, 5);
    });

    it("excludes a one-sided metric as no-verdict, keeping it in the geomean's scope", () => {
      // metric2 was reported by only one target, so no paired samples ever produced
      // a verdict for it. It still belongs to the scope the geomean was asked to
      // cover, so n + excluded.length has to add up to both metrics.
      const { verdicts, metricMeta } = buildInputs([
        { name: "metric1", delta: -5 },
        { name: "metric2", noVerdict: true },
      ]);

      const result = computeGeomean(verdicts, metricMeta);

      expect(result.n).toBe(1);
      expect(result.excluded).toStrictEqual([{ metric: "metric2", reason: "no-verdict" }]);
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
        const { verdicts, metricMeta } = buildInputs([{ name: "metric1", direction, delta }]);

        const result = computeGeomean(verdicts, metricMeta);

        expect(result.value).toBe(0);
        expect(result.n).toBe(0);
        expect(result.excluded).toStrictEqual([{ metric: "metric1", reason }]);
      },
    );
  });

  describe("single gating metric", () => {
    it.each([
      { direction: "lower" as const, delta: -5, expectedValue: -5, precision: 5 },
      { direction: "higher" as const, delta: 5, expectedValue: -4.76, precision: 1 },
      { direction: "lower" as const, delta: 0, expectedValue: 0, precision: 5 },
      { direction: "lower" as const, delta: 100, expectedValue: 100, precision: 1 },
      { direction: "higher" as const, delta: -10, expectedValue: 11.11, precision: 1 },
    ])(
      "returns value ≈ $expectedValue for direction: $direction, delta: $delta",
      ({ direction, delta, expectedValue, precision }) => {
        const { verdicts, metricMeta } = singleGatingMetric(direction, delta);

        const result = computeGeomean(verdicts, metricMeta);

        expect(result.n).toBe(1);
        expect(result.excluded).toStrictEqual([]);
        expect(result.value).toBeCloseTo(expectedValue, precision);
      },
    );
  });

  describe("multiple gating metrics", () => {
    it("computes geomean of multiple direction: lower metrics", () => {
      // metric1: delta = -10 → ρ₁ = 0.9
      // metric2: delta = -5 → ρ₂ = 0.95
      // geomean = (0.9 × 0.95)^(1/2) = 0.8550^(1/2) ≈ 0.9246
      // value = (0.9246 - 1) * 100 ≈ -7.54
      const { verdicts, metricMeta } = buildInputs([
        { name: "metric1", delta: -10 },
        { name: "metric2", delta: -5 },
      ]);

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
      const { verdicts, metricMeta } = buildInputs([
        { name: "metric1", direction: "lower", delta: -10 },
        { name: "metric2", direction: "higher", delta: 10 },
      ]);

      const result = computeGeomean(verdicts, metricMeta);
      expect(result.n).toBe(2);
      expect(result.excluded).toStrictEqual([]);
      expect(result.value).toBeCloseTo(-9.55, 0);
    });

    it.each([
      { badDelta: Number.NaN, reason: "undefined-ratio" },
      // ρ = -0.5 ≤ 0
      { badDelta: -150, reason: "infinite-rho" },
    ])(
      "excludes metric1 as $reason, keeps metric2's ratio in the geomean",
      ({ badDelta, reason }) => {
        // metric2: delta = -5 → ρ = 0.95, the only ratio left in the geomean
        const { verdicts, metricMeta } = buildInputs([
          { name: "metric1", delta: badDelta },
          { name: "metric2", delta: -5 },
        ]);

        const result = computeGeomean(verdicts, metricMeta);

        expect(result.n).toBe(1);
        expect(result.excluded).toStrictEqual([{ metric: "metric1", reason }]);
        expect(result.value).toBeCloseTo(-5, 5);
      },
    );

    it("returns value: 0, n: 0 when all gating metrics are excluded", () => {
      const { verdicts, metricMeta } = buildInputs([
        { name: "metric1", delta: -150 },
        { name: "metric2", delta: Number.NaN },
      ]);

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
    it("computes n correctly with mix of included/excluded metrics", () => {
      const { verdicts, metricMeta } = buildInputs([
        { name: "included1", delta: -5 },
        { name: "included2", delta: -10 },
        { name: "excluded1", delta: Number.NaN },
      ]);

      const result = computeGeomean(verdicts, metricMeta);
      expect(result.n).toBe(2);
      expect(result.excluded).toStrictEqual([{ metric: "excluded1", reason: "undefined-ratio" }]);
    });
  });

  describe("when a gating metric is unstable", () => {
    it("excludes an unstable gating metric even though its ρ is valid", () => {
      // noisy: delta = -50 → ρ = 0.5 would be usable, but the verdict is unjudgeable
      // stable: delta = -5 → ρ = 0.95, the only ratio left in the geomean
      const { verdicts, metricMeta } = buildInputs([
        {
          name: "noisy",
          verdict: {
            verdict: "unstable",
            method: "band",
            delta: -50,
            n: 4,
            usableN: 4,
            noisePct: 250,
            noiseAbs: 25,
          },
        },
        { name: "stable", delta: -5 },
      ]);

      const result = computeGeomean(verdicts, metricMeta);

      expect(result.n).toBe(1);
      expect(result.excluded).toStrictEqual([{ metric: "noisy", reason: "unstable" }]);
      expect(result.value).toBeCloseTo(-5, 5);
    });

    it("reports unstable rather than undefined-ratio when the delta is also NaN", () => {
      const { verdicts, metricMeta } = buildInputs([
        {
          name: "noisy",
          verdict: {
            verdict: "unstable",
            method: "band",
            delta: Number.NaN,
            n: 4,
            usableN: 4,
            noisePct: 300,
            noiseAbs: 30,
          },
        },
      ]);

      const result = computeGeomean(verdicts, metricMeta);

      expect(result).toStrictEqual({
        value: 0,
        n: 0,
        excluded: [{ metric: "noisy", reason: "unstable" }],
        band: 0,
      });
    });
  });

  describe("propagated noise band", () => {
    it.each([
      { noise: [4], expected: 4, formula: "√(4²) ÷ 1" },
      { noise: [3, 4], expected: 2.5, formula: "√(3² + 4²) ÷ 2" },
      {
        noise: [null, 6],
        expected: 3,
        formula: "√(0² + 6²) ÷ 2, the exact metric adding no noise",
      },
    ])("reports band = $formula", ({ noise, expected }) => {
      const { verdicts, metricMeta } = gatingVerdictsWithNoise(noise);

      const result = computeGeomean(verdicts, metricMeta);

      expect(result.band).toBeCloseTo(expected, 10);
    });

    it("carries a byte metric's quantization noise into the band", () => {
      // 4B against 3B is one byte of resolution: ρ = 0.75 drags the geomean to −25%,
      // and the band it is judged against reflects that same byte — √(33.3²) ÷ 1.
      const verdicts = computeVerdicts(
        createSamples(2, 4),
        createSamples(2, 3),
        METRIC_BYTES_LOWER,
      );

      const result = computeGeomean(verdicts, METRIC_BYTES_LOWER);

      expect.soft(result.n).toBe(1);
      expect.soft(result.value).toBeCloseTo(-25, 5);
      expect(result.band).toBeCloseTo(100 / 3, 5);
    });

    it("leaves an excluded metric's noise out of the band", () => {
      // noisy is unjudgeable, so neither its ratio nor its 250% noise counts;
      // steady alone drives the band: √(4²) ÷ 1 = 4
      const { verdicts, metricMeta } = buildInputs([
        {
          name: "noisy",
          verdict: {
            verdict: "unstable",
            method: "band",
            delta: -50,
            n: 4,
            usableN: 4,
            noisePct: 250,
            noiseAbs: 25,
          },
        },
        {
          name: "steady",
          verdict: {
            verdict: "improved",
            method: "band",
            delta: -50,
            n: 4,
            usableN: 4,
            noisePct: 4,
            noiseAbs: 2,
          },
        },
      ]);

      const result = computeGeomean(verdicts, metricMeta);

      expect(result.n).toBe(1);
      expect(result.band).toBeCloseTo(4, 10);
    });
  });
});
