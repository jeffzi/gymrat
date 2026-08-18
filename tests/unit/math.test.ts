import { describe, expect, it } from "vitest";

import { computeHalfRange } from "../../src/math.js";

/** More samples than an engine accepts as arguments to a single call. */
const SAMPLE_COUNT = 300_000;

describe("computeHalfRange", () => {
  describe("when the run holds more samples than a call takes arguments", () => {
    it("returns half the spread between the extremes", () => {
      // 7919 is coprime with 1000, so the values cycle through the whole
      // 0-999 range and neither extreme lands at an end of the array.
      const values = Array.from({ length: SAMPLE_COUNT }, (_, index) => (index * 7919) % 1000);

      const halfRange = computeHalfRange(values);

      expect(halfRange).toBe(499.5);
    });
  });

  describe("when a sample is not finite", () => {
    it.each([
      { description: "NaN among finite samples", values: [1, Number.NaN, 3] },
      { description: "Infinity among finite samples", values: [1, Number.POSITIVE_INFINITY, 3] },
      {
        description: "-Infinity among finite samples",
        values: [1, Number.NEGATIVE_INFINITY, 3],
      },
      { description: "nothing but NaN", values: [Number.NaN, Number.NaN] },
      {
        description: "nothing but Infinity",
        values: [Number.POSITIVE_INFINITY, Number.POSITIVE_INFINITY],
      },
    ])("reports an undefined half-range for $description", ({ values }) => {
      const halfRange = computeHalfRange(values);

      // NaN is the project's undefined-measurement sentinel — it prints
      // blank and is excluded from the geomean, where a dropped sample or an
      // infinite spread would read as a real measurement.
      expect(halfRange).toBeNaN();
    });
  });
});
