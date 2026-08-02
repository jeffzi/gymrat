import { describe, expect, it } from "vitest";

import type { ProgressStep } from "../../src/compare.js";
import { EtaTracker, formatEta } from "../../src/eta.js";

function prepare(label: string): ProgressStep {
  return { kind: "prepare", label };
}

function sample(index: number, total: number, label: string): ProgressStep {
  return { kind: "sample", index, total, label };
}

/** Throws once the sequence is exhausted, rather than returning `undefined`. */
function clockSequence(...times: readonly number[]): () => number {
  let i = 0;
  return () => {
    const t = times[i++];
    if (t === undefined) throw new Error("Clock sequence exhausted");
    return t;
  };
}

describe("EtaTracker", () => {
  describe("record", () => {
    it("returns undefined for a prepare step", () => {
      // Arrange
      const tracker = new EtaTracker(1, clockSequence(0));

      // Act
      const result = tracker.record(prepare("A"));

      // Assert
      expect(result).toBeUndefined();
    });

    it("returns undefined for the very first sample step (no gap measured)", () => {
      // Arrange
      const tracker = new EtaTracker(1, clockSequence(0));

      // Act
      const result = tracker.record(sample(1, 3, "A"));

      // Assert
      expect(result).toBeUndefined();
    });

    it.each([
      {
        name: "returns an estimate at the second sample step once one gap is known",
        targets: 2,
        total: 3,
        expected: 500,
      },
      {
        name: "uses constructor-provided target count, not inferred from index-1 steps",
        targets: 3,
        total: 4,
        expected: 1100,
      },
    ])("$name", ({ targets, total, expected }) => {
      // Arrange — constructor-provided target count, gap of 100 between two index-1 steps
      const tracker = new EtaTracker(targets, clockSequence(0, 100));

      // Act
      tracker.record(sample(1, total, "A")); // t=0, first sample, no gap yet
      const result = tracker.record(sample(1, total, "B")); // t=100, gap=100

      // Assert
      // mean = 100, targetCount from constructor
      // completedSampleSteps before this call = 1, remaining = total*targets - 1
      // estimate = 100 * remaining
      expect(result).toBe(expected);
    });

    it("pools gaps from different targets into a shared mean", () => {
      // Arrange — target A takes 100ms, target B takes 200ms
      const tracker = new EtaTracker(2, clockSequence(0, 100, 300));

      // Act
      tracker.record(sample(1, 3, "A")); // t=0
      tracker.record(sample(1, 3, "B")); // t=100, gap=100 (A's duration)
      const result = tracker.record(sample(2, 3, "A")); // t=300, gap=200 (B's duration)

      // Assert
      // durations = [100, 200], mean = 150 (pooled across targets)
      // targetCount = 2, completedSampleSteps before this call = 2, remaining = 3*2 - 2 = 4
      // estimate = 150 * 4 = 600
      expect(result).toBe(600);
    });

    it("excludes the gap following a prepare step from the mean", () => {
      // Arrange — prepare introduces a 1000ms gap that must not skew the mean
      const tracker = new EtaTracker(1, clockSequence(0, 1000, 1100));

      // Act
      tracker.record(prepare("A")); // t=0
      tracker.record(sample(1, 3, "A")); // t=1000, gap excluded (after prepare)
      const result = tracker.record(sample(2, 3, "A")); // t=1100, gap=100 included

      // Assert
      // durations = [100] (prepare gap excluded), mean = 100
      // targetCount = 1, completedSampleSteps before this call = 1, remaining = 3*1 - 1 = 2
      // estimate = 100 * 2 = 200
      expect(result).toBe(200);
    });

    it("excludes mid-run prepare gaps, keying off kind not position", () => {
      // Arrange — prepare appears between sample steps, not just at the start
      const tracker = new EtaTracker(2, clockSequence(0, 100, 1000, 1100));

      // Act
      tracker.record(sample(1, 3, "A")); // t=0
      tracker.record(prepare("B")); // t=100, prepare returns early
      tracker.record(sample(1, 3, "B")); // t=1000, gap excluded (prevWasPrepare)
      const result = tracker.record(sample(2, 3, "A")); // t=1100, gap=100 included

      // Assert
      // durations = [100] (900ms prepare gap excluded), mean = 100
      // targetCount = 2, completedSampleSteps before this call = 2, remaining = 3*2 - 2 = 4
      // estimate = 100 * 4 = 400
      expect(result).toBe(400);
    });
  });
});

describe("formatEta", () => {
  describe("when below 1000ms", () => {
    it.each([
      { ms: 0, expected: "~1s left" },
      { ms: 500, expected: "~1s left" },
      { ms: 999, expected: "~1s left" },
    ])("formats $ms ms as '$expected' (floor at 1 second)", ({ ms, expected }) => {
      expect(formatEta(ms)).toBe(expected);
    });
  });

  describe("when seconds-only (1000ms to 59998ms)", () => {
    it.each([
      { ms: 1000, expected: "~1s left" },
      { ms: 48200, expected: "~48s left" },
    ])("formats $ms ms as '$expected'", ({ ms, expected }) => {
      expect(formatEta(ms)).toBe(expected);
    });
  });

  describe("when minutes and seconds (59999ms to 3599999ms)", () => {
    it.each([
      { ms: 59999, expected: "~1m left" },
      { ms: 60000, expected: "~1m left" },
      { ms: 130000, expected: "~2m 10s left" },
      { ms: 120000, expected: "~2m left" },
      { ms: 3599999, expected: "~1h left" },
    ])("formats $ms ms as '$expected'", ({ ms, expected }) => {
      expect(formatEta(ms)).toBe(expected);
    });
  });

  describe("when hours and minutes (3600000ms and above)", () => {
    it.each([
      { ms: 3600000, expected: "~1h left" },
      { ms: 3900000, expected: "~1h 5m left" },
      { ms: 7200000, expected: "~2h left" },
      { ms: 7260000, expected: "~2h 1m left" },
    ])("formats $ms ms as '$expected'", ({ ms, expected }) => {
      expect(formatEta(ms)).toBe(expected);
    });
  });
});
