import { stripVTControlCharacters as stripAnsi } from "node:util";

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { LoopOutcome, LoopPrimary } from "../../src/report/loop.js";
import { deriveOutcome, formatLoopHeader, formatVerdictBlock } from "../../src/report/loop.js";
import { renderReport } from "../../src/report/text.js";
import type { MetricComparisons } from "../../src/report/types.js";
import type { MetricEntry } from "../fixtures/comparison-result.js";
import {
  createComparisonResult,
  metricMeta,
  signedRankMetric,
} from "../fixtures/comparison-result.js";

/** The metric under `name`, judged in `direction` with no signal of its own. */
function directedMetric(direction: "lower" | "higher", gating = true): MetricEntry {
  const metric = signedRankMetric({ verdict: "no-signal", delta: 0 });
  return { ...metric, meta: metricMeta("time", { direction, gating }) };
}

/** A run whose single metric regressed, gating or not. */
function regressedMetrics(gating: boolean): MetricComparisons {
  const metric = signedRankMetric({ verdict: "regressed", delta: 4 });
  return { "decode/time": { ...metric, meta: metricMeta("decode", { gating }) } };
}

/** The geomean primary, improving by default. */
function geomeanPrimary(deltaPct = -4.2): LoopPrimary {
  return { kind: "geomean", deltaPct };
}

describe("renderReport", () => {
  describe("when a header override is given", () => {
    it("opens the report with the override instead of the compare header", () => {
      const result = createComparisonResult();

      const output = renderReport(result, { header: "iteration 3 · experiment vs baseline" });

      expect.soft(stripAnsi(output).split("\n")[0]).toBe("iteration 3 · experiment vs baseline");
      expect(stripAnsi(output)).not.toContain("gymrat compare");
    });
  });
});

describe("formatLoopHeader", () => {
  it("names the iteration, what is being compared, and the sample count", () => {
    const header = formatLoopHeader(7, 6);

    expect(stripAnsi(header)).toBe("iteration 7 · experiment vs baseline · 6 paired samples");
  });

  it.each([
    { samples: 1, expected: "· 1 paired sample" },
    { samples: 2, expected: "· 2 paired samples" },
  ])("matches the sample noun to a count of $samples", ({ samples, expected }) => {
    expect(stripAnsi(formatLoopHeader(1, samples))).toContain(expected);
  });

  describe("when rendering with color", () => {
    beforeEach(() => {
      vi.stubEnv("FORCE_COLOR", "1");
    });

    afterEach(() => {
      vi.unstubAllEnvs();
    });

    it("emboldens the iteration label", () => {
      expect(formatLoopHeader(7, 6)).toContain("\x1b[1miteration 7\x1b[22m");
    });

    it("dims each · separator", () => {
      expect(formatLoopHeader(7, 6)).toContain("\x1b[2m · \x1b[22m");
    });
  });
});

describe("formatVerdictBlock", () => {
  it.each([
    { outcome: "improved", word: "IMPROVED" },
    { outcome: "regressed", word: "REGRESSED" },
    { outcome: "no-signal", word: "NO-SIGNAL" },
  ] as const)("states the primary delta beside the $outcome verdict", ({ outcome, word }) => {
    const block = formatVerdictBlock(outcome, geomeanPrimary(), "gymrat keep");

    expect(stripAnsi(block[0] ?? "")).toBe(`primary: -4.2% · verdict: ${word}`);
  });

  it("closes the block with the next step", () => {
    const block = formatVerdictBlock("regressed", geomeanPrimary(3.1), "fix or gymrat discard");

    expect.soft(block).toHaveLength(2);
    expect(stripAnsi(block[1] ?? "")).toBe("Hint: fix or gymrat discard");
  });

  describe("when rendering with color", () => {
    beforeEach(() => {
      vi.stubEnv("FORCE_COLOR", "1");
    });

    afterEach(() => {
      vi.unstubAllEnvs();
    });

    it.each([
      { outcome: "improved", styled: "\x1b[1m\x1b[32mIMPROVED\x1b[39m\x1b[22m" },
      { outcome: "regressed", styled: "\x1b[1m\x1b[31mREGRESSED\x1b[39m\x1b[22m" },
      { outcome: "no-signal", styled: "\x1b[1mNO-SIGNAL\x1b[22m" },
    ] as const)("paints the $outcome verdict word", ({ outcome, styled }) => {
      const block = formatVerdictBlock(outcome, geomeanPrimary(), "gymrat keep");

      expect(block[0]).toContain(styled);
    });
  });
});

describe("deriveOutcome", () => {
  it("reports regressed when a gating metric regressed, whatever the primary did", () => {
    expect(deriveOutcome(regressedMetrics(true), geomeanPrimary(-9))).toBe("regressed");
  });

  it("leaves a regression in a non-gating metric out of the outcome", () => {
    expect(deriveOutcome(regressedMetrics(false), geomeanPrimary(-9))).toBe("improved");
  });

  it.each([
    { primary: { kind: "geomean", deltaPct: -3 }, expected: "improved" },
    { primary: { kind: "geomean", deltaPct: 3 }, expected: "no-signal" },
    { primary: { kind: "geomean", deltaPct: 0 }, expected: "no-signal" },
    { primary: { kind: "metric", name: "lower/time", deltaPct: -3 }, expected: "improved" },
    { primary: { kind: "metric", name: "lower/time", deltaPct: 3 }, expected: "no-signal" },
    { primary: { kind: "metric", name: "higher/time", deltaPct: 3 }, expected: "improved" },
    { primary: { kind: "metric", name: "higher/time", deltaPct: -3 }, expected: "no-signal" },
  ] as const satisfies readonly { primary: LoopPrimary; expected: LoopOutcome }[])(
    "reads a $primary.deltaPct% $primary.kind primary as $expected",
    ({ primary, expected }) => {
      const metrics: MetricComparisons = {
        "lower/time": directedMetric("lower"),
        "higher/time": directedMetric("higher"),
      };

      expect(deriveOutcome(metrics, primary)).toBe(expected);
    },
  );

  it("reports no signal when the run never measured the primary metric", () => {
    const primary: LoopPrimary = { kind: "metric", name: "absent/time", deltaPct: -30 };

    expect(deriveOutcome({ "lower/time": directedMetric("lower") }, primary)).toBe("no-signal");
  });
});
