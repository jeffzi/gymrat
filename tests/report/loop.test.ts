import { stripVTControlCharacters as stripAnsi } from "node:util";

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type {
  LoopOutcome,
  LoopPrimary,
  StatusIteration,
  StatusSummary,
} from "../../src/report/loop.js";
import {
  deriveOutcome,
  formatLoopHeader,
  formatStatusBaseline,
  formatStatusFooter,
  formatStatusHeader,
  formatStatusIteration,
  formatVerdictBlock,
} from "../../src/report/loop.js";
import type { MetricComparisons } from "../../src/report/types.js";
import type { BaselineRecord } from "../../src/session/records.js";
import type { MetricEntry } from "../fixtures/comparison-result.js";
import { metricMeta, signedRankMetric } from "../fixtures/comparison-result.js";
import { sessionRecord } from "../fixtures/session-records.js";

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

const SESSION_ID = "20260808-141530-a3f2";
/** A 40-hex sha whose first seven characters are recognizable on their own. */
const BASELINE_SHA = `a1b2c3d${"e".repeat(33)}`;
const KEEP_COMMIT = `b1b2b3b${"c".repeat(33)}`;

/** An improved iteration numbered 1, settled the way `settle` says. */
function statusIteration(settle: StatusIteration["settle"]): StatusIteration {
  return { seq: 1, deltaPct: -7.2, outcome: "improved", settle };
}

/** A session that measured four iterations, kept one and threw one away. */
function statusSummary(overrides: Partial<StatusSummary> = {}): StatusSummary {
  return {
    iterationCount: 4,
    keepCount: 1,
    discardCount: 1,
    targetReached: false,
    ...overrides,
  };
}

describe("formatStatusHeader", () => {
  it("names the session, the baseline it forked from, the branch, both worktrees, and the adapter", () => {
    // Act
    const lines = formatStatusHeader(
      sessionRecord({ baseline: { ref: "main", sha: BASELINE_SHA } }),
    );

    // Assert
    expect(lines.map(stripAnsi)).toStrictEqual([
      `session ${SESSION_ID} · baseline main@a1b2c3d · adapter metric-lines`,
      `branch gymrat/${SESSION_ID}`,
      "experiment worktree /repo/.gymrat/worktrees/experiment",
      "baseline worktree /repo/.gymrat/worktrees/baseline",
    ]);
  });

  describe("when rendering with color", () => {
    beforeEach(() => {
      vi.stubEnv("FORCE_COLOR", "1");
    });

    afterEach(() => {
      vi.unstubAllEnvs();
    });

    it("emboldens the session it opens on", () => {
      expect(
        formatStatusHeader(sessionRecord({ baseline: { ref: "main", sha: BASELINE_SHA } }))[0],
      ).toContain(`\x1b[1msession ${SESSION_ID}\x1b[22m`);
    });
  });
});

describe("formatStatusIteration", () => {
  it.each([
    {
      desc: "the commit a kept iteration landed as",
      settle: { kind: "kept", commit: KEEP_COMMIT },
      expected: "iteration 1 · ✓ -7.2% · kept b1b2b3b",
    },
    {
      desc: "an iteration thrown away",
      settle: { kind: "discarded" },
      expected: "iteration 1 · ✓ -7.2% · discarded",
    },
    {
      desc: "an iteration nobody settled",
      settle: { kind: "unsettled" },
      expected: "iteration 1 · ✓ -7.2% · unsettled",
    },
    {
      desc: "why a keep was blocked",
      settle: { kind: "keep-blocked", reason: "checks-failed" },
      expected: "iteration 1 · ✓ -7.2% · keep-blocked (checks-failed)",
    },
  ] satisfies { desc: string; settle: StatusIteration["settle"]; expected: string }[])(
    "states $desc",
    ({ settle, expected }) => {
      expect(stripAnsi(formatStatusIteration(statusIteration(settle)))).toBe(expected);
    },
  );

  it.each([
    { outcome: "improved", glyph: "✓" },
    { outcome: "regressed", glyph: "✗" },
    { outcome: "no-signal", glyph: "~" },
  ] as const satisfies { outcome: LoopOutcome; glyph: string }[])(
    "marks a $outcome iteration with $glyph",
    ({ outcome, glyph }) => {
      const entry = { ...statusIteration({ kind: "unsettled" }), outcome };

      expect(stripAnsi(formatStatusIteration(entry))).toBe(
        `iteration 1 · ${glyph} -7.2% · unsettled`,
      );
    },
  );

  describe("when rendering with color", () => {
    beforeEach(() => {
      vi.stubEnv("FORCE_COLOR", "1");
    });

    afterEach(() => {
      vi.unstubAllEnvs();
    });

    it.each([
      { outcome: "improved", styled: "\x1b[32m✓\x1b[39m" },
      { outcome: "regressed", styled: "\x1b[31m✗\x1b[39m" },
    ] as const satisfies { outcome: LoopOutcome; styled: string }[])(
      "paints the $outcome glyph",
      ({ outcome, styled }) => {
        const entry = { ...statusIteration({ kind: "unsettled" }), outcome };

        expect(formatStatusIteration(entry)).toContain(styled);
      },
    );
  });
});

describe("formatStatusBaseline", () => {
  it("states the label beside the median each metric measured", () => {
    // Arrange
    const record: BaselineRecord = {
      type: "baseline",
      at: "2026-08-08T14:15:30.000Z",
      label: "main",
      samples: [
        { total_ms: 15200, alloc_bytes: 1500 },
        { total_ms: 15184, alloc_bytes: 1540 },
      ],
    };

    // Act
    const line = formatStatusBaseline(record);

    // Assert
    expect(stripAnsi(line)).toBe("baseline main · total_ms 15192 · alloc_bytes 1520");
  });

  it("takes each median over the rounds that reported the metric", () => {
    // Arrange
    const record: BaselineRecord = {
      type: "baseline",
      at: "2026-08-08T14:15:30.000Z",
      label: "main",
      samples: [{ total_ms: 100, alloc_bytes: 40 }, { total_ms: 300 }],
    };

    // Act
    const line = formatStatusBaseline(record);

    // Assert
    expect(stripAnsi(line)).toBe("baseline main · total_ms 200 · alloc_bytes 40");
  });
});

describe("formatStatusFooter", () => {
  it.each([
    {
      desc: "a session one iteration in",
      summary: statusSummary({ iterationCount: 1, keepCount: 1, discardCount: 0 }),
      expected: "1 iteration · 1 kept · 0 discarded",
    },
    {
      desc: "a session several iterations in",
      summary: statusSummary(),
      expected: "4 iterations · 1 kept · 1 discarded",
    },
  ])("totals how $desc settled", ({ summary, expected }) => {
    expect(stripAnsi(formatStatusFooter(summary)[0] ?? "")).toBe(expected);
  });

  it.each([
    {
      desc: "counts the iterations against a configured maximum",
      summary: statusSummary({ stop: { maxIterations: 30 } }),
      expected: "stop: 4 of 30 iterations",
    },
    {
      desc: "leaves a configured target pending until it is reached",
      summary: statusSummary({ stop: { targetValue: 95 } }),
      expected: "stop: target pending",
    },
    {
      desc: "calls a reached target reached",
      summary: statusSummary({ stop: { targetValue: 95 }, targetReached: true }),
      expected: "stop: target reached",
    },
    {
      desc: "states both conditions when both are configured",
      summary: statusSummary({ stop: { targetValue: 95, maxIterations: 30 } }),
      expected: "stop: 4 of 30 iterations · target pending",
    },
    {
      desc: "says nothing about stopping when nothing is configured",
      summary: statusSummary(),
      expected: undefined,
    },
  ])("$desc", ({ summary, expected }) => {
    const lines = formatStatusFooter(summary).map(stripAnsi);

    expect(lines.find((line) => line.startsWith("stop:"))).toBe(expected);
  });
});
