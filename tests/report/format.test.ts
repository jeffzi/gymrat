import { describe, it, expect } from "vitest";

import {
  computeColumnWidth,
  countVerdicts,
  formatDelta,
  formatLabel,
  formatNoiseBand,
  formatPValue,
  formatSpread,
  formatTableLine,
  formatValue,
  getGlyph,
  selectHighlights,
} from "../../src/report/format.js";
import type { ComparisonResult } from "../../src/report/types.js";
import type { ApproximateVerdictValue } from "../../src/verdict/verdict.js";

type Metrics = ComparisonResult["metrics"];
type MetricEntry = Metrics[string];

/** One candidate's signed-rank outcome against the shared baseline. */
interface CandidateSpec {
  verdict: ApproximateVerdictValue;
  delta: number;
  noisePct?: number;
}

/**
 * A metric judged once per candidate against the shared baseline.
 *
 * The baseline median and spread are carried once, so the candidate entries
 * differ only in what the pairwise verdict engine returned for each of them.
 */
function metricFor(
  candidates: readonly CandidateSpec[],
  direction: "lower" | "higher" = "lower",
): MetricEntry {
  return {
    baselineMedian: 100,
    baselineSpread: 1,
    candidates: candidates.map(({ verdict, delta, noisePct = 2.5 }) => ({
      median: 100 + delta,
      spread: 1,
      verdict: { verdict, method: "signed-rank", delta, n: 10, p: 0.01, noisePct },
    })),
    meta: { direction, gating: true, exact: false },
  };
}

/** A single-candidate metric whose verdict came from the signed-rank method. */
function approximateMetric(options: {
  verdict: ApproximateVerdictValue;
  delta: number;
  noisePct?: number;
  direction?: "lower" | "higher";
}): MetricEntry {
  const { direction = "lower", ...candidate } = options;
  return metricFor([candidate], direction);
}

/** A metric the candidate never reported, so no verdict could be computed. */
function oneSidedMetric(): MetricEntry {
  return {
    baselineMedian: 100,
    baselineSpread: 1,
    candidates: [{}],
    meta: { direction: "lower", gating: false, exact: false },
  };
}

describe("formatValue", () => {
  describe("when the metric is measured in nanoseconds", () => {
    it.each([
      { tier: "ns", value: 0, expected: "0ns" },
      { tier: "ns", value: 914, expected: "914ns" },
      { tier: "ns", value: 999, expected: "999ns" },
      { tier: "µs", value: 1000, expected: "1.0µs" },
      { tier: "µs", value: 1735, expected: "1.7µs" },
      { tier: "µs", value: 26825, expected: "26.8µs" },
      { tier: "ms", value: 1_000_000, expected: "1.0ms" },
      { tier: "s", value: 2_000_000_000, expected: "2.0s" },
    ])("scales $value to the $tier tier as $expected", ({ value, expected }) => {
      expect(formatValue(value, "ns")).toBe(expected);
    });
  });

  describe("when the metric is measured in bytes", () => {
    it.each([
      { tier: "B", value: 512, expected: "512B" },
      { tier: "B", value: 999, expected: "999B" },
      { tier: "KB", value: 1000, expected: "1.0KB" },
      { tier: "KB", value: 3600, expected: "3.6KB" },
      { tier: "KB", value: 49152, expected: "49.2KB" },
      { tier: "MB", value: 1_000_000, expected: "1.0MB" },
      { tier: "GB", value: 2_000_000_000, expected: "2.0GB" },
    ])("scales $value to the $tier tier as $expected", ({ value, expected }) => {
      expect(formatValue(value, "bytes")).toBe(expected);
    });
  });

  describe("when the metric carries no unit", () => {
    it.each([
      { value: 0, expected: "0" },
      { value: 1200, expected: "1200" },
      { value: 1_100_000, expected: "1100000" },
    ])("renders $value unscaled as $expected", ({ value, expected }) => {
      expect(formatValue(value)).toBe(expected);
    });
  });
});

describe("formatSpread", () => {
  it.each([
    { spread: 0, expected: " ± 0%" },
    { spread: 1, expected: " ± 1%" },
    { spread: 25, expected: " ± 25%" },
  ])("renders $spread as '$expected'", ({ spread, expected }) => {
    expect(formatSpread(spread)).toBe(expected);
  });

  it("renders nothing when the spread is unknown", () => {
    expect(formatSpread(undefined)).toBe("");
  });
});

describe("formatDelta", () => {
  it.each([
    { desc: "signs a regression", delta: 2.2, expected: "+2.2%" },
    { desc: "signs an improvement", delta: -17.9, expected: "-17.9%" },
    { desc: "leaves an exact zero unsigned", delta: 0, expected: "0.0%" },
    { desc: "rounds to one decimal", delta: 30, expected: "+30.0%" },
    {
      desc: "drops the sign of a positive delta that rounds to zero",
      delta: 0.04,
      expected: "0.0%",
    },
    {
      desc: "drops the sign of a negative delta that rounds to zero",
      delta: -0.04,
      expected: "0.0%",
    },
    { desc: "keeps the sign just above the rounding floor", delta: 0.06, expected: "+0.1%" },
    { desc: "keeps the sign just below the rounding floor", delta: -0.06, expected: "-0.1%" },
  ])("$desc: $delta", ({ delta, expected }) => {
    expect(formatDelta(delta)).toBe(expected);
  });

  it("renders nothing when the delta is undefined arithmetic", () => {
    expect(formatDelta(Number.NaN)).toBe("");
  });
});

describe("formatPValue", () => {
  it.each([
    { desc: "collapses zero to the display floor", p: 0, expected: "p<0.001" },
    { desc: "collapses values below the floor", p: 0.0001, expected: "p<0.001" },
    { desc: "keeps three decimals below 0.01", p: 0.002, expected: "p=0.002" },
    { desc: "keeps two decimals at 0.01 and above", p: 0.08, expected: "p=0.08" },
  ])("$desc: $p", ({ p, expected }) => {
    expect(formatPValue(p)).toBe(expected);
  });
});

describe("getGlyph", () => {
  it.each([
    { verdict: "improved" as const, expected: "✓" },
    { verdict: "regressed" as const, expected: "✗" },
    { verdict: "no-signal" as const, expected: "~" },
    { verdict: "unstable" as const, expected: "≈" },
  ])("marks $verdict with $expected", ({ verdict, expected }) => {
    expect(getGlyph(verdict)).toBe(expected);
  });
});

describe("formatNoiseBand", () => {
  it.each([
    { noisePct: 2.5, expected: "±2.5%" },
    { noisePct: 2, expected: "±2.0%" },
    { noisePct: 0.5, expected: "±0.5%" },
    { noisePct: 213.47, expected: "±213.5%" },
  ])("renders a noise band of $noisePct as $expected", ({ noisePct, expected }) => {
    expect(formatNoiseBand(noisePct)).toBe(expected);
  });
});

describe("countVerdicts", () => {
  it.each([
    {
      desc: "counts every verdict class and ignores metrics that have no verdict",
      metrics: {
        "faster/time": approximateMetric({ verdict: "improved", delta: -10 }),
        "also-faster/time": approximateMetric({ verdict: "improved", delta: -5 }),
        "slower/time": approximateMetric({ verdict: "regressed", delta: 8 }),
        "jittery/time": approximateMetric({ verdict: "unstable", delta: 5, noisePct: 300 }),
        "flat/time": approximateMetric({ verdict: "no-signal", delta: 0.2 }),
        "one-sided/time": oneSidedMetric(),
      } as Metrics,
      expected: { improved: 2, regressed: 1, unstable: 1, noSignal: 1 },
    },
    {
      desc: "reports zeros when there are no metrics",
      metrics: {} as Metrics,
      expected: { improved: 0, regressed: 0, unstable: 0, noSignal: 0 },
    },
  ])("$desc", ({ metrics, expected }) => {
    const counts = countVerdicts(metrics, 0);

    expect(counts).toStrictEqual(expected);
  });

  it.each([
    {
      candidateIndex: 0,
      expected: { improved: 1, regressed: 0, unstable: 0, noSignal: 0 },
    },
    {
      candidateIndex: 1,
      expected: { improved: 0, regressed: 1, unstable: 0, noSignal: 0 },
    },
  ])(
    "counts only the verdicts belonging to candidate $candidateIndex",
    ({ candidateIndex, expected }) => {
      const metrics: Metrics = {
        "decode/time": metricFor([
          { verdict: "improved", delta: -10 },
          { verdict: "regressed", delta: 8 },
        ]),
      };

      expect(countVerdicts(metrics, candidateIndex)).toStrictEqual(expected);
    },
  );
});

describe("selectHighlights", () => {
  it("orders regressions by magnitude, then improvements by magnitude, then unstable by noise", () => {
    const metrics: Metrics = {
      "small-improvement/time": approximateMetric({ verdict: "improved", delta: -4 }),
      "quiet-unstable/time": approximateMetric({ verdict: "unstable", delta: 6, noisePct: 210 }),
      "small-regression/time": approximateMetric({ verdict: "regressed", delta: 3 }),
      // Higher is better here, so the negative delta is the larger regression.
      "big-regression/ops": approximateMetric({
        verdict: "regressed",
        delta: -12,
        direction: "higher",
      }),
      "within-noise/time": approximateMetric({ verdict: "no-signal", delta: 0.4 }),
      "big-improvement/time": approximateMetric({ verdict: "improved", delta: -20 }),
      "one-sided/time": oneSidedMetric(),
      "loud-unstable/time": approximateMetric({ verdict: "unstable", delta: 5, noisePct: 300 }),
    };

    const highlights = selectHighlights(metrics, 0);

    expect(highlights.map((highlight) => highlight.name)).toStrictEqual([
      "big-regression/ops",
      "small-regression/time",
      "big-improvement/time",
      "small-improvement/time",
      "loud-unstable/time",
      "quiet-unstable/time",
    ]);
  });

  it("keeps declaration order for metrics of equal magnitude", () => {
    const metrics: Metrics = {
      "second-listed/ops": approximateMetric({
        verdict: "regressed",
        delta: -5,
        direction: "higher",
      }),
      "third-listed/time": approximateMetric({ verdict: "regressed", delta: 5 }),
      "first-listed/time": approximateMetric({ verdict: "regressed", delta: 9 }),
    };

    const highlights = selectHighlights(metrics, 0);

    expect(highlights.map((highlight) => highlight.name)).toStrictEqual([
      "first-listed/time",
      "second-listed/ops",
      "third-listed/time",
    ]);
  });

  it("carries the metric and the candidate slice that earned the highlight", () => {
    const metrics: Metrics = {
      "slower/time": metricFor([
        { verdict: "improved", delta: -10 },
        { verdict: "regressed", delta: 8 },
      ]),
    };

    const highlights = selectHighlights(metrics, 1);

    expect(highlights).toStrictEqual([
      {
        name: "slower/time",
        metric: metrics["slower/time"],
        candidate: metrics["slower/time"]?.candidates[1],
      },
    ]);
  });

  it.each([
    { candidateIndex: 0, expected: ["b/time", "a/time"] },
    { candidateIndex: 1, expected: ["a/time"] },
  ])(
    "ranks candidate $candidateIndex by the verdicts it earned, not its neighbor's",
    ({ candidateIndex, expected }) => {
      const metrics: Metrics = {
        "a/time": metricFor([
          { verdict: "improved", delta: -4 },
          { verdict: "regressed", delta: 3 },
        ]),
        "b/time": metricFor([
          { verdict: "regressed", delta: 6 },
          { verdict: "no-signal", delta: 0.2 },
        ]),
      };

      expect(
        selectHighlights(metrics, candidateIndex).map((highlight) => highlight.name),
      ).toStrictEqual(expected);
    },
  );
});

describe("computeColumnWidth", () => {
  it.each([
    { driver: "the longest cell", header: 6, contents: [11, 24], min: 12, expected: 26 },
    { driver: "the header", header: 24, contents: [11, 9], min: 12, expected: 26 },
    // The gutter is added before the floor applies, so widest+2 wins once it clears the minimum.
    {
      driver: "the widest cell over the minimum",
      header: 10,
      contents: [11, 9],
      min: 12,
      expected: 13,
    },
    { driver: "the minimum, with no rows", header: 6, contents: [], min: 12, expected: 12 },
  ])("sizes the column from $driver", ({ header, contents, min, expected }) => {
    expect(computeColumnWidth(header, contents, min)).toBe(expected);
  });
});

describe("formatTableLine", () => {
  it.each([
    {
      desc: "pads every cell to its column width and separates them with a bar",
      cells: ["metric", "old", "new"],
      widths: [10, 6, 6],
      expected: "metric    │old   │new",
    },
    {
      desc: "pads interior cells that are empty so later columns stay aligned",
      cells: ["geomean", "", "", "-6.0%"],
      widths: [10, 6, 6, 8],
      expected: "geomean   │      │      │-6.0%",
    },
    {
      desc: "trims the padding after a trailing empty cell",
      cells: ["one-sided", "2.048µ ± 2%", ""],
      widths: [12, 14, 10],
      expected: "one-sided   │2.048µ ± 2%   │",
    },
  ])("$desc", ({ cells, widths, expected }) => {
    expect(formatTableLine(cells, widths)).toBe(expected);
  });
});

describe("formatLabel", () => {
  it("wraps the label in ANSI codes for the requested styles when color is on", () => {
    const styled = formatLabel("Hint:", ["yellow", "underline"], true);

    // \x1b[33m = yellow, \x1b[4m = underline
    expect.soft(styled).toContain("\x1b[33m");
    expect.soft(styled).toContain("\x1b[4m");
    expect(styled).toContain("Hint:");
  });

  it("returns the bare label when color is off", () => {
    expect(formatLabel("Hint:", ["yellow", "underline"], false)).toBe("Hint:");
  });
});
