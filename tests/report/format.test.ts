import { afterEach, describe, expect, it, vi } from "vitest";

import {
  computeColumnWidth,
  countVerdicts,
  displayClass,
  formatDelta,
  formatEvidence,
  formatLabel,
  formatMetricCell,
  formatNoiseBand,
  formatPValue,
  formatSpread,
  formatTableLine,
  formatValue,
  getGlyph,
  legendGlosses,
  methodFooterLines,
  selectHighlights,
  truncateLabels,
  verdictSummaryParts,
} from "../../src/report/format.js";
import type { ComparisonResult } from "../../src/report/types.js";
import type {
  ApproximateVerdictValue,
  BandVerdict,
  ExactVerdict,
  SignedRankVerdict,
} from "../../src/verdict/verdict.js";

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
      verdict: {
        verdict,
        method: "signed-rank",
        delta,
        n: 10,
        p: 0.01,
        noisePct,
        noiseAbs: noisePct,
      },
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

/**
 * A single-candidate metric whose verdict fell back to the noise-band method.
 *
 * `n` is `pairedA.length` (total pair count) and `usableN` how many of those
 * pairs survived tie-dropping. When `n < 6` the fallback is due to insufficient
 * samples; when `n >= 6` and `usableN < 6` it is due to too many tied pairs
 * zeroing out the Wilcoxon-usable differences.
 */
function bandMetric(options: {
  verdict?: ApproximateVerdictValue;
  delta?: number;
  n: number;
  usableN?: number;
  noisePct?: number;
  direction?: "lower" | "higher";
}): MetricEntry {
  const {
    verdict = "no-signal",
    delta = -1,
    n,
    usableN = n,
    noisePct = 2.5,
    direction = "lower",
  } = options;
  return {
    baselineMedian: 100,
    baselineSpread: 5,
    candidates: [
      {
        median: 100 + delta,
        spread: 4,
        verdict: {
          verdict,
          method: "band" as const,
          delta,
          n,
          usableN,
          band: noisePct,
          noisePct,
          noiseAbs: noisePct,
        },
      },
    ],
    meta: { direction, gating: true, exact: false },
  };
}

/** A noise-band verdict, tied pairs and all. */
function bandVerdict(overrides: Partial<BandVerdict> = {}): BandVerdict {
  return {
    verdict: "no-signal",
    method: "band",
    delta: -0.5,
    n: 10,
    usableN: 3,
    band: 2.5,
    noisePct: 2.5,
    noiseAbs: 2.5,
    ...overrides,
  };
}

/** A verdict the Wilcoxon signed-rank test produced. */
function signedRankVerdict(overrides: Partial<SignedRankVerdict> = {}): SignedRankVerdict {
  return {
    verdict: "no-signal",
    method: "signed-rank",
    delta: 0.2,
    n: 10,
    p: 0.49,
    noisePct: 2.5,
    noiseAbs: 2.5,
    ...overrides,
  };
}

/** A verdict read straight off a counted metric, with no statistics behind it. */
function exactVerdict(overrides: Partial<ExactVerdict> = {}): ExactVerdict {
  return { verdict: "no-signal", method: "exact", delta: 0, n: 10, ...overrides };
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

describe("formatMetricCell", () => {
  describe("when the spread stays within the median", () => {
    it.each([
      { desc: "an ordinary spread", median: 5, spread: 12, expected: "5B ± 12%" },
      { desc: "a spread at the cap", median: 5, spread: 100, expected: "5B ± 100%" },
    ])("keeps $desc relative", ({ median, spread, expected }) => {
      expect(formatMetricCell(median, spread, "bytes")).toBe(expected);
    });
  });

  describe("when the spread outgrows the median", () => {
    it.each([
      { unit: "bytes" as const, median: 5, spread: 7620, expected: "5B ± 381B" },
      { unit: "ns" as const, median: 1735, spread: 200, expected: "1.7µs ± 3.5µs" },
      { unit: undefined, median: 1200, spread: 150, expected: "1200 ± 1800" },
    ])(
      "restates a $spread% spread in $unit units as '$expected'",
      ({ median, spread, unit, expected }) => {
        expect(formatMetricCell(median, spread, unit)).toBe(expected);
      },
    );
  });
});

describe("formatEvidence", () => {
  it("marks a counted verdict as exact", () => {
    expect(formatEvidence(exactVerdict({ verdict: "improved", delta: -7.9 }))).toBe("(exact)");
  });

  it("leaves a statistical improvement with nothing to add", () => {
    expect(formatEvidence(signedRankVerdict({ verdict: "improved", delta: -10 }))).toBe("");
  });

  it.each([
    { desc: "well below the cap", noisePct: 30, expected: "noise ±30.0%" },
    { desc: "at the cap", noisePct: 100, expected: "noise ±100.0%" },
  ])("states unstable noise $desc as a percentage", ({ noisePct, expected }) => {
    const verdict = signedRankVerdict({ verdict: "unstable", noisePct, noiseAbs: 381 });

    expect(formatEvidence(verdict, "bytes", 5)).toBe(expected);
  });

  it("states unstable noise past the cap in the metric's own units", () => {
    const verdict = signedRankVerdict({ verdict: "unstable", noisePct: 7620, noiseAbs: 381 });

    expect(formatEvidence(verdict, "bytes", 5)).toBe("±381B noise on a 5B median");
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

describe("displayClass", () => {
  it.each([
    {
      desc: "ties left the signed-rank test too few usable pairs",
      verdict: bandVerdict({ n: 10, usableN: 3 }),
      expected: "identical",
    },
    {
      desc: "ties left exactly enough usable pairs",
      verdict: bandVerdict({ n: 10, usableN: 6 }),
      expected: "within-noise",
    },
    {
      desc: "the pair count sits one below the signed-rank floor",
      verdict: bandVerdict({ n: 6, usableN: 5 }),
      expected: "identical",
    },
    {
      desc: "the run was too short to reach the floor at all",
      verdict: bandVerdict({ n: 5, usableN: 5 }),
      expected: "within-noise",
    },
    {
      desc: "the band found an improvement despite the ties",
      verdict: bandVerdict({ verdict: "improved", delta: -10, n: 10, usableN: 3 }),
      expected: "improved",
    },
    {
      desc: "the noise swamped the band",
      verdict: bandVerdict({ verdict: "unstable", n: 10, usableN: 3 }),
      expected: "unstable",
    },
    {
      desc: "the signed-rank test itself found no signal",
      verdict: signedRankVerdict(),
      expected: "within-noise",
    },
    {
      desc: "a counted metric came out unchanged",
      verdict: exactVerdict(),
      expected: "within-noise",
    },
  ])("shows $expected when $desc", ({ verdict, expected }) => {
    expect(displayClass(verdict)).toBe(expected);
  });
});

describe("getGlyph", () => {
  it.each([
    { displayClass: "improved" as const, expected: "✓" },
    { displayClass: "regressed" as const, expected: "✗" },
    { displayClass: "unstable" as const, expected: "≈" },
    { displayClass: "identical" as const, expected: "=" },
    { displayClass: "within-noise" as const, expected: "~" },
  ])("marks $displayClass with $expected", ({ displayClass: shown, expected }) => {
    expect(getGlyph(shown)).toBe(expected);
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

  it("leaves identical metrics out, the way it leaves within-noise ones out", () => {
    const metrics: Metrics = {
      "faster/time": approximateMetric({ verdict: "improved", delta: -10 }),
      "tied/heap": bandMetric({ n: 10, usableN: 3 }),
    };

    const highlights = selectHighlights(metrics, 0);

    expect(highlights.map((highlight) => highlight.name)).toStrictEqual(["faster/time"]);
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

describe("truncateLabels", () => {
  describe("when every label fits the display width", () => {
    it("returns each one verbatim", () => {
      // "feature/short-branch" is exactly the 20-char display width.
      expect(truncateLabels(["main", "feature/short-branch"])).toStrictEqual([
        "main",
        "feature/short-branch",
      ]);
    });
  });

  describe("when a label overflows the display width", () => {
    it("joins its head and tail with a single ellipsis", () => {
      const truncated = truncateLabels(["feature/entity-spawn-fastpath"]);

      expect.soft(truncated).toStrictEqual(["feature/en…-fastpath"]);
      expect(truncated[0]).toHaveLength(20);
    });
  });

  describe("when two labels are identical once truncated", () => {
    it("extends the kept tail until the displayed labels differ", () => {
      // Both share the same 10-char head and 9-char tail, so the 20-char form
      // would name two different branches identically.
      const truncated = truncateLabels([
        "feature/experiment-one-fastpath",
        "feature/exploration-two-fastpath",
      ]);

      expect(truncated).toStrictEqual(["feature/ex…e-fastpath", "feature/ex…o-fastpath"]);
    });
  });
});

describe("formatLabel", () => {
  afterEach(() => {
    vi.unstubAllEnvs();
  });

  it("wraps the label in ANSI codes for the requested styles when color is forced", () => {
    vi.stubEnv("FORCE_COLOR", "1");

    const styled = formatLabel("Hint:", ["yellow", "underline"]);

    // \x1b[33m = yellow, \x1b[4m = underline
    expect.soft(styled).toContain("\x1b[33m");
    expect.soft(styled).toContain("\x1b[4m");
    expect(styled).toContain("Hint:");
  });

  it("returns the bare label when stdout is not a TTY", () => {
    expect(formatLabel("Hint:", ["yellow", "underline"])).toBe("Hint:");
  });
});

describe("verdictSummaryParts", () => {
  afterEach(() => {
    vi.unstubAllEnvs();
  });

  const mixed: Metrics = {
    "faster/time": approximateMetric({ verdict: "improved", delta: -10 }),
    "slower/time": approximateMetric({ verdict: "regressed", delta: 8 }),
    "jittery/time": approximateMetric({ verdict: "unstable", delta: 5, noisePct: 300 }),
    "flat/time": approximateMetric({ verdict: "no-signal", delta: 0.2 }),
    "tied/heap": bandMetric({ n: 10, usableN: 3 }),
  };

  it("returns parts with no ANSI escapes when stdout is not a TTY", () => {
    const parts = verdictSummaryParts(mixed, 0);

    expect(parts.join("")).not.toContain("\x1b[");
  });

  it("tallies identical metrics apart from the ones within noise", () => {
    const parts = verdictSummaryParts(mixed, 0);

    expect.soft(parts.find((p) => p.includes("identical"))).toBe("= 1 identical");
    expect(parts.find((p) => p.includes("within noise"))).toBe("~ 1 within noise");
  });

  it.each([
    { label: "improved", code: "32", color: "green" },
    { label: "regressed", code: "31", color: "red" },
    { label: "unstable", code: "33", color: "yellow" },
    { label: "identical", code: "36", color: "cyan" },
  ])("styles the non-zero $label part $color when color is forced", ({ label, code }) => {
    vi.stubEnv("FORCE_COLOR", "1");

    const parts = verdictSummaryParts(mixed, 0);
    const part = parts.find((p) => p.includes(label));

    expect(part).toContain(`\x1b[${code}m`);
  });

  it.each([{ label: "regressed" }, { label: "identical" }])(
    "dims the zero-count $label part when color is forced",
    ({ label }) => {
      vi.stubEnv("FORCE_COLOR", "1");

      const onlyImproved: Metrics = {
        "faster/time": approximateMetric({ verdict: "improved", delta: -10 }),
      };

      const parts = verdictSummaryParts(onlyImproved, 0);
      const part = parts.find((p) => p.includes(label));

      expect(part).toContain("\x1b[2m");
    },
  );

  it("dims the within-noise part regardless of count when color is forced", () => {
    vi.stubEnv("FORCE_COLOR", "1");

    const parts = verdictSummaryParts(mixed, 0);
    const noisePart = parts.find((p) => p.includes("within noise"));

    expect(noisePart).toContain("\x1b[2m");
  });
});

describe("legendGlosses", () => {
  afterEach(() => {
    vi.unstubAllEnvs();
  });

  it("returns plain text with no ANSI escapes when stdout is not a TTY", () => {
    expect(legendGlosses()).not.toContain("\x1b[");
  });

  it.each([
    { glyph: "✓", code: "32", color: "green" },
    { glyph: "✗", code: "31", color: "red" },
    { glyph: "≈", code: "33", color: "yellow" },
  ])("colors the $glyph glyph $color when color is forced", ({ glyph, code }) => {
    vi.stubEnv("FORCE_COLOR", "1");

    const gloss = legendGlosses();
    const idx = gloss.indexOf(glyph);

    expect(gloss.slice(0, idx)).toContain(`\x1b[${code}m`);
  });
});

describe("methodFooterLines", () => {
  const formatHint = (hint: string): string => `Hint: ${hint}`;

  /** The footer's hint line, if `methodFooterLines` emitted one for these metrics. */
  function hintLineFor(metrics: Metrics): string | undefined {
    return methodFooterLines(metrics, formatHint).find((line) => line.includes("Hint:"));
  }

  afterEach(() => {
    vi.unstubAllEnvs();
  });

  it("returns lines with no ANSI escapes when stdout is not a TTY", () => {
    const metrics: Metrics = {
      "a/time": approximateMetric({ verdict: "improved", delta: -10 }),
    };

    const lines = methodFooterLines(metrics, formatHint);

    for (const line of lines) {
      expect(line).not.toContain("\x1b[");
    }
  });

  it("dims the descriptive verdict line when color is forced", () => {
    vi.stubEnv("FORCE_COLOR", "1");

    const metrics: Metrics = {
      "a/time": approximateMetric({ verdict: "improved", delta: -10 }),
    };

    const lines = methodFooterLines(metrics, formatHint);
    const verdictLine = lines.find((line) => line.includes("Wilcoxon"));

    expect(verdictLine).toContain("\x1b[2m");
  });

  it("does not dim the hint line when color is forced", () => {
    vi.stubEnv("FORCE_COLOR", "1");

    const metrics: Metrics = { "a/time": bandMetric({ n: 4 }) };

    expect(hintLineFor(metrics)).not.toContain("\x1b[2m");
  });

  describe("when all band metrics have too few samples", () => {
    it("shows the sample-shortage hint", () => {
      const metrics: Metrics = {
        "decode/time": bandMetric({ n: 3 }),
        "encode/time": bandMetric({ n: 5 }),
        "parse/time": approximateMetric({ verdict: "improved", delta: -10 }),
      };

      expect(hintLineFor(metrics)).toContain("--samples 6");
    });
  });

  describe("when all band metrics have enough samples but fell back due to ties", () => {
    it("shows no hint, since the = glyph already says the values are identical", () => {
      const metrics: Metrics = {
        "entity.alive_check/heap": bandMetric({ n: 10, usableN: 3 }),
        "iteration.soa_5field/heap": bandMetric({ n: 8, usableN: 2 }),
        "parse/time": approximateMetric({ verdict: "improved", delta: -10 }),
      };

      expect(hintLineFor(metrics)).toBeUndefined();
    });
  });

  describe("when band metrics include both sample-shortage and ties", () => {
    it("shows the sample-shortage hint alone", () => {
      const metrics: Metrics = {
        "decode/time": bandMetric({ n: 3 }),
        "entity.alive_check/heap": bandMetric({ n: 10, usableN: 3 }),
        "parse/time": approximateMetric({ verdict: "improved", delta: -10 }),
      };

      const hintLines = methodFooterLines(metrics, formatHint).filter((line) =>
        line.includes("Hint:"),
      );

      expect(hintLines).toHaveLength(1);
      expect(hintLines[0]).toContain("--samples");
    });
  });
});
