import { afterEach, describe, expect, expectTypeOf, it, vi } from "vitest";

import {
  countVerdicts,
  displayClass,
  footerLines,
  formatDelta,
  formatEvidence,
  formatValue,
  geomeanValueStyle,
  getGlyph,
  scopedGeomeanLabel,
  selectHighlights,
  verdictSummaryParts,
} from "../../src/report/format.js";
import type { ReportOptions } from "../../src/report/types.js";
import type { ApproximateVerdictValue, GeomeanResult } from "../../src/verdict/verdict.js";
import {
  bandMetric,
  bandVerdict,
  exactVerdict,
  geomeanOf,
  signedRankVerdict,
  type Metrics,
  type MetricEntry,
} from "../fixtures/comparison-result.js";

/** Stubs the environment so `styleText` emits ANSI codes. */
function forceColor(): void {
  vi.stubEnv("FORCE_COLOR", "1");
}

/** Stubs the environment so `styleText` never emits ANSI codes. */
function suppressColor(): void {
  vi.stubEnv("FORCE_COLOR", undefined);
  vi.stubEnv("NO_COLOR", "1");
}

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
    meta: { direction, gating: true, exact: false, kind: "other", shortName: "time" },
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
    meta: { direction: "lower", gating: false, exact: false, kind: "other", shortName: "time" },
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
      // A value that rounds up onto a tier boundary belongs to the tier above:
      // printing it below leaves a four-digit magnitude in a three-digit column.
      { tier: "µs", value: 999.5, expected: "1.0µs" },
      { tier: "ms", value: 999_999.6, expected: "1.0ms" },
      { tier: "s", value: 999_950_000, expected: "1.0s" },
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
      // A value that rounds up onto a tier boundary belongs to the tier above:
      // printing it below leaves a four-digit magnitude in a three-digit column.
      { tier: "KB", value: 999.5, expected: "1.0KB" },
      { tier: "MB", value: 999_950, expected: "1.0MB" },
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

  describe("when the value is negative", () => {
    // A negative reading belongs to the tier its magnitude names: printing it
    // raw leaves a seven-digit cell where the column budgets four.
    it.each([
      { unit: "bytes" as const, value: -512, expected: "-512B" },
      { unit: "bytes" as const, value: -3600, expected: "-3.6KB" },
      { unit: "bytes" as const, value: -1_500_000, expected: "-1.5MB" },
      { unit: "ns" as const, value: -1735, expected: "-1.7µs" },
      { unit: "ns" as const, value: -2_000_000_000, expected: "-2.0s" },
      // Rounding onto a tier boundary promotes the magnitude, sign aside.
      { unit: "bytes" as const, value: -999.5, expected: "-1.0KB" },
    ])(
      "picks the tier by magnitude, rendering $value $unit as $expected",
      ({ unit, value, expected }) => {
        expect(formatValue(value, unit)).toBe(expected);
      },
    );
  });

  describe("when the value is non-finite", () => {
    it.each([
      { value: Infinity, unit: "ns" as const, expected: "Infinity" },
      { value: -Infinity, unit: "bytes" as const, expected: "-Infinity" },
      { value: NaN, unit: "ns" as const, expected: "NaN" },
    ])("renders $expected for $value with unit $unit", ({ value, unit, expected }) => {
      expect(formatValue(value, unit)).toBe(expected);
    });
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
    { desc: "fractional", noisePct: 2.5, expected: "noise ±2.5%" },
    { desc: "whole number", noisePct: 2, expected: "noise ±2.0%" },
    { desc: "sub-percent", noisePct: 0.5, expected: "noise ±0.5%" },
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

describe("displayClass", () => {
  it.each([
    {
      desc: "every pair tied, leaving the signed-rank test nothing to work with",
      verdict: bandVerdict({ n: 10, usableN: 0 }),
      expected: "identical",
    },
    {
      desc: "ties left some pairs usable, short of what the signed-rank test needs",
      verdict: bandVerdict({ n: 10, usableN: 3 }),
      expected: "within-noise",
    },
    {
      desc: "ties left exactly enough usable pairs",
      verdict: bandVerdict({ n: 10, usableN: 6 }),
      expected: "within-noise",
    },
    {
      desc: "the pair count sits one below the signed-rank floor",
      verdict: bandVerdict({ n: 6, usableN: 5 }),
      expected: "within-noise",
    },
    {
      desc: "the run was too short to reach the floor at all",
      verdict: bandVerdict({ n: 5, usableN: 5 }),
      expected: "within-noise",
    },
    {
      desc: "one pair left the band nothing but its own floor to judge against",
      verdict: bandVerdict({ n: 1, usableN: 1 }),
      expected: "inconclusive",
    },
    {
      // Two identical readings are worth a `=`; one reading is worth nothing.
      desc: "the run's single pair was a tie",
      verdict: bandVerdict({ n: 1, usableN: 0 }),
      expected: "inconclusive",
    },
    {
      desc: "a second pair gave the band a spread to measure",
      verdict: bandVerdict({ n: 2, usableN: 2 }),
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
    { displayClass: "inconclusive" as const, expected: "?" },
  ])("marks $displayClass with $expected", ({ displayClass: shown, expected }) => {
    expect(getGlyph(shown)).toBe(expected);
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

  it("leaves metrics resting on a single pair out, the way it leaves within-noise ones out", () => {
    const metrics: Metrics = {
      "faster/time": approximateMetric({ verdict: "improved", delta: -10 }),
      "single-pair/time": bandMetric({ n: 1, noisePct: 0.5 }),
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

/** Shorthand matching the local convention: default value=0, n=2, no band. */
const geomeanResult = (overrides: Partial<GeomeanResult> = {}): GeomeanResult =>
  geomeanOf(0, 2, overrides);

describe("scopedGeomeanLabel", () => {
  it.each([
    {
      desc: "nothing was excluded",
      scope: "entity",
      geomean: geomeanResult({ n: 1 }),
      expected: "geomean · entity (1)",
    },
    {
      desc: "the subset lost a metric to an exclusion",
      scope: "entity",
      geomean: geomeanResult({
        n: 1,
        excluded: [{ metric: "entity.spawn/time", reason: "unstable" }],
      }),
      expected: "geomean · entity (1/2)",
    },
    {
      desc: "the subset lost several metrics",
      scope: "memory",
      geomean: geomeanResult({
        n: 13,
        excluded: [
          { metric: "a/heap", reason: "unstable" },
          { metric: "b/heap", reason: "undefined-ratio" },
        ],
      }),
      expected: "geomean · memory (13/15)",
    },
  ])("counts the subset behind the scope when $desc", ({ scope, geomean, expected }) => {
    expect(scopedGeomeanLabel(scope, geomean)).toBe(expected);
  });
});

describe("geomeanValueStyle", () => {
  it.each([
    {
      desc: "an improvement past the band",
      geomean: geomeanResult({ value: -6, band: 5 }),
      expected: ["bold", "green"],
    },
    {
      desc: "a regression past the band",
      geomean: geomeanResult({ value: 6, band: 5 }),
      expected: ["bold", "red"],
    },
    {
      desc: "an improvement inside the band",
      geomean: geomeanResult({ value: -4, band: 5 }),
      expected: ["bold"],
    },
    {
      desc: "a regression inside the band",
      geomean: geomeanResult({ value: 4, band: 5 }),
      expected: ["bold"],
    },
    {
      // The band is the width of the noise, so a value level with it is noise.
      desc: "a value level with the band",
      geomean: geomeanResult({ value: -5, band: 5 }),
      expected: ["bold"],
    },
    {
      desc: "an improvement over a run with no band at all",
      geomean: geomeanResult({ value: -0.2, band: 0 }),
      expected: ["bold", "green"],
    },
    {
      desc: "a geomean with no stable metrics behind it",
      geomean: geomeanResult({ value: Number.NaN, n: 0 }),
      expected: ["bold"],
    },
  ])("styles $desc", ({ geomean, expected }) => {
    expect(geomeanValueStyle(geomean)).toStrictEqual(expected);
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
    "tied/heap": bandMetric({ n: 10, usableN: 0 }),
    "single-pair/time": bandMetric({ n: 1, noisePct: 0.5 }),
  };

  it("returns parts with no ANSI escapes when color is suppressed", () => {
    suppressColor();
    const parts = verdictSummaryParts(mixed, 0);

    expect(parts.join("")).not.toContain("\x1b[");
  });

  it("tallies identical and single-pair metrics apart from the ones within noise", () => {
    suppressColor();
    const parts = verdictSummaryParts(mixed, 0);

    expect.soft(parts.find((p) => p.includes("identical"))).toBe("= 1 identical");
    expect.soft(parts.find((p) => p.includes("inconclusive"))).toBe("? 1 inconclusive");
    expect(parts.find((p) => p.includes("within noise"))).toBe("~ 1 within noise");
  });

  it.each([
    { label: "improved", code: "32", color: "green" },
    { label: "regressed", code: "31", color: "red" },
    { label: "unstable", code: "33", color: "yellow" },
    { label: "identical", code: "36", color: "cyan" },
  ])("styles the non-zero $label part $color when color is forced", ({ label, code }) => {
    forceColor();

    const parts = verdictSummaryParts(mixed, 0);
    const part = parts.find((p) => p.includes(label));

    expect(part).toContain(`\x1b[${code}m`);
  });

  it.each([{ label: "regressed" }, { label: "identical" }])(
    "dims the zero-count $label part when color is forced",
    ({ label }) => {
      forceColor();

      const onlyImproved: Metrics = {
        "faster/time": approximateMetric({ verdict: "improved", delta: -10 }),
      };

      const parts = verdictSummaryParts(onlyImproved, 0);
      const part = parts.find((p) => p.includes(label));

      expect(part).toContain("\x1b[2m");
    },
  );

  it("dims the within-noise part regardless of count when color is forced", () => {
    forceColor();

    const parts = verdictSummaryParts(mixed, 0);
    const noisePart = parts.find((p) => p.includes("within noise"));

    expect(noisePart).toContain("\x1b[2m");
  });
});

describe("footerLines (verbose method lines)", () => {
  const noHint = (hint: string): string => `Hint: ${hint}`;

  /** The verbose method lines, with no hint contribution. */
  function verboseLines(metrics: Metrics): string[] {
    return footerLines(metrics, true, noHint).filter((line) => !line.startsWith("Hint"));
  }

  /** The lines describing the noise-band fallback, in the order they were emitted. */
  function bandLinesFor(metrics: Metrics): string[] {
    return verboseLines(metrics).filter((line) => line.startsWith("noise band"));
  }

  afterEach(() => {
    vi.unstubAllEnvs();
  });

  it("returns lines with no ANSI escapes when color is suppressed", () => {
    suppressColor();
    const metrics: Metrics = {
      "a/time": approximateMetric({ verdict: "improved", delta: -10 }),
    };

    const lines = verboseLines(metrics);

    expect(lines.length).toBeGreaterThan(0);
    for (const line of lines) {
      expect(line).not.toContain("\x1b[");
    }
  });

  it("dims the descriptive verdict line when color is forced", () => {
    forceColor();

    const metrics: Metrics = {
      "a/time": approximateMetric({ verdict: "improved", delta: -10 }),
    };

    const lines = verboseLines(metrics);
    const verdictLine = lines.find((line) => line.includes("Wilcoxon"));

    expect(verdictLine).toContain("\x1b[2m");
  });

  it("leaves the sample-shortage hint to the non-verbose output", () => {
    const metrics: Metrics = { "a/time": bandMetric({ n: 4 }) };

    expect(verboseLines(metrics).join("\n")).not.toContain("Hint");
  });

  const bandCases: { cause: string; metrics: Metrics; expected: string[] }[] = [
    {
      cause: "the run was too short to reach the signed-rank floor",
      metrics: {
        "decode/time": bandMetric({ n: 3 }),
        "encode/time": bandMetric({ n: 5 }),
      },
      expected: ["noise band ±(half-range × K) — n=5 below signed-rank floor (6 pairs)"],
    },
    {
      cause: "ties starved a long-enough run",
      metrics: {
        "entity.alive_check/heap": bandMetric({ n: 10, usableN: 3 }),
        "iteration.soa_5field/heap": bandMetric({ n: 8, usableN: 2 }),
      },
      expected: ["noise band ±(half-range × K) — ties left n=2 usable pairs (6 needed)"],
    },
    {
      cause: "each cause struck a different metric",
      metrics: {
        "decode/time": bandMetric({ n: 3 }),
        "tied/heap": bandMetric({ n: 10, usableN: 3 }),
      },
      expected: [
        "noise band ±(half-range × K) — n=3 below signed-rank floor (6 pairs)",
        "noise band ±(half-range × K) — ties left n=3 usable pairs (6 needed)",
      ],
    },
  ];

  it.each(bandCases)("phrases the band line by cause when $cause", ({ metrics, expected }) => {
    suppressColor();
    expect(bandLinesFor(metrics)).toStrictEqual(expected);
  });
});

describe("footerLines (hint lines)", () => {
  const formatHint = (hint: string): string => `Hint: ${hint}`;

  afterEach(() => {
    vi.unstubAllEnvs();
  });

  it("does not dim the hint line when color is forced", () => {
    forceColor();

    const metrics: Metrics = { "a/time": bandMetric({ n: 4 }) };

    expect(footerLines(metrics, false, formatHint)[0]).not.toContain("\x1b[2m");
  });

  const hintCases: { desc: string; metrics: Metrics; expected: string[] }[] = [
    {
      desc: "every band metric ran short of pairs",
      metrics: {
        "decode/time": bandMetric({ n: 3 }),
        "encode/time": bandMetric({ n: 5 }),
        "parse/time": approximateMetric({ verdict: "improved", delta: -10 }),
      },
      expected: ["Hint: re-run with --samples 6 or more for statistical verdicts"],
    },
    {
      desc: "ties alone starved the signed-rank test",
      metrics: {
        "entity.alive_check/heap": bandMetric({ n: 10, usableN: 3 }),
        "iteration.soa_5field/heap": bandMetric({ n: 8, usableN: 2 }),
        "parse/time": approximateMetric({ verdict: "improved", delta: -10 }),
      },
      expected: [],
    },
    {
      desc: "a shortage and ties struck different metrics",
      metrics: {
        "decode/time": bandMetric({ n: 3 }),
        "entity.alive_check/heap": bandMetric({ n: 10, usableN: 3 }),
        "parse/time": approximateMetric({ verdict: "improved", delta: -10 }),
      },
      expected: ["Hint: re-run with --samples 6 or more for statistical verdicts"],
    },
    {
      desc: "the signed-rank test carried every metric",
      metrics: {
        "parse/time": approximateMetric({ verdict: "improved", delta: -10 }),
      },
      expected: [],
    },
  ];

  it.each(hintCases)("hints when $desc", ({ metrics, expected }) => {
    expect(footerLines(metrics, false, formatHint)).toStrictEqual(expected);
  });
});

describe("ReportOptions", () => {
  it("carries an optional color override", () => {
    expectTypeOf<ReportOptions>().toHaveProperty("color").toEqualTypeOf<boolean | undefined>();
  });
});
