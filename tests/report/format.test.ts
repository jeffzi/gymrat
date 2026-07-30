import { describe, it, expect } from "vitest";

import {
  computeColumnWidth,
  formatDelta,
  formatLabel,
  formatPValue,
  formatSpread,
  formatTableLine,
  formatValue,
  getGlyph,
} from "../../src/report/format.js";

describe("formatValue", () => {
  describe("when the metric is measured in nanoseconds", () => {
    it.each([
      { tier: "n", value: 0, expected: "0n" },
      { tier: "n", value: 914, expected: "914n" },
      { tier: "n", value: 999, expected: "999n" },
      { tier: "µ", value: 1000, expected: "1.0µ" },
      { tier: "µ", value: 1735, expected: "1.735µ" },
      { tier: "m", value: 1_000_000, expected: "1.0m" },
      { tier: "s", value: 2_000_000_000, expected: "2.0s" },
    ])("scales $value to the $tier tier as $expected", ({ value, expected }) => {
      expect(formatValue(value, "ns")).toBe(expected);
    });
  });

  describe("when the metric is measured in bytes", () => {
    it.each([
      { tier: "raw", value: 512, expected: "512" },
      { tier: "raw", value: 999, expected: "999" },
      { tier: "k", value: 1000, expected: "1.0k" },
      { tier: "k", value: 49152, expected: "49.2k" },
      { tier: "M", value: 1_000_000, expected: "1.0M" },
      { tier: "G", value: 2_000_000_000, expected: "2.0G" },
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
  ])("marks $verdict with $expected", ({ verdict, expected }) => {
    expect(getGlyph(verdict)).toBe(expected);
  });
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
  it("pads every cell to its column width and separates them with a bar", () => {
    const line = formatTableLine(["metric", "old", "new"], [10, 6, 6]);

    expect(line).toBe("metric    │old   │new");
  });

  it("pads interior cells that are empty so later columns stay aligned", () => {
    const line = formatTableLine(["geomean", "", "", "-6.0%"], [10, 6, 6, 8]);

    expect(line).toBe("geomean   │      │      │-6.0%");
  });

  it("trims the padding after a trailing empty cell", () => {
    const line = formatTableLine(["one-sided", "2.048µ ± 2%", ""], [12, 14, 10]);

    expect(line).toBe("one-sided   │2.048µ ± 2%   │");
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
