import { stripVTControlCharacters as stripAnsi } from "node:util";

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { renderReport } from "../../src/report/text.js";
import type { ComparisonResult } from "../../src/report/types.js";
import {
  bandVerdict,
  createCandidate,
  createComparisonResult,
  exactVerdict,
  geomeanOf,
  groupedComparison,
  kindMetric,
  memoryKind,
  metricMeta,
  multiCandidateResult,
  nWayKindMetric,
  otherKind,
  signedRankMetric,
  signedRankVerdict,
  singleSampleResult,
  timeKind,
  twoKindResult,
} from "../fixtures/comparison-result.js";

/** The cells of a rendered table line, padding included. */
function cellsOf(line: string): string[] {
  return line.split("│");
}

/** The single rendered line starting with `prefix`, or a failure naming the report. */
function lineStartingWith(report: string, prefix: string): string {
  const line = report.split("\n").find((candidate) => candidate.startsWith(prefix));
  if (line === undefined) {
    throw new Error(`no line starting with ${prefix} in report:\n${report}`);
  }
  return line;
}

/**
 * The first rendered line containing `needle`, or a failure naming the report.
 *
 * A colored line starts with escape codes rather than its text, so the color
 * tests match on content instead of a prefix.
 */
function lineContaining(report: string, needle: string): string {
  const line = report.split("\n").find((candidate) => candidate.includes(needle));
  if (line === undefined) {
    throw new Error(`no line containing ${needle} in report:\n${report}`);
  }
  return line;
}

/**
 * The SGR parameters opened immediately before `marker` in `line`.
 *
 * Only the unbroken run of escape sequences touching the marker counts, so a
 * style opened at the start of the line does not leak into the result.
 *
 * Pass `last` to read the trailing occurrence instead of the leading one, for
 * a marker that repeats within the line.
 */
function stylesAt(line: string, marker: string, options: { last?: boolean } = {}): string[] {
  const index = options.last === true ? line.lastIndexOf(marker) : line.indexOf(marker);
  if (index === -1) {
    throw new Error(`no ${marker} in line: ${JSON.stringify(line)}`);
  }
  const opened = /((?:\x1b\[\d+m)*)$/.exec(line.slice(0, index))?.[1] ?? "";
  return [...opened.matchAll(/\x1b\[(\d+)m/g)].map((match) => match[1] ?? "");
}

/**
 * One entry per report line, coarse enough to read as a layout.
 *
 * A table row collapses to its first cell, a column rule collapses to a marker,
 * and every other line stays as its plain text. A section's top border joins its
 * columns with top-T junctions rather than the crossings of a rule, so it gets
 * its own marker.
 */
function tableShape(report: string): string[] {
  return report.split("\n").map((line) => {
    const bare = stripAnsi(line);
    if (/^─+┼/.test(bare)) {
      return "<rule>";
    }
    if (/^[─┬]+$/.test(bare)) {
      return "<border>";
    }
    if (!bare.includes("│")) {
      return bare.trimEnd();
    }
    return cellsOf(bare)[0]?.trim() ?? "";
  });
}

/** The table region of a report: everything down to the last table row. */
function tableRegion(report: string): string[] {
  const shape = tableShape(report);
  const lines = report.split("\n");
  const last = lines.reduce(
    (found, line, index) => (stripAnsi(line).includes("│") ? index : found),
    -1,
  );
  if (last === -1) {
    throw new Error(`no table rows in report:\n${report}`);
  }
  return shape.slice(0, last + 1);
}

/** A single non-gating kind whose informational tag carries the config source. */
function flatNonGatingResult(overrides: Partial<ComparisonResult> = {}): ComparisonResult {
  return createComparisonResult({
    metrics: {
      "warmup/time": kindMetric({
        kind: "time",
        shortName: "warmup",
        verdict: "improved",
        delta: -10,
        gating: false,
      }),
      "cooldown/time": kindMetric({
        kind: "time",
        shortName: "cooldown",
        verdict: "no-signal",
        delta: 0.3,
        gating: false,
      }),
    },
    candidates: [
      createCandidate({
        kinds: [{ kind: "time", geomean: geomeanOf(-5, 2), groups: [] }],
      }),
    ],
    configKinds: { time: { gating: false } },
    ...overrides,
  });
}

beforeEach(() => {
  vi.stubEnv("NO_COLOR", "1");
  vi.stubEnv("FORCE_COLOR", undefined);
});

afterEach(() => {
  vi.unstubAllEnvs();
});

describe("renderReport", () => {
  describe("when every metric shares one kind", () => {
    /** A single gating kind whose two metrics share a group. */
    function oneKindResult(overrides: Partial<ComparisonResult> = {}): ComparisonResult {
      return createComparisonResult({
        metrics: {
          "entity.alive_check/time": kindMetric({
            kind: "time",
            shortName: "entity.alive_check",
            verdict: "improved",
            delta: -10,
          }),
          "entity.spawn/time": kindMetric({
            kind: "time",
            shortName: "entity.spawn",
            verdict: "regressed",
            delta: 4,
          }),
        },
        candidates: [
          createCandidate({
            kinds: [
              {
                kind: "time",
                geomean: geomeanOf(-3.2, 2),
                groups: [{ group: "entity", geomean: geomeanOf(-3.2, 2) }],
                gatedGeomean: geomeanOf(-3.2, 2),
              },
            ],
          }),
        ],
        ...overrides,
      });
    }

    it("keeps the flat layout, full metric names and one geomean row", () => {
      expect(tableRegion(renderReport(oneKindResult()))).toStrictEqual([
        "gymrat compare · baseline main ↔ perf/faster-decode · 10 paired samples · adapter: mitata",
        "metric",
        "<rule>",
        "entity.alive_check/time",
        "entity.spawn/time",
        "<rule>",
        "geomean (2 stable metrics)",
      ]);
    });

    it("reports no stable metrics when that kind does not gate", () => {
      const result = createComparisonResult({
        metrics: {
          "warmup/time": kindMetric({
            kind: "time",
            shortName: "warmup",
            verdict: "improved",
            delta: -10,
            gating: false,
          }),
        },
        candidates: [
          createCandidate({
            kinds: [{ kind: "time", geomean: geomeanOf(-10, 1), groups: [] }],
          }),
        ],
      });

      const row = lineStartingWith(renderReport(result), "geomean");

      expect(cellsOf(row).map((cell) => cell.trim())).toStrictEqual([
        "geomean",
        "",
        "",
        "—  no stable metrics",
      ]);
    });

    it("paints the flat geomean value by its own band, color on", () => {
      vi.stubEnv("FORCE_COLOR", "1");
      const result = oneKindResult({
        candidates: [
          createCandidate({
            kinds: [
              {
                kind: "time",
                geomean: geomeanOf(-3.2, 2),
                groups: [],
                gatedGeomean: geomeanOf(-3.2, 2, { band: 1 }),
              },
            ],
          }),
        ],
      });

      const line = lineContaining(renderReport(result), "geomean");

      expect(stylesAt(line, "-3.2%")).toStrictEqual(["1", "32"]);
    });

    describe("when the sole kind gates nothing", () => {
      it("shows the informational tag before the table header", () => {
        expect(tableRegion(renderReport(flatNonGatingResult()))).toStrictEqual([
          "gymrat compare · baseline main ↔ perf/faster-decode · 10 paired samples · adapter: mitata",
          "informational — gating off (config: kinds.time.gating = false)",
          "metric",
          "<rule>",
          "warmup/time",
          "cooldown/time",
          "<rule>",
          "geomean",
        ]);
      });

      it.each([
        {
          source: "the kind-level config entry",
          makeResult: () => flatNonGatingResult({ configKinds: { time: { gating: false } } }),
          expected: "informational — gating off (config: kinds.time.gating = false)",
        },
        {
          source: "per-metric overrides alone",
          makeResult: (): ComparisonResult => {
            const { configKinds: _, ...rest } = flatNonGatingResult();
            return rest;
          },
          expected: "informational — gating off",
        },
      ])("credits $source for the informational tag", ({ makeResult, expected }) => {
        const report = renderReport(makeResult());

        expect(lineContaining(report, "informational")).toBe(expected);
      });

      it("dims the informational tag when color is on", () => {
        vi.stubEnv("FORCE_COLOR", "1");

        const tag = lineContaining(renderReport(flatNonGatingResult()), "informational");

        expect(stylesAt(tag, "informational")).toStrictEqual(["2"]);
      });
    });
  });

  describe("when labelling the group and geomean rows", () => {
    /** A single-candidate run of one kind, whose table closes on a flat geomean row. */
    function flatResult(): ComparisonResult {
      return createComparisonResult({
        metrics: { "faster/time": signedRankMetric({ verdict: "improved", delta: -17.5 }) },
      });
    }

    describe("when color is on", () => {
      beforeEach(() => {
        vi.stubEnv("FORCE_COLOR", "1");
      });

      it.each([
        { table: "single-candidate", makeResult: twoKindResult },
        { table: "multi-candidate", makeResult: groupedComparison },
      ])("paints the group sub-header blue in the $table table", ({ makeResult }) => {
        const row = lineContaining(renderReport(makeResult()), "entity");

        expect(stylesAt(row, "entity")).toStrictEqual(["34"]);
      });

      it.each([
        {
          level: "group",
          table: "single-candidate",
          label: "geomean · entity",
          makeResult: twoKindResult,
        },
        {
          level: "kind",
          table: "single-candidate",
          label: "geomean · time",
          makeResult: twoKindResult,
        },
        { level: "flat", table: "single-candidate", label: "geomean", makeResult: flatResult },
        {
          level: "group",
          table: "multi-candidate",
          label: "geomean · entity",
          makeResult: groupedComparison,
        },
        {
          level: "kind",
          table: "multi-candidate",
          label: "geomean · time",
          makeResult: groupedComparison,
        },
        {
          level: "flat",
          table: "multi-candidate",
          label: "geomean",
          makeResult: multiCandidateResult,
        },
      ])("emboldens the $level geomean label in the $table table", ({ label, makeResult }) => {
        const row = lineContaining(renderReport(makeResult()), label);

        expect(stylesAt(row, label)).toStrictEqual(["1"]);
      });
    });
  });

  describe("when every metric behind a geomean landed within noise", () => {
    /**
     * A two-kind run whose every metric landed within noise.
     *
     * Each geomean figure sits far outside its own band, so a rule that reads
     * the band alone would paint all of them green.
     */
    function quietTwoKindResult(): ComparisonResult {
      return createComparisonResult({
        metrics: {
          "entity.alive_check/time": kindMetric({
            kind: "time",
            shortName: "entity.alive_check",
            verdict: "no-signal",
            delta: -9,
          }),
          "entity.spawn/time": kindMetric({
            kind: "time",
            shortName: "entity.spawn",
            verdict: "no-signal",
            delta: -8,
          }),
          "encode/heap": kindMetric({
            kind: "memory",
            shortName: "encode",
            verdict: "no-signal",
            delta: -7,
            gating: false,
            unit: "bytes",
          }),
        },
        candidates: [
          createCandidate({
            kinds: [
              timeKind({
                geomean: geomeanOf(-8.5, 2),
                groups: [{ group: "entity", geomean: geomeanOf(-8.6, 2) }],
                gatedGeomean: geomeanOf(-8.5, 2),
              }),
              memoryKind({ geomean: geomeanOf(-7, 1) }),
            ],
          }),
        ],
        configKinds: { memory: { gating: false } },
      });
    }

    describe("when color is on", () => {
      beforeEach(() => {
        vi.stubEnv("FORCE_COLOR", "1");
      });

      it.each([
        { level: "group", label: "geomean · entity", value: "-8.6%" },
        { level: "kind", label: "geomean · time", value: "-8.5%" },
      ])("leaves the $level geomean value emboldened and uncolored", ({ label, value }) => {
        const line = lineContaining(renderReport(quietTwoKindResult()), label);

        expect(stylesAt(line, value)).toStrictEqual(["1"]);
      });

      it("leaves the flat geomean value emboldened and uncolored", () => {
        const result = createComparisonResult({
          metrics: { "faster/time": signedRankMetric({ verdict: "no-signal", delta: -0.5 }) },
        });

        const line = lineContaining(renderReport(result), "geomean");

        expect(stylesAt(line, "-5.8%")).toStrictEqual(["1"]);
      });

      it("judges each candidate column by that column's own verdicts", () => {
        const result = createComparisonResult({
          metrics: {
            "entity.alive_check/time": nWayKindMetric({
              kind: "time",
              shortName: "entity.alive_check",
              candidates: [
                { verdict: "no-signal", delta: -9, median: 91 },
                { verdict: "improved", delta: -12, median: 88 },
              ],
            }),
            "encode/heap": nWayKindMetric({
              kind: "memory",
              shortName: "encode",
              gating: false,
              candidates: [
                { verdict: "no-signal", delta: -1, median: 99 },
                { verdict: "improved", delta: -2, median: 98 },
              ],
            }),
          },
          candidates: [
            createCandidate({
              label: "candidate-a",
              kinds: [
                timeKind({
                  geomean: geomeanOf(-9, 1),
                  groups: [{ group: "entity", geomean: geomeanOf(-9, 1) }],
                  gatedGeomean: geomeanOf(-9, 1),
                }),
                memoryKind({ geomean: geomeanOf(-1, 1) }),
              ],
            }),
            createCandidate({
              label: "candidate-b",
              kinds: [
                timeKind({
                  geomean: geomeanOf(-12, 1),
                  groups: [{ group: "entity", geomean: geomeanOf(-12, 1) }],
                  gatedGeomean: geomeanOf(-12, 1),
                }),
                memoryKind({ geomean: geomeanOf(-2, 1) }),
              ],
            }),
          ],
          configKinds: { memory: { gating: false } },
        });

        const line = lineContaining(renderReport(result), "geomean · time");

        expect.soft(stylesAt(line, "-9.0%")).toStrictEqual(["1"]);
        expect(stylesAt(line, "-12.0%")).toStrictEqual(["1", "32"]);
      });
    });
  });

  /**
   * Byte-level pins on the whole rendered report.
   *
   * Every other assertion in this file checks a substring or a separator offset,
   * which leaves column widths and inter-cell spacing free to drift while the
   * suite stays green. The report is the product's interface, so that drift is a
   * user-visible output change — these two snapshots are what makes it fail a run
   * instead of shipping.
   *
   * A diff here is never cosmetic. Re-record only when changing the report is the
   * intended change; a refactor that moves these bytes has changed behavior,
   * whatever its intent.
   */
  describe("when rendering a whole report", () => {
    it("matches the recorded bytes for a representative run", async () => {
      const result = createComparisonResult({
        metrics: {
          "decode/text=digits/time": {
            baselineMedian: 1735,
            baselineSpread: 1,
            candidates: [
              {
                median: 1425,
                spread: 1,
                verdict: signedRankVerdict({ verdict: "improved", delta: -17.9, p: 0.002 }),
              },
            ],
            meta: metricMeta("decode/text=digits/time", { unit: "ns" }),
          },
          "decode/text=words/time": {
            baselineMedian: 3065,
            baselineSpread: 1,
            candidates: [
              {
                median: 3093,
                spread: 3,
                verdict: signedRankVerdict({ delta: 0.9, p: 0.49 }),
              },
            ],
            meta: metricMeta("decode/text=words/time", { unit: "ns" }),
          },
          "encode/time": {
            baselineMedian: 914,
            baselineSpread: 1,
            candidates: [
              {
                median: 934,
                spread: 1,
                verdict: signedRankVerdict({ verdict: "regressed", delta: 2.2, p: 0.002 }),
              },
            ],
            meta: metricMeta("encode/time", { unit: "ns" }),
          },
          "encode/heap": {
            baselineMedian: 49152,
            baselineSpread: 0,
            candidates: [
              {
                median: 45261,
                spread: 0,
                verdict: exactVerdict({ verdict: "improved", delta: -7.9 }),
              },
            ],
            meta: metricMeta("encode/heap", { exact: true, unit: "bytes" }),
          },
        },
        candidates: [
          createCandidate({
            kinds: [otherKind(-6, 4)],
          }),
        ],
      });

      await expect(renderReport(result)).toMatchFileSnapshot(
        "../fixtures/report-representative.golden.txt",
      );
    });

    function degenerateResult(): ComparisonResult {
      return createComparisonResult({
        samples: 4,
        adapter: "metric-lines",
        metrics: {
          "zero-median/time": {
            baselineMedian: 0,
            candidates: [
              {
                median: 0,
                verdict: exactVerdict({ n: 4 }),
              },
            ],
            meta: metricMeta("zero-median/time", { exact: true, unit: "ns" }),
          },
          "nan-delta/count": {
            baselineMedian: 0,
            candidates: [
              {
                median: 120,
                verdict: exactVerdict({ delta: Number.NaN, n: 4 }),
              },
            ],
            meta: metricMeta("nan-delta/count", { exact: true }),
          },
          "old-side-only/time": {
            baselineMedian: 2048,
            baselineSpread: 2,
            candidates: [{}],
            meta: metricMeta("old-side-only/time", { unit: "ns" }),
          },
          "throughput/ops": {
            baselineMedian: 1200,
            baselineSpread: 5,
            candidates: [
              {
                median: 1560,
                spread: 4,
                verdict: bandVerdict({ verdict: "improved", delta: 30, n: 4, usableN: 4 }),
              },
            ],
            meta: metricMeta("throughput/ops", { direction: "higher", gating: false }),
          },
        },
        candidates: [
          createCandidate({
            kinds: [
              otherKind(0, 1, {
                excluded: [{ metric: "nan-delta/count", reason: "undefined-ratio" }],
              }),
            ],
          }),
        ],
        worktreesRemoved: 1,
        worktreesLeftBehind: [{ dir: "/tmp/gymrat-abc123", error: "contains modified files" }],
        worktreePruneError: "could not lock config file",
      });
    }

    it("matches the recorded bytes for degenerate inputs and a dirty cleanup", async () => {
      await expect(renderReport(degenerateResult())).toMatchFileSnapshot(
        "../fixtures/report-degenerate.golden.txt",
      );
    });

    function twoCandidateResult(): ComparisonResult {
      return createComparisonResult({
        candidates: [
          createCandidate({
            label: "perf/simd-decode",
            // Both geomeans sit inside their noise band, so the colored twin of
            // this golden pins the styling of a geomean that stays there.
            kinds: [otherKind(-12.4, 3, { band: 30 })],
          }),
          createCandidate({
            label: "perf/lut-decode",
            kinds: [
              otherKind(1.2, 2, {
                excluded: [{ metric: "encode/time", reason: "unstable" }],
                band: 30,
              }),
            ],
          }),
        ],
        metrics: {
          "decode/text=digits/time": {
            baselineMedian: 1735,
            baselineSpread: 1,
            candidates: [
              {
                median: 1425,
                spread: 1,
                verdict: signedRankVerdict({ verdict: "improved", delta: -17.9, p: 0.002 }),
              },
              {
                median: 1698,
                spread: 2,
                verdict: signedRankVerdict({ delta: -2.1, p: 0.32 }),
              },
            ],
            meta: metricMeta("decode/text=digits/time", { unit: "ns" }),
          },
          "encode/time": {
            baselineMedian: 914,
            baselineSpread: 1,
            candidates: [
              {
                median: 934,
                spread: 1,
                verdict: signedRankVerdict({ verdict: "regressed", delta: 2.2, p: 0.002 }),
              },
              {
                median: 1200,
                spread: 12,
                // The band method only runs below six pairs, so this metric was
                // dropped on most rounds — which also pins the n= annotation in
                // an N-way cell.
                verdict: bandVerdict({
                  verdict: "unstable",
                  delta: 31.3,
                  n: 4,
                  usableN: 4,
                  noisePct: 30,
                  noiseAbs: 30,
                }),
              },
            ],
            meta: metricMeta("encode/time", { unit: "ns" }),
          },
          "encode/heap": {
            baselineMedian: 49152,
            baselineSpread: 0,
            // The second candidate never reported this metric, so its cell stays empty.
            candidates: [
              {
                median: 45261,
                spread: 0,
                verdict: exactVerdict({ verdict: "improved", delta: -7.9 }),
              },
              {},
            ],
            meta: metricMeta("encode/heap", { exact: true, unit: "bytes" }),
          },
        },
      });
    }

    it("matches the recorded bytes for a verbose run with two candidates", async () => {
      await expect(renderReport(twoCandidateResult(), { verbose: true })).toMatchFileSnapshot(
        "../fixtures/report-two-candidates.golden.txt",
      );
    });

    it("matches the recorded bytes for a run split into kind sections", async () => {
      await expect(renderReport(twoKindResult())).toMatchFileSnapshot(
        "../fixtures/report-sectioned.golden.txt",
      );
    });

    it("matches the recorded bytes for a run of one paired sample", async () => {
      await expect(renderReport(singleSampleResult())).toMatchFileSnapshot(
        "../fixtures/report-single-sample.golden.txt",
      );
    });

    it("matches the recorded bytes for a colored run of one paired sample", async () => {
      vi.stubEnv("FORCE_COLOR", "1");

      await expect(renderReport(singleSampleResult())).toMatchFileSnapshot(
        "../fixtures/report-single-sample-color.golden.txt",
      );
    });

    it("matches the recorded bytes for a representative colored run", async () => {
      vi.stubEnv("FORCE_COLOR", "1");

      const result = createComparisonResult({
        metrics: {
          "faster/time": signedRankMetric({ verdict: "improved", delta: -17.5, unit: "ns" }),
          "slower/time": signedRankMetric({ verdict: "regressed", delta: 2.4, unit: "ns" }),
          "flat/time": signedRankMetric({ verdict: "no-signal", delta: 0.3, unit: "ns" }),
          "jittery/time": signedRankMetric({ verdict: "unstable", delta: -50, noisePct: 30 }),
        },
        candidates: [
          createCandidate({
            // Inside its noise band, so this golden pins the styling of a
            // geomean that never crosses it.
            kinds: [
              otherKind(-5.8, 3, {
                excluded: [{ metric: "jittery/time", reason: "unstable" }],
                band: 30,
              }),
            ],
          }),
        ],
        worktreesRemoved: 1,
        worktreesLeftBehind: [{ dir: "/tmp/gymrat-abc", error: "is locked" }],
        worktreePruneError: "fatal: not a git repository",
      });

      await expect(renderReport(result)).toMatchFileSnapshot(
        "../fixtures/report-representative-color.golden.txt",
      );
    });

    it("matches the recorded bytes for a verbose degenerate colored run", async () => {
      vi.stubEnv("FORCE_COLOR", "1");

      await expect(renderReport(degenerateResult(), { verbose: true })).toMatchFileSnapshot(
        "../fixtures/report-degenerate-color.golden.txt",
      );
    });

    it("matches the recorded bytes for a verbose two-candidate colored run", async () => {
      vi.stubEnv("FORCE_COLOR", "1");

      await expect(renderReport(twoCandidateResult(), { verbose: true })).toMatchFileSnapshot(
        "../fixtures/report-two-candidates-color.golden.txt",
      );
    });

    it("matches the recorded bytes for a colored run split into kind sections", async () => {
      vi.stubEnv("FORCE_COLOR", "1");

      await expect(renderReport(twoKindResult())).toMatchFileSnapshot(
        "../fixtures/report-sectioned-color.golden.txt",
      );
    });

    it("matches the recorded bytes for a flat non-gating run", async () => {
      await expect(renderReport(flatNonGatingResult())).toMatchFileSnapshot(
        "../fixtures/report-flat-non-gating.golden.txt",
      );
    });

    it("matches the recorded bytes for a flat non-gating colored run", async () => {
      vi.stubEnv("FORCE_COLOR", "1");

      await expect(renderReport(flatNonGatingResult())).toMatchFileSnapshot(
        "../fixtures/report-flat-non-gating-color.golden.txt",
      );
    });
  });
});
